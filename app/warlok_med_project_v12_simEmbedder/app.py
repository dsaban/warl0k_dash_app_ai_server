"""
# app1.py — rebuilt from scratch (v8), includes:
# - Upload docs
# - Build index
# - Ask (QA)
# - Patient record layer (BM25-only, NO SBERT)
# - Pack viewer
#
# Design goals:
# - No domain/ontology hardcode in engine logic
# - Patient tab NEVER returns "empty" silently: shows diagnostics and always returns top-k best-effort matches
#
# Assumptions (based on your project):
# - engine.domain_pack.DomainPack exists
# - engine.pipeline.build_all / ensure_dirs / save_uploaded_docs exist (or you can adapt)
# - engine.qa.answer exists
# - index_dir contains claims.jsonl after build
#
# If your paths differ, adjust ROOT/PACKS_DIR defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import streamlit as st

from engine.domain_pack import DomainPack
from engine.qa import answer

# pipeline helpers (your codebase already has these)
from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all

# Patient layer: BM25-only
from engine.patient_record import PatientRecord
from engine.patient_match_bm25 import (
    load_patient_layer,
    extract_fields_from_notes,
    build_surveillance,
    expand_question_with_patient_context,
)
from engine.retrieve import load_claims
from engine.bm25 import BM25

from engine.patient_state import normalize_patient_snapshot
from engine.triggers import run_triggers

from engine.trigger_evidence import attach_evidence_to_triggers

from engine.population_store import ingest_patient_record, list_latest_by_patient_episode, write_action
from engine.population_metrics import compute_kpis, build_care_gap_queue, patient_timeline

# ----------------------------
# Paths / defaults
# ----------------------------
ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "docs"
INDEX_DIR = ROOT / "data" / "index"
PACKS_DIR = ROOT / "domain_packs"  # contains multiple packs (each has manifest.json)


# ----------------------------
# Utilities
# ----------------------------
def _safe_read_json(path: Path, default):
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return default


def _list_pack_dirs(packs_dir: Path) -> List[Path]:
    if not packs_dir.exists():
        return []
    out = []
    for p in packs_dir.iterdir():
        if p.is_dir() and (p / "manifest.json").exists():
            out.append(p)
    return sorted(out)


def _load_domain(pack_dir: Path) -> DomainPack:
    return DomainPack.load(pack_dir)


def _human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _concept_phrase_rows(patient_layer: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows = []
    texts = []
    for c in (patient_layer.get("concepts") or []):
        cid = c.get("id")
        label = c.get("label", cid)
        edges_hint = c.get("edges_hint") or []
        for ph in (c.get("phrases") or []):
            rows.append(
                {"concept_id": cid, "label": label, "phrase": ph, "edges_hint": edges_hint}
            )
            texts.append(ph)
    return rows, texts


def _patient_query(rec: PatientRecord, patient_layer: Dict[str, Any]) -> str:
    """
    BM25 needs lexical overlap.
    We add pack-provided anchors + extracted field keys to ensure overlap.
    """
    parts = []
    if rec.notes:
        parts.append(rec.notes.strip())

    for k, v in (rec.structured or {}).items():
        parts.append(f"{k}: {v}")

    for k in (rec.extracted or {}).keys():
        parts.append(str(k))

    for t in (patient_layer.get("default_query_terms") or []):
        parts.append(str(t))

    return " | ".join([p for p in parts if str(p).strip()]).strip()


def _bm25_topk_best_effort(docs: List[str], query: str, k: int) -> List[Tuple[int, float]]:
    """
    Always returns top-k indices, even if all scores are zero.
    This prevents empty results and makes debugging clear.
    """
    bm = BM25(docs)
    scores = bm.score(query)
    idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: max(1, k)]
    return [(i, float(scores[i])) for i in idx]


def _patient_match_concepts_bm25(rec: PatientRecord, patient_layer: Dict[str, Any], topk: int = 8):
    rows, texts = _concept_phrase_rows(patient_layer)
    if not texts:
        return []
    q = _patient_query(rec, patient_layer)
    hits = _bm25_topk_best_effort(texts, q, k=min(topk, len(texts)))
    out = []
    for idx, score in hits:
        r = rows[idx]
        out.append(
            {
                "concept_id": r["concept_id"],
                "label": r["label"],
                "phrase": r["phrase"],
                "score": score,
                "edges_hint": ", ".join(r.get("edges_hint") or []),
            }
        )
    return out, q


def _patient_match_claims_bm25(rec: PatientRecord, patient_layer: Dict[str, Any], index_dir: Path, topk: int = 12):
    claims = load_claims(index_dir)
    if not claims:
        return []
    texts = [c.text for c in claims]
    q = _patient_query(rec, patient_layer)
    hits = _bm25_topk_best_effort(texts, q, k=min(topk, len(texts)))
    out = []
    for idx, score in hits:
        c = claims[idx]
        out.append(
            {
                "claim_id": c.id,
                "score": score,
                "edges": ", ".join(c.edge_types or []),
                "doc": c.doc,
                "text": c.text,
            }
        )
    return out, q


# ----------------------------
# Streamlit App
# ----------------------------
def main():
    st.set_page_config(page_title="WARLOK Med ChainGraph (v12)", layout="wide")

    st.title("WARLOK Medical ChainGraph (v12) with Population Surveillance")
    st.caption("Evidence-first, frame-constrained, auditable medical Q/A + patient record mapping (BM25-only).")

    ensure_dirs(ROOT)

    # Pack selection
    pack_dirs = _list_pack_dirs(PACKS_DIR)
    if not pack_dirs:
        st.error(f"No domain packs found in: {PACKS_DIR} (expected subfolders with manifest.json)")
        st.stop()

    pack_names = [p.name for p in pack_dirs]
    default_pack = pack_names[0]
    pack_name = st.sidebar.selectbox("Domain Pack", pack_names, index=pack_names.index(default_pack))
    pack_dir = PACKS_DIR / pack_name

    try:
        domain = _load_domain(pack_dir)
    except Exception as e:
        st.error(f"Pack load error: {e}")
        st.stop()

    # Tabs
    tab_upload, tab_build, tab_ask, tab_patient, tab_population, tab_pack = st.tabs(
        ["1) Upload docs",
         "2) Build index",
         "3) Ask",
         "4) Patient record (BM25)",
         "5) Population surveillance",
         "6) Pack viewer"]
    )
    
    # tab_upload, tab_build, tab_ask, tab_patient, tab_pack = st.tabs(
    #     ["1) Upload docs", "2) Build index", "3) Ask", "4) Patient record (BM25)", "5) Pack viewer"]
    # )

    # ----------------------------
    # 1) Upload docs
    # ----------------------------
    with tab_upload:
        st.subheader("Upload domain documents (.txt)")
        st.write(f"Docs folder: `{DOCS_DIR}`")

        uploaded = st.file_uploader(
            "Upload one or more .txt files",
            type=["txt"],
            accept_multiple_files=True,
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Save uploaded docs"):
                if not uploaded:
                    st.warning("No files selected.")
                else:
                    saved = save_uploaded_docs(DOCS_DIR, uploaded)
                    st.success(f"Saved {saved} file(s) into {DOCS_DIR}")

        with c2:
            if st.button("Clear docs folder"):
                for p in DOCS_DIR.glob("*"):
                    if p.is_file():
                        p.unlink()
                st.success("Docs folder cleared.")

        # show current docs
        st.markdown("### Current docs")
        docs = sorted(DOCS_DIR.glob("*.txt"))
        if not docs:
            st.info("No docs found yet.")
        else:
            rows = []
            for p in docs:
                rows.append({"file": p.name, "size": _human_bytes(p.stat().st_size)})
            st.dataframe(rows, use_container_width=True)

    # ----------------------------
    # 2) Build index
    # ----------------------------
    with tab_build:
        st.subheader("Build / Rebuild index")
        st.write(f"Index folder: `{INDEX_DIR}`")
        st.write(f"Using pack: `{pack_dir}`")

        run_eval = st.checkbox("Run self-eval after build", value=True)
        eval_limit = st.slider("Eval question limit", min_value=20, max_value=2000, value=200, step=20)

        if st.button("Build now", type="primary"):
            docs = list(DOCS_DIR.glob("*.txt"))
            if not docs:
                st.error("No docs found. Upload docs first.")
            else:
                br = build_all(ROOT, pack_dir, run_eval=run_eval, eval_limit=eval_limit)
                st.success("Build complete.")
                st.json(br.__dict__ if hasattr(br, "__dict__") else br)

        # show index summary
        st.markdown("### Index status")
        claims_path = INDEX_DIR / "claims.jsonl"
        ont_path = INDEX_DIR / "ontology.json"
        colA, colB, colC = st.columns(3)
        colA.metric("claims.jsonl", "✅" if claims_path.exists() else "❌")
        colB.metric("ontology.json", "✅" if ont_path.exists() else "❌")
        colC.metric("docs", str(len(list(DOCS_DIR.glob("*.txt")))))

        if claims_path.exists():
            st.caption(f"claims.jsonl size: {_human_bytes(claims_path.stat().st_size)}")

    # ----------------------------
    # 3) Ask
    # ----------------------------
    with tab_ask:
        st.subheader("Ask a question (frame-constrained)")
        if not (INDEX_DIR / "claims.jsonl").exists():
            st.warning("Index not built yet. Build the index first.")
            st.stop()

        q = st.text_area(
            "Question",
            height=140,
            placeholder="e.g., Explain how placental hormones contribute to insulin resistance during pregnancy and why this becomes pathological in GDM.",
        )

        show_debug = st.checkbox("Show debug", value=True)

        if st.button("Answer", type="primary"):
            if not q.strip():
                st.warning("Please enter a question.")
            else:
                res = answer(INDEX_DIR, q, domain)
                c1, c2 = st.columns([1.1, 0.9])
                with c1:
                    st.markdown(res.markdown)
                with c2:
                    if show_debug:
                        st.subheader("Debug")
                        st.json(res.debug)

    # ----------------------------
    # 4) Patient record (BM25-only)
    # ----------------------------
    with tab_patient:
        st.subheader("Patient record mapping (BM25-only, local, explainable)")
        st.caption("This is decision support. It maps patient text → pack concepts → supporting claims. No diagnosis.")

        if not (INDEX_DIR / "claims.jsonl").exists():
            st.warning("Index not built yet. Build the index first.")
            st.stop()

        patient_layer = load_patient_layer(domain)
        if not patient_layer:
            st.error(
                "patient_layer.json missing OR manifest.json does not include files.patient_layer.\n\n"
                "Fix: add in pack manifest.json:  \"patient_layer\": \"patient_layer.json\""
            )
            st.stop()

        left, right = st.columns([1, 1])

        with left:
            patient_id = st.text_input("Patient ID", value="P-001")
            visit_id = st.text_input("Visit ID", value="V-001")
            notes = st.text_area(
                "Patient notes (free text)",
                height=220,
                placeholder=(
                    "Example:\n"
                    "28 y/o G2P1, BMI: 32\n"
                    "Fasting glucose: 102 mg/dl\n"
                    "OGTT 1h: 185 mg/dl\n"
                    "Concern for insulin resistance during pregnancy."
                ),
            )

            topk_concepts = st.slider("Top concepts", 3, 30, 10, 1)
            topk_claims = st.slider("Top claims", 3, 40, 15, 1)

            analyze = st.button("Analyze patient record", type="primary")

        with right:
            st.markdown("### Pack patient layer status")
            st.write("Concepts:", len(patient_layer.get("concepts") or []))
            st.write("Default query terms:", patient_layer.get("default_query_terms") or [])
            st.write("Field extractors:", len(patient_layer.get("field_extractors") or []))
            st.write("Surveillance templates:", len(patient_layer.get("surveillance_templates") or []))

        if analyze:
            if not notes.strip():
                st.warning("Please enter patient notes.")
            else:
                rec = PatientRecord(patient_id=patient_id, visit_id=visit_id, timestamp="", notes=notes, structured={})

                # Extract structured fields from notes using pack rules
                rec.extracted = extract_fields_from_notes(rec.notes, patient_layer)

                # Best-effort BM25 matches (never empty silently)
                concepts, cq = _patient_match_concepts_bm25(rec, patient_layer, topk=topk_concepts)
                rec.matched_concepts = concepts

                claims, q_claims = _patient_match_claims_bm25(rec, patient_layer, INDEX_DIR, topk=topk_claims)
                rec.matched_claims = claims

                surv = build_surveillance(rec, domain)
                
                # ---- Patient State Graph + Triggers ----
                snapshot = normalize_patient_snapshot(rec, domain)
                tr = run_triggers(snapshot, domain)
                
                # Attach evidence claims to each triggered state/alert
                tr = attach_evidence_to_triggers(
                    index_dir=INDEX_DIR,
                    triggers_out=tr,
                    query_text=cq,  # patient-context BM25 query used for matching
                    per_item_topk=3
                )
                
                st.divider()
                st.markdown("## Patient State Graph + Care Pathway Triggers")
                
                cA, cB = st.columns([1, 1])
                
                with cA:
                    st.markdown("### Derived states")
                    if tr["derived_states"]:
                        for s in tr["derived_states"]:
                            st.markdown(
                                f"**{s.get('name')}**  ·  `{s.get('severity', 'info')}`  · rule `{s.get('rule_id')}`")
                            st.caption(s.get("rationale", ""))
                            
                            ev = s.get("evidence_claims") or []
                            if ev:
                                with st.expander("Evidence claims"):
                                    for e in ev:
                                        st.markdown(
                                            f"- **{e['claim_id']}** ({e.get('doc', '')}) "
                                            f"score={e.get('score', 0):.3f} "
                                            f"edges={', '.join(e.get('edges_matched') or [])}"
                                        )
                                        st.write(e.get("text", ""))
                            else:
                                st.caption("No evidence claim matched for this trigger (edges not found in claims).")
                            
                            st.divider()
                    else:
                        st.caption("No derived states triggered.")
                    
                    # st.markdown("### Derived states")
                    # if tr["derived_states"]:
                    #     st.dataframe(
                    #         [
                    #             {
                    #                 "state": s.get("name"),
                    #                 "severity": s.get("severity", "info"),
                    #                 "rationale": s.get("rationale", ""),
                    #                 "rule": s.get("rule_id"),
                    #                 "evidence_edges": ", ".join(s.get("evidence_edges_any", []) or [])
                    #             }
                    #             for s in tr["derived_states"]
                    #         ],
                    #         use_container_width=True
                    #     )
                    # else:
                    #     st.caption("No derived states triggered.")
                
                with cB:
                    st.markdown("### Alerts")
                    if tr["alerts"]:
                        for a in tr["alerts"]:
                            st.markdown(
                                f"**{a.get('name')}**  ·  `{a.get('severity', 'warning')}`  · rule `{a.get('rule_id')}`")
                            st.caption(a.get("rationale", ""))
                            
                            ev = a.get("evidence_claims") or []
                            if ev:
                                with st.expander("Evidence claims"):
                                    for e in ev:
                                        st.markdown(
                                            f"- **{e['claim_id']}** ({e.get('doc', '')}) "
                                            f"score={e.get('score', 0):.3f} "
                                            f"edges={', '.join(e.get('edges_matched') or [])}"
                                        )
                                        st.write(e.get("text", ""))
                            else:
                                st.caption("No evidence claim matched for this trigger (edges not found in claims).")
                            
                            st.divider()
                    else:
                        st.caption("No alerts triggered.")
                    
                    # st.markdown("### Alerts")
                    # if tr["alerts"]:
                    #     st.dataframe(
                    #         [
                    #             {
                    #                 "alert": a.get("name"),
                    #                 "severity": a.get("severity", "warning"),
                    #                 "rationale": a.get("rationale", ""),
                    #                 "rule": a.get("rule_id"),
                    #                 "evidence_edges": ", ".join(a.get("evidence_edges_any", []) or [])
                    #             }
                    #             for a in tr["alerts"]
                    #         ],
                    #         use_container_width=True
                    #     )
                    # else:
                    #     st.caption("No alerts triggered.")
                
                with st.expander("Snapshot (debug)"):
                    st.json(snapshot)
                with st.expander("Trigger debug"):
                    st.json(tr["debug"])
                
                st.divider()
                st.markdown("## Results")

                # Diagnostics
                with st.expander("Diagnostics (why results look like this)", expanded=True):
                    st.write("Extracted fields:", rec.extracted)
                    st.write("BM25 query used:", cq)

                    # show if overlap likely zero
                    if all((c["score"] == 0.0 for c in (rec.matched_concepts or []))):
                        st.warning(
                            "All concept BM25 scores are 0. This means there is little/no lexical overlap between "
                            "patient text and concept phrases. Fix by adding pack.default_query_terms and/or adding "
                            "more concept.phrases that match clinic wording."
                        )
                    if all((c["score"] == 0.0 for c in (rec.matched_claims or []))):
                        st.warning(
                            "All claim BM25 scores are 0. Fix by adding anchor terms in patient_layer.json "
                            "(default_query_terms) and/or using more medical terms in the patient note."
                        )

                c1, c2 = st.columns([1, 1])

                with c1:
                    st.markdown("### 🧠 Matched concepts")
                    if rec.matched_concepts:
                        st.dataframe(
                            [
                                {
                                    "concept": c["label"],
                                    "phrase": c["phrase"],
                                    "score": round(c["score"], 4),
                                    "edges_hint": c.get("edges_hint", ""),
                                }
                                for c in rec.matched_concepts
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No concepts matched (pack might be missing concepts/phrases).")

                with c2:
                    st.markdown("### 📄 Matched claims")
                    if rec.matched_claims:
                        st.dataframe(
                            [
                                {
                                    "claim_id": c["claim_id"],
                                    "score": round(c["score"], 4),
                                    "edges": c.get("edges", ""),
                                    "doc": c["doc"],
                                }
                                for c in rec.matched_claims
                            ],
                            use_container_width=True,
                        )
                    else:
                        st.info("No claims matched (index may be empty or query has no overlap).")

                st.markdown("### 🩺 Surveillance suggestions (pack-driven)")
                if surv:
                    for s in surv:
                        st.markdown(f"**{s.get('label', s.get('id'))}**")
                        for it in s.get("items", []):
                            st.markdown(f"- {it.get('text')}")
                else:
                    st.caption("No surveillance template triggered by current concept matches.")

                st.divider()
                st.markdown("## Ask using patient context")

                default_q = "Explain the relevant mechanism for this patient's findings and recommended follow-up per evidence."
                q2 = st.text_area("Question", height=120, value=default_q, key="patient_q")

                if st.button("Answer using patient context"):
                    expanded_q = expand_question_with_patient_context(q2, rec)
                    res = answer(INDEX_DIR, expanded_q, domain)

                    a1, a2 = st.columns([1.1, 0.9])
                    with a1:
                        st.markdown(res.markdown)
                    with a2:
                        st.subheader("Debug")
                        st.json(res.debug)

    # ---------------------------- Population surveillance
    # 5) Population surveillance
    # -----------------------------------
    with tab_population:
        st.subheader("Population Surveillance (GDM pilot)")
        st.caption("Operational queue + KPIs + drill-down. Local-only. Evidence-linked triggers.")
        
        if not (INDEX_DIR / "claims.jsonl").exists():
            st.warning("Index not built yet. Build the index first.")
            st.stop()
        
        # ---- Ingest ----
        st.markdown("### Ingest patient record into population store")
        
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            p_id = st.text_input("Patient ID", value="P-1001", key="pop_pid")
        with c2:
            v_id = st.text_input("Visit ID", value="V-001", key="pop_vid")
        with c3:
            ts = st.text_input("Timestamp (optional ISO)", value="", key="pop_ts")
        
        notes = st.text_area("Patient notes", height=170, key="pop_notes")
        
        extra_json = st.text_area(
            "Extra structured JSON (optional)",
            height=120,
            value="{}",
            key="pop_extra"
        )
        
        ing = st.button("Ingest record", type="primary")
        
        if ing:
            try:
                extra = json.loads(extra_json) if extra_json.strip() else {}
            except Exception as e:
                st.error(f"Bad JSON in extra structured: {e}")
                extra = {}
            
            try:
                rec = ingest_patient_record(
                    root=ROOT,
                    index_dir=INDEX_DIR,
                    domain=domain,
                    patient_id=p_id.strip(),
                    visit_id=v_id.strip(),
                    notes=notes or "",
                    extra_structured=extra,
                    timestamp=(ts.strip() or None),
                    per_item_topk=3
                )
                st.success("Ingested.")
                st.json(
                    {"patient_id": rec["patient_id"], "episode_id": rec["episode_id"], "timestamp": rec["timestamp"]})
            except Exception as e:
                st.error(f"Ingest failed: {e}")
        
        st.divider()
        
        # ---- Filters ----
        st.markdown("### Filters")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        
        with f1:
            start_ts = st.text_input("Start TS (optional ISO)", value="", key="pop_start")
        with f2:
            end_ts = st.text_input("End TS (optional ISO)", value="", key="pop_end")
        with f3:
            severities = st.multiselect("Severity", ["high", "warning", "info"], default=["high", "warning"])
        with f4:
            statuses = st.multiselect("Status", ["open", "acknowledged", "resolved"], default=["open", "acknowledged"])
        
        start_ts = start_ts.strip() or None
        end_ts = end_ts.strip() or None
        
        # ---- KPIs ----
        st.markdown("### KPIs")
        kpi = compute_kpis(ROOT, start_ts=start_ts, end_ts=end_ts, severity_filter=severities)
        
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Active episodes", str(kpi.get("episodes_total", 0)))
        k2.metric("High severity (open)", str(kpi.get("counts", {}).get("severity::high", 0)))
        k3.metric("Warnings (open)", str(kpi.get("counts", {}).get("severity::warning", 0)))
        k4.metric("Info states", str(kpi.get("counts", {}).get("severity::info", 0)))
        
        # Common triggers (if present)
        counts = kpi.get("counts", {})
        st.caption("Top alert counts")
        top_alerts = sorted([(k, v) for k, v in counts.items() if k.startswith("alert::")], key=lambda x: x[1],
                            reverse=True)[:10]
        if top_alerts:
            st.dataframe([{"alert": k.replace("alert::", ""), "count": v} for k, v in top_alerts],
                         use_container_width=True)
        else:
            st.info("No alerts in current window.")
        
        st.divider()
        
        # ---- Queue ----
        st.markdown("### Care Gap Queue")
        queue = build_care_gap_queue(
            ROOT,
            start_ts=start_ts,
            end_ts=end_ts,
            severity_filter=severities,
            status_filter=statuses,
            max_rows=500
        )
        
        if queue:
            st.dataframe(queue, use_container_width=True, height=320)
        else:
            st.info("Queue is empty for current filters.")
        
        st.divider()
        
        # ---- Drill-down ----
        st.markdown("### Patient drill-down")
        
        d1, d2 = st.columns([1, 1])
        with d1:
            drill_pid = st.text_input("Patient ID", value=p_id, key="dr_pid")
        with d2:
            drill_eid = st.text_input("Episode ID", value="", key="dr_eid")
        
        if st.button("Load timeline"):
            if not drill_pid.strip() or not drill_eid.strip():
                st.warning("Enter both patient_id and episode_id.")
            else:
                tl = patient_timeline(ROOT, drill_pid.strip(), drill_eid.strip())
                if not tl:
                    st.info("No timeline records found.")
                else:
                    st.success(f"Loaded {len(tl)} record(s). Latest shown below.")
                    latest = tl[-1]
                    st.json({
                        "patient_id": latest.get("patient_id"),
                        "episode_id": latest.get("episode_id"),
                        "visit_id": latest.get("visit_id"),
                        "timestamp": latest.get("timestamp")
                    })
                    
                    st.markdown("#### Latest triggers")
                    alerts = latest.get("alerts") or []
                    if alerts:
                        for a in alerts:
                            st.markdown(f"**{a.get('name')}** · `{a.get('severity')}` · rule `{a.get('rule_id')}`")
                            st.caption(a.get("rationale", ""))
                            ev = a.get("evidence_claims") or []
                            if ev:
                                with st.expander("Evidence"):
                                    for e in ev:
                                        st.markdown(
                                            f"- **{e['claim_id']}** ({e.get('doc', '')}) edges={', '.join(e.get('edges_matched') or [])}")
                                        st.write(e.get("text", ""))
                            st.divider()
                    else:
                        st.caption("No alerts.")
                    
                    st.markdown("#### Actions")
                    a1, a2, a3 = st.columns([1, 1, 2])
                    with a1:
                        ak = st.text_input("alert_key (rule::name)", value="", key="act_key")
                    with a2:
                        stt = st.selectbox("status", ["acknowledged", "resolved", "open"], index=0, key="act_status")
                    with a3:
                        note = st.text_input("note", value="", key="act_note")
                    
                    if st.button("Write action"):
                        if not ak.strip():
                            st.warning("Provide alert_key (copy from queue row).")
                        else:
                            write_action(ROOT, drill_pid.strip(), drill_eid.strip(), ak.strip(), stt, note, user="")
                            st.success("Action saved (append-only).")
    
    # ----------------------------
    # 6) Pack viewer
    # ----------------------------
    with tab_pack:
        st.subheader("Pack viewer")
        st.write(f"Pack dir: `{pack_dir}`")

        manifest = _safe_read_json(pack_dir / "manifest.json", {})
        st.markdown("### manifest.json")
        st.json(manifest)

        files = (manifest.get("files") or {})
        st.markdown("### Pack files")
        rows = []
        for k, v in files.items():
            p = pack_dir / v
            rows.append({"key": k, "file": v, "exists": p.exists()})
        st.dataframe(rows, use_container_width=True)

        # Show raw JSON files quickly
        st.markdown("### Quick open")
        open_key = st.selectbox("Open file", ["(select)"] + list(files.keys()))
        if open_key != "(select)":
            p = pack_dir / files[open_key]
            if p.exists():
                st.code(p.read_text(errors="ignore")[:20000])
            else:
                st.error("File not found.")

    st.sidebar.divider()
    st.sidebar.caption(f"ROOT: {ROOT}")
    st.sidebar.caption(f"DOCS_DIR: {DOCS_DIR}")
    st.sidebar.caption(f"INDEX_DIR: {INDEX_DIR}")
    st.sidebar.caption(f"PACKS_DIR: {PACKS_DIR}")


if __name__ == "__main__":
    main()

# import streamlit as st
# from pathlib import Path
#
# from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
# from engine.domain_pack import DomainPack
# from engine.qa import answer
#
# # BM25 matching
# from engine.patient_match_bm25 import load_patient_layer
#
# #  patient_record
# from engine.patient_record import PatientRecord
# from engine.patient_match_bm25 import (
#     extract_fields_from_notes,
#     match_patient_to_concepts,
#     match_patient_to_claims,
#     build_surveillance,
#     expand_question_with_patient_context,
# )
#
#
# def list_packs(packs_dir: Path):
#     if not packs_dir.exists():
#         return []
#     packs = []
#     for p in packs_dir.iterdir():
#         if p.is_dir() and (p / "manifest.json").exists():
#             packs.append(p)
#     return sorted(packs, key=lambda x: x.name)
#
#
# def main():
#     st.set_page_config(page_title="GDM ChainGraph v9", layout="wide")
#     st.title("GDM ChainGraph v9 — Domain Packs; Patient-Centric")
#
#     ROOT = Path(".").resolve()
#     paths = ensure_dirs(ROOT)
#
#     DOCS_DIR = paths["docs_dir"]
#     INDEX_DIR = paths["index_dir"]
#     EVAL_DIR = paths["eval_dir"]
#
#     PACKS_DIR = ROOT / "domain_packs"
#     packs = list_packs(PACKS_DIR)
#     if not packs:
#         st.error("No packs found. Create: domain_packs/gdm_v1/manifest.json")
#         st.stop()
#
#     st.sidebar.header("Domain Pack")
#     pack_names = [p.name for p in packs]
#     sel = st.sidebar.selectbox("Select pack", pack_names, index=0)
#     PACK_DIR = PACKS_DIR / sel
#
#     try:
#         dp = DomainPack.load(PACK_DIR)
#         st.sidebar.success(f"{dp.manifest.get('name')} v{dp.manifest.get('version')}")
#         st.sidebar.caption(str(PACK_DIR))
#     except Exception as e:
#         st.sidebar.error(f"Pack load error: {e}")
#         st.stop()
#
#     # tab_upload, tab_build, tab_ask, tab_pack = st.tabs(
#     #     ["Upload docs", "Build index", "Ask", "Pack viewer"]
#     # )
#     tab_upload, tab_build, tab_ask, tab_patient, tab_pack = st.tabs(
#         ["Upload docs", "Build index", "Ask", "Patient record", "Pack viewer"]
#     )
#
#     with tab_upload:
#         st.subheader("1) Upload your .txt documents")
#         st.write(f"Saved into: `{DOCS_DIR}`")
#
#         uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
#         overwrite = st.checkbox("Overwrite if same filename exists", value=True)
#
#         if st.button("Save uploaded docs", type="primary"):
#             res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
#             st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
#             if res["saved"]:
#                 st.write("Saved:")
#                 for p in res["saved"]:
#                     st.code(p)
#             if res["skipped"]:
#                 st.write("Skipped:")
#                 st.json(res["skipped"])
#
#         st.divider()
#         files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
#         st.write("Currently in docs/:")
#         st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])
#
#     with tab_build:
#         st.subheader("2) Build / Rebuild index & ontology")
#         st.write(f"Using pack: `{PACK_DIR}`")
#
#         run_eval = st.checkbox("Also generate questions + self-eval", value=True)
#         eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)
#
#         if st.button("Build now", type="primary"):
#             br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
#             if br.ok:
#                 st.success(br.message)
#                 st.json(br.stats)
#             else:
#                 st.error(br.message)
#
#         st.divider()
#         st.write("Index status:")
#         st.write(f"- ontology.json exists: `{(INDEX_DIR / 'ontology.json').exists()}`")
#         st.write(f"- claims.jsonl exists: `{(INDEX_DIR / 'claims.jsonl').exists()}`")
#
#     with tab_ask:
#         st.subheader("3) Ask")
#         if not (INDEX_DIR / "claims.jsonl").exists():
#             st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
#         else:
#             q = st.text_area(
#                 "Question",
#                 height=120,
#                 value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia."
#             )
#             if st.button("Answer", type="primary"):
#                 res = answer(INDEX_DIR, q, dp)
#                 c1, c2 = st.columns([1, 1])
#                 with c1:
#                     st.markdown(res.markdown)
#                 with c2:
#                     st.subheader("Debug")
#                     st.json(res.debug)
#
#     with tab_patient:
#         st.subheader("4) Patient record (BM25, local, explainable)")
#         st.caption("Clinical decision support only — evidence-backed, no diagnosis")
#
#         if not (INDEX_DIR / "claims.jsonl").exists():
#             st.warning("Index not built yet. Build the index first.")
#             st.stop()
#
#         # ---- Patient input ----
#         patient_note = st.text_area(
#             "Patient notes",
#             height=180,
#             placeholder=(
#                 "Example:\n"
#                 "28 y/o G2P1, BMI: 32\n"
#                 "Fasting glucose: 102 mg/dl\n"
#                 "OGTT 1h: 185 mg/dl\n"
#                 "Concern for insulin resistance during pregnancy."
#             )
#         )
#
#         patient_id = st.text_input("Patient ID", value="P-001")
#         visit_id = st.text_input("Visit ID", value="V-001")
#
#         if st.button("Analyze patient record", type="primary") and patient_note.strip():
#
#             rec = PatientRecord(
#                 patient_id=patient_id,
#                 visit_id=visit_id,
#                 timestamp="",
#                 notes=patient_note,
#                 structured={}
#             )
#
#             # ---- Field extraction ----
#             patient_layer = {}
#             try:
#                 # loaded indirectly inside extract function
#                 patient_layer = {}
#             except Exception:
#                 pass
#             patient_layer = load_patient_layer(dp)
#             if not patient_layer:
#                 st.error("patient_layer.json is missing or not referenced in manifest.json (files.patient_layer).")
#                 st.stop()
#
#             # rec.extracted = extract_fields_from_notes(rec.notes, patient_layer)
#
#             rec.extracted = extract_fields_from_notes(rec.notes, dp.manifest.get("patient_layer", {}) and {})
#
#             st.write("Patient layer concepts:", len(patient_layer.get("concepts", [])))
#             st.write("Extracted fields:", rec.extracted)
#             st.write("BM25 query used:", _patient_query(rec, patient_layer)[:400] + (
#                 "..." if len(_patient_query(rec, patient_layer)) > 400 else ""))
#
#             # ---- BM25 matching ----
#             rec.matched_concepts = match_patient_to_concepts(rec, dp)
#             rec.matched_claims = match_patient_to_claims(rec, INDEX_DIR)
#
#             # ---- Surveillance ----
#             surveillance = build_surveillance(rec, dp)
#
#             st.divider()
#
#             # ---- Results ----
#             c1, c2 = st.columns([1, 1])
#
#             with c1:
#                 st.markdown("### 🧠 Matched patient concepts")
#                 if rec.matched_concepts:
#                     st.dataframe(
#                         [
#                             {
#                                 "concept": c["label"],
#                                 "phrase": c["phrase"],
#                                 "score": round(c["score"], 3),
#                                 "edges_hint": ", ".join(c.get("edges_hint", []))
#                             }
#                             for c in rec.matched_concepts
#                         ]
#                     )
#                 else:
#                     st.info("No patient concepts matched.")
#
#             with c2:
#                 st.markdown("### 📄 Matched evidence claims")
#                 if rec.matched_claims:
#                     st.dataframe(
#                         [
#                             {
#                                 "claim_id": c["claim_id"],
#                                 "score": round(c["score"], 3),
#                                 "edges": ", ".join(c.get("edges", [])),
#                                 "doc": c["doc"]
#                             }
#                             for c in rec.matched_claims
#                         ]
#                     )
#                 else:
#                     st.info("No evidence claims matched.")
#
#             if surveillance:
#                 st.markdown("### 🩺 Surveillance & follow-up suggestions")
#                 for s in surveillance:
#                     st.markdown(f"**{s.get('label', s.get('id'))}**")
#                     for it in s.get("items", []):
#                         st.markdown(f"- {it.get('text')}")
#
#             # ---- Ask with patient context ----
#             st.divider()
#             st.markdown("### 🔎 Ask a question using patient context")
#
#             default_q = (
#                 "Explain the pathophysiological mechanism relevant to this patient's findings "
#                 "and outline recommended follow-up."
#             )
#
#             q2 = st.text_area("Question", height=120, value=default_q)
#
#             if st.button("Answer using patient context"):
#                 expanded_q = expand_question_with_patient_context(q2, rec)
#                 res = answer(INDEX_DIR, expanded_q, dp)
#
#                 c3, c4 = st.columns([1, 1])
#                 with c3:
#                     st.markdown(res.markdown)
#                 with c4:
#                     st.subheader("Debug")
#                     st.json(res.debug)
#
#     with tab_pack:
#         st.subheader("Domain Pack Viewer")
#         st.write(f"Pack dir: `{PACK_DIR}`")
#         st.json(dp.manifest)
#
#         with st.expander("Frames"):
#             st.json(dp.frames_raw)
#
#         with st.expander("Routing rules"):
#             st.json(dp.routing_rules)
#
#         with st.expander("Edge patterns (raw)"):
#             st.json(dp.edge_patterns_raw)
#
#         with st.expander("Infer rules (raw)"):
#             st.json(dp.infer_rules_raw)
#
#         with st.expander("Question templates"):
#             st.json(dp.question_templates)
#
#         with st.expander("Junk filters"):
#             st.json(dp.junk_filters)
#
#         with st.expander("Scoring"):
#             st.json(dp.scoring)
#
#
# if __name__ == "__main__":
#     main()
#
# # import streamlit as st
# # from pathlib import Path
# #
# # from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
# # from engine.domain_pack import DomainPack
# # from engine.qa import answer
# #
# #
# # def list_packs(packs_dir: Path):
# #     if not packs_dir.exists():
# #         return []
# #     packs = []
# #     for p in packs_dir.iterdir():
# #         if p.is_dir() and (p / "manifest.json").exists():
# #             packs.append(p)
# #     return sorted(packs, key=lambda x: x.name)
# #
# #
# # def main():
# #     st.set_page_config(page_title="GDM ChainGraph — Domain Packs", layout="wide")
# #     st.title("GDM ChainGraph — Domain Packs (external frames/patterns/routing)")
# #
# #     ROOT = Path(".").resolve()
# #     paths = ensure_dirs(ROOT)
# #     DOCS_DIR = paths["docs_dir"]
# #     INDEX_DIR = paths["index_dir"]
# #     EVAL_DIR = paths["eval_dir"]
# #
# #     PACKS_DIR = ROOT / "domain_packs"
# #     packs = list_packs(PACKS_DIR)
# #
# #     if not packs:
# #         st.error(
# #             "No domain packs found.\n\n"
# #             "Create: `domain_packs/gdm_v1/manifest.json` (and the referenced JSON files)."
# #         )
# #         st.stop()
# #
# #     # Sidebar: choose pack
# #     st.sidebar.header("Domain Pack")
# #     pack_names = [p.name for p in packs]
# #     sel = st.sidebar.selectbox("Select pack", pack_names, index=0)
# #     PACK_DIR = PACKS_DIR / sel
# #
# #     # Load pack safely (no crash loops)
# #     try:
# #         dp = DomainPack.load(PACK_DIR)
# #         st.sidebar.success(f"{dp.manifest.get('name')} v{dp.manifest.get('version')}")
# #         st.sidebar.caption(str(PACK_DIR))
# #     except Exception as e:
# #         st.sidebar.error(f"Pack load error: {e}")
# #         st.stop()
# #
# #     tab_upload, tab_build, tab_ask, tab_pack = st.tabs(
# #         ["Upload docs", "Build index", "Ask", "Pack viewer"]
# #     )
# #
# #     with tab_upload:
# #         st.subheader("1) Upload your .txt documents")
# #         st.write(f"Docs will be saved into: `{DOCS_DIR}`")
# #
# #         uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
# #         overwrite = st.checkbox("Overwrite if same filename exists", value=True)
# #
# #         if st.button("Save uploaded docs", type="primary"):
# #             res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
# #             st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
# #             if res["saved"]:
# #                 st.write("Saved files:")
# #                 for p in res["saved"]:
# #                     st.code(p)
# #             if res["skipped"]:
# #                 st.write("Skipped:")
# #                 st.json(res["skipped"])
# #
# #         st.divider()
# #         st.write("Currently in `docs/`:")
# #         files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
# #         st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])
# #
# #     with tab_build:
# #         st.subheader("2) Build / Rebuild index & ontology (using selected pack)")
# #         st.write(f"Using pack: `{PACK_DIR}`")
# #
# #         run_eval = st.checkbox("Also generate questions + self-eval", value=True)
# #         eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)
# #
# #         if st.button("Build now", type="primary"):
# #             br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
# #             if br.ok:
# #                 st.success(br.message)
# #                 st.json(br.stats)
# #             else:
# #                 st.error(br.message)
# #
# #         st.divider()
# #         st.write("Index files status:")
# #         st.write(f"- ontology.json exists: `{(INDEX_DIR / 'ontology.json').exists()}`")
# #         st.write(f"- claims.jsonl exists: `{(INDEX_DIR / 'claims.jsonl').exists()}`")
# #
# #     with tab_ask:
# #         st.subheader("3) Ask (requires index built)")
# #         if not (INDEX_DIR / "ontology.json").exists():
# #             st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
# #         else:
# #             q = st.text_area(
# #                 "Question",
# #                 height=120,
# #                 value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia."
# #             )
# #
# #             if st.button("Answer", type="primary"):
# #                 # IMPORTANT: pass the loaded DomainPack
# #                 res = answer(INDEX_DIR, q, dp)
# #                 c1, c2 = st.columns([1, 1])
# #                 with c1:
# #                     st.markdown(res.markdown)
# #                 with c2:
# #                     st.subheader("Debug")
# #                     st.json(res.debug)
# #
# #     with tab_pack:
# #         st.subheader("Domain Pack Viewer")
# #         st.write(f"Pack dir: `{PACK_DIR}`")
# #         st.json(dp.manifest)
# #
# #         with st.expander("Frames"):
# #             st.json(dp.frames)
# #
# #         with st.expander("Routing rules"):
# #             st.json(dp.routing_rules)
# #
# #         with st.expander("Edge patterns (raw)"):
# #             st.json(dp.edge_patterns_raw)
# #
# #         with st.expander("Infer rules (raw)"):
# #             st.json(dp.infer_rules_raw)
# #
# #         with st.expander("Junk filters"):
# #             st.json(dp.junk_filters)
# #
# #         with st.expander("Scoring"):
# #             st.json(dp.scoring)
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
# # # import streamlit as st
# # # from pathlib import Path
# # # import json
# # #
# # # from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
# # # from engine.domain_pack import DomainPack
# # # from engine.qa import answer
# # #
# # # ROOT = Path(".").resolve()
# # # paths = ensure_dirs(ROOT)
# # # DOCS_DIR = paths["docs_dir"]
# # # INDEX_DIR = paths["index_dir"]
# # # EVAL_DIR = paths["eval_dir"]
# # # PACKS_DIR = ROOT / "domain_packs"
# # # # ROOT = Path(__file__).parent
# # # # PACKS_DIR = ROOT / "domain_packs"
# # # PACK_DIR = PACKS_DIR / "gdm_v1"   # default
# # #
# # # build_all(ROOT, PACK_DIR, ...)
# # #
# # #
# # # st.set_page_config(page_title="GDM ChainGraph — Domain Packs", layout="wide")
# # # st.title("GDM ChainGraph — Domain Packs (external frames/patterns/routing)")
# # #
# # # def list_packs():
# # #     if not PACKS_DIR.exists():
# # #         return []
# # #     out = []
# # #     for p in PACKS_DIR.iterdir():
# # #         if p.is_dir() and (p / "manifest.json").exists():
# # #             out.append(p)
# # #     return sorted(out, key=lambda x: x.name)
# # #
# # # packs = list_packs()
# # # if not packs:
# # #     st.error("No domain packs found. Create: domain_packs/gdm_v1/manifest.json etc.")
# # #     st.stop()
# # #
# # # pack_names = [p.name for p in packs]
# # # sel = st.sidebar.selectbox("Select Domain Pack", pack_names, index=0)
# # # PACK_DIR = PACKS_DIR / sel
# # #
# # # # quick pack info
# # # try:
# # #     dp = DomainPack.load(PACK_DIR)
# # #     st.sidebar.write(f"**Pack:** {dp.manifest.get('name')} v{dp.manifest.get('version')}")
# # # except Exception as e:
# # #     st.sidebar.error(f"Pack load error: {e}")
# # #     st.stop()
# # #
# # # tab_upload, tab_build, tab_ask, tab_pack = st.tabs(["Upload docs", "Build index", "Ask", "Pack viewer"])
# # #
# # # with tab_upload:
# # #     st.subheader("1) Upload your .txt documents")
# # #     st.write(f"Docs will be saved into: `{DOCS_DIR}`")
# # #
# # #     uploaded = st.file_uploader("Upload .txt docs", type=["txt"], accept_multiple_files=True)
# # #     overwrite = st.checkbox("Overwrite if same filename exists", value=True)
# # #
# # #     if st.button("Save uploaded docs", type="primary"):
# # #         res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
# # #         st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
# # #         if res["saved"]:
# # #             st.write("Saved files:")
# # #             for p in res["saved"]:
# # #                 st.code(p)
# # #         if res["skipped"]:
# # #             st.write("Skipped:")
# # #             st.json(res["skipped"])
# # #
# # #     st.divider()
# # #     st.write("Currently in docs/:")
# # #     files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
# # #     st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])
# # #
# # # with tab_build:
# # #     st.subheader("2) Build / Rebuild index & ontology using selected Domain Pack")
# # #     st.write(f"Using pack: `{PACK_DIR}`")
# # #
# # #     run_eval = st.checkbox("Also generate questions + self-eval", value=True)
# # #     eval_limit = st.slider("Eval question limit", min_value=50, max_value=3000, value=600, step=50)
# # #
# # #     if st.button("Build now", type="primary"):
# # #         br = build_all(ROOT, PACK_DIR, run_eval=run_eval, eval_limit=eval_limit)
# # #         if br.ok:
# # #             st.success(br.message)
# # #             st.json(br.stats)
# # #         else:
# # #             st.error(br.message)
# # #
# # #     st.divider()
# # #     st.write("Index files status:")
# # #     st.write(f"- ontology.json: `{(INDEX_DIR/'ontology.json').exists()}`")
# # #     st.write(f"- claims.jsonl: `{(INDEX_DIR/'claims.jsonl').exists()}`")
# # #
# # # with tab_ask:
# # #     st.subheader("3) Ask (requires index built)")
# # #     if not (INDEX_DIR / "ontology.json").exists():
# # #         st.warning("Index not built yet. Go to **Build index** and click **Build now**.")
# # #     else:
# # #         q = st.text_area(
# # #             "Question",
# # #             height=120,
# # #             value="Based on the evidence presented, should GDM be reframed as a chronic metabolic condition rather than a transient pregnancy complication? Justify your reasoning."
# # #         )
# # #         if st.button("Answer", type="primary"):
# # #             res = answer(INDEX_DIR, q, dp)
# # #             c1, c2 = st.columns([1, 1])
# # #             with c1:
# # #                 st.markdown(res.markdown)
# # #             with c2:
# # #                 st.subheader("Debug")
# # #                 st.json(res.debug)
# # #
# # # with tab_pack:
# # #     st.subheader("Domain Pack Viewer")
# # #     st.write(f"Pack dir: `{PACK_DIR}`")
# # #     st.json(dp.manifest)
# # #
# # #     with st.expander("Frames"):
# # #         st.json(dp.frames)
# # #
# # #     with st.expander("Routing rules"):
# # #         st.json(dp.routing_rules)
# # #
# # #     with st.expander("Edge patterns (raw)"):
# # #         st.json(dp.edge_patterns_raw)
# # #
# # #     with st.expander("Infer rules (raw)"):
# # #         st.json(dp.infer_rules_raw)
# # #
# # #     with st.expander("Junk filters"):
# # #         st.json(dp.junk_filters)
# # #
# # #     with st.expander("Scoring"):
# # #         st.json(dp.scoring)
