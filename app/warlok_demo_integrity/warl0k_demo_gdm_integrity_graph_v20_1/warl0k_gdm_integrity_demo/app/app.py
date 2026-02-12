import streamlit as st
from streamlit_plotly_events import plotly_events
from core.store import DataStore
from core.graph_ui import build_patient_graph
from core.pathways import upsert_tasks_from_results
from core.claim_index import build_claim_index
from core.evidence_index import build_passage_index, resolve_evidence_refs
import networkx as nx
import matplotlib.pyplot as plt


APP_VERSION = "v11-patient-pathways-state-machine"
APP_BUILD_NOTE = "Patient To‑Do + timeline + deterministic pathway triggers"
from core.graph_router import GraphRouter
from core.pathways import PathwayEngine
from core.pathways import parse_ga

from core.retrieval import Retriever
from core.qa import QAEvaluator
from core.patient_state import PatientStateEngine

st.set_page_config(page_title="WARL0K GDM Integrity (v10.2-graph-router-ui-versioned)", layout="wide")

@st.cache_resource
def load_store():
    return DataStore.load_default()

store = load_store()
retriever = Retriever(store.passages)
qa_eval = QAEvaluator(store)
ps_engine = PatientStateEngine(store, retriever)


def render_patient_graph(patient: dict, results: list):
    """Visualize a patient state graph: patient -> episode -> checks -> evidence-locked actions."""
    G = nx.DiGraph()
    pid = patient.get("pid","PATIENT")
    pname = patient.get("name","")
    pnode = f"{pid}\n{pname}".strip()
    G.add_node(pnode, kind="patient")

    # Episode nodes
    ep = "PregnancyEpisode"
    G.add_node(ep, kind="episode")
    G.add_edge(pnode, ep)

    # Add key facts as nodes (optional, compact)
    ga = patient.get("gestational_age_weeks")
    pp = patient.get("postpartum_weeks")
    risk = patient.get("risk_level")
    facts = []
    if risk is not None: facts.append(f"Risk: {risk}")
    if ga is not None: facts.append(f"GA: {ga}")
    if pp is not None: facts.append(f"PP: {pp}")
    if patient.get("history_gdm"): facts.append("Hx: GDM")
    for f in facts[:4]:
        G.add_node(f, kind="fact")
        G.add_edge(ep, f)

    # Event nodes
    for e in patient.get("events", []):
        et = e.get("type","event")
        label = et
        if e.get("name"): label += f"\n{e['name']}"
        if e.get("date"): label += f"\n{e['date']}"
        G.add_node(label, kind="event")
        G.add_edge(ep, label)

    # Check nodes with status
    for r in results:
        sid = r.get("sid")
        title = r.get("title")
        status = r.get("status","UNKNOWN")
        node = f"{sid}\n{title}\n[{status}]"
        G.add_node(node, kind="check", status=status)
        G.add_edge(ep, node)
        rec = r.get("recommendation","")
        if rec:
            rec_node = f"Action:\n{rec[:80]}{'…' if len(rec)>80 else ''}"
            G.add_node(rec_node, kind="action", status=status)
            G.add_edge(node, rec_node)

    # Layout
    pos = nx.spring_layout(G, seed=7, k=0.7)

    # Node styling
    def node_style(n):
        kind = G.nodes[n].get("kind")
        status = G.nodes[n].get("status")
        if kind == "patient": return ("#111827", "white", 1400)
        if kind == "episode": return ("#374151", "white", 1100)
        if kind == "fact": return ("#6B7280", "white", 900)
        if kind == "event": return ("#2563EB", "white", 900)
        if kind == "check":
            col = {"OK":"#16A34A","DUE":"#F59E0B","OVERDUE":"#DC2626","UNKNOWN":"#9CA3AF","NOT_APPLICABLE":"#64748B"}.get(status,"#9CA3AF")
            return (col, "white", 1100)
        if kind == "action":
            col = {"OK":"#22C55E","DUE":"#FBBF24","OVERDUE":"#EF4444","UNKNOWN":"#D1D5DB","NOT_APPLICABLE":"#94A3B8"}.get(status,"#D1D5DB")
            return (col, "black" if status in ("UNKNOWN","NOT_APPLICABLE") else "black", 1000)
        return ("#9CA3AF", "black", 900)

    node_colors, font_colors, sizes = [], [], []
    for n in G.nodes():
        c, fc, s = node_style(n)
        node_colors.append(c)
        font_colors.append(fc)
        sizes.append(s)

    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")

    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=10, width=1.0, alpha=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=sizes, linewidths=1.2, edgecolors="#111827", alpha=0.95)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    st.pyplot(fig, clear_figure=True)

demo_patients = store.patients[:5] if hasattr(store, "patients") else []
st.sidebar.title("WARL0K • GDM Integrity")
st.sidebar.markdown('---')
st.sidebar.markdown(f"### Version: `{APP_VERSION}`")
st.sidebar.markdown('---')
st.sidebar.markdown('## Patient Supervision')
patient_mode = st.sidebar.checkbox('Enable patient mode', value=True)
patient_id = st.sidebar.selectbox('Select patient', options=[p.get('id') for p in demo_patients] if 'demo_patients' in globals() else [], index=0 if ('demo_patients' in globals() and len(demo_patients)>0) else None)
st.sidebar.caption('Patient mode generates claim-backed action items (To‑Do) and a timeline.')
st.sidebar.caption(APP_BUILD_NOTE)
st.sidebar.markdown('**WARL0K GDM Integrity Demo**\n\n- Graph-constrained retrieval (lexicon → claim packs → evidence allowlist)\n- Hybrid BM25 + Semantic rerank\n- Slot-gated PRIMARY/SUPPORT evidence\n\n**How to verify changes:**\n1) Check the version header below.\n2) Open *Graph routing* expander and confirm `allowed_passages > 0`.\n3) If results look unchanged, click *Clear app caches*.\n')
if st.sidebar.button('Clear semantic index (rebuild on next run)'):
    import os
    base = os.path.join(os.path.dirname(__file__), 'data', 'index')
    for fn in ['semantic.vocab.json','semantic.npz']:
        fp = os.path.join(base, fn)
        try:
            if os.path.exists(fp):
                os.remove(fp)
        except Exception:
            pass
    st.sidebar.success('Deleted semantic index files. They will be rebuilt.')
    st.rerun()

if st.sidebar.button('Clear app caches'):
    try:
        st.cache_data.clear()
        st.cache_resource.clear()
    except Exception:
        pass
    st.sidebar.success('Cleared Streamlit caches. Reloading…')
    st.rerun()
mode = st.sidebar.radio("Mode", ["Evidence Search", "Q/A Validator", "Patient State"], index=2)

st.sidebar.markdown("---")
st.sidebar.caption("Integrity rule: answers must be supported by retrieved evidence (or the system refuses).")

if mode == "Evidence Search":
    st.title("Evidence Search (BM25 + TF‑IDF semantic)")
    q = st.text_input("Query", "low risk pregnancy screening window 24 28 weeks OGTT")
    k = st.slider("Top‑K", 3, 15, 8)

    if st.button("Search"):
        results = retriever.search(q, k=k)
        for r in results:
            with st.expander(f"{r['rank']}. {r['doc']} • {r['pid']} • score={r['score']:.3f}"):
                st.write(r["text"])
                st.code(r["highlights"], language="text")

elif mode == "Q/A Validator":
    st.title("Q/A Validator (locked answers + variants)")
    qs = store.questions
    qids = [q["qid"] for q in qs]
    selected = st.selectbox("Question ID", qids, index=0)
    qobj = next(q for q in qs if q["qid"] == selected)

    st.subheader("Question")
    st.write(qobj["question"])

    st.subheader("Locked answer (gold)")
    st.write(qobj["locked_answer"])

    st.subheader("Try a variant / custom answer")
    variant = st.selectbox("Example", ["locked_answer", "contradictory_example", "bad_example", "neutral_example", "entangled_example", "custom"])
    if variant == "custom":
        answer = st.text_area("Answer to validate", value="", height=140)
    else:
        answer = qobj[variant] if variant != "locked_answer" else qobj["locked_answer"]
        st.info(answer)

    k = st.slider("Evidence Top‑K", 3, 20, 10)
    if st.button("Validate"):
        report = qa_eval.validate(qobj, answer, retriever, k=k)

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Result")
            st.metric("Integrity score", f"{report['score']:.2f}")
            st.write("**Verdict:**", report["verdict"])
            st.write("**Reasons:**")
            for x in report["reasons"]:
                st.write(f"- {x}")

        with col2:
            st.subheader("Evidence used")
            for e in report["evidence"]:
                with st.expander(f"{e['rank']}. {e['doc']} • {e['pid']} • score={e['score']:.3f}"):
                    st.write(e["text"])

        st.subheader("Field checks")
        st.json(report["field_checks"])

elif mode == "Patient State":
    st.title("Patient State (Gold care checks + evidence‑locked recommendations)")
    pats = store.patients
    pid = st.selectbox("Patient", [p["pid"] + " — " + p.get("name","") for p in pats], index=0)
    pid = pid.split(" — ")[0]
    patient = next(p for p in pats if p["pid"] == pid)

    st.subheader("Patient snapshot")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk level", str(patient.get("risk_level", "—")))
    c2.metric("GA weeks", str(patient.get("gestational_age_weeks", "—")))
    c3.metric("Postpartum weeks", str(patient.get("postpartum_weeks", "—")))
    c4.metric("History GDM", "Yes" if patient.get("history_gdm") else "No")


    st.markdown("---")
    st.subheader("Checklist")
    results = ps_engine.evaluate_patient(patient, k=8)

    st.subheader("Patient state graph (click to drill down)")
    left_g, right_g = st.columns([2,1])

    claim_index = build_claim_index()

    with left_g:
        fig, node_meta = build_patient_graph(patient, results)
        selected = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=560, override_width="100%")
        selected_id = None
        if selected and isinstance(selected, list) and len(selected) > 0:
            selected_id = selected[0].get("customdata")

    with right_g:
        st.markdown("### Details")
        if selected_id and selected_id in node_meta:
            m = node_meta[selected_id]
            kind = m.get("kind")
            st.write(f"**Selected:** {selected_id}")

            if kind == "check":
                data = m.get("check", {})
                st.write(f"**{data.get('sid')} • {data.get('title')}**")
                st.write(f"Status: **{data.get('status')}**")
                st.write(data.get("description",""))
                st.write("**Recommendation:**", data.get("recommendation",""))

                ents = data.get("entities", [])
                if ents:
                    st.write("**Entities:**")
                    for e in ents:
                        st.write(f"- {e.get('canonical_name')} ({e.get('type')})")

                rcids = data.get("related_claim_ids") or []
                if rcids:
                    st.write("**Linked claims:**")
                    for cid in rcids[:10]:
                        c = claim_index.get(cid)
                        if c:
                            label = c.get("title") or c.get("claim_text") or c.get("summary","")
                            st.write(f"- **{cid}** — {label}")
                        else:
                            st.write(f"- {cid} (not in index)")

                st.write("**Evidence (retrieved):**")
                for ev in (data.get("evidence") or [])[:6]:
                    badge = ""
                    if ev.get("slot_primary"):
                        badge = "PRIMARY"
                    elif ev.get("slot_support"):
                        badge = "SUPPORT"
                    sh = ev.get("slot_hits") or {}
                    sh_txt = ", ".join([k for k,v in sh.items() if v])
                    sb = ev.get('score_bm25')
                    ss = ev.get('score_semantic')
                    extra = ''
                    if sb is not None or ss is not None:
                        extra = f" • bm25={float(sb or 0):.3f} • sem={float(ss or 0):.3f}"
                    st.caption(f"{ev.get('doc','')} • {ev.get('pid','')} • score={ev.get('score',0):.3f}{extra} • {badge} • slots: {sh_txt}")
                    st.write(ev.get("text",""))

            elif kind == "claim":
                cid = m.get("claim_id")
                c = claim_index.get(cid, {})
                st.write(f"**Claim {cid}**")
                if c:
                    st.write(c.get("title") or c.get("claim_text") or c.get("summary",""))
                    evs = c.get("evidence") or c.get("evidence_refs") or []
                    if evs:
                        st.write("**Claim evidence:**")
                        passage_index = build_passage_index()
                        resolved = resolve_evidence_refs(evs, passage_index)
                        for r in resolved[:8]:
                            if r.get('text'):
                                st.caption(f"{r.get('doc','')} • {r.get('pid','')}")
                                st.write(r.get('text',''))
                            else:
                                st.write(f"- {r.get('ref')}")
                else:
                    st.info("Claim not found in claim index (expand claim packs).")

            elif kind == "evidence":
                ev = m.get("evidence", {})
                st.write("**Evidence snippet**")
                st.caption(f"{ev.get('doc','')} • {ev.get('pid','')}")
                st.write(ev.get("text",""))

            else:
                st.json(m)
        else:
            st.info("Click a node to inspect checks, claims, and evidence.")

    st.divider()
    st.subheader("Action Pathways (Supervision Tasks)")
    tasks_df = upsert_tasks_from_results(pid=str(patient.get("pid","")), results=results)
    p_tasks = tasks_df[tasks_df["pid"].astype(str) == str(patient.get("pid",""))].copy()
    if len(p_tasks) == 0:
        st.caption("No active tasks for this patient.")
    else:
        st.dataframe(p_tasks.sort_values(["status","due_date"]), use_container_width=True, hide_index=True)


    for res in results:
        status = res["status"]
        icon = {"DUE":"🟠","OVERDUE":"🔴","OK":"🟢","UNKNOWN":"⚪","NOT_APPLICABLE":"➖"}.get(status,"⚪")
        with st.expander(f"{icon} {res['sid']} • {res['title']} — {status}"):
            st.write(res["description"])
            st.write("**Recommendation:**", res["recommendation"])
            st.write("**Why (evidence‑locked):**")
            for ev in res["evidence"]:
                st.write(f"- {ev['doc']} • {ev['pid']} (score={ev['score']:.3f})")
                st.caption(ev["text"])
