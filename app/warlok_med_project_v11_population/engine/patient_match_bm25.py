from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

from .patient_record import PatientRecord
from .bm25 import BM25
from .retrieve import load_claims
from .domain_pack import DomainPack


def load_patient_layer(domain: DomainPack) -> Dict[str, Any]:
    files = (domain.manifest.get("files") or {})
    p = domain.pack_dir / files.get("patient_layer", "patient_layer.json")
    if not p.exists():
        return {}
    return json.loads(p.read_text(errors="ignore"))


def extract_fields_from_notes(notes: str, patient_layer: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    notes = notes or ""
    for ex in (patient_layer.get("field_extractors") or []):
        rx = ex.get("regex")
        if not rx:
            continue
        m = re.search(rx, notes, flags=re.I)
        if not m:
            continue
        val = m.group(1)
        if ex.get("type") == "number":
            try:
                out[ex.get("field")] = float(val)
            except:
                out[ex.get("field")] = val
        else:
            out[ex.get("field")] = val
    return out


# def _patient_query(rec: PatientRecord) -> str:
#     parts = []
#     if rec.notes:
#         parts.append(rec.notes)
#     for k, v in (rec.structured or {}).items():
#         parts.append(f"{k}: {v}")
#     return " | ".join(parts).strip()

def _patient_query(rec: PatientRecord, patient_layer: Dict[str, Any]) -> str:
    parts = []
    if rec.notes:
        parts.append(rec.notes)
    for k, v in (rec.structured or {}).items():
        parts.append(f"{k}: {v}")

    # add extracted field names as tokens (helps overlap)
    for k in (rec.extracted or {}).keys():
        parts.append(str(k))

    # pack-driven query anchors (critical for BM25 overlap)
    for t in (patient_layer.get("default_query_terms") or []):
        parts.append(t)

    return " | ".join([p for p in parts if str(p).strip()]).strip()

def match_patient_to_concepts(rec: PatientRecord, domain: DomainPack, topk: int = 8) -> List[Dict[str, Any]]:
    layer = load_patient_layer(domain)
    concepts = layer.get("concepts") or []
    if not concepts:
        return []

    rows = []
    docs = []
    for c in concepts:
        cid = c.get("id")
        label = c.get("label", cid)
        edges_hint = c.get("edges_hint") or []
        for ph in (c.get("phrases") or []):
            rows.append({"concept_id": cid, "label": label, "phrase": ph, "edges_hint": edges_hint})
            docs.append(ph)

    # q = _patient_query(rec)
    q = _patient_query(rec, layer)
    
    bm = BM25(docs)
    hits = bm.topk(q, k=min(topk, len(docs)))

    out = []
    for idx, score in hits:
        r = rows[idx]
        out.append({
            "concept_id": r["concept_id"],
            "label": r["label"],
            "phrase": r["phrase"],
            "score": score,
            "edges_hint": r["edges_hint"]
        })
    return out


def match_patient_to_claims(rec: PatientRecord, index_dir: Path, topk: int = 12) -> List[Dict[str, Any]]:
    claims = load_claims(index_dir)
    texts = [c.text for c in claims]
    bm = BM25(texts)
    # q = _patient_query(rec)
    layer = load_patient_layer(DomainPack(index_dir.parent))
    print(f"Loaded patient layer for claims matching: {bool(layer)}")
    q = _patient_query(rec, layer)
    
    hits = bm.topk(q, k=min(topk, len(texts)))

    out = []
    for idx, score in hits:
        c = claims[idx]
        out.append({
            "claim_id": c.id,
            "doc": c.doc,
            "text": c.text,
            "score": score,
            "edges": c.edge_types
        })
    return out


def build_surveillance(rec: PatientRecord, domain: DomainPack) -> List[Dict[str, Any]]:
    layer = load_patient_layer(domain)
    if not layer:
        return []
    matched = {c["concept_id"] for c in (rec.matched_concepts or [])}

    out = []
    for tmpl in (layer.get("surveillance_templates") or []):
        trig = set(tmpl.get("triggers_any_concepts") or [])
        if trig and not (matched & trig):
            continue
        out.append(tmpl)
    return out


def expand_question_with_patient_context(question: str, rec: PatientRecord, max_concepts: int = 5) -> str:
    """
    Generic query expansion: append top concept labels + extracted fields.
    This improves retrieval without changing medical logic.
    """
    parts = [question.strip()]
    if rec.extracted:
        fields = " ".join([f"{k}={v}" for k, v in list(rec.extracted.items())[:8]])
        if fields:
            parts.append(f"[patient_fields: {fields}]")
    if rec.matched_concepts:
        labels = ", ".join([c["label"] for c in rec.matched_concepts[:max_concepts]])
        parts.append(f"[patient_concepts: {labels}]")
    return " ".join(parts)
