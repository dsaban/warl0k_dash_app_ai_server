from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Any

from core.parse_claims import parse_claims_from_text
from core.atomic import build_atomic_claims
from core.tagger import tag_text, strength_score
from core.bm25 import BM25Index

def load_text_files(paths: List[Path]) -> List[Dict[str, Any]]:
    docs = []
    for p in paths:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8", errors="ignore")
        docs.append({"path": str(p), "name": p.name, "text": txt})
    return docs

def build_corpus_from_paths(paths: List[str]) -> Dict[str, Any]:
    docs = load_text_files([Path(p) for p in paths])
    claims_all: List[Dict[str, Any]] = []
    atomic_all: List[Dict[str, Any]] = []

    for d in docs:
        file_stub = Path(d["name"]).stem
        claims = parse_claims_from_text(d["text"], file_stub=file_stub)
        claims_all.extend([asdict(c) for c in claims])

        for c in claims:
            atoms = build_atomic_claims(c.patent_id, c.claim_no, c.text, c.is_independent, d["name"])
            for a in atoms:
                rec = asdict(a)
                rec["tags"] = tag_text(a.text)
                rec["strength"] = strength_score(a.text, a.is_independent)
                atomic_all.append(rec)

    bm25_docs = []
    for a in atomic_all:
        tag_blob = " ".join(
            a.get("tags", {}).get("mechanism", []) +
            a.get("tags", {}).get("crypto", []) +
            a.get("tags", {}).get("constraint", []) +
            a.get("tags", {}).get("threat", [])
        )
        bm25_docs.append(a["text"] + " " + tag_blob)

    bm25 = BM25Index.build(bm25_docs) if bm25_docs else None

    return {"docs": docs, "claims": claims_all, "atomic": atomic_all, "bm25": bm25}
