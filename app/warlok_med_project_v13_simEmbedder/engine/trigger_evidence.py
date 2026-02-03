# engine/trigger_evidence.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .retrieve import load_claims
from .bm25 import BM25


def _edges_overlap(claim_edges: List[str], wanted_edges: List[str]) -> List[str]:
    s = set([e for e in (claim_edges or []) if isinstance(e, str)])
    w = set([e for e in (wanted_edges or []) if isinstance(e, str)])
    return sorted(list(s.intersection(w)))


def attach_evidence_to_triggers(
    index_dir,
    triggers_out: Dict[str, Any],
    query_text: str,
    per_item_topk: int = 3,
    edge_pool_limit: int = 400
) -> Dict[str, Any]:
    """
    For each derived state/alert emitted by triggers, attach best supporting claim snippets.

    Ranking strategy:
      1) filter claims by overlap with item.evidence_edges_any
      2) BM25 rank the filtered pool using query_text (patient-context query)
      3) attach top-k claims as 'evidence_claims'

    Returns a NEW triggers_out dict (does not mutate input).
    """
    claims = load_claims(index_dir)
    if not claims:
        return triggers_out

    # Pre-build BM25 over all claim texts (fast enough for local)
    bm25 = BM25([c.text for c in claims])

    def rank_for_edges(wanted_edges: List[str]) -> List[Dict[str, Any]]:
        if not wanted_edges:
            return []

        # filter by edge overlap first
        candidates_idx: List[int] = []
        overlaps: Dict[int, List[str]] = {}

        for i, c in enumerate(claims):
            ov = _edges_overlap(c.edge_types or [], wanted_edges)
            if ov:
                candidates_idx.append(i)
                overlaps[i] = ov

        if not candidates_idx:
            return []

        # optional pool limit
        if len(candidates_idx) > edge_pool_limit:
            candidates_idx = candidates_idx[:edge_pool_limit]

        # BM25 score across all, then pick candidate subset
        scores = bm25.score(query_text or "")
        candidates_idx.sort(key=lambda i: scores[i], reverse=True)

        out: List[Dict[str, Any]] = []
        for i in candidates_idx[: max(1, per_item_topk)]:
            c = claims[i]
            out.append({
                "claim_id": c.id,
                "doc": c.doc,
                "conf": getattr(c, "conf", None),
                "edges": c.edge_types or [],
                "edges_matched": overlaps.get(i, []),
                "score": float(scores[i]),
                "text": c.text
            })
        return out

    # Create a copy and attach evidence
    new_out = {
        "derived_states": [],
        "alerts": [],
        "debug": dict(triggers_out.get("debug") or {})
    }

    for item in (triggers_out.get("derived_states") or []):
        wanted = item.get("evidence_edges_any", []) or []
        ev = rank_for_edges(wanted)
        it = dict(item)
        it["evidence_claims"] = ev
        new_out["derived_states"].append(it)

    for item in (triggers_out.get("alerts") or []):
        wanted = item.get("evidence_edges_any", []) or []
        ev = rank_for_edges(wanted)
        it = dict(item)
        it["evidence_claims"] = ev
        new_out["alerts"].append(it)

    new_out["debug"]["evidence_attach"] = {
        "per_item_topk": per_item_topk,
        "edge_pool_limit": edge_pool_limit,
        "query_len": len(query_text or "")
    }
    return new_out
