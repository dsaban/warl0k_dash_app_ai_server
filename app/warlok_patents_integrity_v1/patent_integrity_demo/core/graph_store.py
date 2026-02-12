from __future__ import annotations
from typing import Dict, List, Any
import json

def build_graph(atomic_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(nid: str, ntype: str, label: str, meta: Dict[str, Any] | None = None):
        if nid in nodes:
            return
        nodes[nid] = {"id": nid, "type": ntype, "label": label, "meta": meta or {}}

    for a in atomic_records:
        aid = a["atomic_id"]
        add_node(aid, "atomic", f'{a["patent_id"]} C{a["claim_no"]} • {a["text"][:60]}', {
            "patent_id": a["patent_id"],
            "claim_no": a["claim_no"],
            "text": a["text"],
            "evidence": a.get("evidence", {}),
            "tags": a.get("tags", {}),
            "strength": a.get("strength", 0.0),
            "is_independent": a.get("is_independent", False),
        })
        tags = a.get("tags", {})
        for cat in ("mechanism", "crypto", "constraint", "threat"):
            for t in tags.get(cat, []):
                tid = f"{cat}:{t}"
                add_node(tid, cat, t, {})
                edges.append({"source": aid, "target": tid, "type": f"HAS_{cat.upper()}"})

    return {"nodes": list(nodes.values()), "edges": edges}

def save_graph(graph: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)
