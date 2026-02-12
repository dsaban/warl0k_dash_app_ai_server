from typing import Dict, Any, List, Tuple

def build_passage_index(passages: List[Dict[str, Any]]) -> Dict[Tuple[str,str], Dict[str, Any]]:
    """Index by (doc, pid)."""
    idx: Dict[Tuple[str,str], Dict[str, Any]] = {}
    for p in passages:
        doc = str(p.get("doc",""))
        pid = str(p.get("pid",""))
        if doc and pid:
            idx[(doc,pid)] = p
    return idx

def resolve_ref(ref: Any) -> Tuple[str,str]:
    """Best-effort parse of evidence ref formats."""
    if isinstance(ref, dict):
        return str(ref.get("doc","")), str(ref.get("pid",""))
    s = str(ref)
    # common format: "Doc.txt • p00015" or "Doc.txt | p00015"
    m = re.search(r"([^•\|]+)\s*[•\|]\s*(p\d+)", s)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # fallback: split
    parts = re.split(r"[•\|]", s)
    if len(parts) >= 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""

import re
def resolve_evidence_refs(evidence_refs: List[Any], passage_index: Dict[Tuple[str,str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for ref in evidence_refs or []:
        doc, pid = resolve_ref(ref)
        if doc and pid and (doc,pid) in passage_index:
            p = passage_index[(doc,pid)]
            out.append({"doc":doc, "pid":pid, "text":p.get("text",""), "ref":ref})
        else:
            out.append({"doc":doc, "pid":pid, "text":"", "ref":ref})
    return out
