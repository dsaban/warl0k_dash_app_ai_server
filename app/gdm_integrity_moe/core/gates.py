from typing import Dict, List, Any
from .schema import QTYPE_REQUIREMENTS, ENTITY_LEXICON
from .utils import norm

def _etype_name(ent_key: str) -> str:
    etype, _ = ENTITY_LEXICON.get(ent_key, (None, None))
    return etype.name if etype else "Unknown"

def gate_grounding(sentences: List[Dict[str,Any]]) -> List[str]:
    for s in sentences:
        if not s.get("claim_refs"):
            return ["ungrounded_sentence"]
    return []

def gate_required_entities(qtype: str, used_entities: List[str]) -> List[str]:
    req = QTYPE_REQUIREMENTS.get(qtype)
    if not req:
        return []
    have = set(used_entities)
    flags = []
    for e in req.get("required_entities_all", []):
        if e not in have:
            flags.append(f"missing_entity:{e}")
    any_list = req.get("required_entities_any", [])
    if any_list and not any(e in have for e in any_list):
        flags.append("missing_any_entity:" + "|".join(any_list))
    return flags

def gate_required_edges(qtype: str, used_edges: List[Dict[str,str]]) -> List[str]:
    req = QTYPE_REQUIREMENTS.get(qtype)
    if not req:
        return []
    have = set((e["src"], e["edge"], e["dst"]) for e in used_edges)
    flags = []
    for src, edge, dst in req.get("required_edges", []):
        ok = False
        for hs, he, hd in have:
            # allow matching by entity type name for src/dst (e.g., "Hormone")
            if (hs == src or _etype_name(hs) == src) and he == edge and (hd == dst or _etype_name(hd) == dst):
                ok = True
                break
        if not ok:
            flags.append(f"missing_edge:{src}-{edge}-{dst}")
    return flags

def gate_drift(qtype: str, answer_text: str) -> List[str]:
    req = QTYPE_REQUIREMENTS.get(qtype)
    if not req:
        return []
    blocks = [norm(x) for x in req.get("drift_block", [])]
    t = norm(answer_text)
    hits = sum(1 for b in blocks if b and b in t)
    return ["drift_terms_present"] if hits >= 2 else []

def gate_redundancy(sentences: List[Dict[str,Any]]) -> List[str]:
    seen = set()
    for s in sentences:
        refs = tuple(s.get("claim_refs", []))
        if refs in seen:
            return ["redundant_claims"]
        seen.add(refs)
    return []

def evaluate_all(
    qtype: str,
    answer_sentences: List[Dict[str,Any]],
    used_entities: List[str],
    used_edges: List[Dict[str,str]],
    answer_text: str
) -> Dict[str,Any]:
    flags = []
    flags += gate_grounding(answer_sentences)
    flags += gate_required_entities(qtype, used_entities)
    flags += gate_required_edges(qtype, used_edges)
    flags += gate_drift(qtype, answer_text)
    flags += gate_redundancy(answer_sentences)
    return {"decision": "ALLOW" if not flags else "ABSTAIN", "flags": flags}
