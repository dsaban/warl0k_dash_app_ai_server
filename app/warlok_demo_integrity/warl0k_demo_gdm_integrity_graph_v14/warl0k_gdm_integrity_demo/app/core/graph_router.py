from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple, Optional

def _split_synonyms(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]

@dataclass
class GraphRouter:
    """Route a free-text query into entity_ids -> claim packs -> allowed evidence passages."""
    lexicon_rows: List[Dict[str, Any]]
    claim_packs: List[Dict[str, Any]]
    claims: Dict[str, Dict[str, Any]]

    def __post_init__(self):
        # build synonym index: phrase -> entity_id
        self.phrase_to_entity: List[Tuple[str,str]] = []
        for r in self.lexicon_rows or []:
            eid = str(r.get("entity_id","")) or ""
            canon = str(r.get("canonical_name","")) or ""
            syns = _split_synonyms(r.get("synonyms","")) + ([canon.lower()] if canon else [])
            # keep longer phrases first to avoid tiny matches
            for p in syns:
                if len(p) >= 3:
                    self.phrase_to_entity.append((p, eid))
        self.phrase_to_entity.sort(key=lambda x: len(x[0]), reverse=True)

        # pack index
        self.pack_by_id = {str(p.get("pack_id")): p for p in (self.claim_packs or [])}

    def detect_entities(self, query: str, max_entities: int = 8) -> List[str]:
        q = (query or "").lower()
        found: List[str] = []
        for phrase, eid in self.phrase_to_entity:
            if eid in found:
                continue
            if phrase in q:
                found.append(eid)
                if len(found) >= max_entities:
                    break
        return found

    def select_packs(self, entity_ids: List[str], max_packs: int = 3) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        e = set(entity_ids)
        scored = []
        for p in self.claim_packs or []:
            pe = set(p.get("entity_ids") or [])
            inter = len(e & pe)
            if inter > 0:
                scored.append((inter, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _,p in scored[:max_packs]]

    def allowed_passages(self, query: str) -> Dict[str, Any]:
        entity_ids = self.detect_entities(query)
        packs = self.select_packs(entity_ids)
        claim_ids: Set[str] = set()
        for p in packs:
            for cid in (p.get("claim_ids") or []):
                claim_ids.add(str(cid))
        allowed: Set[Tuple[str,str]] = set()
        for cid in claim_ids:
            c = self.claims.get(cid)
            if not c:
                continue
            for ev in (c.get("evidence") or []):
                doc = str(ev.get("doc",""))
                pid = str(ev.get("pid",""))
                if doc and pid:
                    allowed.add((doc,pid))
        return {
            "entity_ids": entity_ids,
            "pack_ids": [str(p.get("pack_id")) for p in packs],
            "claim_ids": sorted(list(claim_ids)),
            "allowed": allowed,
        }
