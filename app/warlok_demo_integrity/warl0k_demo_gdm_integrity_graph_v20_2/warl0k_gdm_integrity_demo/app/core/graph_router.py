from dataclasses import dataclass
from typing import Dict, Any, List, Set, Tuple

def _split_synonyms(s: str) -> List[str]:
    if not s:
        return []
    return [x.strip().lower() for x in str(s).split("|") if x.strip()]

def _norm(s: str) -> str:
    return (s or "").lower().strip()

@dataclass
class GraphRouter:
    """Route a free-text query into entity_ids -> claim packs -> allowed evidence passages.

    v10.1 improvements:
    - fallback keyword→pack routing when lexicon entity match is empty
    - supports multi-word phrase scanning with safer boundaries
    """
    lexicon_rows: List[Dict[str, Any]]
    claim_packs: List[Dict[str, Any]]
    claims: Dict[str, Dict[str, Any]]

    def __post_init__(self):
        # build phrase index: phrase -> entity_id
        self.phrase_to_entity: List[Tuple[str,str]] = []
        for r in self.lexicon_rows or []:
            eid = str(r.get("entity_id","")) or ""
            canon = str(r.get("canonical_name","")) or ""
            syns = _split_synonyms(r.get("synonyms","")) + ([canon.lower()] if canon else [])
            for p in syns:
                p = _norm(p)
                if len(p) >= 3 and eid:
                    self.phrase_to_entity.append((p, eid))
        # longer phrases first
        self.phrase_to_entity.sort(key=lambda x: len(x[0]), reverse=True)

        self.packs = self.claim_packs or []

        # keyword → pack_id fallback (edit these as you add packs)
        self.kw_pack = [
            ("screen", "PK_SCREENING"),
            ("24-28", "PK_SCREENING"),
            ("24–28", "PK_SCREENING"),
            ("ogtt", "PK_SCREENING"),
            ("threshold", "PK_OGTT_THRESHOLDS"),
            ("cut-off", "PK_OGTT_THRESHOLDS"),
            ("cutoff", "PK_OGTT_THRESHOLDS"),
            ("fasting", "PK_OGTT_THRESHOLDS"),
            ("one-hour", "PK_OGTT_THRESHOLDS"),
            ("two-hour", "PK_OGTT_THRESHOLDS"),
            ("postpartum", "PK_POSTPARTUM"),
            ("6 weeks", "PK_POSTPARTUM"),
            ("4-12", "PK_POSTPARTUM"),
            ("follow-up", "PK_POSTPARTUM"),
            ("risk", "PK_RISKS"),
            ("macrosomia", "PK_RISKS"),
            ("preeclampsia", "PK_RISKS"),
            ("hypoglycemia", "PK_RISKS"),
            ("management", "PK_MANAGEMENT"),
            ("insulin", "PK_MANAGEMENT"),
            ("metformin", "PK_MANAGEMENT"),
        ]

    def detect_entities(self, query: str, max_entities: int = 10) -> List[str]:
        q = _norm(query)
        found: List[str] = []
        # simple substring scan (works well for medical phrases)
        for phrase, eid in self.phrase_to_entity:
            if eid in found:
                continue
            if phrase in q:
                found.append(eid)
                if len(found) >= max_entities:
                    break
        return found

    def select_packs_by_entities(self, entity_ids: List[str], max_packs: int = 4) -> List[Dict[str, Any]]:
        if not entity_ids:
            return []
        e = set(entity_ids)
        scored = []
        for p in self.packs:
            pe = set(p.get("entity_ids") or [])
            inter = len(e & pe)
            if inter > 0:
                scored.append((inter, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _,p in scored[:max_packs]]

    def select_packs_by_keywords(self, query: str, max_packs: int = 3) -> List[Dict[str, Any]]:
        q = _norm(query)
        pack_ids: List[str] = []
        for kw, pid in self.kw_pack:
            if kw in q and pid not in pack_ids:
                pack_ids.append(pid)
        if not pack_ids:
            return []
        # match by pack_id or tag field
        chosen=[]
        for p in self.packs:
            pid = str(p.get("pack_id","")) or ""
            if pid in pack_ids:
                chosen.append(p)
        return chosen[:max_packs]

    def allowed_passages(self, query: str) -> Dict[str, Any]:
        entity_ids = self.detect_entities(query)
        packs = self.select_packs_by_entities(entity_ids)

        # fallback: if no entity packs selected, use keyword routing
        used_fallback = False
        if not packs:
            packs = self.select_packs_by_keywords(query)
            used_fallback = True if packs else False

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
            "used_fallback": used_fallback,
        }
