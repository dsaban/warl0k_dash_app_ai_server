import re
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

# --- Slot patterns (lightweight, explainable) ---

_WEEK_RANGE_PAT = re.compile(r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*(?:weeks?|wks?)\b", re.IGNORECASE)
_WEEK_SINGLE_PAT = re.compile(r"\b(\d{1,2})\s*(?:weeks?|wks?)\b", re.IGNORECASE)

def _has_ogtt(text: str) -> bool:
    t = text.lower()
    return ("ogtt" in t) or ("oral glucose tolerance" in t)

def _has_75g(text: str) -> bool:
    t = text.lower()
    return ("75 g" in t) or ("75g" in t)

def _has_postpartum_window(text: str) -> bool:
    t = text.lower()
    # allow "4-12 weeks" or "6 weeks" style
    if _WEEK_RANGE_PAT.search(t):
        a,b = _WEEK_RANGE_PAT.search(t).groups()
        try:
            a=int(a); b=int(b)
            return (a <= 12 and b >= 4)  # overlaps 4–12
        except:
            pass
    # fallback: mentions postpartum and some weeks
    return ("postpartum" in t and bool(_WEEK_SINGLE_PAT.search(t)))

def _has_24_28_window(text: str) -> bool:
    t = text.lower()
    m = _WEEK_RANGE_PAT.search(t)
    if not m:
        # sometimes written "between 24-28 weeks of gestation"
        if "24" in t and "28" in t and "week" in t:
            return True
        return False
    a,b = m.groups()
    try:
        a=int(a); b=int(b)
        # accept exact or containing range
        return (a <= 24 and b >= 28) or (a==24 and b==28)
    except:
        return False

def _has_cutoffs_ada(text: str) -> bool:
    t = text.lower()
    # common trio for 75g OGTT
    return ("92" in t and "180" in t and "153" in t)

@dataclass
class SlotResult:
    qtype: str
    slot_hits: Dict[str, bool]
    slot_score: float
    primary: bool
    support: bool

class SlotGuard:
    """Explainable slot-based gating for integrity: primary evidence must satisfy required slots."""

    def schema(self, qtype: str) -> Tuple[List[str], List[str]]:
        # required slots, optional slots
        if qtype == "screening_window":
            return ["weeks_24_28", "test_ogtt"], ["load_75g"]
        if qtype == "postpartum_followup":
            return ["postpartum_timing", "test_ogtt"], ["load_75g", "long_term_1_3y"]
        if qtype == "ogtt_thresholds":
            return ["test_ogtt", "load_75g", "cutoffs"], []
        if qtype == "mechanism_ir":
            return ["insulin_resistance", "beta_cell"], ["placental_hormones"]
        if qtype == "risks":
            return ["maternal_risk", "neonatal_risk"], ["pathway"]
        return [], []

    def evaluate_passage(self, text: str, qtype: str) -> SlotResult:
        req, opt = self.schema(qtype)
        hits: Dict[str, bool] = {}

        t = text or ""
        tl = t.lower()

        # Screening
        hits["weeks_24_28"] = _has_24_28_window(t)
        hits["test_ogtt"] = _has_ogtt(t)
        hits["load_75g"] = _has_75g(t)

        # Postpartum
        hits["postpartum_timing"] = _has_postpartum_window(t)
        hits["long_term_1_3y"] = ("1" in tl and "3" in tl and ("year" in tl or "years" in tl))

        # Thresholds
        hits["cutoffs"] = _has_cutoffs_ada(t)

        # Mechanism / Risks (light)
        hits["insulin_resistance"] = ("insulin resistance" in tl)
        hits["beta_cell"] = ("beta" in tl and "cell" in tl) or ("insulin secretion" in tl) or ("insulin" in tl and "compens" in tl)
        hits["placental_hormones"] = ("placent" in tl and ("hormone" in tl or "lactogen" in tl or "progesterone" in tl or "cortisol" in tl))

        hits["maternal_risk"] = any(x in tl for x in ["preeclampsia","hypertension","cesarean","c-section","caesarean"])
        hits["neonatal_risk"] = any(x in tl for x in ["macrosomia","hypoglyc","shoulder dystocia","respiratory"])
        hits["pathway"] = any(x in tl for x in ["placenta","hyperinsulin","fetal insulin"])

        # scoring: required hit = 1, optional hit = 0.5
        req_hits = sum(1 for s in req if hits.get(s, False))
        opt_hits = sum(1 for s in opt if hits.get(s, False))
        slot_score = 0.0
        if req:
            slot_score = req_hits / float(len(req)) + 0.25 * (opt_hits / float(max(1,len(opt))))
        primary = (req_hits == len(req)) if req else False
        support = (req_hits >= max(1, len(req)//2)) if req else False

        # keep only relevant hits
        rel = {k:v for k,v in hits.items() if k in set(req+opt)}
        return SlotResult(qtype=qtype, slot_hits=rel, slot_score=float(slot_score), primary=primary, support=support)


def _extract_week_ranges(text: str):
    ranges=[]
    for m in _WEEK_RANGE_PAT.finditer(text or ""):
        try:
            a=int(m.group(1)); b=int(m.group(2))
            ranges.append((a,b))
        except: 
            continue
    return ranges

def _extract_threshold_triple(text: str):
    t=text.lower()
    # crude: look for 92,180,153 anywhere
    if _has_cutoffs_ada(text):
        return (92,180,153)
    return None

def detect_conflicts(evidence: List[Dict[str, Any]], qtype: str) -> Dict[str, Any]:
    """Detect simple numeric conflicts among PRIMARY evidence."""
    prim = [e for e in (evidence or []) if e.get("slot_primary")]
    out = {"has_conflict": False, "kind": None, "details": ""}
    if len(prim) < 2:
        return out

    if qtype == "postpartum_followup":
        all_ranges=[]
        for e in prim:
            all_ranges += _extract_week_ranges(e.get("text",""))
        # if we have multiple distinct ranges, flag
        uniq = sorted(set(all_ranges))
        if len(uniq) >= 2:
            out.update({
                "has_conflict": True,
                "kind": "postpartum_timing_ranges",
                "details": f"Multiple postpartum week ranges in PRIMARY evidence: {uniq}"
            })
        return out

    if qtype == "ogtt_thresholds":
        triples=[]
        for e in prim:
            tr=_extract_threshold_triple(e.get("text",""))
            if tr: triples.append(tr)
        uniq=sorted(set(triples))
        if len(uniq) >= 2:
            out.update({
                "has_conflict": True,
                "kind": "ogtt_cutoffs",
                "details": f"Multiple OGTT cutoff sets in PRIMARY evidence: {uniq}"
            })
        return out

    return out

def unit_coverage_penalty(answer: str, evidence_text: str) -> float:
    """Penalize if answer uses units not present in evidence."""
    a = (answer or "").lower()
    e = (evidence_text or "").lower()
    pen = 0.0
    if "mg/dl" in a and "mg/dl" not in e:
        pen += 0.10
    if "mmol" in a and "mmol" not in e:
        pen += 0.10
    return pen
