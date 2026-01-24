from typing import Dict, List
from .utils import norm
from .schema import QTYPE_REQUIREMENTS

def schema_boost(text: str, required_entities: List[str], section: str = "") -> float:
    t = norm(text)
    s = norm(section)
    boost = 0.0

    # section priors (light)
    if "risk" in s and any(e == "Ethnicity" for e in required_entities):
        boost += 0.25
    if "pathophys" in s and any(e in required_entities for e in ["PregnancyInducedInsulinResistance", "BetaCellCompensation"]):
        boost += 0.25
    if "metabolism" in s and any(e in required_entities for e in ["PregnancyInducedInsulinResistance"]):
        boost += 0.15

    for e in required_entities:
        key = e.lower()
        if "placental" in key and "placental" in t:
            boost += 0.25
        if "ethnicity" in key and ("ethnicity" in t or "race" in t):
            boost += 0.25
        if "insulin" in key and "insulin" in t:
            boost += 0.15
        if "beta" in key and "beta" in t:
            boost += 0.15
        if "gdm" in key and ("gdm" in t or "gestational diabetes" in t):
            boost += 0.20
    return float(boost)

def drift_penalty(qtype: str, text: str) -> float:
    req = QTYPE_REQUIREMENTS.get(qtype)
    if not req:
        return 0.0
    blocks = [norm(x) for x in req.get("drift_block", [])]
    t = norm(text)
    hits = sum(1 for b in blocks if b and b in t)
    return float(0.6 * hits)

def anchor_penalty(qtype: str, text: str) -> float:
    req = QTYPE_REQUIREMENTS.get(qtype)
    if not req:
        return 0.0
    anchors = [norm(a) for a in req.get("anchors", [])]
    if not anchors:
        return 0.0
    t = norm(text)
    hit = sum(1 for a in anchors if a and a in t)
    if hit >= 2:
        return 0.0
    if hit == 1:
        return 0.5
    return 1.0

def score_breakdown(qtype: str, text: str, base_bm25: float, required_entities: List[str], section: str = "") -> Dict[str, float]:
    boost = schema_boost(text, required_entities, section=section)
    pen = drift_penalty(qtype, text) + anchor_penalty(qtype, text)
    total = float(base_bm25 + boost - pen)
    return {"bm25": float(base_bm25), "boost": float(boost), "penalty": float(pen), "total": total}
