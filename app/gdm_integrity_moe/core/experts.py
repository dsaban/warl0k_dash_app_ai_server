from dataclasses import dataclass
from typing import List, Dict, Any
from .schema import ENTITY_LEXICON
from .utils import norm

@dataclass
class Candidate:
    expert_name: str
    sentences: List[Dict[str, Any]]
    used_entities: List[str]
    used_edges: List[Dict[str, str]]
    blueprint: Dict[str, Any]
    retrieval_scores: List[float]

def _etype(ent_key: str) -> str:
    et, _ = ENTITY_LEXICON.get(ent_key, (None, None))
    return et.name if et else "Unknown"

def _edge(src: str, edge: str, dst: str) -> Dict[str, str]:
    return {"src": src, "edge": edge, "dst": dst}

def infer_edges_from_claim(claim) -> List[Dict[str,str]]:
    s = norm(getattr(claim, "support_text", claim.sentence))
    ents = set(claim.entities)
    edges: List[Dict[str,str]] = []

    hormones = [e for e in ents if _etype(e) == "Hormone"]
    if hormones and "PregnancyInducedInsulinResistance" in ents and "insulin resistance" in s:
        edges.append(_edge(hormones[0], "INDUCES", "PregnancyInducedInsulinResistance"))

    if "PregnancyInducedInsulinResistance" in ents and "BetaCellCompensation" in ents and any(k in s for k in ["compens", "increase insulin secretion", "overcome"]):
        edges.append(_edge("PregnancyInducedInsulinResistance", "INCREASES", "BetaCellCompensation"))

    if "BetaCellDysfunction" in ents and "GestationalDiabetesMellitus" in ents and any(k in s for k in ["inadequate","insufficient","fails","cannot compensate","dysfunction"]):
        edges.append(_edge("BetaCellDysfunction", "LEADS_TO", "GestationalDiabetesMellitus"))

    if "Ethnicity" in ents and "GestationalDiabetesMellitus" in ents and ("independent risk factor" in s or "independently associated" in s):
        edges.append(_edge("Ethnicity", "IS_INDEPENDENT_RISK_FACTOR_FOR", "GestationalDiabetesMellitus"))

    if "Ethnicity" in ents and ("PopulationT2DPrevalence" in ents or "Type2DiabetesMellitus" in ents) and any(k in s for k in ["linked to","correlates","associated","background population","population prevalence"]):
        edges.append(_edge("Ethnicity", "CORRELATES_WITH", "PopulationT2DPrevalence"))

    return edges

def _mk_sent(claim) -> Dict[str,Any]:
    return {
        "text": claim.sentence,
        "claim_refs": [claim.claim_id],
        "entities": claim.entities,
        "edges": infer_edges_from_claim(claim),
    }

def _dedup(xs):
    seen=set(); out=[]
    for x in xs:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def _pick_by_entity(claims, ranked, entity, n=1):
    out=[]
    for i, sc in ranked:
        if entity in claims[i].entities:
            out.append((i, sc))
            if len(out) >= n: break
    return out

def _pick_by_phrase(claims, ranked, phrases, n=1):
    out=[]
    for i, sc in ranked:
        t = norm(getattr(claims[i], "support_text", claims[i].sentence))
        if any(norm(p) in t for p in phrases):
            out.append((i, sc))
            if len(out) >= n: break
    return out

def expert_mechanism(claims, ranked) -> Candidate:
    picks=[]
    picks += _pick_by_phrase(claims, ranked, ["placental", "hpl", "progesterone", "cortisol", "growth hormone", "estrogen", "prolactin"], 1)
    picks += _pick_by_entity(claims, ranked, "PregnancyInducedInsulinResistance", 1)
    picks += _pick_by_entity(claims, ranked, "BetaCellCompensation", 1)
    picks += _pick_by_entity(claims, ranked, "BetaCellDysfunction", 1)
    picks += _pick_by_entity(claims, ranked, "GestationalDiabetesMellitus", 1)
    picks = _dedup([i for i,_ in picks])[:6]

    sents=[]; ents=[]; edges=[]; scores=[]
    pickset=set(picks)
    for i, sc in ranked:
        if i in pickset:
            s = _mk_sent(claims[i])
            sents.append(s)
            ents += claims[i].entities
            edges += s["edges"]
            scores.append(float(sc))

    return Candidate(
        expert_name="mechanism",
        sentences=sents,
        used_entities=list(dict.fromkeys(ents)),
        used_edges=edges,
        blueprint={"question_type":"mechanism_hormone_to_gdm","sentences":[{"claim_refs":x["claim_refs"],"entities":x["entities"],"edges":x["edges"]} for x in sents]},
        retrieval_scores=scores
    )

def expert_risk_ethnicity(claims, ranked) -> Candidate:
    picks=[]
    picks += _pick_by_entity(claims, ranked, "Ethnicity", 2)
    picks += _pick_by_phrase(claims, ranked, ["independent risk factor","independently associated"], 1)
    picks += _pick_by_phrase(claims, ranked, ["background population", "population prevalence", "type 2 diabetes"], 1)
    picks += _pick_by_entity(claims, ranked, "GestationalDiabetesMellitus", 1)
    picks = _dedup([i for i,_ in picks])[:6]

    sents=[]; ents=[]; edges=[]; scores=[]
    pickset=set(picks)
    for i, sc in ranked:
        if i in pickset:
            s = _mk_sent(claims[i])
            sents.append(s)
            ents += claims[i].entities
            edges += s["edges"]
            scores.append(float(sc))

    return Candidate(
        expert_name="risk_ethnicity",
        sentences=sents,
        used_entities=list(dict.fromkeys(ents)),
        used_edges=edges,
        blueprint={"question_type":"risk_ethnicity_t2d_link","sentences":[{"claim_refs":x["claim_refs"],"entities":x["entities"],"edges":x["edges"]} for x in sents]},
        retrieval_scores=scores
    )

def expert_generic(claims, ranked) -> Candidate:
    if not ranked:
        return Candidate("generic", [], [], [], {"question_type":"generic","sentences":[]}, [])
    i, sc = ranked[0]
    s = _mk_sent(claims[i])
    return Candidate(
        expert_name="generic",
        sentences=[s],
        used_entities=list(dict.fromkeys(claims[i].entities)),
        used_edges=s["edges"],
        blueprint={"question_type":"generic","sentences":[{"claim_refs":s["claim_refs"],"entities":s["entities"],"edges":s["edges"]}]},
        retrieval_scores=[float(sc)]
    )

EXPERTS = {
    "mechanism": expert_mechanism,
    "risk_ethnicity": expert_risk_ethnicity,
    "generic": expert_generic,
}
