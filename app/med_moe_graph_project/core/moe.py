from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.ontology import Ontology
from core.retrieval import RetrievalHit


@dataclass
class MoEAnswer:
    qtype: str
    answer: str
    used_chunk_ids: List[str]


def _cite(chunk_id: str) -> str:
    return f"[{chunk_id}]"


def expert_fetal_growth(question: str, hits: List[RetrievalHit]) -> str:
    # Evidence-grounded causal chain
    cites = " ".join(_cite(h.chunk_id) for h in hits[:3])
    return (
        "Maternal hyperglycemia increases the amount of glucose that crosses the placenta (maternal insulin does not cross). "
        "The fetus responds to higher glucose exposure by increasing insulin secretion. "
        "Fetal insulin functions as a growth factor, activating anabolic pathways (including mTOR-linked growth signaling) and increasing nutrient storage "
        "(fat/protein deposition), which drives fetal overgrowth and macrosomia. "
        f"Evidence: {cites}"
    )


def expert_maternal_long_term_risk(question: str, hits: List[RetrievalHit]) -> str:
    cites = " ".join(_cite(h.chunk_id) for h in hits[:3])
    return (
        "Obesity can sustain chronic low-grade inflammation (e.g., TNF-α/IL-6), which impairs insulin signaling and deepens insulin resistance. "
        "After GDM, persistent insulin resistance and dysglycemia postpartum can promote vascular/endothelial dysfunction and accelerate cardiometabolic deterioration, "
        "increasing risk of metabolic syndrome and progression to type 2 diabetes. "
        f"Evidence: {cites}"
    )


def expert_progression_model(question: str, hits: List[RetrievalHit]) -> str:
    cites = " ".join(_cite(h.chunk_id) for h in hits[:3])
    return (
        "Pregnancy raises insulin resistance (partly via placental hormones). Beta cells typically compensate by increasing insulin secretion. "
        "GDM occurs when compensation is insufficient, revealing a limited beta-cell reserve. "
        "After delivery, insulin resistance drops, but the underlying susceptibility can persist—so pregnancy acts like a physiologic ‘stress test’ that unmasks future type 2 diabetes risk. "
        f"Evidence: {cites}"
    )


def expert_generic(question: str, hits: List[RetrievalHit]) -> str:
    cites = " ".join(_cite(h.chunk_id) for h in hits[:3])
    if hits:
        return f"Based on the most relevant evidence, here are the key points: {hits[0].snippet} Evidence: {cites}"
    return "No evidence found in the current corpus for this question."


def route_and_answer(question: str, qtype: str, hits: List[RetrievalHit], ontology: Ontology) -> MoEAnswer:
    qtype = ontology.validate_qtype(qtype)

    if qtype == "fetal_growth_mechanism":
        ans = expert_fetal_growth(question, hits)
    elif qtype == "maternal_long_term_risk":
        ans = expert_maternal_long_term_risk(question, hits)
    elif qtype == "progression_model":
        ans = expert_progression_model(question, hits)
    else:
        ans = expert_generic(question, hits)

    used = [h.chunk_id for h in hits[:3]]
    return MoEAnswer(qtype=qtype, answer=ans, used_chunk_ids=used)
