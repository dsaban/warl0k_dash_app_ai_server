from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.ontology import Ontology


@dataclass
class IntegrityResult:
    ok: bool
    missing: List[str]
    notes: str


def _presence(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def check_answer_integrity(answer: str, qtype: str, ontology: Ontology) -> IntegrityResult:
    spec = ontology.qtypes[ontology.validate_qtype(qtype)]
    missing: List[str] = []

    # Simple checklist heuristics (customize per qtype)
    t = answer.lower()

    if qtype == "fetal_growth_mechanism":
        if "placenta" not in t or "cross" not in t:
            missing.append("maternal_glucose_crosses_placenta")
        if "fetal" not in t or "insulin" not in t:
            missing.append("fetal_hyperinsulinemia")
        if "growth" not in t and "anabolic" not in t and "storage" not in t:
            missing.append("insulin_is_growth_factor")
        if "macrosomia" not in t and "overgrowth" not in t:
            missing.append("net_anabolic_storage")

    elif qtype == "maternal_long_term_risk":
        if not _presence(answer, ["inflamm", "cytok", "tnf", "il-6"]):
            missing.append("obesity_inflammation")
        if "insulin resistance" not in t:
            missing.append("persistent_insulin_resistance")
        if "endothel" not in t and "vascular" not in t:
            missing.append("endothelial_dysfunction")
        if "metabolic syndrome" not in t and "type 2" not in t:
            missing.append("metabolic_syndrome_progression")

    elif qtype == "progression_model":
        if "pregnan" not in t and "placental" not in t:
            missing.append("pregnancy_ir_rises")
        if "beta" not in t or "compens" not in t:
            missing.append("beta_cell_compensation")
        if "fail" not in t and "insufficient" not in t and "decomp" not in t:
            missing.append("failure_leads_hyperglycemia")
        if "postpartum" not in t and "future" not in t:
            missing.append("postpartum_unmasks_risk")

    # unknown / management: skip strict checks

    ok = len(missing) == 0
    notes = "" if ok else f"Missing checklist items: {', '.join(missing)}"
    return IntegrityResult(ok=ok, missing=missing, notes=notes)
