# core/role.py
from dataclasses import dataclass
from typing import List, Tuple
from .utils import normalize_text

MATERNAL = ("maternal","woman","women","mother","pregnancy","postpartum","beta","beta-cell","insulin secretion","cardiometabolic","cardiovascular","dyslipidemia","hypertension","metabolic syndrome","endothelial")
OFFSPRING = ("offspring","child","fetal","fetus","neonatal","infant","birth weight","macrosomia","childhood")
MONITORING = ("cgm","monitoring","hba1c","hypoglycemia","smbg","sensor","adherence","cost","time-in-range","tir")

@dataclass
class RoleScores:
    maternal: float
    offspring: float
    monitoring: float
    target: str
    alignment: float

def _count_hits(text: str, terms: Tuple[str, ...]) -> int:
    t = " " + normalize_text(text) + " "
    hits = 0
    for w in terms:
        if f" {w} " in t:
            hits += 1
    return hits

def role_alignment(chunks: List[str], target_role: str) -> RoleScores:
    agg = {"maternal": 0.0, "offspring": 0.0, "monitoring": 0.0}
    for ch in chunks[:4]:
        agg["maternal"] += _count_hits(ch, MATERNAL)
        agg["offspring"] += _count_hits(ch, OFFSPRING)
        agg["monitoring"] += _count_hits(ch, MONITORING)

    if target_role == "maternal":
        align = agg["maternal"] - max(agg["offspring"], agg["monitoring"])
    elif target_role == "offspring":
        align = agg["offspring"] - max(agg["maternal"], agg["monitoring"])
    elif target_role == "monitoring":
        align = agg["monitoring"] - max(agg["maternal"], agg["offspring"])
    else:
        align = 999.0

    return RoleScores(
        maternal=float(agg["maternal"]),
        offspring=float(agg["offspring"]),
        monitoring=float(agg["monitoring"]),
        target=target_role,
        alignment=float(align),
    )
