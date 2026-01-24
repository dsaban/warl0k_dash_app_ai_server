from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass(frozen=True)
class EntityType:
    name: str

Condition = EntityType("Condition")
PhysiologicProcess = EntityType("PhysiologicProcess")
Hormone = EntityType("Hormone")
Mechanism = EntityType("Mechanism")
RiskFactor = EntityType("RiskFactor")

# Expand these as your corpus grows.
ENTITY_LEXICON: Dict[str, Tuple[EntityType, List[str]]] = {
    "GestationalDiabetesMellitus": (Condition, ["gdm", "gestational diabetes", "gestational diabetes mellitus"]),
    "Type2DiabetesMellitus": (Condition, ["type 2 diabetes", "t2d", "type ii diabetes"]),
    "Prediabetes": (Condition, ["prediabetes", "pre-diabetes"]),

    "PregnancyInducedInsulinResistance": (PhysiologicProcess, ["insulin resistance during pregnancy", "insulin resistance in pregnancy", "pregnancy-induced insulin resistance", "insulin resistance"]),
    "BetaCellCompensation": (PhysiologicProcess, ["beta-cell compensation", "compensatory increase in insulin secretion", "increased insulin secretion", "compensate by increasing insulin secretion"]),
    "BetaCellDysfunction": (Mechanism, ["beta-cell dysfunction", "beta cell dysfunction", "inadequate compensatory response", "insufficient insulin secretion", "fails to compensate"]),

    "PlacentalLactogen": (Hormone, ["human placental lactogen", "placental lactogen", "hpl"]),
    "PlacentalGrowthHormone": (Hormone, ["placental growth hormone"]),
    "Progesterone": (Hormone, ["progesterone"]),
    "Estrogen": (Hormone, ["estrogen", "oestrogen"]),
    "Cortisol": (Hormone, ["cortisol"]),
    "Prolactin": (Hormone, ["prolactin"]),

    "Ethnicity": (RiskFactor, ["ethnicity", "ethnic", "race", "racial"]),
    "PopulationT2DPrevalence": (Mechanism, ["background population rates of type 2 diabetes", "background prevalence of type 2 diabetes", "population prevalence of type 2 diabetes"]),
}

# Simple cue words for edge hints.
EDGE_CUES = {
    "INDUCES": ["induces", "drives", "causes", "leads to", "results in"],
    "INCREASES": ["increases", "raises", "elevates"],
    "COUNTERACTS": ["compensates", "counteracts", "overcome"],
    "LEADS_TO": ["leads to", "results in", "causes"],
    "IS_INDEPENDENT_RISK_FACTOR_FOR": ["independent risk factor", "independently associated"],
    "CORRELATES_WITH": ["linked to", "correlates with", "associated with", "reflects"],
}

# Hard requirements + drift blocks for qtypes.
QTYPE_REQUIREMENTS = {
    "mechanism_hormone_to_gdm": {
        "required_entities_all": ["PregnancyInducedInsulinResistance", "BetaCellCompensation", "GestationalDiabetesMellitus"],
        "required_entities_any": ["PlacentalLactogen","PlacentalGrowthHormone","Progesterone","Estrogen","Cortisol","Prolactin","BetaCellDysfunction"],
        "required_edges": [
            ("Hormone", "INDUCES", "PregnancyInducedInsulinResistance"),
            ("PregnancyInducedInsulinResistance", "INCREASES", "BetaCellCompensation"),
            ("BetaCellDysfunction", "LEADS_TO", "GestationalDiabetesMellitus"),
        ],
        "drift_block": ["cgm","hba1c","smbg","hypoglycemia","cost","adherence"],
        "anchors": ["placental", "hormone", "insulin resistance", "beta", "insulin secretion", "compens"],
    },
    "risk_ethnicity_t2d_link": {
        "required_entities_all": ["Ethnicity", "GestationalDiabetesMellitus"],
        "required_entities_any": ["PopulationT2DPrevalence", "Type2DiabetesMellitus"],
        "required_edges": [
            ("Ethnicity", "IS_INDEPENDENT_RISK_FACTOR_FOR", "GestationalDiabetesMellitus"),
            ("Ethnicity", "CORRELATES_WITH", "PopulationT2DPrevalence"),
        ],
        "drift_block": ["cgm","hba1c","smbg","hypoglycemia","cost","adherence"],
        "anchors": ["ethnicity", "independent", "background", "population", "type 2"],
    },
}
