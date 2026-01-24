from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FrameSpec:
    id: str
    name: str
    required_edges: List[str]
    min_required_covered: int
    allowed_node_types: List[str]
    blocked_node_types: List[str] = field(default_factory=list)
    min_steps: int = 3
    max_steps: int = 6

FRAMES: Dict[str, FrameSpec] = {
    "A": FrameSpec(
        "A","Pregnancy physiology → GDM (normal → pathology)",
        ["INDUCES_RESISTANCE","REQUIRES_COMPENSATION","FAILURE_LEADS_TO_GDM"],
        min_required_covered=2,
        allowed_node_types=["hormone","condition","tissue","pathway","risk_factor","threshold_or_timing"],
        blocked_node_types=["intervention","test","drug","guideline_org"],
        min_steps=3, max_steps=6
    ),
    "B": FrameSpec(
        "B","Maternal hyperglycemia → fetal macrosomia (Pedersen/HAPO style)",
        ["CROSSES_PLACENTA","STIMULATES_INSULIN","PROMOTES_GROWTH"],
        min_required_covered=2,
        allowed_node_types=["condition","tissue","outcome_fetal","pathway","threshold_or_timing"],
        blocked_node_types=["intervention","test","drug","guideline_org"],
        min_steps=3, max_steps=6
    ),
    "C": FrameSpec(
        "C","Epidemiology / risk factors",
        ["INCREASES_RISK","VARIES_WITH_BACKGROUND_T2DM"],
        min_required_covered=1,
        allowed_node_types=["risk_factor","condition","threshold_or_timing"],
        min_steps=3, max_steps=6
    ),
    "D": FrameSpec(
        "D","Postpartum progression & follow-up (GDM → T2DM)",
        ["POSTPARTUM_OGTT","FOLLOWUP_GAP","MISSED_DETECTION"],
        min_required_covered=1,
        allowed_node_types=["condition","test","threshold_or_timing","risk_factor","outcome_maternal","guideline_org"],
        min_steps=3, max_steps=6
    ),
    "E": FrameSpec(
        "E","Guidelines comparison (screening/timing/thresholds)",
        ["SCREEN_24_28_WEEKS","THRESHOLD_CRITERIA","POSTPARTUM_OGTT"],
        min_required_covered=1,
        allowed_node_types=["guideline_org","test","threshold_or_timing","condition"],
        min_steps=3, max_steps=6
    ),
    "F": FrameSpec(
        "F","Intervention & monitoring (targets → intervention → outcomes)",
        ["MONITORING_TARGETS","TREATED_BY","IMPROVES_OUTCOMES"],
        min_required_covered=1,
        allowed_node_types=["intervention","test","condition","threshold_or_timing","drug","outcome_fetal","outcome_longterm","outcome_maternal"],
        min_steps=3, max_steps=6
    ),
    # "G": FrameSpec(
    #   id="G",
    #   name="Reframing: transient pregnancy complication vs chronic metabolic risk",
    #   required_edges=["RESOLVES_AFTER_DELIVERY","BETA_CELL_DECLINE_OR_LIMIT","POSTPARTUM_T2DM_RISK","POSTPARTUM_OGTT"],
    #   min_required_covered=2,
    #   allowed_node_types=["condition","risk_factor","outcome_maternal","test","pathway","threshold_or_timing"],
    #   blocked_node_types=["intervention","drug"],
    #   min_steps=4,
    #   max_steps=6
    # ),
    "G": FrameSpec(
        id="G",
        name="Reframing: transient pregnancy complication vs chronic metabolic risk",
        required_edges=[
            "RESOLVES_AFTER_DELIVERY",
            "BETA_CELL_LIMITATION",
            "PERSISTENT_INSULIN_RESISTANCE",
            "POSTPARTUM_T2DM_RISK",
            "POSTPARTUM_OGTT"
        ],
        min_required_covered=3,
        allowed_node_types=[
            "condition",
            "risk_factor",
            "outcome_maternal",
            "test",
            "pathway",
            "guideline_org"
        ],
        blocked_node_types=[
            "intervention",
            "drug",
            "outcome_fetal",
            "threshold_or_timing"
        ],
        min_steps=4,
        max_steps=6
    ),


}
