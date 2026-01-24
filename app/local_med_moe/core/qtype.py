# core/qtype.py
from dataclasses import dataclass
from typing import Tuple
from .utils import normalize_text

@dataclass
class QTypeProfile:
    name: str
    required_all: Tuple[str, ...]
    required_any: Tuple[str, ...]
    forbidden: Tuple[str, ...]
    anchors: Tuple[str, ...]
    target_role: str
    schema_id: str
    needs_entailment: bool = False
    needs_failure_explanation: bool = False

def _has_any(q: str, keys):
    return any(k in q for k in keys)

def detect_qtype(user_query: str) -> QTypeProfile:
    q = normalize_text(user_query)

    # Monitoring
    if _has_any(q, ["cgm", "continuous glucose monitoring", "hba1c", "hypoglycemia", "smbg", "time in range", "sensor"]):
        return QTypeProfile(
            name="monitoring",
            required_all=(),
            required_any=("cgm","monitoring","hba1c","hypoglycemia","smbg","sensor"),
            forbidden=(),
            anchors=("continuous glucose monitoring HbA1c hypoglycemia SMBG adherence cost",),
            target_role="monitoring",
            schema_id="monitoring"
        )

    # NEW: placental hormones -> insulin resistance -> pathological in GDM
    mentions_placenta = _has_any(q, ["placental", "placenta", "hpl", "human placental lactogen",
                                    "progesterone", "cortisol", "growth hormone", "estrogen"])
    mentions_ir = _has_any(q, ["insulin resistance", "pregnancy-induced insulin resistance"])
    asks_pathologic = _has_any(q, ["pathological", "becomes pathological", "become pathological", "in gdm", "gdm"])

    if mentions_placenta and mentions_ir and asks_pathologic:
        return QTypeProfile(
            name="placental_hormones_pathologic_gdm",
            required_all=("insulin", "resistance"),
            required_any=(
                "placental", "hormone", "hpl", "progesterone", "cortisol", "growth hormone", "estrogen",
                "anti-insulin", "insulin resistance",
                "beta", "beta-cell", "insulin secretion", "compensation",
                "inadequate", "insufficient", "cannot", "fail", "failure", "dysfunction",
                "hyperglycemia", "gdm", "pathological"
            ),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
            anchors=(
                "placental hormones human placental lactogen progesterone cortisol growth hormone cause insulin resistance",
                "physiologic insulin resistance pregnancy purpose glucose shunting to fetus",
                "beta-cell compensation increased insulin secretion during pregnancy",
                "GDM occurs when beta-cell compensation is inadequate leading to hyperglycemia"
            ),
            target_role="maternal",
            schema_id="placental_hormones_pathologic_gdm",
            needs_failure_explanation=True
        )

    # NEW: Maternal beta-cell failure mechanism (why adaptation fails)
    asks_failure = _has_any(q, ["why does", "why", "fail", "fails", "failure", "in certain women", "certain women", "some women", "adaptation fail"])
    mentions_beta = _has_any(q, ["beta-cell", "beta cell", "insulin secretion", "pancreatic"])
    mentions_ir2 = _has_any(q, ["insulin resistance", "pregnancy-induced insulin resistance", "pregnancy induced insulin resistance"])

    if asks_failure and mentions_beta and mentions_ir2:
        return QTypeProfile(
            name="maternal_beta_cell_failure",
            required_all=("insulin", "resistance"),
            required_any=(
                "pregnancy", "insulin resistance", "placental", "hormone",
                "beta", "beta-cell", "insulin secretion", "compensation",
                "inadequate", "insufficient", "cannot", "fail", "failure", "dysfunction",
                "hyperglycemia", "gdm"
            ),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
            anchors=(
                "pregnancy insulin resistance placental hormones increase insulin demand",
                "beta-cell compensation increased insulin secretion to overcome pregnancy insulin resistance",
                "inadequate beta-cell compensation beta-cell dysfunction insufficient insulin secretion failure in some women",
                "why compensation fails obesity inflammation genetic susceptibility impaired beta-cell reserve"
            ),
            target_role="maternal",
            schema_id="maternal_beta_cell_failure",
            needs_failure_explanation=True
        )

    # Classification / “described as”
    if _has_any(q, ["described as", "classified", "considered", "defined", "represents"]) or "prediabet" in q:
        return QTypeProfile(
            name="classification",
            required_all=(),
            required_any=("classified","considered","defined","described as","represents","prediabet"),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
            anchors=("classified considered defined described as represents prediabetes",),
            target_role="maternal",
            schema_id="classification",
            needs_entailment=True
        )

    # Maternal inflammation → cardiometabolic
    if _has_any(q, ["obesity-driven", "inflammation", "cardiometabolic", "cardiovascular", "long-term", "postpartum"]) and "gdm" in q:
        return QTypeProfile(
            name="maternal_inflammation_cardiometabolic",
            required_all=("gdm",),
            required_any=("obesity","inflammation","cytokine","adipose","cardiometabolic","cardiovascular","dyslipidemia","hypertension","endothelial","postpartum","long-term"),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
            anchors=(
                "obesity inflammation cytokines adipose postpartum cardiometabolic risk after gestational diabetes",
                "endothelial dysfunction dyslipidemia hypertension insulin resistance postpartum after GDM",
            ),
            target_role="maternal",
            schema_id="maternal_inflammation_cardiometabolic"
        )

    # Maternal progression model (general)
    if _has_any(q, ["clinical model", "serve as a model", "transition", "progression", "from insulin resistance", "overt diabetes", "beta-cell", "beta cell"]):
        return QTypeProfile(
            name="maternal_progression_model",
            required_all=("insulin","resistance"),
            required_any=("model","transition","progression","beta","beta-cell","insulin secretion","compensation","failure","postpartum","type 2","overt"),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
            anchors=(
                "pregnancy insulin resistance beta-cell compensation failure postpartum progression type 2 diabetes",
                "GDM stress test beta-cell dysfunction postpartum conversion to type 2 diabetes",
            ),
            target_role="maternal",
            schema_id="maternal_progression_model"
        )

    # Offspring outcomes
    if _has_any(q, ["offspring", "fetus", "fetal", "neonatal", "infant", "child", "birth weight", "macrosomia"]):
        return QTypeProfile(
            name="offspring_outcomes",
            required_all=(),
            required_any=("offspring","fetal","neonatal","infant","child","macrosomia","birth weight"),
            forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
            anchors=("offspring fetal neonatal infant child macrosomia birth weight outcomes",),
            target_role="offspring",
            schema_id="offspring_outcomes"
        )

    return QTypeProfile(
        name="generic",
        required_all=(),
        required_any=(),
        forbidden=(),
        anchors=(),
        target_role="generic",
        schema_id="generic"
    )

# # core/qtype.py
# from dataclasses import dataclass
# from typing import Tuple
# from .utils import normalize_text
#
# @dataclass
# class QTypeProfile:
#     name: str
#     required_all: Tuple[str, ...]
#     required_any: Tuple[str, ...]
#     forbidden: Tuple[str, ...]
#     anchors: Tuple[str, ...]
#     target_role: str
#     schema_id: str
#     needs_entailment: bool = False
#     needs_failure_explanation: bool = False  # NEW
#
# def _has_any(q: str, keys):
#     return any(k in q for k in keys)
#
# def detect_qtype(user_query: str) -> QTypeProfile:
#     q = normalize_text(user_query)
#
#     # Monitoring
#     if _has_any(q, ["cgm", "continuous glucose monitoring", "hba1c", "hypoglycemia", "smbg", "time in range", "sensor"]):
#         return QTypeProfile(
#             name="monitoring",
#             required_all=(),
#             required_any=("cgm","monitoring","hba1c","hypoglycemia","smbg","sensor"),
#             forbidden=(),
#             anchors=("continuous glucose monitoring HbA1c hypoglycemia SMBG adherence cost",),
#             target_role="monitoring",
#             schema_id="monitoring"
#         )
#
#     # NEW: Maternal beta-cell failure mechanism (your latest review)
#     asks_failure = _has_any(q, ["why does", "why", "fail", "fails", "failure", "in certain women", "certain women", "some women", "adaptation fail"])
#     mentions_beta = _has_any(q, ["beta-cell", "beta cell", "insulin secretion", "pancreatic"])
#     mentions_ir = _has_any(q, ["insulin resistance", "pregnancy-induced insulin resistance", "pregnancy induced insulin resistance"])
#
#     if asks_failure and mentions_beta and mentions_ir:
#         return QTypeProfile(
#             name="maternal_beta_cell_failure",
#             required_all=("insulin", "resistance"),
#             required_any=(
#                 "pregnancy", "insulin resistance", "placental", "hormone",
#                 "beta", "beta-cell", "insulin secretion", "compensation",
#                 "inadequate", "insufficient", "cannot", "fail", "failure", "dysfunction",
#                 "hyperglycemia", "gdm"
#             ),
#             forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
#             anchors=(
#                 "pregnancy insulin resistance placental hormones increase insulin demand",
#                 "beta-cell compensation increased insulin secretion to overcome pregnancy insulin resistance",
#                 "inadequate beta-cell compensation beta-cell dysfunction insufficient insulin secretion failure in some women",
#                 "why compensation fails obesity inflammation genetic susceptibility impaired beta-cell reserve"
#             ),
#             target_role="maternal",
#             schema_id="maternal_beta_cell_failure",
#             needs_failure_explanation=True
#         )
#
#     # Classification / “described as”
#     if _has_any(q, ["described as", "classified", "considered", "defined", "represents"]) or "prediabet" in q:
#         return QTypeProfile(
#             name="classification",
#             required_all=(),
#             required_any=("classified","considered","defined","described as","represents","prediabet"),
#             forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
#             anchors=("classified considered defined described as represents prediabetes",),
#             target_role="maternal",
#             schema_id="classification",
#             needs_entailment=True
#         )
#
#     # Maternal inflammation → cardiometabolic
#     if _has_any(q, ["obesity-driven", "inflammation", "cardiometabolic", "cardiovascular", "long-term", "postpartum"]) and "gdm" in q:
#         return QTypeProfile(
#             name="maternal_inflammation_cardiometabolic",
#             required_all=("gdm",),
#             required_any=("obesity","inflammation","cytokine","adipose","cardiometabolic","cardiovascular","dyslipidemia","hypertension","endothelial","postpartum","long-term"),
#             forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
#             anchors=(
#                 "obesity inflammation cytokines adipose postpartum cardiometabolic risk after gestational diabetes",
#                 "endothelial dysfunction dyslipidemia hypertension insulin resistance postpartum after GDM",
#             ),
#             target_role="maternal",
#             schema_id="maternal_inflammation_cardiometabolic"
#         )
#
#     # Maternal progression model (general)
#     if _has_any(q, ["clinical model", "serve as a model", "transition", "progression", "from insulin resistance", "overt diabetes", "beta-cell", "beta cell"]):
#         return QTypeProfile(
#             name="maternal_progression_model",
#             required_all=("insulin","resistance"),
#             required_any=("model","transition","progression","beta","beta-cell","insulin secretion","compensation","failure","postpartum","type 2","overt"),
#             forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
#             anchors=(
#                 "pregnancy insulin resistance beta-cell compensation failure postpartum progression type 2 diabetes",
#                 "GDM stress test beta-cell dysfunction postpartum conversion to type 2 diabetes",
#             ),
#             target_role="maternal",
#             schema_id="maternal_progression_model"
#         )
#
#     # Offspring outcomes
#     if _has_any(q, ["offspring", "fetus", "fetal", "neonatal", "infant", "child", "birth weight", "macrosomia"]):
#         return QTypeProfile(
#             name="offspring_outcomes",
#             required_all=(),
#             required_any=("offspring","fetal","neonatal","infant","child","macrosomia","birth weight"),
#             forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
#             anchors=("offspring fetal neonatal infant child macrosomia birth weight outcomes",),
#             target_role="offspring",
#             schema_id="offspring_outcomes"
#         )
#
#     return QTypeProfile(
#         name="generic",
#         required_all=(),
#         required_any=(),
#         forbidden=(),
#         anchors=(),
#         target_role="generic",
#         schema_id="generic"
#     )
#
# # # core/qtype.py
# # from dataclasses import dataclass
# # from typing import Tuple
# # from .utils import normalize_text
# #
# # @dataclass
# # class QTypeProfile:
# #     name: str
# #     required_all: Tuple[str, ...]
# #     required_any: Tuple[str, ...]
# #     forbidden: Tuple[str, ...]
# #     anchors: Tuple[str, ...]
# #     target_role: str
# #     schema_id: str
# #     needs_entailment: bool = False
# #
# # def detect_qtype(user_query: str) -> QTypeProfile:
# #     q = normalize_text(user_query)
# #
# #     if any(k in q for k in ["cgm", "continuous glucose monitoring", "hba1c", "hypoglycemia", "smbg", "time in range", "sensor"]):
# #         return QTypeProfile(
# #             name="monitoring",
# #             required_all=(),
# #             required_any=("cgm","monitoring","hba1c","hypoglycemia","smbg","sensor"),
# #             forbidden=(),
# #             anchors=("continuous glucose monitoring HbA1c hypoglycemia SMBG adherence cost",),
# #             target_role="monitoring",
# #             schema_id="monitoring"
# #         )
# #
# #     if any(k in q for k in ["described as", "classified", "considered", "defined", "represents"]) or "prediabet" in q:
# #         return QTypeProfile(
# #             name="classification",
# #             required_all=(),
# #             required_any=("classified","considered","defined","described as","represents","prediabet"),
# #             forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
# #             anchors=("classified considered defined described as represents prediabetes",),
# #             target_role="maternal",
# #             schema_id="classification",
# #             needs_entailment=True
# #         )
# #
# #     if any(k in q for k in ["obesity-driven", "inflammation", "cardiometabolic", "cardiovascular", "long-term", "postpartum"]) and "gdm" in q:
# #         return QTypeProfile(
# #             name="maternal_inflammation_cardiometabolic",
# #             required_all=("gdm",),
# #             required_any=("obesity","inflammation","cytokine","adipose","cardiometabolic","cardiovascular","dyslipidemia","hypertension","endothelial","postpartum","long-term"),
# #             forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
# #             anchors=(
# #                 "obesity inflammation cytokines adipose postpartum cardiometabolic risk after gestational diabetes",
# #                 "endothelial dysfunction dyslipidemia hypertension insulin resistance postpartum after GDM",
# #             ),
# #             target_role="maternal",
# #             schema_id="maternal_inflammation_cardiometabolic"
# #         )
# #
# #     if any(k in q for k in ["clinical model", "serve as a model", "transition", "progression", "from insulin resistance", "overt diabetes", "beta-cell", "beta cell"]):
# #         return QTypeProfile(
# #             name="maternal_progression_model",
# #             required_all=("insulin","resistance"),
# #             required_any=("model","transition","progression","beta","beta-cell","insulin secretion","compensation","failure","postpartum","type 2","overt"),
# #             forbidden=("cgm","monitoring","hba1c","hypoglycemia","smbg","adherence","cost"),
# #             anchors=(
# #                 "pregnancy insulin resistance beta-cell compensation failure postpartum progression type 2 diabetes",
# #                 "GDM stress test beta-cell dysfunction postpartum conversion to type 2 diabetes",
# #             ),
# #             target_role="maternal",
# #             schema_id="maternal_progression_model"
# #         )
# #
# #     if any(k in q for k in ["offspring", "fetus", "fetal", "neonatal", "infant", "child", "birth weight", "macrosomia"]):
# #         return QTypeProfile(
# #             name="offspring_outcomes",
# #             required_all=(),
# #             required_any=("offspring","fetal","neonatal","infant","child","macrosomia","birth weight"),
# #             forbidden=("cgm","monitoring","hba1c","hypoglycemia","adherence","cost"),
# #             anchors=("offspring fetal neonatal infant child macrosomia birth weight outcomes",),
# #             target_role="offspring",
# #             schema_id="offspring_outcomes"
# #         )
# #
# #     return QTypeProfile(
# #         name="generic",
# #         required_all=(),
# #         required_any=(),
# #         forbidden=(),
# #         anchors=(),
# #         target_role="generic",
# #         schema_id="generic"
# #     )
