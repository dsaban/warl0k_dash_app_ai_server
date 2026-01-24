# core/schema.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .utils import normalize_text
from .entailment import split_sentences

@dataclass
class Slot:
    name: str
    required_terms: Tuple[str, ...]
    guidance: str
    critical: bool = True

SCHEMAS: Dict[str, List[Slot]] = {
    # NEW: Placental hormones -> IR -> pathological in GDM
    "placental_hormones_pathologic_gdm": [
        Slot(
            "Placental hormones → insulin resistance",
            ("placental","hormone","hpl","human placental lactogen","progesterone","cortisol","growth hormone","estrogen","anti-insulin","insulin resistance"),
            "Explain that placental hormones drive insulin resistance during pregnancy.",
            True
        ),
        Slot(
            "Physiological purpose",
            ("fetus","fetal","glucose","nutrient","shunting","supply","transfer"),
            "Explain the physiological purpose: ensuring nutrient/glucose availability to the fetus.",
            False
        ),
        Slot(
            "Beta-cell compensation",
            ("beta","beta-cell","insulin secretion","compensat","increase in insulin","insulin secretion increases"),
            "Explain beta-cell compensation (increased insulin secretion) to maintain normoglycemia.",
            True
        ),
        Slot(
            "Pathological transition in GDM",
            ("inadequate","insufficient","cannot","fail","failure","dysfunction","impaired","hypergly","gdm"),
            "Explain why it becomes pathological in GDM: compensation fails → hyperglycemia.",
            True
        ),
    ],

    # Deep mechanism schema (why adaptation fails)
    "maternal_beta_cell_failure": [
        Slot("Pregnancy-induced insulin resistance", ("pregnancy", "insulin resistance"), "State that pregnancy induces insulin resistance (often via placental hormones).", True),
        Slot("Increased insulin demand", ("increase", "insulin", "demand"), "Explain higher insulin demand during pregnancy.", False),
        Slot("Beta-cell compensation", ("beta", "insulin secretion", "compensat"), "Explain compensatory rise in insulin secretion by beta-cells.", True),
        Slot("Failure in susceptible women", ("inadequate", "insufficient", "cannot", "fail", "failure", "dysfunction", "impaired"), "Explain why compensation fails in some women (limited beta-cell reserve/dysfunction).", True),
        Slot("Result: hyperglycemia / GDM", ("hypergly", "gdm", "glucose"), "Connect failed compensation to hyperglycemia and GDM.", True),
        Slot("Susceptibility factors (if present)", ("obesity", "inflammation", "genetic", "risk factor", "prior", "age"), "Mention susceptibility factors only if text supports them.", False),
    ],

    "maternal_progression_model": [
        Slot("Mechanism", ("insulin","resistance","pregnancy"), "Explain pregnancy-induced insulin resistance (stress test).", True),
        Slot("Beta-cell response", ("beta","insulin secretion","compensation","failure"), "Explain compensation vs failure leading to GDM.", True),
        Slot("Postpartum progression", ("postpartum","type 2","overt","progression","conversion","risk"), "Explain postpartum progression to overt diabetes.", True),
    ],

    "maternal_inflammation_cardiometabolic": [
        Slot("Mechanism", ("obesity","inflammation","insulin resistance"), "Explain obesity-driven inflammation worsening insulin resistance.", True),
        Slot("Pathway", ("cytokine","adipose","endothelial","dyslipidemia","hypertension"), "Connect inflammation to vascular/metabolic dysfunction.", True),
        Slot("Outcome", ("postpartum","long-term","cardiovascular","cardiometabolic","type 2","risk"), "Tie to long-term maternal cardiometabolic risk after GDM.", True),
    ],

    "classification": [
        Slot("Classification evidence", ("classified","considered","defined","described as","represents","prediabet"), "Find explicit classification wording (not just risk).", True),
    ],

    "monitoring": [
        Slot("Monitoring effect", ("cgm","monitoring","hba1c","hypoglycemia"), "Summarize what CGM changes and evidence.", True),
    ],

    "offspring_outcomes": [
        Slot("Offspring outcomes", ("offspring","child","fetal","neonatal","infant"), "Summarize offspring consequences.", True),
    ],

    "generic": [
        Slot("Answer", (), "Summarize supported evidence.", True),
    ],
}

def _sent_score_for_terms(sentence_norm: str, terms: Tuple[str, ...]) -> int:
    score = 0
    for t in terms:
        tn = normalize_text(t)
        if tn and tn in sentence_norm:
            score += 1
    return score

def _best_sentence(evidence_chunks: List[str], terms: Tuple[str, ...], used_norm: set) -> str:
    if not evidence_chunks:
        return ""

    best_s, best_score = "", 0
    for ch in evidence_chunks:
        for s in split_sentences(ch):
            sn = normalize_text(s)
            if sn in used_norm:
                continue
            if not terms:
                return s
            score = _sent_score_for_terms(sn, terms)
            if score > best_score:
                best_score = score
                best_s = s

    return best_s if best_score > 0 else ""

def fill_schema(schema_id: str, evidence_chunks: List[str]) -> Dict:
    slots = SCHEMAS.get(schema_id, SCHEMAS["generic"])
    used = set()
    filled, missing = [], []

    for sl in slots:
        sent = _best_sentence(evidence_chunks, sl.required_terms, used)
        if sent:
            used.add(normalize_text(sent))
            filled.append({"slot": sl.name, "sentence": sent, "guidance": sl.guidance, "critical": sl.critical})
        else:
            missing.append({"slot": sl.name, "guidance": sl.guidance, "critical": sl.critical})

    crit_total = sum(1 for s in slots if s.critical)
    crit_filled = sum(1 for f in filled if f.get("critical"))
    crit_cov = crit_filled / max(1, crit_total)

    overall_cov = len(filled) / max(1, len(slots))
    return {
        "schema_id": schema_id,
        "filled": filled,
        "missing": missing,
        "coverage": float(overall_cov),
        "critical_coverage": float(crit_cov),
    }

# # core/schema.py
# from dataclasses import dataclass
# from typing import Dict, List, Tuple
# from .utils import normalize_text
# from .entailment import split_sentences
#
# @dataclass
# class Slot:
#     name: str
#     required_terms: Tuple[str, ...]
#     guidance: str
#     critical: bool = True  # NEW
#
# SCHEMAS: Dict[str, List[Slot]] = {
#     # NEW: Deep mechanism schema
#     "maternal_beta_cell_failure": [
#         Slot("Pregnancy-induced insulin resistance", ("pregnancy", "insulin resistance"), "State that pregnancy induces insulin resistance (often via placental hormones).", True),
#         Slot("Increased insulin demand", ("increase", "insulin", "demand"), "Explain higher insulin demand during pregnancy.", False),
#         Slot("Beta-cell compensation", ("beta", "insulin secretion", "compensat"), "Explain compensatory rise in insulin secretion by beta-cells.", True),
#         Slot("Failure in susceptible women", ("inadequate", "insufficient", "cannot", "fail", "failure", "dysfunction", "impaired"), "Explain why compensation fails in some women (limited beta-cell reserve/dysfunction).", True),
#         Slot("Result: hyperglycemia / GDM", ("hypergly", "gdm", "glucose"), "Connect failed compensation to hyperglycemia and GDM.", True),
#         Slot("Susceptibility factors (if present)", ("obesity", "inflammation", "genetic", "risk factor", "prior", "age"), "Mention susceptibility factors only if text supports them.", False),
#     ],
#
#     "maternal_progression_model": [
#         Slot("Mechanism", ("insulin","resistance","pregnancy"), "Explain pregnancy-induced insulin resistance (stress test).", True),
#         Slot("Beta-cell response", ("beta","insulin secretion","compensation","failure"), "Explain compensation vs failure leading to GDM.", True),
#         Slot("Postpartum progression", ("postpartum","type 2","overt","progression","conversion","risk"), "Explain postpartum progression to overt diabetes.", True),
#     ],
#
#     "maternal_inflammation_cardiometabolic": [
#         Slot("Mechanism", ("obesity","inflammation","insulin resistance"), "Explain obesity-driven inflammation worsening insulin resistance.", True),
#         Slot("Pathway", ("cytokine","adipose","endothelial","dyslipidemia","hypertension"), "Connect inflammation to vascular/metabolic dysfunction.", True),
#         Slot("Outcome", ("postpartum","long-term","cardiovascular","cardiometabolic","type 2","risk"), "Tie to long-term maternal cardiometabolic risk after GDM.", True),
#     ],
#
#     "classification": [
#         Slot("Classification evidence", ("classified","considered","defined","described as","represents","prediabet"), "Find explicit classification wording (not just risk).", True),
#     ],
#
#     "monitoring": [
#         Slot("Monitoring effect", ("cgm","monitoring","hba1c","hypoglycemia"), "Summarize what CGM changes and evidence.", True),
#     ],
#
#     "offspring_outcomes": [
#         Slot("Offspring outcomes", ("offspring","child","fetal","neonatal","infant"), "Summarize offspring consequences.", True),
#     ],
#
#     "generic": [
#         Slot("Answer", (), "Summarize supported evidence.", True),
#     ],
# }
#
# def _sent_score_for_terms(sentence_norm: str, terms: Tuple[str, ...]) -> int:
#     # allow partial matches like "compensat" "hypergly"
#     score = 0
#     for t in terms:
#         tn = normalize_text(t)
#         if tn and tn in sentence_norm:
#             score += 1
#     return score
#
# def _best_sentence(evidence_chunks: List[str], terms: Tuple[str, ...], used_norm: set) -> str:
#     # If no terms: pick first unused sentence
#     if not evidence_chunks:
#         return ""
#
#     best_s, best_score = "", 0
#     for ch in evidence_chunks:
#         for s in split_sentences(ch):
#             sn = normalize_text(s)
#             if sn in used_norm:
#                 continue
#             if not terms:
#                 return s
#             score = _sent_score_for_terms(sn, terms)
#             if score > best_score:
#                 best_score = score
#                 best_s = s
#
#     return best_s if best_score > 0 else ""
#
# def fill_schema(schema_id: str, evidence_chunks: List[str]) -> Dict:
#     slots = SCHEMAS.get(schema_id, SCHEMAS["generic"])
#     used = set()
#     filled, missing = [], []
#
#     for sl in slots:
#         sent = _best_sentence(evidence_chunks, sl.required_terms, used)
#         if sent:
#             used.add(normalize_text(sent))
#             filled.append({"slot": sl.name, "sentence": sent, "guidance": sl.guidance, "critical": sl.critical})
#         else:
#             missing.append({"slot": sl.name, "guidance": sl.guidance, "critical": sl.critical})
#
#     # Coverage counts critical slots heavier
#     crit_total = sum(1 for s in slots if s.critical)
#     crit_filled = sum(1 for f in filled if f.get("critical"))
#     crit_cov = crit_filled / max(1, crit_total)
#
#     overall_cov = len(filled) / max(1, len(slots))
#     return {
#         "schema_id": schema_id,
#         "filled": filled,
#         "missing": missing,
#         "coverage": float(overall_cov),
#         "critical_coverage": float(crit_cov),
#     }
#
# # # core/schema.py
# # from dataclasses import dataclass
# # from typing import Dict, List, Tuple
# # from .utils import normalize_text
# # from .entailment import split_sentences
# #
# # @dataclass
# # class Slot:
# #     name: str
# #     required_terms: Tuple[str, ...]
# #     guidance: str
# #
# # SCHEMAS: Dict[str, List[Slot]] = {
# #     "maternal_progression_model": [
# #         Slot("Mechanism", ("insulin","resistance","pregnancy"), "Explain pregnancy-induced insulin resistance (stress test)."),
# #         Slot("Beta-cell response", ("beta","insulin secretion","compensation","failure"), "Explain compensation vs failure leading to GDM/hyperglycemia."),
# #         Slot("Progression", ("postpartum","type 2","overt","progression","conversion","risk"), "Explain postpartum progression to overt diabetes."),
# #     ],
# #     "maternal_inflammation_cardiometabolic": [
# #         Slot("Mechanism", ("obesity","inflammation","insulin resistance"), "Explain obesity-driven inflammation worsening insulin resistance."),
# #         Slot("Pathway", ("cytokine","adipose","endothelial","dyslipidemia","hypertension"), "Connect inflammation to vascular/metabolic dysfunction."),
# #         Slot("Outcome", ("postpartum","long-term","cardiovascular","cardiometabolic","type 2","risk"), "Tie to long-term maternal cardiometabolic risk after GDM."),
# #     ],
# #     "classification": [
# #         Slot("Classification evidence", ("classified","considered","defined","described as","represents","prediabet"), "Find explicit classification wording (not just risk)."),
# #     ],
# #     "monitoring": [
# #         Slot("Monitoring effect", ("cgm","monitoring","hba1c","hypoglycemia"), "Summarize what CGM changes and evidence."),
# #     ],
# #     "offspring_outcomes": [
# #         Slot("Offspring outcomes", ("offspring","child","fetal","neonatal","infant"), "Summarize offspring consequences."),
# #     ],
# #     "generic": [
# #         Slot("Answer", (), "Summarize supported evidence."),
# #     ],
# # }
# #
# # def _best_sentence_for_terms(evidence_chunks: List[str], terms: Tuple[str, ...]) -> str:
# #     if not evidence_chunks:
# #         return ""
# #     if not terms:
# #         for ch in evidence_chunks:
# #             ss = split_sentences(ch)
# #             if ss:
# #                 return ss[0]
# #         return ""
# #
# #     terms_n = [normalize_text(t) for t in terms]
# #     best_s, best_score = "", 0
# #     for ch in evidence_chunks:
# #         for s in split_sentences(ch):
# #             sn = normalize_text(s)
# #             score = sum(1 for t in terms_n if t in sn)
# #             if score > best_score:
# #                 best_score = score
# #                 best_s = s
# #     return best_s if best_score > 0 else ""
# #
# # def fill_schema(schema_id: str, evidence_chunks: List[str]) -> Dict:
# #     slots = SCHEMAS.get(schema_id, SCHEMAS["generic"])
# #     filled, missing = [], []
# #     for sl in slots:
# #         sent = _best_sentence_for_terms(evidence_chunks, sl.required_terms)
# #         if sent:
# #             filled.append({"slot": sl.name, "sentence": sent, "guidance": sl.guidance})
# #         else:
# #             missing.append({"slot": sl.name, "guidance": sl.guidance})
# #
# #     coverage = len(filled) / max(1, len(slots))
# #     return {"schema_id": schema_id, "filled": filled, "missing": missing, "coverage": float(coverage)}
