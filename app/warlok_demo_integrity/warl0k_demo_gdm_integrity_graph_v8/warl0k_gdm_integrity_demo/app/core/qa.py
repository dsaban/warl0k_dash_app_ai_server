import re
from typing import Dict, Any, List

class QAEvaluator:
    """
    Integrity evaluator:
      - retrieves evidence (top-k)
      - checks required fields by question type
      - verifies the key tokens appear in evidence (support check)
    This is intentionally conservative: missing evidence => fail/low score.
    """
    def __init__(self, store):
        self.store = store

    def _infer_qtype(self, question: str) -> str:
        q = question.lower()
        if "postpartum" in q or "after a pregnancy" in q:
            return "postpartum_followup"
        if "low-risk" in q and "screen" in q:
            return "screening_window"
        if "threshold" in q or "ogtt" in q and "diagnos" in q:
            return "ogtt_thresholds"
        if "insulin resistance" in q:
            return "mechanism_ir"
        if "risks" in q:
            return "risks"
        return "generic"

    def _required_fields(self, qtype: str) -> Dict[str, List[str]]:
        # each field is satisfied if any of its tokens appear in the answer
        if qtype == "screening_window":
            return {"weeks":["24","28"], "test":["ogtt","75"]}
        if qtype == "postpartum_followup":
            return {"timing":["4","12"], "test":["ogtt","75"], "long_term":["1","3"]}
        if qtype == "ogtt_thresholds":
            return {"approach":["75","ogtt"], "cutoffs":["92","180","153"]}
        if qtype == "mechanism_ir":
            return {"cause":["placental","hormone"], "compensation":["beta","insulin"], "failure":["inadequate","insufficient","hyperglycemia"]}
        if qtype == "risks":
            return {"maternal":["preeclampsia","hypertension","cesarean"], "neonatal":["macrosomia","hypoglycemia"], "pathway":["placenta","hyperinsulinemia"]}
        return {}

    def validate(self, qobj: Dict[str, Any], answer: str, retriever, k: int = 10) -> Dict[str, Any]:
        qtype = self._infer_qtype(qobj["question"])
        required = self._required_fields(qtype)

        evidence = retriever.search(qobj["question"], k=k)
        evidence_text = " ".join([e["text"].lower() for e in evidence])

        field_checks = {}
        missing = []
        supported_missing = []
        ans_l = answer.lower()

        # Field presence + evidence support
        for field, tokens in required.items():
            present = any(t.lower() in ans_l for t in tokens)
            supported = any(t.lower() in evidence_text for t in tokens)
            field_checks[field] = {"present_in_answer": present, "supported_in_evidence": supported, "tokens": tokens}
            if not present:
                missing.append(field)
            if present and not supported:
                supported_missing.append(field)

        # Simple hallucination detector: numbers in answer that don't appear in evidence
        nums = re.findall(r"\b\d+(\.\d+)?\b", answer)
        bad_nums = []
        for n in nums:
            if n and n not in evidence_text:
                # allow common filler years etc? keep strict.
                bad_nums.append(n)
        bad_nums = sorted(set(bad_nums))

        reasons = []
        score = 1.0
        if missing:
            reasons.append(f"Missing required fields for qtype={qtype}: {', '.join(missing)}")
            score -= 0.35
        if supported_missing:
            reasons.append(f"Answer fields not supported by retrieved evidence: {', '.join(supported_missing)}")
            score -= 0.35
        if bad_nums:
            reasons.append(f"Answer contains numeric claims not found in retrieved evidence: {', '.join(bad_nums[:8])}")
            score -= 0.20

        # clamp
        score = max(0.0, min(1.0, score))

        if score >= 0.75 and not missing and not supported_missing:
            verdict = "PASS (evidence-supported)"
        elif score >= 0.45:
            verdict = "WARN (partial/weak support)"
        else:
            verdict = "FAIL (insufficient/unsupported)"

        return {
            "qid": qobj["qid"],
            "qtype": qtype,
            "score": float(score),
            "verdict": verdict,
            "reasons": reasons or ["No major integrity violations detected under current checks."],
            "evidence": evidence,
            "field_checks": field_checks,
        }
