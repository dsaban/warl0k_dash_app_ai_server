# core/entailment.py
import re
from .utils import normalize_text

def split_sentences(text: str):
    t = (text or "").strip()
    if not t:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) > 12:
            out.append(p)
    return out

CLASSIFY_VERBS = ("classified","considered","defined","described as","represents","is a","is an")

def entailment_check_classification(target_phrase: str, evidence_chunks, min_score: float = 0.18):
    """
    Very light entailment:
    - Look for a sentence that contains a classification verb AND overlaps with the target.
    - Distinguish classification vs mere risk language.
    """
    target = normalize_text(target_phrase)
    target_terms = set(re.findall(r"[a-z0-9\-]+", target))

    best = {"score": 0.0, "sentence": "", "support_type": "none"}
    for ch in (evidence_chunks or []):
        for s in split_sentences(ch):
            sn = normalize_text(s)
            st = set(re.findall(r"[a-z0-9\-]+", sn))
            overlap = len(target_terms.intersection(st)) / max(1, len(target_terms))
            has_classify = any(v in sn for v in CLASSIFY_VERBS)
            has_risk = any(x in sn for x in ["risk","increased risk","raises the chance","more likely","associated with"])

            score = overlap
            if has_classify:
                score += 0.25
            if has_risk and not has_classify:
                score -= 0.10

            if score > best["score"]:
                best = {
                    "score": float(score),
                    "sentence": s,
                    "support_type": "classification" if (has_classify and score >= min_score) else ("risk_only" if has_risk else "none")
                }

    best["passed"] = bool(best["support_type"] == "classification" and best["score"] >= min_score)
    return best
