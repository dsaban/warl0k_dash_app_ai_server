
import json, re
from pathlib import Path
import numpy as np

CLAIMS_PATH = Path("claims_200.jsonl")
EVAL_PATH = Path("eval_set_100.jsonl")
FREE_ANSWERS_PATH = Path("free_answers.jsonl")  # {"qid":"Q001","answer":"..."}

def hash_ngrams(text, n=3, dim=2048):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    toks = text.split()
    v = np.zeros(dim, dtype=np.float32)
    if len(toks) < n:
        toks = toks + [""]*(n-len(toks))
    for i in range(len(toks)-n+1):
        ng = toks[i] + " " + toks[i+1] + " " + toks[i+2]
        v[hash(ng) % dim] += 1.0
    for t in toks:
        v[hash(t) % dim] += 0.3
    return v / (np.linalg.norm(v) + 1e-9)

def extract_numbers_units(text):
    nums=[]
    for m in re.finditer(r'(\d{2,3})\s*mg/dL', text, re.I):
        nums.append(("mg/dL", int(m.group(1))))
    for m in re.finditer(r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s*weeks', text, re.I):
        nums.append(("weeks_range", (int(m.group(1)), int(m.group(2)))))
    return nums

def load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def build_claim_matrix(claims):
    C = np.stack([hash_ngrams(c["text"]) for c in claims])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C

def best_claim_for_sentence(claims, C, sent):
    v = hash_ngrams(sent)
    sims = C @ v
    i = int(np.argmax(sims))
    return claims[i], float(sims[i])

def support_metrics(claims, C, answer, support_thr=0.55):
    sents = [s for s in re.split(r'(?<=[\.\?\!])\s+', answer.strip()) if len(s)>10]
    if not sents:
        return {"support_rate":0.0,"contradictions":0,"unsupported":1,"sentences":0}
    unsupported=0
    contradictions=0
    for s in sents:
        claim, sim = best_claim_for_sentence(claims, C, s)
        if sim < support_thr:
            unsupported += 1
            continue
        a_nums = extract_numbers_units(s)
        c_nums = extract_numbers_units(claim.get("evidence","") + " " + claim.get("text",""))
        if a_nums and c_nums:
            for typ,val in a_nums:
                c_vals=[v for t,v in c_nums if t==typ]
                if c_vals and val not in c_vals:
                    contradictions += 1
                    break
    return {
        "support_rate": 1.0 - (unsupported/len(sents)),
        "contradictions": contradictions,
        "unsupported": unsupported,
        "sentences": len(sents)
    }

def integrity_label(m):
    if m["sentences"]==0: return "bad"
    if m["contradictions"]>0: return "contradictory"
    if m["support_rate"] < 0.66: return "neutral" if m["support_rate"] >= 0.34 else "bad"
    if m["sentences"] > 3: return "entangled"
    return "good"

if __name__ == "__main__":
    claims = load_jsonl(CLAIMS_PATH)
    C = build_claim_matrix(claims)
    eval_rows = load_jsonl(EVAL_PATH)

    free_map={}
    if FREE_ANSWERS_PATH.exists():
        for r in load_jsonl(FREE_ANSWERS_PATH):
            free_map[r["qid"]] = r.get("answer","")

    report=[]
    for r in eval_rows:
        qid=r["qid"]
        locked=r["locked_answer"]
        free=free_map.get(qid,"")

        m_locked=support_metrics(claims,C,locked)
        row={
            "qid": qid,
            "locked_label": integrity_label(m_locked),
            "locked_support": round(m_locked["support_rate"],3),
            "locked_contra": m_locked["contradictions"],
        }
        if free:
            m_free=support_metrics(claims,C,free)
            row.update({
                "free_label": integrity_label(m_free),
                "free_support": round(m_free["support_rate"],3),
                "free_contra": m_free["contradictions"],
            })
        report.append(row)

    Path("report.json").write_text(json.dumps(report, indent=2))
    from collections import Counter
    print("Locked summary:", dict(Counter([r["locked_label"] for r in report])))
    if any("free_label" in r for r in report):
        print("Free summary:", dict(Counter([r["free_label"] for r in report if "free_label" in r])))
    print("Wrote report.json")
