
import json, re
from pathlib import Path
import numpy as np

def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def hash_ngrams(text: str, n: int = 3, dim: int = 2048) -> np.ndarray:
    """
    NumPy-only embedder: hashed token 3-grams + unigram boost.
    Note: Python's built-in hash() is process-dependent. For strict reproducibility
    across runs, swap to a stable hash (e.g., hashlib.md5).
    """
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    toks = text.split()
    v = np.zeros(dim, dtype=np.float32)
    if len(toks) < n:
        toks = toks + [""] * (n - len(toks))
    for i in range(len(toks) - n + 1):
        ng = toks[i] + " " + toks[i+1] + " " + toks[i+2]
        v[hash(ng) % dim] += 1.0
    for t in toks:
        v[hash(t) % dim] += 0.3
    return v / (np.linalg.norm(v) + 1e-9)

def build_claim_matrix(claims, dim: int = 2048):
    C = np.stack([hash_ngrams(c["text"], dim=dim) for c in claims])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C

def retrieve(claims, C, question: str, k: int = 6):
    q = hash_ngrams(question, dim=C.shape[1])
    sims = C @ q
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [claims[i] for i in idx], sims[idx]

def compose_answer(claims_k, max_sentences: int = 3) -> str:
    kind_order = {"atom":0,"rel_atom":1,"sentence":2,"gen_atom":3}
    seen=set(); out=[]
    for c in sorted(claims_k, key=lambda x:(kind_order.get(x.get("kind",""),9), len(x.get("text","")))):
        t=(c.get("text","") or "").strip()
        if not t:
            continue
        tl=t.lower()
        if tl in seen:
            continue
        seen.add(tl)
        if not t.endswith(('.', '!', '?')):
            t += '.'
        out.append(t)
        if len(out) >= max_sentences:
            break
    return " ".join(out)

def extract_numbers_units(text: str):
    nums=[]
    for m in re.finditer(r'(\d{2,3})\s*mg/dL', text, re.I):
        nums.append(("mg/dL", int(m.group(1))))
    for m in re.finditer(r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s*weeks', text, re.I):
        nums.append(("weeks_range", (int(m.group(1)), int(m.group(2)))))
    return nums

def best_claim_for_sentence(claims, C, sent: str):
    v = hash_ngrams(sent, dim=C.shape[1])
    sims = C @ v
    i = int(np.argmax(sims))
    return claims[i], float(sims[i])

def support_metrics(claims, C, answer: str, support_thr: float = 0.55):
    sents = [s for s in re.split(r'(?<=[\.\?\!])\s+', (answer or '').strip()) if len(s) > 10]
    if not sents:
        return {"support_rate":0.0,"contradictions":0,"unsupported":1,"sentences":0, "matches":[]}
    unsupported=0
    contradictions=0
    matches=[]
    for s in sents:
        claim, sim = best_claim_for_sentence(claims, C, s)
        matches.append((s, claim, sim))
        if sim < support_thr:
            unsupported += 1
            continue
        a_nums = extract_numbers_units(s)
        c_nums = extract_numbers_units((claim.get("evidence","") or "") + " " + (claim.get("text","") or ""))
        if a_nums and c_nums:
            for typ, val in a_nums:
                c_vals=[v for t,v in c_nums if t==typ]
                if c_vals and val not in c_vals:
                    contradictions += 1
                    break
    return {
        "support_rate": 1.0 - (unsupported/len(sents)),
        "contradictions": contradictions,
        "unsupported": unsupported,
        "sentences": len(sents),
        "matches": matches
    }

def integrity_label(m):
    if m["sentences"]==0:
        return "bad"
    if m["contradictions"]>0:
        return "contradictory"
    if m["support_rate"] < 0.66:
        return "neutral" if m["support_rate"] >= 0.34 else "bad"
    if m["sentences"] > 3:
        return "entangled"
    return "good"

def batch_score(eval_rows, claims, C, free_map=None):
    free_map = free_map or {}
    report=[]
    for r in eval_rows:
        qid=r["qid"]
        locked=r.get("locked_answer","")
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
    return report
