
import json, re
from pathlib import Path
import numpy as np
import hashlib

def _stable_hash_int(s: str) -> int:
    # Stable across runs/machines (unlike Python's built-in hash()).
    return int(hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest(), 16)


# ---------------- I/O ----------------
def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

# ---------------- Retrieval embedder (NumPy-only) ----------------
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
        v[_stable_hash_int(ng) % dim] += 1.0
    for t in toks:
        v[_stable_hash_int(t) % dim] += 0.3
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

# ---------------- Heuristic integrity metrics ----------------
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
    sims=[]
    for s in sents:
        claim, sim = best_claim_for_sentence(claims, C, s)
        sims.append(sim)
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
    sents_n = len(sents)
    return {
        "support_rate": 1.0 - (unsupported/sents_n),
        "contradictions": contradictions,
        "unsupported": unsupported,
        "sentences": sents_n,
        "matches": matches,
        "sim_mean": float(np.mean(sims)) if sims else 0.0,
        "sim_min": float(np.min(sims)) if sims else 0.0,
        "sim_max": float(np.max(sims)) if sims else 0.0,
        "sim_std": float(np.std(sims)) if sims else 0.0,
    }

def integrity_label_heuristic(m):
    if m["sentences"]==0:
        return "bad"
    if m["contradictions"]>0:
        return "contradictory"
    if m["support_rate"] < 0.66:
        return "neutral" if m["support_rate"] >= 0.34 else "bad"
    if m["sentences"] > 3:
        return "entangled"
    return "good"

# ---------------- Learned evaluator (NumPy-only classifier) ----------------
LABELS = ["good","neutral","entangled","bad","contradictory"]
LABEL2ID = {l:i for i,l in enumerate(LABELS)}

def _token_stats(text: str):
    toks = re.findall(r"[a-z0-9]+", (text or "").lower())
    n = len(toks)
    uniq = len(set(toks)) if n else 0
    return n, (uniq / (n + 1e-9))

def _numeric_stats(text: str):
    mg = len(re.findall(r'\d{2,3}\s*mg/dL', text or "", flags=re.I))
    wk = len(re.findall(r'\d{1,2}\s*(?:[-–]\s*\d{1,2}\s*)?weeks', text or "", flags=re.I))
    return mg, wk

def features_for_answer(claims, C, answer: str):
    m = support_metrics(claims, C, answer)
    tok_n, uniq_ratio = _token_stats(answer)
    mg_n, wk_n = _numeric_stats(answer)

    sents = [s for s,_,_ in m["matches"]]
    sent_lens = np.array([len(re.findall(r"[a-z0-9]+", s.lower())) for s in sents], dtype=np.float32) if sents else np.zeros(1, dtype=np.float32)

    x = np.array([
        m["support_rate"],              # 0
        float(m["contradictions"]),     # 1
        float(m["unsupported"]),        # 2
        float(m["sentences"]),          # 3
        m.get("sim_mean", 0.0),         # 4
        m.get("sim_min", 0.0),          # 5
        m.get("sim_max", 0.0),          # 6
        m.get("sim_std", 0.0),          # 7
        float(tok_n),                   # 8
        float(uniq_ratio),              # 9
        float(mg_n),                    # 10
        float(np.std(sent_lens)) if sent_lens.size else 0.0,  # 11
    ], dtype=np.float32)
    return x, m

def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-9)

def train_softmax_classifier(X, y, num_classes: int, lr: float = 0.15, epochs: int = 800, l2: float = 1e-3, seed: int = 7):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W = rng.normal(0, 0.01, size=(d, num_classes)).astype(np.float32)
    b = np.zeros((1, num_classes), dtype=np.float32)

    Y = np.zeros((n, num_classes), dtype=np.float32)
    Y[np.arange(n), y] = 1.0

    for _ in range(epochs):
        logits = X @ W + b
        P = _softmax(logits)
        dlog = (P - Y) / n
        gW = X.T @ dlog + l2 * W
        gb = np.sum(dlog, axis=0, keepdims=True)
        W -= lr * gW
        b -= lr * gb
    return W, b

def fit_learned_evaluator(eval_rows, claims, C):
    Xs=[]
    ys=[]
    for r in eval_rows:
        locked = r.get("locked_answer","")
        x,_ = features_for_answer(claims, C, locked)
        Xs.append(x); ys.append(LABEL2ID["good"])

        v = r.get("variants", {})
        for lab in ["neutral","entangled","bad","contradictory"]:
            ans = v.get(lab,"")
            x,_ = features_for_answer(claims, C, ans)
            Xs.append(x); ys.append(LABEL2ID[lab])

        # Hard negative: realistic contradiction (subtle swaps)
        hn = make_hard_negative(locked)
        x,_ = features_for_answer(claims, C, hn)
        Xs.append(x); ys.append(LABEL2ID["contradictory"])

    X = np.stack(Xs).astype(np.float32)
    y = np.array(ys, dtype=np.int64)

    mu = X.mean(axis=0, keepdims=True)
    sig = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sig

    W, b = train_softmax_classifier(Xn, y, num_classes=len(LABELS))
    return {"W": W, "b": b, "mu": mu.astype(np.float32), "sig": sig.astype(np.float32)}

def predict_learned(model, x):
    x = x.astype(np.float32)[None, :]
    Xn = (x - model["mu"]) / model["sig"]
    logits = Xn @ model["W"] + model["b"]
    p = _softmax(logits)[0]
    lab = LABELS[int(np.argmax(p))]
    return lab, p

def score_answer_learned(model, claims, C, answer: str):
    x, m = features_for_answer(claims, C, answer)
    lab, p = predict_learned(model, x)
    return lab, p, m


# ---------------- Hard-negative generator (realistic contradictions) ----------------
_COMMON_THRESHOLDS = {
    # Common values seen across diagnostic criteria; used to create subtle swaps.
    "fasting": [92, 95, 100],
    "1h": [180, 190],
    "2h": [153, 155],
    "3h": [140],
}

def make_hard_negative(answer: str) -> str:
    """
    Create a realistic 'almost correct' contradictory variant by swapping common
    diagnostic numbers and/or glucose load terms (75g vs 100g) and screening window.
    This is intentionally subtle to train the evaluator on real failure modes.
    """
    if not answer:
        return answer

    out = answer

    # Swap glucose load terms if present
    out = re.sub(r"\b75\s*g\b", "100 g", out, flags=re.I)
    out = re.sub(r"\b100\s*g\b", "75 g", out, flags=re.I)

    # Swap screening window 24-28 weeks ↔ 20-24 weeks (common drift)
    out = re.sub(r"\b24\s*[-–]\s*28\s*weeks\b", "20-24 weeks", out, flags=re.I)
    out = re.sub(r"\b20\s*[-–]\s*24\s*weeks\b", "24-28 weeks", out, flags=re.I)

    # Subtle mg/dL swaps among common thresholds
    def swap_mg(match):
        val = int(match.group(1))
        # pick a nearby plausible alternative different from val
        candidates = [92,95,100,180,190,153,155,140]
        # choose closest different
        best = None
        bestd = 10**9
        for c in candidates:
            if c == val:
                continue
            d = abs(c - val)
            if d < bestd:
                bestd = d
                best = c
        if best is None:
            best = val + 5
        return f"{best} mg/dL"

    out = re.sub(r"(\d{2,3})\s*mg/dL", swap_mg, out, flags=re.I)
    return out

# ---------------- Batch scoring utility ----------------
def batch_score(eval_rows, claims, C, free_map=None, learned_model=None):
    free_map = free_map or {}
    report=[]
    for r in eval_rows:
        qid=r["qid"]
        locked=r.get("locked_answer","")
        free=free_map.get(qid,"")

        m_locked=support_metrics(claims,C,locked)
        row={
            "qid": qid,
            "locked_label_heur": integrity_label_heuristic(m_locked),
            "locked_support": round(m_locked["support_rate"],3),
            "locked_contra": m_locked["contradictions"],
        }

        if learned_model is not None:
            lab_l, p_l, _ = score_answer_learned(learned_model, claims, C, locked)
            row["locked_label_learned"] = lab_l
            row["locked_prob_learned"] = [float(x) for x in p_l]

        if free:
            m_free=support_metrics(claims,C,free)
            row.update({
                "free_label_heur": integrity_label_heuristic(m_free),
                "free_support": round(m_free["support_rate"],3),
                "free_contra": m_free["contradictions"],
            })
            if learned_model is not None:
                lab_f, p_f, _ = score_answer_learned(learned_model, claims, C, free)
                row["free_label_learned"] = lab_f
                row["free_prob_learned"] = [float(x) for x in p_f]

        report.append(row)
    return report
