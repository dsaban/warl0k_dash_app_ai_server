import argparse, json
import numpy as np

from core.learned_rerank import RERANK_FEATURES, hashing_embed, cosine

LABEL = {"entail": 1.0, "neutral": 0.0, "contradict": -1.0}

def vec(feat):
    return np.array([float(feat.get(k,0.0)) for k in RERANK_FEATURES], dtype=np.float32)

def compute_feat(question: str, support: str):
    qv = hashing_embed(question, 512)
    cv = hashing_embed(support, 512)
    cosv = cosine(qv, cv)

    t = support.lower()
    ind = 1.0 if ("independent risk factor" in t or "independently associated" in t) else 0.0
    horm = 1.0 if any(k in t for k in ["placental","hpl","progesterone","cortisol","growth hormone","estrogen","prolactin"]) else 0.0
    return {"cos_q_claim": cosv, "has_independent_risk_phrase": ind, "has_hormone_phrase": horm}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.08)
    args = ap.parse_args()

    d = np.load(args.index, allow_pickle=True)
    claim_id = list(d["claim_id"])
    claim_support = list(d["claim_support"])
    id_to_idx = {cid: i for i, cid in enumerate(claim_id)}

    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    pairs = []
    for r in rows:
        q = r.get("question", "")
        for cid in r.get("support_claim_ids", []):
            pairs.append((q, cid, "entail"))
        for cid in r.get("neutral_claim_ids", []):
            pairs.append((q, cid, "neutral"))
        for cid in r.get("contradict_claim_ids", []):
            pairs.append((q, cid, "contradict"))

    if not pairs:
        raise RuntimeError("No claim supervision found. Add support_claim_ids / neutral_claim_ids / contradict_claim_ids to train_pairs.jsonl")

    w = np.zeros((len(RERANK_FEATURES),), dtype=np.float32)
    b = np.float32(0.0)

    for ep in range(args.epochs):
        np.random.shuffle(pairs)
        loss = 0.0
        for q, cid, lab in pairs:
            if cid not in id_to_idx:
                continue
            i = id_to_idx[cid]
            supp = claim_support[i]

            feat = compute_feat(q, supp)
            x = vec(feat)
            y = LABEL[lab]
            pred = float(np.dot(w, x) + b)
            err = (pred - y)

            w -= args.lr * (2.0 * err) * x
            b -= args.lr * (2.0 * err)
            loss += err * err

        print(f"epoch {ep+1}/{args.epochs} mse={loss/max(1,len(pairs)):.4f}")

    np.savez(args.out, rerank_w=w, rerank_b=b, rerank_dim=np.int32(512))
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
