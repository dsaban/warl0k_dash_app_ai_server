import argparse, json
import numpy as np
from core.moe_router import MoERouter, EXPERT_ORDER, hash_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.08)
    args = ap.parse_args()

    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    train = [r for r in rows if r.get("target_expert") in EXPERT_ORDER]
    if not train:
        raise RuntimeError("No rows with target_expert matching expert names.")

    router = MoERouter(dim=128)

    for ep in range(args.epochs):
        np.random.shuffle(train)
        loss = 0.0
        for r in train:
            x = hash_features(r["question"], router.dim)
            t = EXPERT_ORDER.index(r["target_expert"])

            logits = router.W @ x + router.b
            logits = logits - np.max(logits)
            p = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)

            loss += float(-np.log(p[t] + 1e-9))

            g = p
            g[t] -= 1.0
            router.W -= args.lr * (g[:, None] * x[None, :])
            router.b -= args.lr * g

        print(f"epoch {ep+1}/{args.epochs} loss={loss/len(train):.4f}")

    router.use_learned = True
    router.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
