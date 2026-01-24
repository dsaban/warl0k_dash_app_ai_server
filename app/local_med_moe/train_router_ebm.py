#!/usr/bin/env python3
import os, glob, argparse
import numpy as np

from core.config import AppConfig
from core.pipeline import LocalMedMoEPipeline
from core.synthdata import synth_qa_from_chunks

def load_docs(folder: str) -> str:
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "\n\n".join(texts)

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def one_hot(k: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    v[k] = 1.0
    return v

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs")
    ap.add_argument("--out", default="trained_weights.npz")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr_router", type=float, default=0.03)
    ap.add_argument("--lr_ebm", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=2000)
    args = ap.parse_args()

    cfg = AppConfig()
    docs_text = load_docs(args.docs)
    if not docs_text.strip():
        raise RuntimeError(f"No .txt docs in {args.docs}/")

    pipe = LocalMedMoEPipeline(docs_text, cfg, weights_path=None)

    # synth dataset
    data = synth_qa_from_chunks(pipe.chunks[: min(80, len(pipe.chunks))], n_per_chunk=2, seed=13)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(data)
    data = data[: min(len(data), args.max_samples)]

    # posture routing targets (keep simple)
    target_expert = {
        "good": 0,
        "neutral": 1,
        "bad": 2,
        "irrelevant_support": 1,
        "off_topic": 1,
        "drift_monitoring": 1,
        "wrong_subject": 1,
    }

    # margins
    mg = 1.0
    mb = 2.2
    mirr = 2.0
    moff = 2.1

    def clamp_expert(k: int) -> int:
        return int(max(0, min(cfg.n_experts - 1, k)))

    def ebm_step(qv, av, ev, sign: float):
        # E(q,a,e) = - u · tanh(W(q+a+e)) - b
        z = (qv + av + ev).astype(np.float32)
        h = np.tanh(pipe.ebm.W @ z).astype(np.float32)
        dh = (1.0 - h * h).astype(np.float32)

        grad_u_E = -h
        grad_W_E = -np.outer(pipe.ebm.u * dh, z)
        grad_b_E = -1.0

        pipe.ebm.u += args.lr_ebm * sign * grad_u_E
        pipe.ebm.W += args.lr_ebm * sign * grad_W_E
        pipe.ebm.b += np.float32(args.lr_ebm * sign * grad_b_E)

    for ep in range(1, args.epochs + 1):
        rng.shuffle(data)
        loss_r, loss_e = 0.0, 0.0
        n_r, n_e = 0, 0

        for ex in data:
            q = ex.get("q", "")
            label = ex.get("label", "neutral")
            evidence = ex.get("evidence", "")

            qv = pipe._qvec(q)
            ev = pipe._qvec(evidence)
            x = (qv + 0.7 * ev).astype(np.float32)

            # --- router CE ---
            t = clamp_expert(target_expert.get(label, 1))
            logits = (pipe.moe.Wr @ x + pipe.moe.br).astype(np.float32)
            p = softmax(logits)
            ce = float(-np.log(p[t] + 1e-9))
            loss_r += ce; n_r += 1

            g = (p - one_hot(t, cfg.n_experts)).astype(np.float32)
            pipe.moe.Wr -= args.lr_router * (g[:, None] * x[None, :])
            pipe.moe.br -= args.lr_router * g

            # --- EBM hinge ---
            moe_out = pipe.moe.forward(x)
            a_vec = moe_out["expert_vecs"][t].astype(np.float32)
            E = float(pipe.ebm.energy(qv, a_vec, ev))

            if label == "good":
                L = max(0.0, E - mg)
                if L > 0: ebm_step(qv, a_vec, ev, sign=-1.0)
                loss_e += L; n_e += 1
            elif label == "bad":
                L = max(0.0, mb - E)
                if L > 0: ebm_step(qv, a_vec, ev, sign=+1.0)
                loss_e += L; n_e += 1
            elif label == "irrelevant_support":
                L = max(0.0, mirr - E)
                if L > 0: ebm_step(qv, a_vec, ev, sign=+1.0)
                loss_e += L; n_e += 1
            elif label in ("off_topic","drift_monitoring","wrong_subject"):
                L = max(0.0, moff - E)
                if L > 0: ebm_step(qv, a_vec, ev, sign=+1.0)
                loss_e += L; n_e += 1

        print(f"Epoch {ep}/{args.epochs} | router_CE={loss_r/max(1,n_r):.4f} | ebm_hinge={loss_e/max(1,n_e):.4f} | samples={len(data)}")

    np.savez(
        args.out,
        Wr=pipe.moe.Wr.astype(np.float32),
        br=pipe.moe.br.astype(np.float32),
        ebmW=pipe.ebm.W.astype(np.float32),
        ebmU=pipe.ebm.u.astype(np.float32),
        ebmB=np.array(pipe.ebm.b, dtype=np.float32),
    )
    print(f"Saved weights: {args.out}")

if __name__ == "__main__":
    main()
