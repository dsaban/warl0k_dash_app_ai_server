import argparse, json
import numpy as np

from core.retrieve import DualBM25Retriever
from core.moe_router import MoERouter
from core.ebm_ranker import EBMRanker, feat_vec
from core.pipeline import IntegrityMoEPipeline

LABEL_Y = {"good": 0, "neutral": 1, "bad": 2}

def load_index(path: str):
    d = np.load(path, allow_pickle=True)

    class Chunk: pass
    class Claim: pass

    chunks = []
    for i in range(len(d["chunk_id"])):
        c = Chunk()
        c.chunk_id = d["chunk_id"][i]
        c.file = d["chunk_file"][i]
        c.text = d["chunk_text"][i]
        c.span = tuple(d["chunk_span"][i])
        c.token_count = int(d["chunk_tok"][i])
        c.section = d["chunk_section"][i] if "chunk_section" in d else ""
        chunks.append(c)

    claims = []
    for i in range(len(d["claim_id"])):
        c = Claim()
        c.claim_id = d["claim_id"][i]
        c.file = d["claim_file"][i]
        c.chunk_id = d["claim_chunk"][i]
        c.sentence = d["claim_sentence"][i]
        c.span = tuple(d["claim_span"][i])
        c.support_text = d["claim_support"][i]
        c.entities = list(d["claim_entities"][i]) if d["claim_entities"][i] is not None else []
        c.edge_hints = list(d["claim_edges"][i]) if d["claim_edges"][i] is not None else []
        claims.append(c)

    return chunks, claims

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--top_chunks", type=int, default=8)
    ap.add_argument("--max_claims", type=int, default=24)
    args = ap.parse_args()

    chunks, claims = load_index(args.index)
    retriever = DualBM25Retriever(chunks, claims)
    router = MoERouter()
    ebm = EBMRanker()
    pipe = IntegrityMoEPipeline(chunks, claims, retriever, router=router, ebm=ebm, rerank_path="")

    rows = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    train = [r for r in rows if r.get("label") in LABEL_Y]
    if not train:
        raise RuntimeError("No rows with label good/neutral/bad.")

    for ep in range(args.epochs):
        np.random.shuffle(train)
        tot = 0.0
        for r in train:
            out = pipe.infer(r["question"], top_chunks=args.top_chunks, max_claims=args.max_claims)
            best = out.get("best")
            if not best:
                continue

            y = LABEL_Y[r["label"]]
            f = feat_vec(best["features"])
            E = float(np.dot(ebm.w, f) + ebm.b)

            target = (-1.0 if y == 0 else (0.5 if y == 1 else 2.0))
            err = (E - target)

            ebm.w -= args.lr * (2.0 * err) * f
            ebm.b -= args.lr * (2.0 * err)
            tot += err * err

        print(f"epoch {ep+1}/{args.epochs} mse={tot/max(1,len(train)):.4f}")

    ebm.save(args.out)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
