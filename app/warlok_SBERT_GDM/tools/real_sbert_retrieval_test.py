import os, json, argparse
import numpy as np
from sentence_transformers import SentenceTransformer

def read_lines(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return [ln.strip() for ln in f if ln.strip()]

def load_questions(path):
    # Supports either:
    # (a) JSONL lines of {"id":..,"q":..}
    # (b) A single JSON list
    txt = open(path, "r", encoding="utf-8").read().strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    qs = []
    for ln in txt.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        qs.append(json.loads(ln))
    return qs

def dot_topk(q_emb, doc_embs, k):
    # embeddings normalized -> cosine == dot
    scores = doc_embs @ q_emb
    if k >= len(scores):
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, kth=k-1)[:k]
        idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--corpus", required=True, help="text file, 1 evidence row per line")
    ap.add_argument("--questions", required=True, help="jsonl or json list")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_len", type=int, default=256, help="token truncation inside transformer")
    args = ap.parse_args()

    print(f"[SBERT] loading model: {args.model}")
    model = SentenceTransformer(args.model)

    corpus = read_lines(args.corpus)
    questions = load_questions(args.questions)

    print(f"[SBERT] corpus rows: {len(corpus)}")
    print(f"[SBERT] questions: {len(questions)}")
    print(f"[SBERT] encoding corpus (batch={args.batch}, max_len={args.max_len}) ...")

    # normalize_embeddings=True => dot product is cosine similarity (fast)
    doc_emb = model.encode(
        corpus,
        batch_size=args.batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        # max_length=args.max_len,
    ).astype(np.float32)

    print("[SBERT] done corpus embedding.")
    print("=" * 90)

    for qi, qobj in enumerate(questions, 1):
        qtext = qobj.get("q") or qobj.get("question") or str(qobj)
        qid = qobj.get("id", f"Q{qi:03d}")

        q_emb = model.encode(
            [qtext],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            # max_length=args.max_len,
        )[0].astype(np.float32)

        idx, sc = dot_topk(q_emb, doc_emb, args.topk)

        print(f"\n[{qid}] {qtext}")
        for rank, (i, s) in enumerate(zip(idx, sc), 1):
            row = corpus[int(i)]
            row = row.replace("\t", " ").strip()
            if len(row) > 240:
                row = row[:240] + "..."
            print(f"{rank:02d}. score={float(s):.4f} | {row}")

    print("\n[SBERT] finished.")

if __name__ == "__main__":
    main()
