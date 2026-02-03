from __future__ import annotations

import argparse, re
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A


def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")


def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        if 60 <= len(s) <= 360 and "http" not in s.lower():
            out.append(s)
    return out


EVIDENCE_TERMS = [
    "ogtt","75","24","28","fasting","hba1c","threshold","diagnos",
    "postpartum","6","12","follow-up","type 2","surveillance",
    "insulin","metformin","lifestyle","diet","exercise","monitor","targets","smbg","cgm",
    "macrosomia","hypoglyc","preeclamps","shoulder","nicu","polycyth","hyperbilirub",
    "placent","insulin resistance","beta","hyperglyc","fetal insulin",
]

def is_evidence_sentence(s: str) -> bool:
    t = s.lower()
    if len(t) < 70:
        return False
    for w in EVIDENCE_TERMS:
        if w in t:
            return True
    return False


def l2norm_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def build_bank(docs: List[Path], max_sents: int, evidence_only: bool) -> List[str]:
    sents = []
    for p in docs:
        ss = split_sentences(read_text(p))
        sents.extend(ss)
    if evidence_only:
        sents = [s for s in sents if is_evidence_sentence(s)]
    # de-dup while preserving order
    seen = set()
    uniq = []
    for s in sents:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq[:max_sents] if max_sents > 0 else uniq


def topk_cosine(qv: np.ndarray, bank_embs: np.ndarray, k: int) -> List[Tuple[int, float]]:
    sims = bank_embs @ qv
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, kth=k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[int(i)])) for i in idx]


def main(args):
    docs = [Path(x) for x in args.docs]
    print("[bank] building...")
    bank_sents = build_bank(docs, max_sents=args.bank_max_sents, evidence_only=args.evidence_only)
    print(f"[bank] sentences={len(bank_sents)} (evidence_only={args.evidence_only})")

    print("[model] loading...")
    model = GdmSentenceTransformerV2A.load(Path(args.models_dir), name=args.name)
    print(f"[model] loaded: {args.name}")

    print("[bank] embedding...")
    embs = []
    for i, s in enumerate(bank_sents):
        if args.print_steps and i and (i % 800 == 0):
            print(f"  embed {i}/{len(bank_sents)}")
        embs.append(model.embed(s).astype(np.float32))
    bank_embs = l2norm_rows(np.vstack(embs).astype(np.float32))
    print(f"[bank] emb matrix: {bank_embs.shape}")

    # queries
    if args.query:
        queries = [args.query]
    else:
        qfile = Path(args.query_file)
        queries = [line.strip() for line in qfile.read_text(errors="ignore").splitlines() if line.strip()]
        print(f"[query] loaded {len(queries)} from {qfile}")

    for qi, q in enumerate(queries, 1):
        qv = model.embed(q).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)

        hits = topk_cosine(qv, bank_embs, k=args.k)

        print("\n" + "=" * 90)
        print(f"[Q{qi}] {q}")
        print("-" * 90)
        for rank, (idx, score) in enumerate(hits, 1):
            print(f"{rank:02d}. score={score:.4f} | {bank_sents[idx]}")
    print("\n[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--name", required=True)

    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("-k", type=int, default=10)

    ap.add_argument("--bank_max_sents", type=int, default=6000)
    ap.add_argument("--evidence_only", action="store_true")
    ap.add_argument("--print_steps", action="store_true")

    ap.add_argument("--query", default="")
    ap.add_argument("--query_file", default="tests/demo_questions.txt")

    args = ap.parse_args()
    main(args)
