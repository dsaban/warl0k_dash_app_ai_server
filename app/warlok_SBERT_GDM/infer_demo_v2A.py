from __future__ import annotations

import argparse, json, re
from pathlib import Path
from typing import List, Tuple
import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A


def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        if 60 <= len(s) <= 320 and "http" not in s.lower():
            out.append(s)
    return out


def build_sentence_bank(doc_paths: List[Path]) -> Tuple[List[str], List[str]]:
    sents, meta = [], []
    for p in doc_paths:
        t = p.read_text(errors="ignore")
        ss = split_sentences(t)
        sents.extend(ss)
        meta.extend([p.name] * len(ss))
    return sents, meta


def topk_cosine(query_emb: np.ndarray, embs: np.ndarray, k: int = 10):
    sims = embs @ query_emb
    k = min(k, sims.shape[0])
    idx = np.argpartition(-sims, kth=k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return [(int(i), float(sims[i])) for i in idx]


def hint_match(text: str, hints: List[str]) -> int:
    t = text.lower()
    score = 0
    for h in hints:
        if h.lower() in t:
            score += 1
    return score


def main(args):
    model = GdmSentenceTransformerV2A.load(Path(args.models_dir), name=args.name)
    print(f"[load] model={args.name} vocab={len(model.vocab)} max_len={model.max_len} layers={model.n_layers}")

    doc_paths = [Path(p) for p in args.docs]
    sents, meta = build_sentence_bank(doc_paths)
    print(f"[bank] sentences={len(sents)}")

    # Pre-embed bank (cache this later if you want)
    bank_embs = []
    for i, s in enumerate(sents):
        if i % 500 == 0 and i > 0:
            print(f"[embed] {i}/{len(sents)}")
        e = model.embed(s).astype(np.float32)
        e = e / (np.linalg.norm(e) + 1e-9)
        bank_embs.append(e)
    bank_embs = np.vstack(bank_embs).astype(np.float32)

    demos = []
    for line in Path(args.demo).read_text(encoding="utf-8").splitlines():
        if line.strip():
            demos.append(json.loads(line))
    print(f"[demo] questions={len(demos)}")

    hit10 = 0
    rel_ok = 0

    for d in demos:
        q = d["q"]
        qemb = model.embed(q).astype(np.float32)
        qemb = qemb / (np.linalg.norm(qemb) + 1e-9)

        top = topk_cosine(qemb, bank_embs, k=args.k)
        top_texts = [sents[i] for i, _ in top]

        best_i, best_sim = top[0]
        best_text = sents[best_i]

        gold = d.get("gold_hint", [])
        bad = d.get("bad_hint", [])

        gold_best = max(hint_match(t, gold) for t in top_texts) if gold else 0
        bad_best = max(hint_match(t, bad) for t in top_texts) if bad else 0
        if gold_best > bad_best and gold_best > 0:
            hit10 += 1

        pred_lab, probs = model.predict_relation(q, best_text)
        exp = d.get("expected_relation", "entailment")
        ok = (pred_lab == "contradiction") if exp == "contradiction" else (pred_lab != "contradiction")
        if ok:
            rel_ok += 1

        print("\n" + "=" * 80)
        print(f"{d['id']} topic={d['topic']}")
        print("Q:", q)
        print(f"Top1 cosine={best_sim:.4f} doc={meta[best_i]}")
        print("Top1:", best_text[:220] + ("..." if len(best_text) > 220 else ""))
        print("NLI pred:", pred_lab, "probs:", np.round(probs, 3).tolist())
        print("hint(gold_best, bad_best):", gold_best, bad_best)

        for pt in d.get("pair_tests", []):
            pq = pt["q"]

            def find_by_hint(hints):
                for s in sents:
                    if hint_match(s, hints) >= max(1, len(hints) // 2):
                        return s
                return None

            good_a = find_by_hint(pt.get("good_answer_hint", [])) or best_text
            bad_a = find_by_hint(pt.get("bad_answer_hint", [])) or best_text

            good_lab, good_probs = model.predict_relation(pq, good_a)
            bad_lab, bad_probs = model.predict_relation(pq, bad_a)

            print("\n[probe]")
            print("probe Q:", pq)
            print("good pred:", good_lab, "probs:", np.round(good_probs, 3).tolist())
            print("bad  pred:", bad_lab,  "probs:", np.round(bad_probs, 3).tolist())

    print("\n" + "#" * 80)
    print(f"Top-{args.k} hint hit-rate: {hit10}/{len(demos)} = {hit10/len(demos):.3f}")
    print(f"Relation gate check: {rel_ok}/{len(demos)} = {rel_ok/len(demos):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="models")
    ap.add_argument("--name", default="gdm_sbert_v2A")
    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--demo", default="tests/demo_questions.jsonl")
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    main(args)
