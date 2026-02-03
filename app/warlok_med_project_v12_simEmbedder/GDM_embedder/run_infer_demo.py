#!/usr/bin/env python3
"""
End-to-end demo:
1) Load saved model package (from training output)
2) Embed a pipe of phrases
3) Compute similarity matrix (pipe vs pipe)
4) Do top-k retrieval from a candidate list

Run:
  python3 run_infer_demo.py --prefix gdm_sbert_numpy
"""
import argparse
import numpy as np
from gdm_engine_numpy import GDMEmbeddingEngine, cosine_batch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gdm_sbert_numpy")
    args = ap.parse_args()

    eng = GDMEmbeddingEngine.load(prefix=args.prefix)
    print("Model loaded from prefix:", args.prefix)

    pipe = [
        "when should screening for gestational diabetes be performed",
        "what is the postpartum follow up test for gestational diabetes",
        "maternal hyperglycemia leads to fetal macrosomia via fetal hyperinsulinemia",
        "management may include diet therapy and insulin when needed",
    ]

    E = eng.embed_pipe(pipe)
    print("Pipe embeddings:", E.shape)

    # similarity matrix pipe vs pipe
    S = np.zeros((len(pipe), len(pipe)), dtype=np.float32)
    for i in range(len(pipe)):
        S[i] = cosine_batch(E, E[i])
    print("\nSimilarity matrix (pipe vs pipe):")
    np.set_printoptions(precision=3, suppress=True)
    print(S)

    # retrieval
    candidates = [
        "maternal hyperglycemia increases glucose transfer to fetus and raises fetal insulin, promoting growth",
        "maternal glucose crosses the placenta, but insulin does not",
        "maternal hyperglycemia increase fetal size via fetal hyperinsulinemia",
        "diet and exercise are first line management for gestational diabetes",
        "screening for gestational diabetes is recommended at 24-28 weeks gestation",
        "postpartum glucose testing is recommended to assess for persistent diabetes",
        "fetal size increasing is a result of elevated maternal insulin levels",
        "oral glucose tolerance test is used for diagnosis",
        "postpartum testing is recommended after a gestational diabetes pregnancy",
        "screening may use oral glucose tolerance testing in pregnancy",
        "contradiction: gestational diabetes reduces the risk of macrosomia",
        "insulin is used if diet and exercise are insufficient to control glucose",
    ]
    # q = "why does maternal hyperglycemia increase fetal size"
    # q = "what is the recommended test for gestational diabetes postpartum"
    q = "when is screening for gestational diabetes performed"
    # q = "how is gestational diabetes managed"
    # q = "how does gestational diabetes lead to fetal macrosomia"
    # q = "how does gestational diabetes affect the fetus"
    # q = "what is the recommended postpartum test after gestational diabetes"
    top = eng.topk(q, candidates, k=3)
    print("\nTop-3 for query:", q)
    for sim, txt in top:
        print(f"  sim={sim:.3f} | {txt}")

if __name__ == "__main__":
    main()
