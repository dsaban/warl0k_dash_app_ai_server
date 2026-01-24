# GDM Integrity MoE (BM25 + ClaimGraph + MoE + EBM + Learning)

This is a local, integrity-first Q/A system for nuanced GDM questions.

## What it does
- Builds a **dual index**:
  - **ChunkNodes** (topic chunks, BM25 retrieval)
  - **ClaimNodes** (sentence claims, hard grounding)
- Answers by:
  - Retrieving top chunks (BM25)
  - Ranking claims inside those chunks (BM25 + schema boosts - penalties)
  - Running multiple **experts** (MoE) that assemble candidate answers using only claims
  - Scoring candidates with an **EBM** (energy-based ranker)
  - Enforcing **hard integrity gates** (must include required entities/edges, no drift, all sentences cited)
- Optional learning:
  - Train **Router** (select expert weights)
  - Train **EBM** (learn candidate ranking)
  - Train **Claim Re-ranker** (learn entail/neutral/contradict claim support)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
