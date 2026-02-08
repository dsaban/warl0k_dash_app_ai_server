# WARL0K GDM Integrity Demo (from scratch)

This is a **proof-first** evidence-based demo for gestational diabetes (GDM):

- Hybrid retrieval: **BM25 + TF‑IDF semantic** (SBERT‑lite) over your provided docs
- Locked Q/A set (gold + variants) for integrity testing
- **Patient State** tab: evidence‑locked care checks (screening/postpartum follow‑up)

## Quick start

```bash
cd app
streamlit run app.py
```

## Data

- `app/data/raw_docs/` – your source documents
- `app/data/passages.jsonl` – 3-sentence passages built from the docs
- `app/data/claims/claims_seed.jsonl` – seeded claim packs (auto-selected evidence)
- `app/data/gold/questions.csv` – locked questions + variants
- `app/data/patients/patients_demo.jsonl` – demo patient timelines
- `app/data/state_checks/state_checks.jsonl` – gold patient-state checks

## Integrity philosophy

The UI is built around one rule:

> If a statement cannot be supported by retrieved evidence, the system should **warn or refuse**.

You can tighten checks in `app/core/qa.py` and add more patient checks in
`app/data/state_checks/state_checks.jsonl`.
