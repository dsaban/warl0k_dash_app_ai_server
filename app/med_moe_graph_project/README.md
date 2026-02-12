# Local Medical MoE + Graph Explorer (Refined Schema)

This project is a **clean rebuild** of the local medical MoE pipeline with a stronger, explicit **graph schema** and a UI that surfaces:

- **Ranked evidence phrases/chunks** for a question
- **Detected QType** (from a *fixed* taxonomy)
- **Tags, entities, and connections** used to justify retrieval and answer
- A **graph neighborhood explorer** so the UI is not “answer-only” but also “why/how the answer was formed”

## Key rules (your requirements)

1. **Stop adding new QTypes dynamically**
   - QTypes are defined in `config/ontology.json`.
   - If a question doesn’t match, it falls back to `unknown`.

2. **Entities are NOT created from scratch during inference**
   - Entity extraction is **lexicon-based** (`data/lexicon/entities.json`).
   - Unknown terms are ignored (optionally logged), preventing uncontrolled growth.

3. **UI must show tags + node/edge connections**
   - Tab **Ask** shows: answer + ranked evidence + tags/entities + connection table.
   - Tab **Graph Explorer** shows: neighborhood and edge lists, plus a small graph plot.

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional: build graph from docs
python -m core.cli ingest --docs data/docs --out data/graph/graph.json

# run UI
streamlit run app/app1.py
```

## What’s inside

- `core/ontology.py` — fixed QTypes + Tags loader and validator
- `core/graph.py` — lightweight typed graph (no heavy dependencies)
- `core/ingest.py` — chunking docs + lexicon entity match + tag assignment
- `core/retrieval.py` — TF-IDF style scoring + tag/entity boosting
- `core/moe.py` — expert routing by QType + evidence-grounded synthesis
- `core/integrity.py` — checklist-based answer integrity guard
- `core/pipeline.py` — the single entry point: `LocalMedMoEPipeline.infer(question)`
- `app/app.py` — Streamlit UI

## Notes

- This is **not** a medical device. It is a research/demo pipeline for structured answering.
- Replace `data/docs/*.md` with your real corpus. Re-run `ingest` to rebuild the graph.
