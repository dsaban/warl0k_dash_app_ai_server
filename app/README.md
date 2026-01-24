# GDM Graph-Gated MoE + EBM (Python-only)

This is a **GDM-only** reasoning project with:

- Locked ontology (fixed QTypes, tags, relations)
- Locked entity lexicon (no inference-time new entities)
- Graph-gated Mixture of Experts (MoE)
- Energy-Based scoring penalties (EBM-style)
- Minimal contrastive training over **causal paths** (not an LLM)

## Run
pip install -r requirements.txt
streamlit run ui/app.py

## Train
python training/train.py
