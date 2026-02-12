# Patent Integrity Graph Explorer (Streamlit Demo)

A lightweight GDM-like demo for **evidence-locked patent claim mining**:
- Loads a small corpus (defaults to `/mnt/data/0.txt ... /mnt/data/9.txt`)
- Parses claim-like sections (heuristic)
- Decomposes into **AtomicClaims**
- Tags AtomicClaims (mechanisms / crypto / constraints / threats)
- Provides **BM25 lexical search**, evidence packs, a KG view, and a novelty/draft builder.

## Run
```bash
cd patent_integrity_demo
streamlit run app.py
```

## Requirements
Install:
```bash
pip install -r requirements.txt
```
