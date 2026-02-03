# GDM NumPy-only SBERT-lite (Slim but Deep)

This pack gives you:

- **NumPy-only training** (manual backprop) on 4 provided GDM docs
- Word vocab
- Synthetic NLI-like pairs (entailment / neutral / contradiction)
- **Fast inference engine** for real-time similarity in your WARL0K medical integrity pipeline

## Files

- `train_gdm_sbert_numpy.py`  
  Trains the model and writes a **model package** to the current directory:

  - `gdm_sbert_numpy_weights.npz`
  - `gdm_sbert_numpy_vocab.txt`
  - `gdm_sbert_numpy_config.npz`

- `gdm_engine_numpy.py`  
  Loads the saved package and provides:
  - `embed(text)`
  - `similarity(a,b)`
  - `embed_pipe([phrases])`
  - `build_index(phrases)`
  - `topk_from_index(query, phrases, embs)`
  - `topk(query, phrases)`

- `run_infer_demo.py`  
  End-to-end demo: embed a pipe, similarity matrix, top-k retrieval.

## Train

```bash
python3 train_gdm_sbert_numpy.py
```

## Inference demo

```bash
python3 run_infer_demo.py --prefix gdm_sbert_numpy
```

Or engine CLI:

```bash
python3 gdm_engine_numpy.py --prefix gdm_sbert_numpy --demo
python3 gdm_engine_numpy.py --prefix gdm_sbert_numpy --pipe "phrase one" "phrase two"
```

## Notes

- This is intentionally "slim but deep": **2 gated layers + attention pooling**, CPU-friendly.
- For production, precompute candidate embeddings once:
  - `phrases, embs = engine.build_index(phrases)`
  - then `engine.topk_from_index(query, phrases, embs)`
