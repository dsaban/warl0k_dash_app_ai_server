"""
Train and persist SBERT-lite model locally (NumPy-only).

Usage:
  python sbertlite_train.py
Writes:
  sbert_lite_model.npz
"""
from pathlib import Path
import core

BASE_DIR = Path(__file__).parent
claims = core.load_jsonl(BASE_DIR/"claims_200.jsonl")
eval_rows = core.load_jsonl(BASE_DIR/"eval_set_100.jsonl")

model = core.train_sbert_lite_from_eval(eval_rows, claims, epochs=100, batch_size=8, in_dim=4096, out_dim=256, lr=0.05, seed=7)
model.save(BASE_DIR/"sbert_lite_model.npz")
print("Saved:", BASE_DIR/"sbert_lite_model.npz")
