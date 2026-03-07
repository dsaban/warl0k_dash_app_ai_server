# warlok_gw/model_io.py
from __future__ import annotations
import numpy as np, json, time, hashlib
from pathlib import Path
from typing  import Dict, Any, Tuple

WEIGHT_KEYS = ("Wz","Uz","bz","Wr","Ur","br","Wh","Uh","bh","Wc","bc")
ATK_LABELS  = ("none","reorder","drop","replay","timewarp","splice")

_EXPECTED_SHAPES = {
    "Wz":(64,9),"Uz":(64,64),"bz":(64,),
    "Wr":(64,9),"Ur":(64,64),"br":(64,),
    "Wh":(64,9),"Uh":(64,64),"bh":(64,),
    "Wc":(6,64),"bc":(6,),
}

def save_weights(params, path, extra_meta=None):
    path = Path(path)
    missing = [k for k in WEIGHT_KEYS if k not in params]
    if missing:
        raise KeyError(f"Missing weight keys: {missing}")
    arrays = {k: np.asarray(params[k], dtype=np.float32) for k in WEIGHT_KEYS}
    np.savez(path, **arrays)
    # Write metadata as a JSON sidecar — avoids object-array pickle issues
    meta = {
        "saved_at":   time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "atk_labels": list(ATK_LABELS),
        "H": int(np.asarray(params["Wz"]).shape[0]),
        "D": int(np.asarray(params["Wz"]).shape[1]),
        "C": int(np.asarray(params["Wc"]).shape[0]),
        **(extra_meta or {}),
    }
    Path(str(path).replace(".npz","_meta.json")).write_text(json.dumps(meta, indent=2))
    print(f"[model_io] saved → {path}  ({path.stat().st_size/1024:.1f} KB)")
    return path

def load_weights(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    raw    = np.load(path, allow_pickle=False)
    params = {k: raw[k] for k in WEIGHT_KEYS}
    meta   = {}
    meta_p = Path(str(path).replace(".npz","_meta.json"))
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            pass
    return params, meta

def validate(params):
    for k, expected in _EXPECTED_SHAPES.items():
        arr = np.asarray(params[k])
        if arr.shape != expected:
            raise ValueError(f"Shape mismatch '{k}': expected {expected}, got {arr.shape}")
    return True

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
