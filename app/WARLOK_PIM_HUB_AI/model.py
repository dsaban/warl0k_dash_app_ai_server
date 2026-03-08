# warlok/model.py — Multi-label GRU, featuriser, weight registry
import hashlib, time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from crypto import sigmoid, softmax1d, H

# ── Label registry ────────────────────────────────────────────────────────────
ATK_LABELS = ["none", "reorder", "drop", "replay", "timewarp", "splice"]
N_CLASSES  = len(ATK_LABELS)

ATK_COLORS = {
    "none":     "#4ade80",
    "reorder":  "#fb923c",
    "drop":     "#f87171",
    "replay":   "#c084fc",
    "timewarp": "#fbbf24",
    "splice":   "#38bdf8",
}

FIELD_TO_ATKS: Dict[str, List[str]] = {
    "dt_ms":          ["timewarp"],
    "op_code":        ["splice"],
    "step_idx":       ["reorder"],
    "global_counter": ["replay"],
    "window_id":      ["reorder"],
    "os_meas":        ["splice"],
    "session_id":     ["replay"],
}

WINDOW_SIZE = 48
RNN_IN_DIM  = 15   # 9 original + 6 proof features

# ── 15-dim featuriser ─────────────────────────────────────────────────────────
def featurise(trace: List[Dict], max_dt: float = 200.0,
              window_size: int = WINDOW_SIZE) -> np.ndarray:
    """
    Convert trace list → (window_size, 15) float32 array.

    Dims 0–8  : original features
      0: os_meas
      1: dt_ms / max_dt
      2: step / (window_size-1)
      3: counter / window_size
      4: accept bit
      5: READ one-hot
      6: WRITE one-hot
      7: CONTROL one-hot
      8: position t / (window_size-1)

    Dims 9–14 : proof-in-motion features
      9:  acc_divergence        (0=clean, >0=history mismatch)
      10: root_delta_norm       (XOR of consecutive window roots, normalised)
      11: membership_depth      (log2(window_size), constant=5.58 for 48)
      12: leaf_hash_norm        (first 4 bytes of leaf_hash as float in [0,1])
      13: window_boundary       (1.0 if this msg closes a window)
      14: anchor_age_norm       (normalised epoch age)
    """
    import math
    op_map = {"READ": 0, "WRITE": 1, "CONTROL": 2}
    X = np.zeros((window_size, RNN_IN_DIM), dtype=np.float32)
    depth_norm = math.log2(max(window_size, 2)) / 8.0

    for t, r in enumerate(trace[:window_size]):
        oi = op_map.get(r.get("op","WRITE"), 1)
        # Original 9
        X[t, 0] = float(r.get("meas", 0.0))
        X[t, 1] = float(r.get("dt_ms", 0)) / max(max_dt, 1.0)
        X[t, 2] = float(r.get("step", 0)) / max(window_size - 1, 1)
        X[t, 3] = float(r.get("ctr",  0)) / max(window_size, 1)
        X[t, 4] = 1.0 if r.get("decision") == "ACCEPT" else 0.0
        X[t, 5 + oi] = 1.0
        X[t, 8] = float(t) / max(window_size - 1, 1)
        # Proof features
        X[t, 9]  = float(r.get("acc_divergence", 0.0))
        X[t, 10] = float(r.get("root_delta_norm", 0.0))
        X[t, 11] = depth_norm
        lh = r.get("leaf_hash_norm", 0.0)
        X[t, 12] = float(lh) if isinstance(lh, float) else 0.0
        X[t, 13] = 1.0 if r.get("window_boundary", False) else 0.0
        X[t, 14] = float(r.get("anchor_age_norm", 0.0))
    return X

def make_multihot(label_indices: List[int]) -> np.ndarray:
    v = np.zeros(N_CLASSES, dtype=np.float32)
    for i in label_indices:
        v[i] = 1.0
    return v

def infer_attack_labels(edits: List[Dict], named_attacks: List[str]) -> List[int]:
    active = set()
    for a in named_attacks:
        if a != "none":
            active.add(a)
    for e in edits:
        for lbl in FIELD_TO_ATKS.get(str(e.get("field", "")), []):
            active.add(lbl)
    if not active:
        active.add("none")
    return sorted([ATK_LABELS.index(a) for a in active if a in ATK_LABELS])

# ── GRU implementation ────────────────────────────────────────────────────────
def init_rnn(seed: int = 0, hdim: int = 96) -> dict:
    rng = np.random.RandomState(seed); s = 0.06
    D, H_, C = RNN_IN_DIM, hdim, N_CLASSES
    p = {}
    for g in ("z", "r", "h"):
        p[f"W{g}"] = (rng.randn(H_, D) * s).astype(np.float32)
        p[f"U{g}"] = (rng.randn(H_, H_) * s).astype(np.float32)
        p[f"b{g}"] = np.zeros(H_, dtype=np.float32)
    p["Wc"] = (rng.randn(C, H_) * s).astype(np.float32)
    p["bc"] = np.zeros(C, dtype=np.float32)
    return p

def rnn_forward(Xb: np.ndarray, p: dict) -> np.ndarray:
    """(B,T,D) → (B,H) mean-pooled context."""
    B, T, _ = Xb.shape; H_ = p["Wz"].shape[0]
    h = np.zeros((B, H_), dtype=np.float32)
    s = np.zeros((B, H_), dtype=np.float32)
    for t in range(T):
        x = Xb[:, t, :]
        z = sigmoid(x @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
        r = sigmoid(x @ p["Wr"].T + h @ p["Ur"].T + p["br"])
        g = np.tanh(x @ p["Wh"].T + (r*h) @ p["Uh"].T + p["bh"])
        h = (1-z)*h + z*g; s += h
    return s / T

def rnn_step_multilabel(p, Xb, Yb, opt, lr):
    """One Adam step for multi-label BCE. Returns scalar loss."""
    B, T, D = Xb.shape; H_ = p["Wz"].shape[0]
    hs = np.zeros((B,T,H_), dtype=np.float32)
    zs = np.zeros((B,T,H_), dtype=np.float32)
    rs = np.zeros((B,T,H_), dtype=np.float32)
    gs = np.zeros((B,T,H_), dtype=np.float32)
    hp_arr = np.zeros((B,T,H_), dtype=np.float32)
    h = np.zeros((B,H_), dtype=np.float32)

    for t in range(T):
        x = Xb[:, t, :]
        hp_arr[:, t, :] = h
        z = sigmoid(x @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
        r = sigmoid(x @ p["Wr"].T + h @ p["Ur"].T + p["br"])
        g = np.tanh(x @ p["Wh"].T + (r*h) @ p["Uh"].T + p["bh"])
        h = (1-z)*h + z*g
        hs[:,t,:]=h; zs[:,t,:]=z; rs[:,t,:]=r; gs[:,t,:]=g

    ctx    = hs.mean(axis=1)
    logits = ctx @ p["Wc"].T + p["bc"]
    probs  = sigmoid(logits)
    eps    = 1e-7
    loss   = float(-np.mean(
        Yb * np.log(probs+eps) + (1-Yb) * np.log(1-probs+eps)
    ))

    dlog   = (probs - Yb) / B
    grads  = {"Wc": dlog.T @ ctx, "bc": dlog.sum(0)}
    dctx   = dlog @ p["Wc"]
    dh_ctx = dctx / T

    dWz=np.zeros_like(p["Wz"]); dUz=np.zeros_like(p["Uz"]); dbz=np.zeros_like(p["bz"])
    dWr=np.zeros_like(p["Wr"]); dUr=np.zeros_like(p["Ur"]); dbr=np.zeros_like(p["br"])
    dWh=np.zeros_like(p["Wh"]); dUh=np.zeros_like(p["Uh"]); dbh=np.zeros_like(p["bh"])
    dhn = np.zeros((B,H_), dtype=np.float32)

    for t in reversed(range(T)):
        x=Xb[:,t,:]; hp=hp_arr[:,t,:]
        z,r,g = zs[:,t,:], rs[:,t,:], gs[:,t,:]
        dh = dh_ctx + dhn
        dg=dh*z; dz=dh*(g-hp); dhp=dh*(1-z)
        dag=dg*(1-g*g)
        dWh+=dag.T@x; dUh+=dag.T@(r*hp); dbh+=dag.sum(0)
        dhp+=(dag@p["Uh"])*r; dr=(dag@p["Uh"])*hp
        dar=dr*r*(1-r)
        dWr+=dar.T@x; dUr+=dar.T@hp; dbr+=dar.sum(0); dhp+=dar@p["Ur"]
        daz=dz*z*(1-z)
        dWz+=daz.T@x; dUz+=daz.T@hp; dbz+=daz.sum(0); dhp+=daz@p["Uz"]
        dhn = dhp

    grads.update({"Wz":dWz,"Uz":dUz,"bz":dbz,
                  "Wr":dWr,"Ur":dUr,"br":dbr,
                  "Wh":dWh,"Uh":dUh,"bh":dbh})

    opt["t"] += 1; b1,b2,eps_=0.9,0.999,1e-8; t_=opt["t"]
    for k, gv in grads.items():
        if k not in opt["m"]:
            opt["m"][k]=np.zeros_like(gv); opt["v"][k]=np.zeros_like(gv)
        opt["m"][k] = b1*opt["m"][k] + (1-b1)*gv
        opt["v"][k] = b2*opt["v"][k] + (1-b2)*(gv*gv)
        p[k] -= lr * opt["m"][k]/(1-b1**t_) / (np.sqrt(opt["v"][k]/(1-b2**t_)) + eps_)
    return loss

def train(labelled_traces: List[Tuple[np.ndarray, List[int]]],
          epochs: int = 60, seed: int = 0, hdim: int = 96,
          lr: float = 0.006, batch: int = 32,
          log_cb=None) -> Tuple[dict, List[float]]:
    Xs = [x for x,_ in labelled_traces]
    Ys = [make_multihot(li) for _,li in labelled_traces]
    X  = np.stack(Xs).astype(np.float32)
    Y  = np.stack(Ys).astype(np.float32)
    N  = X.shape[0]
    p  = init_rnn(seed, hdim)
    opt = {"t":0,"m":{},"v":{}}
    losses = []
    for ep in range(1, epochs+1):
        idx=np.random.permutation(N); ep_loss=0.0; nb=0
        for s in range(0, N, batch):
            b = idx[s:s+batch]
            if len(b) < 2: continue
            ep_loss += rnn_step_multilabel(p, X[b], Y[b], opt, lr); nb+=1
        avg = ep_loss / max(nb,1); losses.append(avg)
        if log_cb and (ep==1 or ep%max(1,epochs//8)==0):
            log_cb(f"epoch {ep:3d}/{epochs}  loss={avg:.4f}")
    return p, losses

def predict(p: dict, trace: List[Dict],
            threshold: float = 0.35) -> Dict[str, Any]:
    X    = featurise(trace)[None, :, :]
    ctx  = rnn_forward(X, p)
    logits = (ctx @ p["Wc"].T + p["bc"])[0]
    probs  = sigmoid(logits)
    detected = [ATK_LABELS[i] for i in range(N_CLASSES) if probs[i] >= threshold]
    if not detected:
        detected = [ATK_LABELS[int(np.argmax(probs))]]
    return {
        "detected":  detected,
        "probs":     {lbl: float(probs[i]) for i,lbl in enumerate(ATK_LABELS)},
        "threshold": threshold,
        "top_label": ATK_LABELS[int(np.argmax(probs))],
        "top_conf":  float(probs[int(np.argmax(probs))]),
    }

def serialise_weights(p: dict) -> bytes:
    import io, pickle
    buf = io.BytesIO(); pickle.dump({k: v.tolist() for k,v in p.items()}, buf)
    return buf.getvalue()

def deserialise_weights(data: bytes) -> dict:
    import io, pickle
    buf = io.BytesIO(data); raw = pickle.load(buf)
    return {k: np.array(v, dtype=np.float32) for k,v in raw.items()}

def weights_hash(p: dict) -> bytes:
    return H(serialise_weights(p))

# ── Weight Registry ───────────────────────────────────────────────────────────
@dataclass
class WeightRecord:
    version:            str
    weights_hash:       bytes
    weights_data:       bytes   # serialised
    anchor_fingerprint: str
    trained_on:         int     # n samples
    final_loss:         float
    created_at:         int
    session_key:        str
    losses:             List[float] = field(default_factory=list)

class WeightRegistry:
    """
    In-memory weight registry with hot/warm/cold tiers.
    hot_cache   → dict keyed by session_key, instant access
    archive     → list of all records (cold)
    """
    def __init__(self):
        self.hot_cache: Dict[str, WeightRecord] = {}
        self.archive:   List[WeightRecord]       = []
        self._version_counter = 0

    def store(self, session_key: str, p: dict, losses: List[float],
              anchor_fp: str, n_samples: int) -> WeightRecord:
        self._version_counter += 1
        rec = WeightRecord(
            version            = f"1.{self._version_counter}.0",
            weights_hash       = weights_hash(p),
            weights_data       = serialise_weights(p),
            anchor_fingerprint = anchor_fp,
            trained_on         = n_samples,
            final_loss         = losses[-1] if losses else 0.0,
            created_at         = int(time.time()),
            session_key        = session_key,
            losses             = losses,
        )
        self.hot_cache[session_key] = rec
        self.archive.append(rec)
        return rec

    def load(self, session_key: str) -> Optional[dict]:
        rec = self.hot_cache.get(session_key)
        if rec is None: return None
        return deserialise_weights(rec.weights_data)

    def verify(self, session_key: str, p: dict) -> bool:
        rec = self.hot_cache.get(session_key)
        if rec is None: return False
        import hmac as _hmac
        return _hmac.compare_digest(rec.weights_hash, weights_hash(p))

    def list_records(self) -> List[Dict]:
        return [
            {"version": r.version, "session_key": r.session_key,
             "anchor_fp": r.anchor_fingerprint[:12], "samples": r.trained_on,
             "loss": f"{r.final_loss:.4f}",
             "created": time.strftime("%H:%M:%S", time.localtime(r.created_at))}
            for r in reversed(self.archive)
        ]

# Global registry singleton (shared across Streamlit session)
_REGISTRY = WeightRegistry()

def get_registry() -> WeightRegistry:
    return _REGISTRY
