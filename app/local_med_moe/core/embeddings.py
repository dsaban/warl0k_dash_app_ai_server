# core/embeddings.py
import numpy as np
from .utils import tokenize

class HashingEmbedder:
    def __init__(self, dim=512):
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros((self.dim,), dtype=np.float32)
        toks = tokenize(text)
        if not toks:
            return v
        for t in toks:
            h = hash(t)
            idx = h % self.dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            v[idx] += sign
        # normalize
        n = np.linalg.norm(v)
        if n > 1e-9:
            v /= n
        return v

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-9))

class Projector:
    def __init__(self, out_dim: int, in_dim: int, seed: int = 5):
        rng = np.random.default_rng(seed)
        self.P = rng.normal(0, 0.12, size=(out_dim, in_dim)).astype(np.float32)

    def proj(self, x: np.ndarray) -> np.ndarray:
        return (self.P @ x).astype(np.float32)
