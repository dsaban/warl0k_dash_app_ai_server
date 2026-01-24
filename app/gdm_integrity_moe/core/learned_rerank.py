import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from .utils import norm

RERANK_FEATURES = [
    "cos_q_claim",
    "has_independent_risk_phrase",
    "has_hormone_phrase",
]

def hashing_embed(text: str, dim: int = 512) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for w in norm(text).split():
        v[hash(w) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-9
    return v / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))

def _vec(f: Dict[str,float]) -> np.ndarray:
    return np.array([float(f.get(k,0.0)) for k in RERANK_FEATURES], dtype=np.float32)

@dataclass
class RerankModel:
    w: np.ndarray
    b: float
    dim: int = 512

    def score_pair(self, question: str, claim_support: str) -> float:
        qv = hashing_embed(question, self.dim)
        cv = hashing_embed(claim_support, self.dim)
        cosv = cosine(qv, cv)

        t = norm(claim_support)
        has_ind = 1.0 if ("independent risk factor" in t or "independently associated" in t) else 0.0
        has_horm = 1.0 if any(k in t for k in ["placental","hpl","progesterone","cortisol","growth hormone","estrogen","prolactin"]) else 0.0

        feat = {
            "cos_q_claim": cosv,
            "has_independent_risk_phrase": has_ind,
            "has_hormone_phrase": has_horm,
        }
        return float(np.dot(self.w, _vec(feat)) + self.b)

def load_reranker(path: str) -> RerankModel:
    d = np.load(path, allow_pickle=False)
    w = d["rerank_w"].astype(np.float32)
    b = float(d["rerank_b"])
    dim = int(d.get("rerank_dim", 512))
    return RerankModel(w=w, b=b, dim=dim)
