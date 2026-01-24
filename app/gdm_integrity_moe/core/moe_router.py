import numpy as np
from dataclasses import dataclass
from typing import Dict
from .utils import norm

EXPERT_ORDER = ["mechanism","risk_ethnicity","generic"]

def _softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

def hash_features(text: str, dim: int = 128) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for w in norm(text).split():
        v[hash(w) % dim] += 1.0
    n = np.linalg.norm(v) + 1e-9
    return v / n

@dataclass
class RouterOut:
    weights: Dict[str,float]
    conf: float
    logits: Dict[str,float]

class MoERouter:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.W = np.zeros((len(EXPERT_ORDER), dim), dtype=np.float32)
        self.b = np.zeros((len(EXPERT_ORDER),), dtype=np.float32)
        self.use_learned = False

    def load(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.W = d["router_W"].astype(np.float32)
        self.b = d["router_b"].astype(np.float32)
        self.use_learned = True

    def save(self, path: str):
        np.savez(path, router_W=self.W, router_b=self.b)

    def route(self, question: str, qtype: str) -> RouterOut:
        q = norm(question)
        base = np.zeros((len(EXPERT_ORDER),), dtype=np.float32)

        # rule priors
        if qtype == "mechanism_hormone_to_gdm":
            base[EXPERT_ORDER.index("mechanism")] += 2.0
        if qtype == "risk_ethnicity_t2d_link":
            base[EXPERT_ORDER.index("risk_ethnicity")] += 2.0

        if any(k in q for k in ["placental","hpl","progesterone","cortisol","growth hormone","estrogen","prolactin"]):
            base[EXPERT_ORDER.index("mechanism")] += 1.0
        if any(k in q for k in ["ethnicity","race","population","prevalence","background","independent risk"]):
            base[EXPERT_ORDER.index("risk_ethnicity")] += 1.0

        if self.use_learned:
            x = hash_features(q, self.dim)
            base = base + (self.W @ x + self.b)

        p = _softmax(base)
        conf = float(np.max(p))
        weights = {EXPERT_ORDER[i]: float(p[i]) for i in range(len(EXPERT_ORDER))}
        logits = {EXPERT_ORDER[i]: float(base[i]) for i in range(len(EXPERT_ORDER))}
        return RouterOut(weights=weights, conf=conf, logits=logits)
