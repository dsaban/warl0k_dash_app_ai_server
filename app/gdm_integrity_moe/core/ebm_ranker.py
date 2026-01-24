import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

FEATURES = [
    "avg_retrieval",
    "min_retrieval",
    "n_sent",
    "n_entities",
    "n_edges",
    "miss_entities",
    "miss_edges",
    "drift",
    "redundant",
]

def feat_vec(f: Dict[str,float]) -> np.ndarray:
    return np.array([float(f.get(k,0.0)) for k in FEATURES], dtype=np.float32)

@dataclass
class EBMOut:
    energy: float
    feat: Dict[str,float]

class EBMRanker:
    def __init__(self):
        self.w = np.zeros((len(FEATURES),), dtype=np.float32)
        self.b = np.float32(0.0)

        # sensible priors
        self.w[FEATURES.index("miss_entities")] = 2.5
        self.w[FEATURES.index("miss_edges")] = 2.5
        self.w[FEATURES.index("drift")] = 2.0
        self.w[FEATURES.index("redundant")] = 1.0
        self.w[FEATURES.index("avg_retrieval")] = -2.0
        self.w[FEATURES.index("min_retrieval")] = -1.0
        self.w[FEATURES.index("n_edges")] = -0.6
        self.w[FEATURES.index("n_entities")] = -0.3

    def load(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.w = d["ebm_w"].astype(np.float32)
        self.b = d["ebm_b"].astype(np.float32)

    def save(self, path: str):
        np.savez(path, ebm_w=self.w, ebm_b=self.b)

    def features(self, candidate, gate_eval: Dict[str,Any]) -> Dict[str,float]:
        scores = candidate.retrieval_scores or [0.0]
        flags = gate_eval.get("flags", []) if gate_eval else []

        miss_ent = sum(1 for f in flags if f.startswith("missing_entity") or f.startswith("missing_any_entity"))
        miss_edge = sum(1 for f in flags if f.startswith("missing_edge"))
        drift = 1.0 if "drift_terms_present" in flags else 0.0
        redundant = 1.0 if "redundant_claims" in flags else 0.0

        return {
            "avg_retrieval": float(sum(scores)/max(1,len(scores))),
            "min_retrieval": float(min(scores)),
            "n_sent": float(len(candidate.sentences)),
            "n_entities": float(len(candidate.used_entities)),
            "n_edges": float(len(candidate.used_edges)),
            "miss_entities": float(miss_ent),
            "miss_edges": float(miss_edge),
            "drift": float(drift),
            "redundant": float(redundant),
        }

    def energy(self, feat: Dict[str,float]) -> float:
        x = feat_vec(feat)
        return float(np.dot(self.w, x) + self.b)

    def score(self, candidate, gate_eval: Dict[str,Any]) -> EBMOut:
        f = self.features(candidate, gate_eval)
        return EBMOut(energy=self.energy(f), feat=f)
