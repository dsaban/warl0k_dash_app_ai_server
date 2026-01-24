# core/ebm.py
import numpy as np

class EnergyModel:
    """
    E(q,a,e) = - u · tanh(W(q+a+e)) - b
    Lower energy = better
    """
    def __init__(self, d: int, seed: int = 11):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.22, size=(d, d)).astype(np.float32)
        self.u = rng.normal(0, 0.22, size=(d,)).astype(np.float32)
        self.b = np.float32(0.0)

    def energy(self, qv: np.ndarray, av: np.ndarray, ev: np.ndarray) -> float:
        z = (qv + av + ev).astype(np.float32)
        h = np.tanh(self.W @ z)
        E = -float(np.dot(self.u, h)) - float(self.b)
        return float(E)
