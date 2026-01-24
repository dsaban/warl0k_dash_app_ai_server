# core/moe.py
import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-9)

class Expert:
    def __init__(self, d_in: int, hidden: int, seed: int):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.25, size=(hidden, d_in)).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.25, size=(d_in, hidden)).astype(np.float32)
        self.b2 = np.zeros((d_in,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ x + self.b1)
        y = np.tanh(self.W2 @ h + self.b2)
        return y.astype(np.float32)

class MoE:
    def __init__(self, d_in: int, n_experts: int, hidden: int = 64, seed: int = 7):
        self.d_in = d_in
        self.n_experts = n_experts
        rng = np.random.default_rng(seed)
        self.Wr = rng.normal(0, 0.15, size=(n_experts, d_in)).astype(np.float32)
        self.br = np.zeros((n_experts,), dtype=np.float32)
        self.experts = [Expert(d_in, hidden, seed=seed + 100 + i) for i in range(n_experts)]

    def forward(self, x: np.ndarray) -> dict:
        logits = (self.Wr @ x + self.br).astype(np.float32)
        w = softmax(logits)
        expert_vecs = [ex.forward(x) for ex in self.experts]
        # router confidence = max prob
        router_conf = float(np.max(w))
        return {"weights": w, "router_conf": router_conf, "expert_vecs": expert_vecs}
