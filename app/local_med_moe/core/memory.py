# core/memory.py
import numpy as np

class GatedRNNMemory:
    """
    Simple GRU-like memory + dot attention over recent states.
    Not trained by default; acts as a state compressor.
    """
    def __init__(self, d: int, max_steps: int = 180, seed: int = 21):
        rng = np.random.default_rng(seed)
        self.d = d
        self.max_steps = max_steps

        self.Wz = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)
        self.Uz = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)
        self.Wr = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)
        self.Ur = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)
        self.Wh = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)
        self.Uh = rng.normal(0, 0.25, size=(d, d)).astype(np.float32)

        self.h = np.zeros((d,), dtype=np.float32)
        self.hist = []

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def reset(self):
        self.h[:] = 0.0
        self.hist = []

    def step(self, x: np.ndarray):
        h = self.h
        z = self._sigmoid(self.Wz @ x + self.Uz @ h)
        r = self._sigmoid(self.Wr @ x + self.Ur @ h)
        h_tilde = np.tanh(self.Wh @ x + self.Uh @ (r * h))
        self.h = (1 - z) * h + z * h_tilde

        self.hist.append(self.h.copy())
        if len(self.hist) > self.max_steps:
            self.hist.pop(0)

    def attend(self, q: np.ndarray):
        if not self.hist:
            return {"context": np.zeros_like(q), "weights": []}
        H = np.stack(self.hist, axis=0)  # (T, d)
        # dot attention
        scores = H @ q
        scores = scores - np.max(scores)
        w = np.exp(scores)
        w = w / (np.sum(w) + 1e-9)
        ctx = (w[:, None] * H).sum(axis=0)
        return {"context": ctx.astype(np.float32), "weights": w.astype(np.float32)}
