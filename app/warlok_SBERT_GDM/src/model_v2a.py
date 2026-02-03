from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
from pathlib import Path
import json
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def layer_norm(x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    xn = (x - mu) / np.sqrt(var + eps)
    return xn * g + b


def l2norm(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + eps)


LABELS = ["entailment", "neutral", "contradiction", "positive_qa"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}


@dataclass
class GdmSentenceTransformerV2A:
    """
    NumPy-only Sentence-Transformer-ish encoder:
      - word-level vocab
      - 4 Transformer-lite layers
      - dynamic head/global attention gates per layer
      - gated pooling
      - embedding head (192-dim) + relation head (4-way)

    Note: This file provides forward/inference + save/load.
    Training script can update selected params without full backprop through all transformer weights.
    """
    vocab: Dict[str, int]

    max_len: int = 384
    d_model: int = 160
    n_heads: int = 5
    d_ff: int = 384
    n_layers: int = 4
    emb_dim: int = 192
    n_classes: int = 4

    # Embeddings
    W_emb: np.ndarray = None  # (V,d)
    W_pos: np.ndarray = None  # (L,d)

    # Per-layer transformer weights
    Wq: List[np.ndarray] = None
    Wk: List[np.ndarray] = None
    Wv: List[np.ndarray] = None
    Wo: List[np.ndarray] = None

    # Gating (v2 signature)
    W_gate_h: List[np.ndarray] = None  # (d, heads)
    b_gate_h: List[np.ndarray] = None  # (heads,)
    W_gate_g: List[np.ndarray] = None  # (d,1)
    b_gate_g: List[np.ndarray] = None  # (1,)

    # FFN
    W_ff1: List[np.ndarray] = None  # (d, d_ff)
    b_ff1: List[np.ndarray] = None
    W_ff2: List[np.ndarray] = None  # (d_ff, d)
    b_ff2: List[np.ndarray] = None

    # LayerNorm params
    ln1_g: List[np.ndarray] = None
    ln1_b: List[np.ndarray] = None
    ln2_g: List[np.ndarray] = None
    ln2_b: List[np.ndarray] = None

    # Pool gate
    w_pool: np.ndarray = None  # (d,)
    b_pool: float = 0.0

    # Heads
    W_proj: np.ndarray = None  # (d, emb_dim)
    b_proj: np.ndarray = None
    W_cls: np.ndarray = None   # (d, C)
    b_cls: np.ndarray = None
    W_cls_gate: np.ndarray = None  # (d, C)
    b_cls_gate: np.ndarray = None

    @staticmethod
    def _rand(shape, scale=0.02, rng=None):
        if rng is None:
            rng = np.random.default_rng(7)
        return (rng.standard_normal(shape).astype(np.float32) * scale)

    @classmethod
    def init(cls, vocab: Dict[str, int], *, seed=7, **kw) -> "GdmSentenceTransformerV2A":
        rng = np.random.default_rng(seed)
        m = cls(vocab=vocab, **kw)
        V = len(vocab)

        m.W_emb = cls._rand((V, m.d_model), rng=rng)
        m.W_pos = cls._rand((m.max_len, m.d_model), rng=rng)

        m.Wq = []; m.Wk = []; m.Wv = []; m.Wo = []
        m.W_gate_h = []; m.b_gate_h = []; m.W_gate_g = []; m.b_gate_g = []
        m.W_ff1 = []; m.b_ff1 = []; m.W_ff2 = []; m.b_ff2 = []
        m.ln1_g = []; m.ln1_b = []; m.ln2_g = []; m.ln2_b = []

        for _ in range(m.n_layers):
            m.Wq.append(cls._rand((m.d_model, m.d_model), rng=rng))
            m.Wk.append(cls._rand((m.d_model, m.d_model), rng=rng))
            m.Wv.append(cls._rand((m.d_model, m.d_model), rng=rng))
            m.Wo.append(cls._rand((m.d_model, m.d_model), rng=rng))

            m.W_gate_h.append(cls._rand((m.d_model, m.n_heads), rng=rng))
            m.b_gate_h.append(np.zeros((m.n_heads,), dtype=np.float32))
            m.W_gate_g.append(cls._rand((m.d_model, 1), rng=rng))
            m.b_gate_g.append(np.zeros((1,), dtype=np.float32))

            m.W_ff1.append(cls._rand((m.d_model, m.d_ff), rng=rng))
            m.b_ff1.append(np.zeros((m.d_ff,), dtype=np.float32))
            m.W_ff2.append(cls._rand((m.d_ff, m.d_model), rng=rng))
            m.b_ff2.append(np.zeros((m.d_model,), dtype=np.float32))

            m.ln1_g.append(np.ones((m.d_model,), dtype=np.float32))
            m.ln1_b.append(np.zeros((m.d_model,), dtype=np.float32))
            m.ln2_g.append(np.ones((m.d_model,), dtype=np.float32))
            m.ln2_b.append(np.zeros((m.d_model,), dtype=np.float32))

        m.w_pool = cls._rand((m.d_model,), rng=rng).reshape(-1)
        m.b_pool = 0.0

        m.W_proj = cls._rand((m.d_model, m.emb_dim), rng=rng)
        m.b_proj = np.zeros((m.emb_dim,), dtype=np.float32)

        m.W_cls = cls._rand((m.d_model, m.n_classes), rng=rng)
        m.b_cls = np.zeros((m.n_classes,), dtype=np.float32)

        m.W_cls_gate = cls._rand((m.d_model, m.n_classes), rng=rng)
        m.b_cls_gate = np.zeros((m.n_classes,), dtype=np.float32)

        return m

    def tokenize(self, text: str) -> np.ndarray:
        toks = []
        for w in text.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%', '-'])
            if not w:
                continue
            toks.append(self.vocab.get(w, self.vocab.get("<unk>", 1)))
        if not toks:
            toks = [self.vocab.get("<unk>", 1)]
        toks = toks[: self.max_len]
        if len(toks) < 2:
            toks.append(self.vocab.get("<pad>", 0))
        return np.array(toks, dtype=np.int32)

    def encode(self, ids: np.ndarray) -> np.ndarray:
        T = ids.shape[0]
        return (self.W_emb[ids] + self.W_pos[:T]).astype(np.float32)

    def _mh_attn(self, x: np.ndarray, l: int) -> np.ndarray:
        # x: (T,d)
        T, d = x.shape
        h = self.n_heads
        dh = d // h  # 160//5=32

        Q = x @ self.Wq[l]
        K = x @ self.Wk[l]
        V = x @ self.Wv[l]

        Q = Q.reshape(T, h, dh).transpose(1, 0, 2)  # (h,T,dh)
        K = K.reshape(T, h, dh).transpose(1, 0, 2)
        V = V.reshape(T, h, dh).transpose(1, 0, 2)

        scores = (Q @ K.transpose(0, 2, 1)) / np.sqrt(dh)  # (h,T,T)
        A = softmax(scores, axis=-1)
        H = A @ V  # (h,T,dh)

        # dynamic gates from context
        ctx = x.mean(axis=0)  # (d,)
        g_head = sigmoid(ctx @ self.W_gate_h[l] + self.b_gate_h[l])  # (h,)
        g_glob = float(sigmoid(ctx @ self.W_gate_g[l] + self.b_gate_g[l]))  # scalar
        H = H * g_head.reshape(h, 1, 1) * g_glob

        out = H.transpose(1, 0, 2).reshape(T, d)
        return out @ self.Wo[l]

    def pool(self, x: np.ndarray) -> np.ndarray:
        # gated pooling across tokens
        alpha = sigmoid(x @ self.w_pool + self.b_pool).reshape(-1, 1)  # (T,1)
        num = (alpha * x).sum(axis=0)
        den = float(alpha.sum() + 1e-9)
        return (num / den).astype(np.float32)

    def forward(self, ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = self.encode(ids)
        for l in range(self.n_layers):
            xn = layer_norm(x, self.ln1_g[l], self.ln1_b[l])
            x = x + self._mh_attn(xn, l)
            xn2 = layer_norm(x, self.ln2_g[l], self.ln2_b[l])
            h = gelu(xn2 @ self.W_ff1[l] + self.b_ff1[l])
            x = x + (h @ self.W_ff2[l] + self.b_ff2[l])

        pooled = self.pool(x)  # (d,)
        emb = l2norm((pooled @ self.W_proj + self.b_proj).astype(np.float32))

        gate = sigmoid(pooled @ self.W_cls_gate + self.b_cls_gate)  # (C,)
        logits = (pooled @ self.W_cls + self.b_cls) * gate
        return emb, logits.astype(np.float32), pooled

    def embed(self, text: str) -> np.ndarray:
        ids = self.tokenize(text)
        emb, _, _ = self.forward(ids)
        return emb

    def predict_relation(self, q: str, a: str) -> Tuple[str, np.ndarray]:
        # Pair encoding: "q [SEP] a"
        pair = q.strip() + " [SEP] " + a.strip()
        ids = self.tokenize(pair)
        _, logits, _ = self.forward(ids)
        probs = softmax(logits.reshape(1, -1), axis=1).reshape(-1)
        lab = LABELS[int(np.argmax(probs))]
        return lab, probs

    def save(self, out_dir: Path, name: str = "gdm_sbert_v2A") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)

        # vocab
        with open(out_dir / f"{name}_vocab.txt", "w", encoding="utf-8") as f:
            for w, i in sorted(self.vocab.items(), key=lambda x: x[1]):
                f.write(w + "\n")

        cfg = {
            "max_len": self.max_len,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "n_layers": self.n_layers,
            "emb_dim": self.emb_dim,
            "n_classes": self.n_classes,
            "labels": LABELS,
        }
        (out_dir / f"{name}_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        np.savez_compressed(
            out_dir / f"{name}_weights.npz",
            W_emb=self.W_emb, W_pos=self.W_pos,
            Wq=np.array(self.Wq, dtype=object),
            Wk=np.array(self.Wk, dtype=object),
            Wv=np.array(self.Wv, dtype=object),
            Wo=np.array(self.Wo, dtype=object),
            W_gate_h=np.array(self.W_gate_h, dtype=object),
            b_gate_h=np.array(self.b_gate_h, dtype=object),
            W_gate_g=np.array(self.W_gate_g, dtype=object),
            b_gate_g=np.array(self.b_gate_g, dtype=object),
            W_ff1=np.array(self.W_ff1, dtype=object),
            b_ff1=np.array(self.b_ff1, dtype=object),
            W_ff2=np.array(self.W_ff2, dtype=object),
            b_ff2=np.array(self.b_ff2, dtype=object),
            ln1_g=np.array(self.ln1_g, dtype=object),
            ln1_b=np.array(self.ln1_b, dtype=object),
            ln2_g=np.array(self.ln2_g, dtype=object),
            ln2_b=np.array(self.ln2_b, dtype=object),
            w_pool=self.w_pool, b_pool=np.array([self.b_pool], dtype=np.float32),
            W_proj=self.W_proj, b_proj=self.b_proj,
            W_cls=self.W_cls, b_cls=self.b_cls,
            W_cls_gate=self.W_cls_gate, b_cls_gate=self.b_cls_gate
        )

    @classmethod
    def load(cls, models_dir: Path, name: str = "gdm_sbert_v2A") -> "GdmSentenceTransformerV2A":
        vocab = {}
        with open(models_dir / f"{name}_vocab.txt", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                w = line.strip()
                if w:
                    vocab[w] = i

        cfg = json.loads((models_dir / f"{name}_config.json").read_text(encoding="utf-8"))
        w = np.load(models_dir / f"{name}_weights.npz", allow_pickle=True)

        def _tolist(x): return list(x.tolist())

        m = cls(
            vocab=vocab,
            max_len=int(cfg["max_len"]),
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            d_ff=int(cfg["d_ff"]),
            n_layers=int(cfg["n_layers"]),
            emb_dim=int(cfg["emb_dim"]),
            n_classes=int(cfg["n_classes"]),
        )
        m.W_emb = w["W_emb"].astype(np.float32)
        m.W_pos = w["W_pos"].astype(np.float32)
        m.Wq = _tolist(w["Wq"]); m.Wk = _tolist(w["Wk"]); m.Wv = _tolist(w["Wv"]); m.Wo = _tolist(w["Wo"])
        m.W_gate_h = _tolist(w["W_gate_h"]); m.b_gate_h = _tolist(w["b_gate_h"])
        m.W_gate_g = _tolist(w["W_gate_g"]); m.b_gate_g = _tolist(w["b_gate_g"])
        m.W_ff1 = _tolist(w["W_ff1"]); m.b_ff1 = _tolist(w["b_ff1"])
        m.W_ff2 = _tolist(w["W_ff2"]); m.b_ff2 = _tolist(w["b_ff2"])
        m.ln1_g = _tolist(w["ln1_g"]); m.ln1_b = _tolist(w["ln1_b"])
        m.ln2_g = _tolist(w["ln2_g"]); m.ln2_b = _tolist(w["ln2_b"])
        m.w_pool = w["w_pool"].astype(np.float32).reshape(-1)
        m.b_pool = float(w["b_pool"][0])
        m.W_proj = w["W_proj"].astype(np.float32); m.b_proj = w["b_proj"].astype(np.float32)
        m.W_cls = w["W_cls"].astype(np.float32); m.b_cls = w["b_cls"].astype(np.float32)
        m.W_cls_gate = w["W_cls_gate"].astype(np.float32); m.b_cls_gate = w["b_cls_gate"].astype(np.float32)
        return m
