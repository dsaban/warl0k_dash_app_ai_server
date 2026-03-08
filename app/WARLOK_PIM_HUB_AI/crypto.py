# warlok/crypto.py — Core cryptographic primitives
import hashlib, hmac, base64, secrets
import numpy as np
from typing import List, Tuple, Optional

# ── Hashing ───────────────────────────────────────────────────────────────────
def H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    out, t, c = b"", b"", 1
    while len(out) < length:
        t = hmac.new(key, t + info + bytes([c]), hashlib.sha256).digest()
        out += t; c += 1
    return out[:length]

def mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def b64e(b: bytes) -> bytes: return base64.urlsafe_b64encode(b)
def b64d(b: bytes) -> bytes: return base64.urlsafe_b64decode(b)
def bhex(b: bytes, n: int = 16) -> str: return b.hex()[:n]

def rand_bytes(n: int = 16) -> bytes: return secrets.token_bytes(n)
def rand_hex(n: int = 16) -> str:    return secrets.token_hex(n)

# ── Stream cipher (XOR + HKDF) ────────────────────────────────────────────────
def xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out, ctr, i = bytearray(), 0, 0
    while i < len(data):
        blk = hkdf(key, nonce + ctr.to_bytes(4, "big"), 32)
        for bb in blk:
            if i >= len(data): break
            out.append(data[i] ^ bb); i += 1
        ctr += 1
    return bytes(out)

# ── Activation helpers ────────────────────────────────────────────────────────
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def softmax1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x); e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def softmax2d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True); e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

# ── Merkle Tree ───────────────────────────────────────────────────────────────
def merkle_build(leaves: List[bytes]) -> Tuple[List[List[bytes]], bytes]:
    """
    Build a binary Merkle tree over `leaves`.
    Returns (levels, root) where levels[0] = leaf hashes.
    Odd-length levels duplicate the last element.
    """
    if not leaves:
        return [[H(b"EMPTY")]], H(b"EMPTY")
    level = [H(l) for l in leaves]
    levels = [level]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level = level + [level[-1]]
        level = [H(level[i] + level[i+1]) for i in range(0, len(level), 2)]
        levels.append(level)
    return levels, levels[-1][0]

def merkle_proof(levels: List[List[bytes]], leaf_idx: int) -> List[Tuple[str, bytes]]:
    """
    Return membership proof path for leaf at `leaf_idx`.
    Each element is ("left"|"right", sibling_hash).
    """
    path = []
    idx = leaf_idx
    for level in levels[:-1]:
        if len(level) % 2 == 1:
            level = level + [level[-1]]
        sibling_idx = idx ^ 1
        side = "right" if idx % 2 == 0 else "left"
        path.append((side, level[sibling_idx]))
        idx //= 2
    return path

def merkle_verify(leaf: bytes, proof: List[Tuple[str, bytes]], root: bytes) -> bool:
    """Verify a membership proof."""
    current = H(leaf)
    for side, sibling in proof:
        if side == "right":
            current = H(current + sibling)
        else:
            current = H(sibling + current)
    return current == root

# ── Running Accumulator ───────────────────────────────────────────────────────
class RunningAccumulator:
    """
    Hash-chained accumulator. Each update folds in a new leaf.
    acc_i = H(acc_{i-1} ∥ leaf_i ∥ counter_i.to_bytes)
    """
    def __init__(self, init_value: bytes):
        self.value = init_value
        self.count = 0

    def update(self, leaf: bytes) -> bytes:
        self.value = H(self.value + leaf + self.count.to_bytes(4, "big"))
        self.count += 1
        return self.value

    def divergence(self, expected: bytes) -> float:
        """0.0 = identical, 1.0 = completely different."""
        if self.value == expected:
            return 0.0
        diff = sum(a != b for a, b in zip(self.value, expected))
        return diff / len(self.value)
