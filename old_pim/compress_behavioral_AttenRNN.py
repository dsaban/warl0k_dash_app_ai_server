import numpy as np
import hashlib

# -----------------------------------------------------
# 1. INPUT TEXT (can be long)
# -----------------------------------------------------

text = """
Warl0k nano-models do not need to reconstruct the text.
They only need to generate a deterministic compressed signature
that proves the sequence is authentic, consistent, and verified.
This matches the WARL0K principle: "Proof without storing secrets."
"""

# Raw size
raw_bytes = len(text.encode("utf-8"))

# Simple cryptographic hash for comparison
hash_hex = hashlib.sha256(text.encode("utf-8")).hexdigest()

# -----------------------------------------------------
# 2. VOCAB & ENCODING
# -----------------------------------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int32)
seq_len = len(indices)

# One-hot matrix (streamed)
X = np.zeros((seq_len, vocab_size), dtype=np.float32)
X[np.arange(seq_len), indices] = 1.0

# -----------------------------------------------------
# 3. TINY RNN + ATTENTION (WARL0K style)
# -----------------------------------------------------

np.random.seed(42)

hidden_size = 8      # tiny WARL0K state
attn_size   = 8

# RNN weights (fixed for all sessions/windows)
W_xh = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.05
W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.05
b_h  = np.zeros((hidden_size,), dtype=np.float32)

# Attention weights (fixed)
W_att = np.random.randn(hidden_size, attn_size).astype(np.float32) * 0.05
v_att = np.random.randn(attn_size).astype(np.float32) * 0.05


def init_hidden_from_seed(seed: int) -> np.ndarray:
    """
    Deterministic session-dependent initial hidden state.
    Same seed -> same h0, different seed -> different h0.
    """
    rng = np.random.RandomState(seed)
    h0 = rng.randn(hidden_size).astype(np.float32) * 0.1
    return np.tanh(h0)


def rnn_forward(X_window: np.ndarray, h0: np.ndarray) -> np.ndarray:
    """
    RNN over a window of tokens, starting from h0.
    X_window: (T, vocab_size)
    h0:       (hidden_size,)
    Returns:
      H: (T, hidden_size)
    """
    T, _ = X_window.shape
    H = np.zeros((T, hidden_size), dtype=np.float32)
    h = h0.copy()

    for t in range(T):
        h = np.tanh(X_window[t] @ W_xh + h @ W_hh + b_h)
        H[t] = h

    return H


def attention(H: np.ndarray):
    """
    Standard additive attention over time:
    H: (T, hidden_size)
    Returns:
      signature: (hidden_size,)
      alphas:    (T,)
    """
    scores = np.tanh(H @ W_att) @ v_att
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    alphas = exp_scores / np.sum(exp_scores)
    signature = (alphas[:, None] * H).sum(axis=0)
    return signature, alphas


def warlok_signature(session_seed: int, window_start: int, window_size: int):
    """
    Compute a WARL0K-style behavioral signature for:
      - given session seed
      - given window (counter range) over the SAME TEXT.
    """
    # clip window
    end = min(window_start + window_size, seq_len)
    X_window = X[window_start:end]          # (T, vocab_size)
    h0 = init_hidden_from_seed(session_seed)
    H = rnn_forward(X_window, h0)
    signature, alphas = attention(H)
    return signature


# -----------------------------------------------------
# 4. DEMONSTRATIONS
# -----------------------------------------------------

print("=== STATIC HASH (CONTENT ONLY) ===")
print("SHA-256(text):", hash_hex)
print()

# A. SAME text, DIFFERENT session seeds (different h0),
#    full sequence window
print("=== SAME TEXT, DIFFERENT SESSIONS (SEEDS) ===")
sig_s1 = warlok_signature(session_seed=1, window_start=0, window_size=seq_len)
sig_s2 = warlok_signature(session_seed=2, window_start=0, window_size=seq_len)
sig_s3 = warlok_signature(session_seed=3, window_start=0, window_size=seq_len)

def l2(a, b): return float(np.linalg.norm(a - b))

print("||sig(seed=1) - sig(seed=2)||:", l2(sig_s1, sig_s2))
print("||sig(seed=1) - sig(seed=3)||:", l2(sig_s1, sig_s3))
print("First 4 dims seed=1:", np.round(sig_s1[:4], 6))
print("First 4 dims seed=2:", np.round(sig_s2[:4], 6))
print("First 4 dims seed=3:", np.round(sig_s3[:4], 6))
print()

# B. SAME text, SAME session seed, DIFFERENT windows (counter offsets)
print("=== SAME TEXT & SESSION, DIFFERENT WINDOWS (COUNTERS) ===")
window_size = 59  # arbitrary window length

sig_w1 = warlok_signature(session_seed=1, window_start=0,          window_size=window_size)
sig_w2 = warlok_signature(session_seed=1, window_start=10,         window_size=window_size)
sig_w3 = warlok_signature(session_seed=1, window_start=20,         window_size=window_size)

print("||sig(win=0..59)  - sig(win=10..69)|| :", l2(sig_w1, sig_w2))
print("||sig(win=0..59)  - sig(win=20..79)|| :", l2(sig_w1, sig_w3))
print("First 4 dims win0:", np.round(sig_w1[:4], 6))
print("First 4 dims win10:", np.round(sig_w2[:4], 6))
print("First 4 dims win20:", np.round(sig_w3[:4], 6))
print()

# C. To show stability: SAME text, SAME seed, SAME window -> SAME signature
print("=== STABILITY CHECK (SAME EVERYTHING) ===")
sig_repeat = warlok_signature(session_seed=1, window_start=0, window_size=seq_len)
#  print result
print("||sig(seed=1) - sig_repeat||:", l2(sig_s1, sig_repeat))
