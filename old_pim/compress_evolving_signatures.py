import numpy as np, hashlib

# -----------------------------------------------------
# 1. BASE TEXT & TAMPERED TEXT
# -----------------------------------------------------

text = """
Warl0k nano-models do not need to reconstruct the text.
They only need to generate a deterministic compressed signature
that proves the sequence is authentic, consistent, and verified.
This matches the WARL0K principle: "Proof without storing secrets."
"""

# One small modification in the text
tampered_text = text.replace("deterministic", "d3terministic", 1)

# Joint vocab so both texts use the same embedding space
chars = sorted(list(set(text + tampered_text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}

def text_to_X(t: str):
    idx = np.array([char_to_idx[ch] for ch in t], dtype=np.int32)
    X = np.zeros((len(idx), vocab_size), dtype=np.float32)
    X[np.arange(len(idx)), idx] = 1.0
    return X

X     = text_to_X(text)
X_tam = text_to_X(tampered_text)
seq_len = X.shape[0]

# -----------------------------------------------------
# 2. TINY RNN + ATTENTION (SHARED MODEL)
# -----------------------------------------------------

np.random.seed(42)
hidden_size = 8
attn_size   = 8

W_xh  = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.05
W_hh  = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.05
b_h   = np.zeros((hidden_size,), dtype=np.float32)
W_att = np.random.randn(hidden_size, attn_size).astype(np.float32) * 0.05
v_att = np.random.randn(attn_size).astype(np.float32) * 0.05

def init_hidden_from_seed(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    h0 = rng.randn(hidden_size).astype(np.float32) * 0.1
    return np.tanh(h0)

def rnn_forward(X_window: np.ndarray, h0: np.ndarray) -> np.ndarray:
    T, _ = X_window.shape
    H = np.zeros((T, hidden_size), dtype=np.float32)
    h = h0.copy()
    for t in range(T):
        h = np.tanh(X_window[t] @ W_xh + h @ W_hh + b_h)
        H[t] = h
    return H

def attention(H: np.ndarray):
    scores = np.tanh(H @ W_att) @ v_att
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    alphas = exp_scores / np.sum(exp_scores)
    signature = (alphas[:, None] * H).sum(axis=0)
    return signature, alphas

def warlok_signature(text_X: np.ndarray, session_seed: int, window_start: int, window_size: int):
    seq_len = text_X.shape[0]
    end = min(window_start + window_size, seq_len)
    X_window = text_X[window_start:end]
    h0 = init_hidden_from_seed(session_seed)
    H = rnn_forward(X_window, h0)
    sig, _ = attention(H)
    return sig

# -----------------------------------------------------
# 3. DEVICE & GATEWAY SIGNATURES OVER WINDOWS
# -----------------------------------------------------

def device_signatures(text_X, session_seed, window_size, step, start_offset=0):
    """
    Device-side sliding windows:
      windows: [start_offset, start_offset+step, ...]
      each window has 'window_size' tokens.
    """
    sigs = []
    starts = list(range(start_offset, text_X.shape[0] - window_size + 1, step))
    for s in starts:
        sig = warlok_signature(text_X, session_seed, s - start_offset, window_size)
        sigs.append(sig)
    return np.array(sigs), starts

def l2(a, b):
    return float(np.linalg.norm(a - b))

def gateway_verify(text_X, session_seed, window_size, step, received_sigs, starts):
    """
    Gateway recomputes expected signatures with its own view
    (start_offset = 0) and compares to received_sigs.
    """
    expected_sigs, _ = device_signatures(text_X, session_seed, window_size, step, start_offset=0)
    ok = True
    for i, s in enumerate(starts):
        d = l2(received_sigs[i], expected_sigs[i])
        if d > 1e-6:
            print(f"  MISMATCH at window {i} (start={s}), L2={d:.6f}")
            ok = False
    if ok:
        print("  All windows OK (device == gateway)")
    return ok

# -----------------------------------------------------
# 4. RUN SCENARIOS
# -----------------------------------------------------

session_seed = 123
window_size = 40   # tokens per verification window
step        = 10   # sliding step (counter stride)

print("=== BASELINE: honest device & gateway, same text ===")
dev_sigs, starts = device_signatures(X, session_seed, window_size, step)
print(f"  Number of windows: {len(starts)}")
gateway_verify(X, session_seed, window_size, step, dev_sigs, starts)

print("\nStatic SHA-256(original text):", hashlib.sha256(text.encode('utf-8')).hexdigest()[:32], "...")

# 1) TEXT TAMPERING
print("\n=== SCENARIO 1: Device text tampered (one word changed) ===")
dev_sigs_tam, starts_tam = device_signatures(X_tam, session_seed, window_size, step)
gateway_verify(X, session_seed, window_size, step, dev_sigs_tam, starts)
print("Static SHA-256(tampered text):", hashlib.sha256(tampered_text.encode('utf-8')).hexdigest()[:32], "...")

# 2) COUNTER / WINDOW DESYNC
print("\n=== SCENARIO 2: Counter / window desync (device offset by +5 tokens) ===")
dev_sigs_shifted, shifted_starts = device_signatures(X[5:], session_seed, window_size, step, start_offset=5)
print("  Gateway expects windows starting at:", starts)
print("  Device actually uses windows starting at:", shifted_starts)
gateway_verify(X, session_seed, window_size, step, dev_sigs_shifted, shifted_starts)

# 3) REPLAY
print("\n=== SCENARIO 3: Replay old signature for a later window ===")
replayed_sigs = dev_sigs.copy()
if len(replayed_sigs) >= 2:
    # Device lies: sends signature of first window as if it was the last
    replayed_sigs[-1] = dev_sigs[0]
gateway_verify(X, session_seed, window_size, step, replayed_sigs, starts)
