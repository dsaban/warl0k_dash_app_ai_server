import numpy as np

# =======================================================
# CONFIG
# =======================================================

MS_BYTES  = 4           # 4 bytes master secret -> 8 nibbles
SEQ_LEN   = 16          # 16 ASCII tokens OS (8 real + 8 padded/rotated)
VOCAB     = [chr(65 + i) for i in range(16)]  # 'A'..'P' as nibble tokens
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)

np.random.seed(42)

# =======================================================
# 1. EXACT OS <-> MS MAPPING (NO TRAINING)
# =======================================================

def obfuscate_ms(ms_bytes, key):
    """
    ms_bytes: array of MS_BYTES integers in [0,255]
    key: integer in [0,15]
    Returns a 16-char OS string.
    """
    tokens = []

    # First 8 tokens: nibble(hi,lo) + key
    for v in ms_bytes:
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        hi_o = (hi + key) & 0xF
        lo_o = (lo + key) & 0xF
        tokens.append(VOCAB[hi_o])
        tokens.append(VOCAB[lo_o])

    # Second 8 tokens: same pattern but rotated with (key+1)
    for v in ms_bytes:
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        hi_o = (hi + key + 1) & 0xF
        lo_o = (lo + key + 1) & 0xF
        tokens.append(VOCAB[hi_o])
        tokens.append(VOCAB[lo_o])

    return ''.join(tokens)  # length 16


def deobfuscate_os(os_str, key):
    """
    Reverse of obfuscate_ms.
    Takes the FIRST 8 tokens (real ones) and inverts nibble+key.
    Returns ms_bytes (np.array of MS_BYTES, int32).
    """
    assert len(os_str) >= 2 * MS_BYTES, "OS too short for MS decode"
    tokens = os_str[:2 * MS_BYTES]  # first 8 tokens
    ms_bytes = []

    for i in range(0, len(tokens), 2):
        hi_o = char_to_idx[tokens[i]]
        lo_o = char_to_idx[tokens[i + 1]]
        hi = (hi_o - key) & 0xF
        lo = (lo_o - key) & 0xF
        v = (hi << 4) | lo
        ms_bytes.append(v)

    return np.array(ms_bytes, dtype=np.int32)


# =======================================================
# 2. BEHAVIORAL SIGNATURE (RNN + ATTENTION)
#    (seed, counter, window) + OS one-hot
# =======================================================

hidden_size = 8
attn_size   = 8

# Fixed tiny RNN + attention weights (no training; deterministic)
W_xh  = np.random.randn(VOCAB_SIZE, hidden_size).astype(np.float32) * 0.05
W_hh  = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.05
b_h   = np.zeros((hidden_size,), dtype=np.float32)
W_att = np.random.randn(hidden_size, attn_size).astype(np.float32) * 0.05
v_att = np.random.randn(attn_size).astype(np.float32) * 0.05


def os_to_onehot(os_str):
    idxs = np.array([char_to_idx[c] for c in os_str], dtype=np.int32)
    X = np.zeros((len(idxs), VOCAB_SIZE), dtype=np.float32)
    X[np.arange(len(idxs)), idxs] = 1.0
    return X


def init_hidden_from_seed_and_counter(seed, counter):
    """
    Deterministic init of h0 based on (seed, counter).
    """
    mix = seed * 10007 + counter * 7919
    rng = np.random.RandomState(mix & 0xFFFFFFFF)
    h0 = rng.randn(hidden_size).astype(np.float32) * 0.1
    return np.tanh(h0)


def rnn_forward(X_window, h0):
    T = X_window.shape[0]
    H = np.zeros((T, hidden_size), dtype=np.float32)
    h = h0.copy()
    for t in range(T):
        h = np.tanh(X_window[t] @ W_xh + h @ W_hh + b_h)
        H[t] = h
    return H


def attention(H):
    scores = np.tanh(H @ W_att) @ v_att
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    alphas = exp_scores / np.sum(exp_scores)
    sig = (alphas[:, None] * H).sum(axis=0)
    return sig, alphas


def behavioral_signature(os_str, seed, counter, window_start, window_size):
    """
    Compute WARL0K-style behavioral signature over a window of the OS.
    window_start, window_size act like (position, window) / counter.
    """
    X = os_to_onehot(os_str)
    T = X.shape[0]
    end = min(window_start + window_size, T)
    Xw = X[window_start:end]

    h0 = init_hidden_from_seed_and_counter(seed, counter)
    H  = rnn_forward(Xw, h0)
    sig, _ = attention(H)
    return sig


def l2(a, b):
    return float(np.linalg.norm(a - b))


# =======================================================
# 3. FULL PIM DEMO
# =======================================================

if __name__ == "__main__":
    # -------- Enrollment / setup --------
    # Generate master secret bytes
    ms_bytes_true = np.random.randint(0, 256, size=MS_BYTES, dtype=np.int32)
    key = np.random.randint(0, 16)  # nibble key

    os_str = obfuscate_ms(ms_bytes_true, key)

    print("=== MASTER SECRET & OBFUSCATED OS ===")
    print("MS bytes:        ", ms_bytes_true)
    print("Obfuscation key: ", key)
    print("OS (16 ASCII):   ", os_str)

    # Exact reconstruction (no training)
    ms_bytes_recon = deobfuscate_os(os_str, key)
    print("\n=== EXACT RECONSTRUCTION (OS -> MS) ===")
    print("Reconstructed MS bytes: ", ms_bytes_recon)
    print("Match? ", np.array_equal(ms_bytes_true, ms_bytes_recon))

    # -------- Behavioral baseline --------
    seed    = 123
    counter = 7
    window_start = 4   # where in OS we start
    window_size  = 8   # how many tokens in this verification window

    sig_ref = behavioral_signature(os_str, seed, counter, window_start, window_size)

    print("\n=== BASELINE BEHAVIORAL SIGNATURE ===")
    print("sig_ref:", np.round(sig_ref, 4))

    # Helper for deviation tests
    def test_behavior(desc, os_current, seed_, counter_, w_start_, w_size_):
        sig = behavioral_signature(os_current, seed_, counter_, w_start_, w_size_)
        err = l2(sig, sig_ref)
        print(f"\n--- {desc} ---")
        print("sig:", np.round(sig, 4))
        print("L2 vs baseline:", err)

    # 1) Wrong seed
    test_behavior("Wrong SEED", os_str, seed+1, counter, window_start, window_size)

    # 2) Wrong counter
    test_behavior("Wrong COUNTER", os_str, seed, counter+5, window_start, window_size)

    # 3) Wrong window (start shifted by +2)
    test_behavior("Wrong WINDOW START (+2)", os_str, seed, counter, window_start+2, window_size)

    # 4) Wrong window size
    test_behavior("Wrong WINDOW SIZE (+2)", os_str, seed, counter, window_start, window_size+2)

    # 5) Tampered OS (flip first char)
    tampered_list = list(os_str)
    idx0 = char_to_idx[tampered_list[0]]
    tampered_list[0] = VOCAB[(idx0 + 1) % len(VOCAB)]
    os_tampered = ''.join(tampered_list)

    ms_tampered = deobfuscate_os(os_tampered, key)
    print("\n=== TAMPERED OS ===")
    print("Original OS:   ", os_str)
    print("Tampered OS:   ", os_tampered)
    print("MS from tampered OS:", ms_tampered)
    print("Matches original MS? ", np.array_equal(ms_bytes_true, ms_tampered))

    test_behavior("Tampered OS (same seed/counter/window)", os_tampered, seed, counter, window_start, window_size)
