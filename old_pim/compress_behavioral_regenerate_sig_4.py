import numpy as np

# =======================================================
# CONFIG
# =======================================================

MS_BYTES    = 4           # 4 bytes master secret
SEQ_LEN     = 16          # 16 ASCII tokens OS
VOCAB       = [chr(65 + i) for i in range(60)]  # 'A'..'P'
shuffle = VOCAB.copy()
np.random.shuffle(shuffle)
VOCAB = shuffle
char_to_idx = {c: i for i, c in enumerate(VOCAB)}
VOCAB_SIZE  = len(VOCAB)

BEH_HIDDEN  = 16
BEH_ATTN    = 8
SIG_DIM     = 8

np.random.seed(42)

# =======================================================
# 1. OBFUSCATION OS <-> MS (STRUCTURE, NOT THE MODEL)
# =======================================================

def obfuscate_ms(ms_bytes, key):
    """
    ms_bytes: MS_BYTES integers in [0,255]
    key: integer in [0,15]
    Returns OS as 16 ASCII tokens (string).
    """
    tokens = []

    # First 2*MS_BYTES tokens: hi/lo nibble + key
    for v in ms_bytes:
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        hi_o = (hi + key) & 0xF
        lo_o = (lo + key) & 0xF
        tokens.append(VOCAB[hi_o])
        tokens.append(VOCAB[lo_o])

    # Remaining tokens: padded pattern (key+1) to reach SEQ_LEN
    while len(tokens) < SEQ_LEN:
        tokens.append(VOCAB[(key + 1) & 0xF])

    return ''.join(tokens)


def deobfuscate_os(os_str, key):
    """
    Structural inverse of obfuscation for sanity check.
    For training we *don't use this* — that's the point of the learned model.
    """
    tokens = os_str[:2 * MS_BYTES]
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
# 2. IDENTITY MODEL: LEARNED LINEAR SECRET RECONSTRUCTOR
#    (NO RNN HERE – PURE SPEED & EXACTNESS)
# =======================================================

class IdentityLinearPIM:
    """
    Learns an exact linear map:
        flatten(one-hot(OS)) -> MS (float vector)
    by constructing W and b analytically from a single sample.
    We then save W,b as the "nano-model" weights.
    """

    def __init__(self, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE, ms_bytes=MS_BYTES):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.ms_bytes = ms_bytes
        self.input_dim = seq_len * vocab_size  # pos + token joint one-hot
        self.ms_dim = ms_bytes

        # Will be set at "training" time
        self.W = np.zeros((self.ms_dim, self.input_dim), dtype=np.float32)
        self.b = np.zeros(self.ms_dim, dtype=np.float32)

    def _flatten_one_hot(self, os_str):
        """
        Encode OS as one-hot over (position, token).
        x: shape (input_dim,), with exactly seq_len ones.
        """
        x = np.zeros(self.input_dim, dtype=np.float32)
        for pos, ch in enumerate(os_str):
            j = char_to_idx[ch]  # 0..VOCAB_SIZE-1
            idx = pos * self.vocab_size + j
            x[idx] = 1.0
        return x

    def train_single(self, os_str, ms_bytes):
        """
        Given ONE OS and its MS, construct a W,b such that:
            forward(os_str) == ms_bytes (exact, after rounding to int).
        """
        x = self._flatten_one_hot(os_str)
        count = np.sum(x)  # should be seq_len

        # We'll make b = 0, and for *exactly* the active positions in x,
        # we set columns so that W @ x = ms.
        self.W[:] = 0.0
        self.b[:] = 0.0

        # Distribute the MS vector over columns that correspond to the os_str's tokens
        # so that sum over positions = ms_bytes.
        ms_vec = ms_bytes.astype(np.float32)  # shape (MS_BYTES,)
        per_col = ms_vec / count

        active_indices = np.where(x == 1.0)[0]
        for idx in active_indices:
            self.W[:, idx] = per_col  # every active column contributes equally

    def forward(self, os_str):
        x = self._flatten_one_hot(os_str)
        ms_hat = self.W @ x + self.b
        return ms_hat  # float vector of length MS_BYTES

    def reconstruct_bytes(self, os_str):
        ms_hat = self.forward(os_str)
        # Quantize to byte range [0,255]
        ms_int = np.clip(ms_hat, 0, 255).astype(np.int32)
        return ms_int


# =======================================================
# 3. BEHAVIORAL SIGNATURE MODEL (RNN + ATTENTION)
# =======================================================

# random but fixed weights (this *is* the model you would train more fully)
W_xh  = np.random.randn(VOCAB_SIZE, BEH_HIDDEN).astype(np.float32) * 0.05
W_hh  = np.random.randn(BEH_HIDDEN, BEH_HIDDEN).astype(np.float32) * 0.05
b_h   = np.zeros((BEH_HIDDEN,), dtype=np.float32)
W_att = np.random.randn(BEH_HIDDEN, BEH_ATTN).astype(np.float32) * 0.05
v_att = np.random.randn(BEH_ATTN).astype(np.float32) * 0.05

# For signature head: concat(context, seed_norm, counter_norm, window_norm)
W_sig = np.random.randn(SIG_DIM, BEH_HIDDEN + 3).astype(np.float32) * 0.1
b_sig = np.zeros(SIG_DIM, dtype=np.float32)


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
    h0 = rng.randn(BEH_HIDDEN).astype(np.float32) * 0.1
    return np.tanh(h0)


def rnn_forward(X_window, h0):
    T = X_window.shape[0]
    H = np.zeros((T, BEH_HIDDEN), dtype=np.float32)
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
    context = (alphas[:, None] * H).sum(axis=0)
    return context, alphas


def behavioral_signature(os_str, seed, counter, window_start, window_size):
    """
    Full signature:
      - one-hot OS
      - slice by window_start/size
      - RNN forward from h0(seed,counter)
      - attention over hidden states -> context
      - concat context + (seed_norm,counter_norm,window_norm)
      - linear -> tanh
    """
    X = os_to_onehot(os_str)
    T = X.shape[0]
    end = min(window_start + window_size, T)
    Xw = X[window_start:end]

    h0 = init_hidden_from_seed_and_counter(seed, counter)
    H  = rnn_forward(Xw, h0)
    context, _ = attention(H)

    p_seed    = (seed % 1000) / 1000.0
    p_counter = (counter % 100) / 100.0
    p_window  = (window_start % 100) / 100.0
    params_vec = np.array([p_seed, p_counter, p_window], dtype=np.float32)

    combined = np.concatenate([context, params_vec], axis=0)  # (BEH_HIDDEN + 3,)
    sig = np.tanh(W_sig @ combined + b_sig)                   # (SIG_DIM,)
    return sig


def l2(a, b):
    return float(np.linalg.norm(a - b))


# =======================================================
# 4. FULL DEMO
# =======================================================

if __name__ == "__main__":
    # ---------- Enrollment ----------
    ms_bytes_true = np.random.randint(0, 256, size=MS_BYTES, dtype=np.int32)
    key = np.random.randint(0, 16)

    os_str = obfuscate_ms(ms_bytes_true, key)

    print("=== ENROLLMENT ===")
    print("MS bytes:        ", ms_bytes_true)
    print("Obfuscation key: ", key)
    print("OS (16 ASCII):   ", os_str)

    # Identity model "training" (compute weights)
    id_model = IdentityLinearPIM()
    id_model.train_single(os_str, ms_bytes_true)

    # ---------- Identity Reconstruction ----------
    ms_bytes_recon = id_model.reconstruct_bytes(os_str)
    print("\n=== IDENTITY RECONSTRUCTION (MODEL) ===")
    print("Reconstructed MS bytes: ", ms_bytes_recon)
    print("Exact match? ", np.array_equal(ms_bytes_true, ms_bytes_recon))

    # For sanity, compare to structural inverse (not used by model)
    ms_struct = deobfuscate_os(os_str, key)
    print("Structural inverse bytes: ", ms_struct)
    print("Structural == true?      ", np.array_equal(ms_struct, ms_bytes_true))

    # ---------- Behavioral Baseline ----------
    seed    = 123
    counter = 7
    window_start = 4
    window_size  = 8

    sig_ref = behavioral_signature(os_str, seed, counter, window_start, window_size)

    print("\n=== BASELINE BEHAVIORAL SIGNATURE ===")
    print("sig_ref:", np.round(sig_ref, 4))

    def check_behavior(desc, os_current, seed_, counter_, w_start_, w_size_):
        sig = behavioral_signature(os_current, seed_, counter_, w_start_, w_size_)
        err = l2(sig, sig_ref)
        print(f"\n--- {desc} ---")
        print("sig:", np.round(sig, 4))
        print("L2 vs baseline:", err)

    # ---------- Deviations ----------
    # Wrong seed
    check_behavior("Wrong SEED", os_str, seed + 1, counter, window_start, window_size)

    # Wrong counter
    check_behavior("Wrong COUNTER", os_str, seed, counter + 5, window_start, window_size)

    # Wrong window (start shifted)
    check_behavior("Wrong WINDOW START (+2)", os_str, seed, counter, window_start + 2, window_size)

    # Wrong window size
    check_behavior("Wrong WINDOW SIZE (+2)", os_str, seed, counter, window_start, window_size + 2)

    # ---------- Tampered OS ----------
    tampered_list = list(os_str)
    idx0 = char_to_idx[tampered_list[0]]
    tampered_list[0] = VOCAB[(idx0 + 1) % len(VOCAB)]
    os_tampered = ''.join(tampered_list)

    ms_bytes_t = id_model.reconstruct_bytes(os_tampered)

    print("\n=== TAMPERED OS ===")
    print("Original OS:     ", os_str)
    print("Tampered OS:     ", os_tampered)
    print("MS from tampered:", ms_bytes_t)
    print("Still matches true MS? ", np.array_equal(ms_bytes_true, ms_bytes_t))

    check_behavior("Tampered OS (same seed/counter/window)",
                   os_tampered, seed, counter, window_start, window_size)
