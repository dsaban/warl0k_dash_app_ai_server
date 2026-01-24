import numpy as np
import hashlib

# ============================================
# 1. GLOBAL CONFIG
# ============================================

MS_DIM      = 8   # master secret vector size
OS_DIM      = 8   # obfuscated secret size (= MS_DIM here)
HIDDEN_DIM  = 16  # nano-model hidden size
LR          = 0.001
EPOCHS      = 2000
N_SAMPLES   = 1000

np.random.seed(4)

# ============================================
# 2. MASTER SECRET & OBFUSCATION FUNCTION
# ============================================

# Fix a single "true" master secret (identity) for this demo
MASTER_SECRET = np.random.randn(MS_DIM).astype(np.float32)
print(f"True MASTER_SECRET: {MASTER_SECRET}\n")
def generate_os(ms, seed, counter, window_start):
    """
    Obfuscate the master secret into an OS (Obfuscated Secret)
    using seed + counter + window_start as a deterministic "behavioral" perturbation.
    This simulates an OS generated at the far peer during enrollment.
    """
    # Deterministic RNG based on params
    combined = seed * 1000003 + counter * 9176 + window_start * 13
    rng = np.random.RandomState(combined & 0xFFFFFFFF)

    noise = rng.randn(MS_DIM).astype(np.float32) * 0.05  # small noise
    # Simple param-based mixing (behavioral pattern)
    mix = np.array([
        np.sin(0.1 * counter),
        np.cos(0.1 * window_start),
        np.sin(0.1 * (seed % 100)),
        np.cos(0.05 * (seed % 200)),
        np.sin(0.2 * (counter + window_start)),
        np.cos(0.15 * counter),
        np.sin(0.07 * window_start),
        np.cos(0.13 * (counter - window_start))
    ], dtype=np.float32) * 0.05
	
    return ms + noise + mix  # OS is distorted MS with behavioral pattern


def encode_params(seed, counter, window_start, max_seed=1000, max_counter=20, max_window=128):
    """
    Encode (seed, counter, window_start) into a small param vector.
    This ties chain, seed, and window into the nano-model inputs.
    """
    return np.array([
        (seed % max_seed) / max_seed,
        counter / max_counter,
        window_start / max_window
    ], dtype=np.float32)


# ============================================
# 3. NANO-MODEL: OS + PARAMS -> MASTER SECRET
#    (Tiny MLP with manual backprop)
# ============================================

INPUT_DIM = OS_DIM + 3  # OS vector + (seed, counter, window) encoding
OUTPUT_DIM = MS_DIM

# Initialize weights
W1 = (np.random.randn(INPUT_DIM, HIDDEN_DIM) * 0.1).astype(np.float32)
b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
W2 = (np.random.randn(HIDDEN_DIM, OUTPUT_DIM) * 0.1).astype(np.float32)
b2 = np.zeros(OUTPUT_DIM, dtype=np.float32)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t * t

def forward_nano(x):
    """
    x: (batch, INPUT_DIM)
    returns:
      y: (batch, OUTPUT_DIM)
      cache: intermediates for backprop
    """
    z1 = x @ W1 + b1
    h  = tanh(z1)
    y  = h @ W2 + b2
    return y, (x, z1, h)

def backward_nano(cache, dy):
    """
    cache from forward_nano, dy: dL/dy (batch, OUTPUT_DIM)
    returns gradients dW1, db1, dW2, db2
    """
    global W1, W2
    x, z1, h = cache

    # Gradients for second layer
    dW2 = h.T @ dy                          # (HIDDEN_DIM, OUTPUT_DIM)
    db2 = dy.sum(axis=0)                    # (OUTPUT_DIM,)

    # Backprop to hidden
    dh  = dy @ W2.T                         # (batch, HIDDEN_DIM)
    dz1 = dh * dtanh(z1)                    # (batch, HIDDEN_DIM)

    # Gradients for first layer
    dW1 = x.T @ dz1                         # (INPUT_DIM, HIDDEN_DIM)
    db1 = dz1.sum(axis=0)                   # (HIDDEN_DIM,)

    return dW1, db1, dW2, db2

def mse_loss(pred, target):
    return np.mean((pred - target) ** 2)

def train_nano_model():
    global W1, b1, W2, b2

    # Generate training data: multiple (seed, counter, window) combinations for the SAME MS
    X_inputs = []
    Y_targets = []

    for _ in range(N_SAMPLES):
        seed         = np.random.randint(1, 500)
        counter      = np.random.randint(0, 16)
        window_start = np.random.randint(0, 64)

        os_vec   = generate_os(MASTER_SECRET, seed, counter, window_start)
        p_vec    = encode_params(seed, counter, window_start)
        inp      = np.concatenate([os_vec, p_vec], axis=0)

        X_inputs.append(inp)
        Y_targets.append(MASTER_SECRET)

    X = np.stack(X_inputs, axis=0)  # (N_SAMPLES, INPUT_DIM)
    Y = np.stack(Y_targets, axis=0) # (N_SAMPLES, OUTPUT_DIM)

    batch_size = 64
    n_batches  = N_SAMPLES // batch_size

    for epoch in range(EPOCHS):
        # Shuffle
        perm = np.random.permutation(N_SAMPLES)
        X = X[perm]
        Y = Y[perm]

        epoch_loss = 0.0

        for i in range(n_batches):
            xb = X[i * batch_size:(i + 1) * batch_size]
            yb = Y[i * batch_size:(i + 1) * batch_size]

            # Forward
            pred, cache = forward_nano(xb)
            loss = mse_loss(pred, yb)
            epoch_loss += loss

            # dL/dy
            dy = (pred - yb) / batch_size

            # Backward
            dW1, db1, dW2, db2 = backward_nano(cache, dy)

            # SGD step
            W1 -= LR * dW1
            b1 -= LR * db1
            W2 -= LR * dW2
            b2 -= LR * db2

        if (epoch + 1) % 100 == 0:
            print(f"[NanoModel] Epoch {epoch+1}/{EPOCHS}, avg loss = {epoch_loss / n_batches:.6f}")

    print("Nano-model training complete.\n")


def reconstruct_ms(os_vec, seed, counter, window_start):
    """
    Run the trained nano-model to reconstruct MS from
    OS + params. This is what peers do at validation time.
    """
    p_vec = encode_params(seed, counter, window_start)
    inp   = np.concatenate([os_vec, p_vec], axis=0)[None, :]
    pred, _ = forward_nano(inp)
    return pred[0]


# ============================================
# 4. BEHAVIORAL SIGNATURE OVER TEXT (RNN+ATTN)
#    Reuses the style of your evolution code
# ============================================

TEXT = """
Warl0k PIM nano-models map obfuscated secrets to master secrets.
They fold session seed, counter, and window into behavior so that
any tampering with the chain, timing, or content breaks validation.
"""

# Build vocab
chars = sorted(list(set(TEXT)))
vocab_size = len(chars)
c2i = {ch: i for i, ch in enumerate(chars)}

def text_to_onehot(t):
    idx = np.array([c2i[ch] for ch in t], dtype=np.int32)
    X = np.zeros((len(idx), vocab_size), dtype=np.float32)
    X[np.arange(len(idx)), idx] = 1.0
    return X

X_text   = text_to_onehot(TEXT)
SEQ_LEN  = X_text.shape[0]

hidden_size = 8
attn_size   = 8

W_xh = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.05
W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.05
b_h  = np.zeros((hidden_size,), dtype=np.float32)

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
    return signature

def behavioral_signature(seed, window_start, window_size):
    end = min(window_start + window_size, SEQ_LEN)
    Xw  = X_text[window_start:end]
    h0  = init_hidden_from_seed(seed)
    H   = rnn_forward(Xw, h0)
    sig = attention(H)
    return sig


def l2(a, b):
    return float(np.linalg.norm(a - b))


# ============================================
# 5. PIM VALIDATION SCENARIOS
# ============================================

def scenario_honest():
    print("=== SCENARIO 0: Honest peer (correct OS, seed, counter, window, text) ===")

    seed         = 123
    counter      = 5
    window_start = 10
    window_size  = 40

    # Far peer creates OS and PIM signature
    os_vec = generate_os(MASTER_SECRET, seed, counter, window_start)
    sig    = behavioral_signature(seed, window_start, window_size)

    # Verifier peer reconstructs MS and recomputes signature
    ms_hat     = reconstruct_ms(os_vec, seed, counter, window_start)
    print(f"Reconstructed MASTER_SECRET: {ms_hat}")
    sig_expect = behavioral_signature(seed, window_start, window_size)

    ms_err  = l2(ms_hat, MASTER_SECRET)
    sig_err = l2(sig, sig_expect)

    print(f"Master secret reconstruction error: {ms_err:.6f}")
    print(f"True MASTER_SECRET               : {MASTER_SECRET}")
    print(f"Behavioral signature error        : {sig_err:.6f}")

    ok_ms  = ms_err  < 0.05
    ok_sig = sig_err < 1e-6

    print(f"MS valid?  {ok_ms}")
    print(f"SIG valid? {ok_sig}\n")


def scenario_tampered_os():
    print("=== SCENARIO 1: Tampered OS (secret changed) ===")

    seed         = 123
    counter      = 5
    window_start = 10
    window_size  = 40

    # Honest OS
    os_vec = generate_os(MASTER_SECRET, seed, counter, window_start)

    # Attacker tampers OS
    tampered_os = os_vec.copy()
    tampered_os += np.random.randn(OS_DIM).astype(np.float32) * 0.2

    sig = behavioral_signature(seed, window_start, window_size)

    # Verifier tries to reconstruct
    ms_hat = reconstruct_ms(tampered_os, seed, counter, window_start)
    print(f"Reconstructed MASTER_SECRET (from tampered OS): {ms_hat}")
    sig_expect = behavioral_signature(seed, window_start, window_size)

    ms_err  = l2(ms_hat, MASTER_SECRET)
    sig_err = l2(sig, sig_expect)

    print(f"Master secret reconstruction error (tampered OS): {ms_err:.6f}")
    print(f"Behavioral signature error                    : {sig_err:.6f}")
    print("MS valid?  ", ms_err < 0.05)
    print("SIG valid? ", sig_err < 1e-6, "\n")


def scenario_wrong_counter():
    print("=== SCENARIO 2: Wrong counter (chain desync) ===")

    seed         = 123
    counter      = 5
    window_start = 10
    window_size  = 40

    os_vec = generate_os(MASTER_SECRET, seed, counter, window_start)
    sig    = behavioral_signature(seed, window_start, window_size)

    # Verifier believes counter is 6 (off by 1)
    wrong_counter = counter + 1

    ms_hat = reconstruct_ms(os_vec, seed, wrong_counter, window_start)
    print(f"Reconstructed MASTER_SECRET (wrong counter): {ms_hat}")
    sig_expect = behavioral_signature(seed, window_start, window_size)  # verifier still uses correct text window

    ms_err  = l2(ms_hat, MASTER_SECRET)
    sig_err = l2(sig, sig_expect)

    print(f"Master secret reconstruction error (wrong counter): {ms_err:.6f}")
    print(f"Behavioral signature error                      : {sig_err:.6f}")
    print("MS valid?  ", ms_err < 0.03)
    print("SIG valid? ", sig_err < 1e-6, "\n")


def scenario_wrong_seed():
    print("=== SCENARIO 3: Wrong seed (session mismatch) ===")

    seed         = 123
    counter      = 5
    window_start = 10
    window_size  = 40

    os_vec = generate_os(MASTER_SECRET, seed, counter, window_start)
    sig    = behavioral_signature(seed, window_start, window_size)

    wrong_seed = 999  # verifier uses different seed

    ms_hat = reconstruct_ms(os_vec, wrong_seed, counter, window_start)
    print(f"Reconstructed MASTER_SECRET (wrong seed): {ms_hat}")
    sig_expect = behavioral_signature(wrong_seed, window_start, window_size)

    ms_err  = l2(ms_hat, MASTER_SECRET)
    sig_err = l2(sig, sig_expect)

    print(f"Master secret reconstruction error (wrong seed): {ms_err:.6f}")
    print(f"Behavioral signature L2 difference            : {sig_err:.6f}")
    print("MS valid?  ", ms_err < 0.1)
    print("SIG valid? ", sig_err < 1e-6, "\n")


def scenario_text_tamper():
    print("=== SCENARIO 4: Text tampering (content changed) ===")

    seed         = 123
    counter      = 5
    window_start = 10
    window_size  = 40

    os_vec = generate_os(MASTER_SECRET, seed, counter, window_start)

    # Honest signature on original text
    sig_honest = behavioral_signature(seed, window_start, window_size)

    # Tamper the text by flipping one character
    tampered = list(TEXT)
    for i, ch in enumerate(tampered):
        if ch.isalpha():
            tampered[i] = 'X' if ch != 'X' else 'Y'
            break
    tampered_text = "".join(tampered)

    # Rebuild one-hot for tampered text locally (attacker changed payload)
    chars2 = sorted(list(set(tampered_text)))
    vocab2 = len(chars2)
    c2i2   = {ch: i for i, ch in enumerate(chars2)}

    idx2 = np.array([c2i2[ch] for ch in tampered_text], dtype=np.int32)
    X2   = np.zeros((len(idx2), vocab2), dtype=np.float32)
    X2[np.arange(len(idx2)), idx2] = 1.0

    def behavioral_signature_tampered(seed, window_start, window_size):
        end = min(window_start + window_size, len(idx2))
        Xw  = X2[window_start:end]
        h0  = init_hidden_from_seed(seed)
        H   = rnn_forward(Xw, h0)
        return attention(H)

    sig_tampered = behavioral_signature_tampered(seed, window_start, window_size)

    ms_hat = reconstruct_ms(os_vec, seed, counter, window_start)
    print(f"Reconstructed MASTER_SECRET (text tamper): {ms_hat}")

    ms_err  = l2(ms_hat, MASTER_SECRET)
    sig_err = l2(sig_honest, sig_tampered)

    print(f"Master secret reconstruction error (text tamper) : {ms_err:.6f}")
    print(f"Behavioral signature L2 (original vs tampered)   : {sig_err:.6f}")
    print("MS valid?  ", ms_err < 0.04)
    print("SIG valid? ", sig_err < 1e-6, "\n")


# ============================================
# 6. RUN DEMO
# ============================================

if __name__ == "__main__":
    print("Training nano-model to map OS + (seed,counter,window) -> MASTER_SECRET ...")
    train_nano_model()

    scenario_honest()
    scenario_tampered_os()
    scenario_wrong_counter()
    scenario_wrong_seed()
    scenario_text_tamper()
