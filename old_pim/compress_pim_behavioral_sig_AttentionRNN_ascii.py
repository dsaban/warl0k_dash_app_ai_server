import numpy as np

# ============================================
# 1. GLOBAL CONFIG
# ============================================

MS_DIM     = 8          # master secret length (vector of 8 floats)
SEQ_LEN    = 32         # length of obfuscated secret (32 ASCII tokens)
HIDDEN_DIM = 32         # RNN hidden size
ATTN_DIM   = 16         # attention size

N_SAMPLES  = 1000        # training samples (each has its own MS + secret)
EPOCHS     = 100         # increase for better convergence
LR         = 0.01      # learning rate

np.random.seed(42)

# ============================================
# 2. ASCII VOCAB (PRINTABLE)
# ============================================

ASCII_CODES = list(range(32, 127))  # ' ' to '~'
VOCAB_SIZE  = len(ASCII_CODES)

idx_to_char = {i: chr(ASCII_CODES[i]) for i in range(VOCAB_SIZE)}
char_to_idx = {ch: i for i, ch in idx_to_char.items()}

# We use only the FIRST 16 printable chars as "code tokens"
CODE_TOKENS = [idx_to_char[i] for i in range(16)]  # 16 symbols for base-16 nibbles


def indices_to_string(idxs):
    return "".join(idx_to_char[int(i)] for i in idxs)


# ============================================
# 3. DETERMINISTIC MS -> 32-TOKEN SECRET
#    (ONE-HOT PHASE)
# ============================================

def encode_ms_to_tokens(ms_int):
    """
    ms_int: (MS_DIM,) ints in [0,255]
    We encode each 8-bit int into 2 nibbles (base-16).
    - First 16 tokens: 2 tokens per dimension = 16 tokens.
    - Next 16 tokens: checksum-based padding (same token repeated).
    Total = 32 tokens (fixed SEQ_LEN).
    """
    tokens = []
    for v in ms_int:
        hi = (v >> 4) & 0xF   # high nibble
        lo = v & 0xF          # low nibble
        tokens.append(CODE_TOKENS[hi])
        tokens.append(CODE_TOKENS[lo])

    # 16 tokens so far
    checksum = int(ms_int.sum() % 16)
    pad_token = CODE_TOKENS[checksum]

    # Add 16 extra copies of checksum token (behavioral tail)
    tokens.extend([pad_token] * 16)
    assert len(tokens) == SEQ_LEN

    idxs = np.array([char_to_idx[ch] for ch in tokens], dtype=np.int32)
    return idxs


def generate_master_and_secret():
    """
    Generate:
      - master secret as 8 floats in [0,1]
      - obfuscated secret as 32 ASCII token indices
    """
    ms_int = np.random.randint(0, 256, size=MS_DIM, dtype=np.int32)
    ms = ms_int.astype(np.float32) / 255.0  # normalized master secret
    idxs = encode_ms_to_tokens(ms_int)
    return ms, idxs


def one_hot(idxs):
    """
    idxs: (T,) int array of token indices
    returns X: (T, VOCAB_SIZE) one-hot
    """
    T = len(idxs)
    X = np.zeros((T, VOCAB_SIZE), dtype=np.float32)
    X[np.arange(T), idxs] = 1.0
    return X


# ============================================
# 4. RNN + ATTENTION REGENERATIVE MODEL
#    Input:  32 x ASCII one-hot sequence
#    Output: master secret vector (MS_DIM,)
# ============================================

# RNN parameters
W_xh = (np.random.randn(VOCAB_SIZE, HIDDEN_DIM) * 0.1).astype(np.float32)
W_hh = (np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1).astype(np.float32)
b_h  = np.zeros(HIDDEN_DIM, dtype=np.float32)

# Attention parameters
W_att = (np.random.randn(HIDDEN_DIM, ATTN_DIM) * 0.1).astype(np.float32)
v_att = (np.random.randn(ATTN_DIM) * 0.1).astype(np.float32)

# Output mapping (context -> MS)
W_out = (np.random.randn(HIDDEN_DIM, MS_DIM) * 0.1).astype(np.float32)
b_out = np.zeros(MS_DIM, dtype=np.float32)


def tanh(x):
    return np.tanh(x)


def dtanh(x):
    t = np.tanh(x)
    return 1.0 - t * t


def forward_rnn_attention(x_seq):
    """
    Forward pass: RNN + attention + linear head
    x_seq: (T, VOCAB_SIZE)
    Returns:
      y: (MS_DIM,)
      cache: intermediates for backprop
    """
    T = x_seq.shape[0]

    # ----- RNN -----
    h_list = []
    pre_h_list = []
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)

    for t in range(T):
        pre_h = x_seq[t] @ W_xh + h @ W_hh + b_h
        h = tanh(pre_h)
        pre_h_list.append(pre_h)
        h_list.append(h)

    H = np.stack(h_list, axis=0)  # (T, HIDDEN_DIM)

    # ----- Attention -----
    pre_u_list = []
    u_list = []
    scores = np.zeros(T, dtype=np.float32)

    for t in range(T):
        pre_u = H[t] @ W_att            # (ATTN_DIM,)
        u = tanh(pre_u)
        s = float(np.dot(u, v_att))     # scalar
        pre_u_list.append(pre_u)
        u_list.append(u)
        scores[t] = s

    # Softmax over time
    scores_shift = scores - np.max(scores)
    exp_scores = np.exp(scores_shift)
    alphas = exp_scores / np.sum(exp_scores)

    # Context vector
    context = np.sum(alphas[:, None] * H, axis=0)  # (HIDDEN_DIM,)

    # ----- Output -----
    y = context @ W_out + b_out  # (MS_DIM,)

    cache = (x_seq, h_list, pre_h_list, H, scores, alphas, pre_u_list, u_list, context)
    return y, cache


def backward_rnn_attention(cache, dy):
    """
    Backward pass: compute gradients of all parameters.
    dy: dL/dy, shape (MS_DIM,)
    """
    (x_seq, h_list, pre_h_list, H, scores, alphas, pre_u_list, u_list, context) = cache
    T = H.shape[0]

    # Init grads
    dW_out = np.zeros_like(W_out)
    db_out = np.zeros_like(b_out)
    dW_att = np.zeros_like(W_att)
    dv_att = np.zeros_like(v_att)
    dW_xh  = np.zeros_like(W_xh)
    dW_hh  = np.zeros_like(W_hh)
    db_h   = np.zeros_like(b_h)

    # ----- Output layer -----
    dW_out += np.outer(context, dy)
    db_out += dy
    dcontext = W_out @ dy         # (HIDDEN_DIM,)

    # ----- Context / attention -----
    dH = np.zeros_like(H)
    dalphas = np.zeros_like(alphas)

    # context = sum_t alphas_t * h_t
    for t in range(T):
        h_t = H[t]
        dH[t] += alphas[t] * dcontext
        dalphas[t] += np.dot(dcontext, h_t)

    # softmax backward
    dot_ga = float(np.dot(dalphas, alphas))
    dscores = alphas * (dalphas - dot_ga)

    for t in range(T):
        dscore_t = dscores[t]
        u_t = u_list[t]
        pre_u_t = pre_u_list[t]
        h_t = H[t]

        dv_att += u_t * dscore_t
        du_t = v_att * dscore_t
        dpre_u_t = dtanh(pre_u_t) * du_t

        dW_att += np.outer(h_t, dpre_u_t)
        dH[t]  += W_att @ dpre_u_t

    # ----- BPTT through RNN -----
    dh_next = np.zeros(HIDDEN_DIM, dtype=np.float32)

    for t in reversed(range(T)):
        pre_h_t = pre_h_list[t]
        x_t = x_seq[t]
        h_prev = np.zeros(HIDDEN_DIM, dtype=np.float32) if t == 0 else h_list[t - 1]

        dh = dH[t] + dh_next
        dpre_h = dtanh(pre_h_t) * dh

        dW_xh += np.outer(x_t, dpre_h)
        dW_hh += np.outer(h_prev, dpre_h)
        db_h  += dpre_h

        dh_next = W_hh @ dpre_h

    return dW_xh, dW_hh, db_h, dW_att, dv_att, dW_out, db_out


def mse_loss(pred, target):
    # 0.5 * mean squared error
    return 0.5 * np.mean((pred - target) ** 2)


# ============================================
# 5. TRAINING
# ============================================

def train_model():
    global W_xh, W_hh, b_h, W_att, v_att, W_out, b_out

    # Build dataset
    masters = []
    secrets = []
    for _ in range(N_SAMPLES):
        ms, idxs = generate_master_and_secret()
        masters.append(ms)
        secrets.append(one_hot(idxs))

    masters = np.stack(masters, axis=0)  # (N, MS_DIM)

    for epoch in range(EPOCHS):
        perm = np.random.permutation(N_SAMPLES)
        total_loss = 0.0

        for i in perm:
            X = secrets[i]      # (32, VOCAB_SIZE)
            ms = masters[i]     # (MS_DIM,)

            # Forward
            y, cache = forward_rnn_attention(X)
            loss = mse_loss(y, ms)
            total_loss += loss

            # dL/dy for 0.5 * mean((y-ms)^2)
            dy = (y - ms) / MS_DIM

            # Backward
            dW_xh, dW_hh, db_h, dW_att, dv_att, dW_out, db_out = backward_rnn_attention(cache, dy)

            # SGD step
            W_xh  -= LR * dW_xh
            W_hh  -= LR * dW_hh
            b_h   -= LR * db_h
            W_att -= LR * dW_att
            v_att -= LR * dv_att
            W_out -= LR * dW_out
            b_out -= LR * db_out

        avg_loss = total_loss / N_SAMPLES
        if (epoch + 1) % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch+1}/{EPOCHS}] avg loss = {avg_loss:.6f}")

    print("Training complete.\n")


# ============================================
# 6. DEMO: NEW MASTER + SECRET
# ============================================

def demo_reconstruction():
    ms_true, idxs = generate_master_and_secret()
    X = one_hot(idxs)
    y_pred, _ = forward_rnn_attention(X)

    secret_str = indices_to_string(idxs)
    l2_err = float(np.linalg.norm(y_pred - ms_true))

    print("=== DEMO: New master + 32-token obfuscated secret ===")
    print("Obfuscated secret (32 ASCII tokens):")
    print(secret_str)
    print("\nTrue master secret (normalized [0,1]):")
    print(np.round(ms_true, 4))
    print("\nReconstructed master secret:")
    print(np.round(y_pred, 4))
    print(f"\nL2 reconstruction error: {l2_err:.6f}")


# ============================================
# 7. RUN
# ============================================

if __name__ == "__main__":
    print("Training RNN+attention nano-model to map 32-token one-hot secrets -> master secrets...")
    train_model()
    demo_reconstruction()
