import numpy as np

# ============================================================
# CONFIG
# ============================================================
np.random.seed(0)

VOCAB_SIZE   = 16        # ASCII-ish tokens 'A'..'P'
MS_DIM       = 8         # master secret dimension
HIDDEN_DIM   = 16        # GRU hidden size
ATTN_DIM     = 8         # attention hidden size
SEQ_LEN      = 20        # length of OS chain per window
NUM_WINDOWS  = 128        # number of training windows
LR           = 0.01
EPOCHS       = 1000       # keep modest for demo
CLIP_NORM    = 5.0

# ============================================================
# 1. DATA GENERATION
#    - One MS (vector)
#    - Many windows, each is a chain of OS tokens mapping to same MS
# ============================================================

def generate_master_secret(ms_dim=MS_DIM):
    # Master secret as continuous vector [-1,1]
    return np.random.uniform(-1.0, 1.0, size=(ms_dim,)).astype(np.float32)

def ms_to_seed(ms_vec):
    # Simple deterministic int from MS for seeding RNG
    return int(np.round((ms_vec * 1000).sum())) & 0xFFFFFFFF

def generate_window_tokens(ms_vec, window_id, seq_len=SEQ_LEN):
    """
    Generate a sequence of token indices representing OS chain
    for a given MS and window_id.
    Behavior is deterministic from (MS, window_id, t).
    """
    base_seed = ms_to_seed(ms_vec) ^ (window_id * 9973)
    tokens = []
    for t in range(seq_len):
        # Derive a per-position seed
        local_seed = base_seed ^ (t * 7919)
        rng = np.random.RandomState(local_seed)
        token = rng.randint(0, VOCAB_SIZE)
        tokens.append(token)
    return np.array(tokens, dtype=np.int32)  # shape (T,)

def build_sample(ms_vec, window_id, seq_len=SEQ_LEN):
    """
    Build one training sample:
      X: (T, input_dim) with:
         - one-hot token (VOCAB_SIZE)
         - position (t / (T-1))
         - window_id normalized
      y: MS target (MS_DIM,)
    """
    tokens = generate_window_tokens(ms_vec, window_id, seq_len)
    T = len(tokens)
    X = np.zeros((T, VOCAB_SIZE + 2), dtype=np.float32)
    max_window = max(1, NUM_WINDOWS - 1)
    for t in range(T):
        # One-hot token
        X[t, tokens[t]] = 1.0
        # Position feature [0,1]
        X[t, VOCAB_SIZE + 0] = t / max(1, (T - 1))
        # Window feature [0,1]
        X[t, VOCAB_SIZE + 1] = window_id / max_window
    y = ms_vec.astype(np.float32)
    return X, y, tokens

def build_dataset(ms_vec, num_windows=NUM_WINDOWS):
    X_list, y_list, tokens_list = [], [], []
    for w in range(num_windows):
        X, y, toks = build_sample(ms_vec, w, SEQ_LEN)
        X_list.append(X)
        y_list.append(y)
        tokens_list.append(toks)
    return X_list, y_list, tokens_list

# ============================================================
# 2. GRU + ATTENTION MODEL
# ============================================================

def init_model(input_dim, hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM, ms_dim=MS_DIM):
    params = {}
    # GRU parameters: z, r, h_tilde
    params["W_z"] = np.random.randn(hidden_dim, input_dim) * 0.1
    params["U_z"] = np.random.randn(hidden_dim, hidden_dim) * 0.1
    params["b_z"] = np.zeros((hidden_dim,), dtype=np.float32)

    params["W_r"] = np.random.randn(hidden_dim, input_dim) * 0.1
    params["U_r"] = np.random.randn(hidden_dim, hidden_dim) * 0.1
    params["b_r"] = np.zeros((hidden_dim,), dtype=np.float32)

    params["W_h"] = np.random.randn(hidden_dim, input_dim) * 0.1
    params["U_h"] = np.random.randn(hidden_dim, hidden_dim) * 0.1
    params["b_h"] = np.zeros((hidden_dim,), dtype=np.float32)

    # Attention
    params["W_att"] = np.random.randn(attn_dim, hidden_dim) * 0.1
    params["v_att"] = np.random.randn(attn_dim) * 0.1

    # MS reconstruction head
    params["W_ms"] = np.random.randn(ms_dim, hidden_dim) * 0.1
    params["b_ms"] = np.zeros((ms_dim,), dtype=np.float32)

    return params

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def gru_forward(X, params):
    """
    X: (T, input_dim)
    Returns:
      hs: (T, hidden_dim)
      cache: dict with intermediates for backprop
    """
    W_z, U_z, b_z = params["W_z"], params["U_z"], params["b_z"]
    W_r, U_r, b_r = params["W_r"], params["U_r"], params["b_r"]
    W_h, U_h, b_h = params["W_h"], params["U_h"], params["b_h"]

    T, input_dim = X.shape
    hidden_dim = W_z.shape[0]
    hs = np.zeros((T, hidden_dim), dtype=np.float32)
    z_list, r_list, h_tilde_list = [], [], []
    a_z_list, a_r_list, a_h_list = [], [], []

    h_prev = np.zeros((hidden_dim,), dtype=np.float32)

    for t in range(T):
        x_t = X[t]

        a_z = W_z @ x_t + U_z @ h_prev + b_z
        a_r = W_r @ x_t + U_r @ h_prev + b_r
        z_t = sigmoid(a_z)
        r_t = sigmoid(a_r)

        a_h = W_h @ x_t + U_h @ (r_t * h_prev) + b_h
        h_tilde = np.tanh(a_h)

        h_t = (1 - z_t) * h_prev + z_t * h_tilde

        hs[t] = h_t
        z_list.append(z_t)
        r_list.append(r_t)
        h_tilde_list.append(h_tilde)
        a_z_list.append(a_z)
        a_r_list.append(a_r)
        a_h_list.append(a_h)

        h_prev = h_t

    cache = {
        "X": X,
        "hs": hs,
        "z": z_list,
        "r": r_list,
        "h_tilde": h_tilde_list,
        "a_z": a_z_list,
        "a_r": a_r_list,
        "a_h": a_h_list
    }
    return hs, cache

def attention_forward(hs, params):
    """
    hs: (T, hidden_dim)
    Returns:
      context: (hidden_dim,)
      alphas: (T,)
      cache: dict
    """
    W_att, v_att = params["W_att"], params["v_att"]
    T, hidden_dim = hs.shape
    attn_dim = W_att.shape[0]

    u_list = []
    scores = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        h_t = hs[t]
        a_t = W_att @ h_t            # (attn_dim,)
        u_t = np.tanh(a_t)           # (attn_dim,)
        s_t = v_att @ u_t            # scalar
        u_list.append(u_t)
        scores[t] = s_t

    alphas = softmax(scores)         # (T,)
    context = np.sum(alphas[:, None] * hs, axis=0)  # (hidden_dim,)

    cache = {
        "hs": hs,
        "u": u_list,
        "scores": scores,
        "alphas": alphas
    }
    return context, alphas, cache

def ms_head_forward(context, params):
    W_ms, b_ms = params["W_ms"], params["b_ms"]
    ms_hat = W_ms @ context + b_ms
    return ms_hat

# ============================================================
# 3. BACKPROP
# ============================================================

def init_grads(params):
    grads = {}
    for k, v in params.items():
        grads[k] = np.zeros_like(v)
    return grads

def attention_backward(dcontext, cache, params):
    """
    Inputs:
      dcontext: (hidden_dim,) gradient from MS head
    Returns:
      dhs: (T, hidden_dim) gradients wrt hs from attention
      dW_att, dv_att
    """
    W_att, v_att = params["W_att"], params["v_att"]
    hs = cache["hs"]
    u_list = cache["u"]
    alphas = cache["alphas"]
    T, hidden_dim = hs.shape
    attn_dim = W_att.shape[0]

    dW_att = np.zeros_like(W_att)
    dv_att = np.zeros_like(v_att)
    dhs = np.zeros_like(hs)

    # context = sum_t alpha_t * h_t
    # => d(alpha_t) from context: dcontext dot h_t
    d_alpha = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        d_alpha[t] = np.dot(dcontext, hs[t])
        dhs[t] += alphas[t] * dcontext  # from context contribution

    # alphas = softmax(scores)
    # Jacobian: dscore_t = alpha_t * (d_alpha[t] - sum_k d_alpha[k] * alpha_k)
    sum_dalpha_alpha = np.sum(d_alpha * alphas)
    dscores = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        dscores[t] = alphas[t] * (d_alpha[t] - sum_dalpha_alpha)

    # scores_t = v_att @ u_t, u_t = tanh(W_att @ h_t)
    for t in range(T):
        u_t = u_list[t]
        h_t = hs[t]
        ds_t = dscores[t]

        # d v_att
        dv_att += ds_t * u_t

        # d u_t
        du_t = ds_t * v_att              # (attn_dim,)

        # u_t = tanh(a_t), a_t = W_att h_t
        da_t = du_t * (1.0 - u_t**2)     # (attn_dim,)

        # dW_att
        dW_att += da_t[:, None] @ h_t[None, :]

        # d h_t from attention
        dhs[t] += W_att.T @ da_t

    return dhs, dW_att, dv_att

def gru_backward(dhs_total, cache, params):
    """
    dhs_total: (T, hidden_dim) gradients wrt hs from higher layers (attention etc.)
    Returns:
      grads for GRU params
    """
    W_z, U_z, b_z = params["W_z"], params["U_z"], params["b_z"]
    W_r, U_r, b_r = params["W_r"], params["U_r"], params["b_r"]
    W_h, U_h, b_h = params["W_h"], params["U_h"], params["b_h"]

    X = cache["X"]
    hs = cache["hs"]
    z_list = cache["z"]
    r_list = cache["r"]
    h_tilde_list = cache["h_tilde"]
    a_z_list = cache["a_z"]
    a_r_list = cache["a_r"]
    a_h_list = cache["a_h"]

    T, input_dim = X.shape
    hidden_dim = hs.shape[1]

    dW_z = np.zeros_like(W_z)
    dU_z = np.zeros_like(U_z)
    db_z = np.zeros_like(b_z)

    dW_r = np.zeros_like(W_r)
    dU_r = np.zeros_like(U_r)
    db_r = np.zeros_like(b_r)

    dW_h = np.zeros_like(W_h)
    dU_h = np.zeros_like(U_h)
    db_h = np.zeros_like(b_h)

    dh_next = np.zeros((hidden_dim,), dtype=np.float32)

    for t in reversed(range(T)):
        x_t = X[t]
        h_t = hs[t]
        h_prev = np.zeros((hidden_dim,), dtype=np.float32) if t == 0 else hs[t-1]

        z_t = z_list[t]
        r_t = r_list[t]
        h_tilde = h_tilde_list[t]
        a_z = a_z_list[t]
        a_r = a_r_list[t]
        a_h = a_h_list[t]

        # total dh from future + higher layers
        dh = dh_next + dhs_total[t]

        # h_t = (1 - z_t) * h_prev + z_t * h_tilde
        dh_tilde = dh * z_t
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z_t)

        # h_tilde = tanh(a_h)
        da_h = dh_tilde * (1 - h_tilde**2)

        dW_h += da_h[:, None] @ x_t[None, :]
        dU_h += da_h[:, None] @ (r_t * h_prev)[None, :]
        db_h += da_h

        # contribution to dh_prev via U_h and r_t
        dh_prev += (U_h.T @ da_h) * r_t
        dr = (U_h.T @ da_h) * h_prev

        # r_t = sigmoid(a_r)
        da_r = dr * r_t * (1 - r_t)
        dW_r += da_r[:, None] @ x_t[None, :]
        dU_r += da_r[:, None] @ h_prev[None, :]
        db_r += da_r

        dh_prev += U_r.T @ da_r

        # z_t = sigmoid(a_z)
        da_z = dz * z_t * (1 - z_t)
        dW_z += da_z[:, None] @ x_t[None, :]
        dU_z += da_z[:, None] @ h_prev[None, :]
        db_z += da_z

        dh_prev += U_z.T @ da_z

        dh_next = dh_prev

    grads = {
        "W_z": dW_z, "U_z": dU_z, "b_z": db_z,
        "W_r": dW_r, "U_r": dU_r, "b_r": db_r,
        "W_h": dW_h, "U_h": dU_h, "b_h": db_h
    }
    return grads

# ============================================================
# 4. TRAINING LOOP
# ============================================================

def train_model(X_list, y_list, params, epochs=EPOCHS, lr=LR):
    num_samples = len(X_list)
    for epoch in range(epochs):
        total_loss = 0.0
        grads_sum = init_grads(params)

        for i in range(num_samples):
            X = X_list[i]      # (T, input_dim)
            y = y_list[i]      # (MS_DIM,)

            # Forward
            hs, cache_gru = gru_forward(X, params)
            context, alphas, cache_att = attention_forward(hs, params)
            ms_hat = ms_head_forward(context, params)

            # Loss (MSE)
            diff = ms_hat - y
            loss = 0.5 * np.mean(diff**2)
            total_loss += loss

            # Backward
            dms_hat = diff / MS_DIM
            dW_ms = np.outer(dms_hat, context)
            db_ms = dms_hat
            dcontext = params["W_ms"].T @ dms_hat

            # Attention backward
            dhs_att, dW_att, dv_att = attention_backward(dcontext, cache_att, params)

            # GRU backward
            cache_gru["X"] = X  # ensure X is in cache
            dhs_total = dhs_att
            grads_gru = gru_backward(dhs_total, cache_gru, params)

            # Accumulate grads
            grads_sum["W_ms"] += dW_ms
            grads_sum["b_ms"] += db_ms
            grads_sum["W_att"] += dW_att
            grads_sum["v_att"] += dv_att

            for k in grads_gru:
                grads_sum[k] += grads_gru[k]

        # Average grads
        for k in grads_sum:
            grads_sum[k] /= num_samples

        # Clip gradients
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_sum.values()))
        if total_norm > CLIP_NORM:
            scale = CLIP_NORM / (total_norm + 1e-8)
            for k in grads_sum:
                grads_sum[k] *= scale

        # Update params
        for k in params:
            params[k] -= lr * grads_sum[k]

        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss/num_samples:.6f}")

    return params

# ============================================================
# 5. CHAIN VALIDATION / PIM CHECK
# ============================================================

def forward_chain(ms_vec, window_id, params):
    """
    Simulate validation of one window:
      - generate OS chain for (ms_vec, window_id)
      - run GRU + Attn
      - return ms_hat, context, alphas
    """
    X, y_true, toks = build_sample(ms_vec, window_id, SEQ_LEN)
    hs, _ = gru_forward(X, params)
    context, alphas, _ = attention_forward(hs, params)
    ms_hat = ms_head_forward(context, params)
    return ms_hat, context, alphas, X, y_true, toks

def l2(a, b):
    return float(np.linalg.norm(a - b))

# ============================================================
# 6. MAIN DEMO
# ============================================================

if __name__ == "__main__":
    # 1) Generate master secret for one peer
    MS = generate_master_secret()
    print("Master Secret (MS):", np.round(MS, 3))

    # 2) Build dataset: many windows mapping to same MS
    X_list, y_list, tokens_list = build_dataset(MS, NUM_WINDOWS)
    input_dim = X_list[0].shape[1]
    print("Input dim:", input_dim, "  Num windows:", len(X_list))

    # 3) Init and train model
    params = init_model(input_dim)
    params = train_model(X_list, y_list, params, epochs=EPOCHS, lr=LR)

    # 4) Pick one legit window and validate
    legit_window = 5
    ms_hat, context, alphas, X_legit, y_true, toks_legit = forward_chain(MS, legit_window, params)

    print("\n=== LEGIT WINDOW VALIDATION ===")
    print("Window ID:", legit_window)
    print("True MS:     ", np.round(y_true, 3))
    print("Reconst MS:  ", np.round(ms_hat, 3))
    print("L2(MS_hat, MS):", l2(ms_hat, y_true))

    # Use context as a behavioral signature for this window
    sig_ref = context.copy()
    print("Signature (ref) first 5 dims:", np.round(sig_ref[:5], 4))

    # 5) Tamper scenarios

    # (a) Shuffle tokens in the same window (behavior broken)
    toks_shuffled = toks_legit.copy()
    np.random.shuffle(toks_shuffled)

    # build a fake X from shuffled tokens but same positional + window features
    T = len(toks_shuffled)
    X_shuf = np.zeros_like(X_legit)
    max_window = max(1, NUM_WINDOWS - 1)
    for t in range(T):
        X_shuf[t, toks_shuffled[t]] = 1.0
        X_shuf[t, VOCAB_SIZE + 0] = t / max(1, (T - 1))
        X_shuf[t, VOCAB_SIZE + 1] = legit_window / max_window

    hs_shuf, _ = gru_forward(X_shuf, params)
    context_shuf, alphas_shuf, _ = attention_forward(hs_shuf, params)
    ms_hat_shuf = ms_head_forward(context_shuf, params)

    print("\n=== TAMPERED: SHUFFLED OS IN WINDOW ===")
    print("Reconst MS:        ", np.round(ms_hat_shuf, 3))
    print("L2(MS_hat, MS):    ", l2(ms_hat_shuf, y_true))
    print("L2(signature,ref): ", l2(context_shuf, sig_ref))

    # (b) Wrong window id (behavioral param drift)
    wrong_window = legit_window + 7
    ms_hat_wrong, context_wrong, alphas_wrong, _, _, _ = forward_chain(MS, wrong_window, params)

    print("\n=== TAMPERED: WRONG WINDOW ID ===")
    print("Window ID used:", wrong_window)
    print("Reconst MS:        ", np.round(ms_hat_wrong, 3))
    print("L2(MS_hat, MS):    ", l2(ms_hat_wrong, y_true))
    print("L2(signature,ref): ", l2(context_wrong, sig_ref))

    # (c) Truncated chain (early stop)
    trunc_len = SEQ_LEN // 2
    X_trunc = X_legit[:trunc_len]
    hs_trunc, _ = gru_forward(X_trunc, params)
    context_trunc, alphas_trunc, _ = attention_forward(hs_trunc, params)
    ms_hat_trunc = ms_head_forward(context_trunc, params)

    print("\n=== TAMPERED: TRUNCATED CHAIN ===")
    print("Used first", trunc_len, "OS only.")
    print("Reconstruct MS:        ", np.round(ms_hat_trunc, 3))
    print("L2(MS_hat, MS):    ", l2(ms_hat_trunc, y_true))
    print("L2(signature,ref): ", l2(context_trunc, sig_ref))
