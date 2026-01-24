import numpy as np

# =========================================
# CONFIG
# =========================================
np.random.seed(0)

VOCAB_SIZE        = 16     # discrete token bins
MS_DIM            = 16      # master secret dimension
HIDDEN_DIM        = 64     # GRU hidden size
ATTN_DIM          = 16     # attention hidden size
SEQ_LEN           = 20     # length of each OS chain
N_IDENTITIES      = 4      # how many different MS identities
N_WINDOWS_PER_ID  = 64     # chains per identity
LR                = 0.01
EPOCHS            = 300     # can increase if you want
CLIP_NORM         = 5.0

# =========================================
# 1. DATA GENERATION
#    OS_t = linear(MS) + noise → quantized token + measurement
# =========================================

def generate_master_secrets(n_ids=N_IDENTITIES, ms_dim=MS_DIM):
    return np.random.uniform(-1.0, 1.0, size=(n_ids, ms_dim)).astype(np.float32)

MS_all = generate_master_secrets()

# Fixed linear measurement matrices per time-step
A_t = np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32)  # (T, MS_DIM)

def generate_os_chain(ms_vec, window_id, seq_len=SEQ_LEN):
    """
    For a given MS and window, generate:
      - continuous measurements m[t] = normalized(A_t[t] @ MS + noise)
      - discrete tokens = quantized buckets of m[t]
    """
    zs = A_t @ ms_vec  # (T,)
    rng = np.random.RandomState(window_id * 1337 + int((ms_vec * 100).sum()))
    noise = rng.normal(scale=0.05, size=seq_len).astype(np.float32)
    zs = zs + noise

    # Normalize per-chain
    mean = zs.mean()
    std = zs.std() + 1e-6
    m = (zs - mean) / std  # (T,)

    # Map m approx in [-3,3] to tokens 0..VOCAB_SIZE-1
    scaled = (m + 3.0) / 6.0
    scaled = np.clip(scaled, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

def build_X(tokens, m, window_id, seq_len=SEQ_LEN):
    """
    Build input features per timestep:
      - one-hot token [VOCAB_SIZE]
      - continuous measurement m[t]
      - normalized position t/(T-1)
      - normalized window id
    """
    T = len(tokens)
    input_dim = VOCAB_SIZE + 3
    X = np.zeros((T, input_dim), dtype=np.float32)
    max_window = max(1, N_WINDOWS_PER_ID - 1)

    for t in range(T):
        X[t, tokens[t]]      = 1.0
        X[t, VOCAB_SIZE]     = m[t]
        X[t, VOCAB_SIZE + 1] = t / max(1, T - 1)
        X[t, VOCAB_SIZE + 2] = window_id / max_window

    return X

def build_dataset():
    X_list = []
    y_list = []
    for id_idx in range(N_IDENTITIES):
        ms = MS_all[id_idx]
        for w in range(N_WINDOWS_PER_ID):
            global_wid = id_idx * N_WINDOWS_PER_ID + w
            tokens, m = generate_os_chain(ms, window_id=global_wid)
            X = build_X(tokens, m, w)
            X_list.append(X)
            y_list.append(ms)
    return X_list, y_list

X_list, y_list = build_dataset()
print("Num samples:", len(X_list), "X shape:", X_list[0].shape)

# =========================================
# 2. GRU + ATTENTION MODEL
# =========================================

def init_model(input_dim, hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM, ms_dim=MS_DIM):
    params = {}
    # GRU
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
        z_list.append(z_t); r_list.append(r_t); h_tilde_list.append(h_tilde)
        a_z_list.append(a_z); a_r_list.append(a_r); a_h_list.append(a_h)
        h_prev = h_t

    cache = {
        "X": X,
        "hs": hs,
        "z": z_list,
        "r": r_list,
        "h_tilde": h_tilde_list,
        "a_z": a_z_list,
        "a_r": a_r_list,
        "a_h": a_h_list,
    }
    return hs, cache

def attention_forward(hs, params):
    W_att, v_att = params["W_att"], params["v_att"]
    T, hidden_dim = hs.shape
    u_list = []
    scores = np.zeros((T,), dtype=np.float32)

    for t in range(T):
        h_t = hs[t]
        a_t = W_att @ h_t
        u_t = np.tanh(a_t)
        s_t = v_att @ u_t
        u_list.append(u_t)
        scores[t] = s_t

    alphas = softmax(scores)
    context = np.sum(alphas[:, None] * hs, axis=0)

    cache = {
        "hs": hs,
        "u": u_list,
        "scores": scores,
        "alphas": alphas,
    }
    return context, alphas, cache

def ms_head_forward(context, params):
    W_ms, b_ms = params["W_ms"], params["b_ms"]
    return W_ms @ context + b_ms

def init_grads(params):
    return {k: np.zeros_like(v) for k, v in params.items()}

def attention_backward(dcontext, cache, params):
    W_att, v_att = params["W_att"], params["v_att"]
    hs = cache["hs"]
    u_list = cache["u"]
    alphas = cache["alphas"]

    T, hidden_dim = hs.shape
    dW_att = np.zeros_like(W_att)
    dv_att = np.zeros_like(v_att)
    dhs = np.zeros_like(hs)

    d_alpha = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        d_alpha[t] = np.dot(dcontext, hs[t])
        dhs[t] += alphas[t] * dcontext

    sum_dalpha_alpha = np.sum(d_alpha * alphas)
    dscores = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        dscores[t] = alphas[t] * (d_alpha[t] - sum_dalpha_alpha)

    for t in range(T):
        u_t = u_list[t]
        h_t = hs[t]
        ds_t = dscores[t]

        dv_att += ds_t * u_t

        du_t = ds_t * v_att
        da_t = du_t * (1.0 - u_t**2)

        dW_att += da_t[:, None] @ h_t[None, :]
        dhs[t] += W_att.T @ da_t

    return dhs, dW_att, dv_att

def gru_backward(dhs_total, cache, params):
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
        h_prev = np.zeros((hidden_dim,), dtype=np.float32) if t == 0 else hs[t - 1]

        z_t = z_list[t]
        r_t = r_list[t]
        h_tilde = h_tilde_list[t]

        dh = dh_next + dhs_total[t]

        dh_tilde = dh * z_t
        dz = dh * (h_tilde - h_prev)
        dh_prev = dh * (1 - z_t)

        da_h = dh_tilde * (1 - h_tilde**2)

        dW_h += da_h[:, None] @ x_t[None, :]
        dU_h += da_h[:, None] @ (r_t * h_prev)[None, :]
        db_h += da_h

        dh_prev += (U_h.T @ da_h) * r_t
        dr = (U_h.T @ da_h) * h_prev

        da_r = dr * r_t * (1 - r_t)
        dW_r += da_r[:, None] @ x_t[None, :]
        dU_r += da_r[:, None] @ h_prev[None, :]
        db_r += da_r

        dh_prev += U_r.T @ da_r

        da_z = dz * z_t * (1 - z_t)
        dW_z += da_z[:, None] @ x_t[None, :]
        dU_z += da_z[:, None] @ h_prev[None, :]
        db_z += da_z

        dh_prev += U_z.T @ da_z

        dh_next = dh_prev

    grads = {
        "W_z": dW_z, "U_z": dU_z, "b_z": db_z,
        "W_r": dW_r, "U_r": dU_r, "b_r": db_r,
        "W_h": dW_h, "U_h": dU_h, "b_h": db_h,
    }
    return grads

# =========================================
# 3. TRAINING WITH SIMPLE ADAM
# =========================================

def train_model(X_list, y_list, params, epochs=EPOCHS, lr=LR):
    num_samples = len(X_list)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        total_loss = 0.0
        grads_sum = init_grads(params)

        for i in range(num_samples):
            X = X_list[i]
            y = y_list[i]

            hs, cache_gru = gru_forward(X, params)
            context, alphas, cache_att = attention_forward(hs, params)
            ms_hat = ms_head_forward(context, params)

            diff = ms_hat - y
            loss = 0.5 * np.mean(diff**2)
            total_loss += loss

            dms_hat = diff / MS_DIM
            dW_ms = np.outer(dms_hat, context)
            db_ms = dms_hat
            dcontext = params["W_ms"].T @ dms_hat

            dhs_att, dW_att, dv_att = attention_backward(dcontext, cache_att, params)
            cache_gru["X"] = X
            grads_gru = gru_backward(dhs_att, cache_gru, params)

            grads_sum["W_ms"] += dW_ms
            grads_sum["b_ms"] += db_ms
            grads_sum["W_att"] += dW_att
            grads_sum["v_att"] += dv_att

            for k in grads_gru:
                grads_sum[k] += grads_gru[k]

        for k in grads_sum:
            grads_sum[k] /= num_samples

        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_sum.values()))
        if total_norm > CLIP_NORM:
            scale = CLIP_NORM / (total_norm + 1e-8)
            for k in grads_sum:
                grads_sum[k] *= scale

        t = epoch + 1
        for k in params:
            g = grads_sum[k]
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
            m_hat = m[k] / (1 - beta1**t)
            v_hat = v[k] / (1 - beta2**t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch+1}/{epochs}, loss={total_loss/num_samples:.6f}")

    return params

params = init_model(X_list[0].shape[1])
params = train_model(X_list, y_list, params, epochs=EPOCHS, lr=LR)

# =========================================
# 4. EVALUATION
# =========================================

def l2(a, b):
    return float(np.linalg.norm(a - b))

errs = []
for i in range(len(X_list)):
    X = X_list[i]; y = y_list[i]
    hs, _ = gru_forward(X, params)
    context, _, _ = attention_forward(hs, params)
    ms_hat = ms_head_forward(context, params)
    errs.append(l2(ms_hat, y))

print("\nReconstruction L2 stats across all chains:")
print("  min L2:", min(errs))
print("  max L2:", max(errs))
print("  avg L2:", sum(errs)/len(errs))
