import numpy as np

# =========================================
# CONFIG
# =========================================
np.random.seed(0)

VOCAB_SIZE        = 16     # discrete token bins
MS_DIM            = 8      # master secret dimension
HIDDEN_DIM        = 64     # GRU hidden size
ATTN_DIM          = 16     # attention hidden size
SEQ_LEN           = 20     # length of each OS chain
N_IDENTITIES      = 4      # how many different MS identities
N_WINDOWS_PER_ID  = 32     # chains per identity (positives)
LR                = 0.01
EPOCHS            = 400    # adjust as needed
CLIP_NORM         = 5.0
LAMBDA_MS         = 0.0    # weight of MS regression loss (auxiliary)

# =========================================
# 1. DATA GENERATION
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

    mean = zs.mean()
    std = zs.std() + 1e-6
    m = (zs - mean) / std  # (T,)

    # Map m ~ [-3,3] to tokens 0..VOCAB_SIZE-1
    scaled = (m + 3.0) / 6.0
    scaled = np.clip(scaled, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

def build_X(tokens, m, window_local_id, seq_len=SEQ_LEN):
    """
    Features per timestep:
      - one-hot token [VOCAB_SIZE]
      - measurement m[t]
      - normalized position t/(T-1)
      - normalized window_id (local 0..N_WINDOWS_PER_ID-1)
    """
    T = len(tokens)
    input_dim = VOCAB_SIZE + 4
    X = np.zeros((T, input_dim), dtype=np.float32)
    max_window = max(1, N_WINDOWS_PER_ID - 1)

    for t in range(T):
        X[t, tokens[t]]      = 1.0
        X[t, VOCAB_SIZE]     = m[t]
        X[t, VOCAB_SIZE + 1] = t / max(1, T - 1)
        # X[t, VOCAB_SIZE + 2] = window_local_id / max_window
        # New: id_idx normalized
        X[t, VOCAB_SIZE + 3] = window_local_id / max(1, N_IDENTITIES - 1)
    
    return X

def build_dataset_with_negatives():
    """
    For each identity i and window w:
      - Positive sample: legit chain (y_cls = 1, MS = MS_i)
      - Negative 1: shuffled time order
      - Negative 2: wrong window from other identity
      - Negative 3: truncated chain (half length)
    """
    X_all = []
    y_ms_all = []
    y_cls_all = []    # 1 for legit, 0 for tampered
    meta_all = []     # (identity, window, kind)

    total_windows = N_IDENTITIES * N_WINDOWS_PER_ID

    for id_idx in range(N_IDENTITIES):
        ms = MS_all[id_idx]
        for w in range(N_WINDOWS_PER_ID):
            global_wid = id_idx * N_WINDOWS_PER_ID + w

            # ---- Positive ----
            toks_pos, m_pos = generate_os_chain(ms, window_id=global_wid)
            X_pos = build_X(toks_pos, m_pos, window_local_id=w)
            X_all.append(X_pos)
            y_ms_all.append(ms)
            y_cls_all.append(1)
            meta_all.append((id_idx, w, "legit"))

            # ---- Neg 1: shuffled time ----
            idxs = np.arange(SEQ_LEN)
            np.random.shuffle(idxs)
            toks_shuf = toks_pos[idxs]
            m_shuf    = m_pos[idxs]
            X_shuf = build_X(toks_shuf, m_shuf, window_local_id=w)
            X_all.append(X_shuf)
            y_ms_all.append(ms)
            y_cls_all.append(0)
            meta_all.append((id_idx, w, "shuffled"))

            # ---- Neg 2: wrong window (different identity/window) ----
            # pick some other global window
            other_global = (global_wid + np.random.randint(1, total_windows)) % total_windows
            other_id = other_global // N_WINDOWS_PER_ID
            other_local_w = other_global % N_WINDOWS_PER_ID
            ms_other = MS_all[other_id]
            toks_wrong, m_wrong = generate_os_chain(ms_other, window_id=other_global)
            X_wrong = build_X(toks_wrong, m_wrong, window_local_id=other_local_w)
            X_all.append(X_wrong)
            y_ms_all.append(ms)     # we still associate with original MS_i; regression only used for positives
            y_cls_all.append(0)
            meta_all.append((id_idx, w, "wrong_window"))

            # ---- Neg 3: truncated chain ----
            trunc_len = SEQ_LEN // 2
            X_trunc = X_pos[:trunc_len]
            X_all.append(X_trunc)
            y_ms_all.append(ms)
            y_cls_all.append(0)
            meta_all.append((id_idx, w, "truncated"))

    return X_all, y_ms_all, y_cls_all, meta_all

X_all, y_ms_all, y_cls_all, meta_all = build_dataset_with_negatives()
print("Num samples:", len(X_all), " X[0].shape:", X_all[0].shape)

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

    # Behavior validity classifier head (scalar)
    params["W_beh"] = np.random.randn(1, hidden_dim) * 0.1  # (1,H)
    params["b_beh"] = np.zeros((1,), dtype=np.float32)

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

def beh_head_forward(context, params):
    W_beh, b_beh = params["W_beh"], params["b_beh"]  # (1,H), (1,)
    z = W_beh @ context + b_beh     # shape (1,)
    logit = np.squeeze(z)           # scalar
    prob = sigmoid(logit)
    return logit, prob

# =========================================
# 3. BACKPROP HELPERS
# =========================================

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

def bce_loss_and_grad(logit, y):
    """
    y in {0,1}, logit scalar, p = sigmoid(logit)
    return loss, dlogit
    """
    p = sigmoid(logit)
    eps = 1e-8
    loss = -(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    dlogit = p - y
    return loss, dlogit

# =========================================
# 4. TRAINING (ADAM)
# =========================================

def train_model(X_all, y_ms_all, y_cls_all, params, epochs=EPOCHS, lr=LR):
    num_samples = len(X_all)
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8

    for epoch in range(epochs):
        total_loss = 0.0
        grads_sum = init_grads(params)

        for i in range(num_samples):
            X = X_all[i]
            y_ms = y_ms_all[i]
            y_cls = y_cls_all[i]

            # forward
            hs, cache_gru = gru_forward(X, params)
            context, alphas, cache_att = attention_forward(hs, params)
            ms_hat = ms_head_forward(context, params)
            logit, prob = beh_head_forward(context, params)

            # classification loss
            loss_cls, dlogit = bce_loss_and_grad(logit, y_cls)

            # MS loss only for positives
            if y_cls == 1:
                diff_ms = ms_hat - y_ms
                loss_ms = 0.5 * np.mean(diff_ms**2)
                dms_hat = diff_ms / MS_DIM
            else:
                loss_ms = 0.0
                dms_hat = np.zeros_like(ms_hat)

            loss = loss_cls + LAMBDA_MS * loss_ms
            total_loss += loss

            # backprop heads
            dW_ms = np.outer(dms_hat, context)
            db_ms = dms_hat
            dcontext_ms = params["W_ms"].T @ dms_hat

            dW_beh = dlogit * context[None, :]    # (1,H)
            db_beh = np.array([dlogit], dtype=np.float32)
            w_beh_row = params["W_beh"][0]        # (H,)
            dcontext_beh = w_beh_row * dlogit     # (H,)

            dcontext = dcontext_ms + dcontext_beh

            # attention backprop
            dhs_att, dW_att, dv_att = attention_backward(dcontext, cache_att, params)

            # GRU backprop
            cache_gru["X"] = X
            grads_gru = gru_backward(dhs_att, cache_gru, params)

            grads_sum["W_ms"] += dW_ms
            grads_sum["b_ms"] += db_ms
            grads_sum["W_beh"] += dW_beh
            grads_sum["b_beh"] += db_beh
            grads_sum["W_att"] += dW_att
            grads_sum["v_att"] += dv_att

            for k in grads_gru:
                grads_sum[k] += grads_gru[k]

        # average
        for k in grads_sum:
            grads_sum[k] /= num_samples

        # clip
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_sum.values()))
        if total_norm > CLIP_NORM:
            scale = CLIP_NORM / (total_norm + 1e-8)
            for k in grads_sum:
                grads_sum[k] *= scale

        # Adam update
        t = epoch + 1
        for k in params:
            g = grads_sum[k]
            m[k] = beta1 * m[k] + (1 - beta1) * g
            v[k] = beta2 * v[k] + (1 - beta2) * (g * g)
            m_hat = m[k] / (1 - beta1**t)
            v_hat = v[k] / (1 - beta2**t)
            params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

        if (epoch + 1) % max(1, epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch+1}/{epochs}  avg_loss={total_loss/num_samples:.4f}")

    return params

input_dim = X_all[0].shape[1]
params = init_model(input_dim)
params = train_model(X_all, y_ms_all, y_cls_all, params, epochs=EPOCHS, lr=LR)

# =========================================
# 5. EVALUATION ON ONE WINDOW (PIM VIEW)
# =========================================

def l2(a, b):
    return float(np.linalg.norm(a - b))

# pick identity 0, window 5
id_eval = 0
w_eval  = 5
ms_eval = MS_all[id_eval]
global_eval_wid = id_eval * N_WINDOWS_PER_ID + w_eval

# Legit chain
toks_legit, m_legit = generate_os_chain(ms_eval, window_id=global_eval_wid)
X_legit = build_X(toks_legit, m_legit, window_local_id=w_eval)
hs_legit, _ = gru_forward(X_legit, params)
context_legit, _, _ = attention_forward(hs_legit, params)
ms_hat_legit = ms_head_forward(context_legit, params)
logit_legit, p_legit = beh_head_forward(context_legit, params)

print("\n=== LEGIT WINDOW VALIDATION ===")
print("Identity:", id_eval, "Window:", w_eval)
print("True MS:     ", np.round(ms_eval, 3))
print("Reconst MS:  ", np.round(ms_hat_legit, 3))
print("L2(MS_hat,MS):", l2(ms_hat_legit, ms_eval))
print("Behavior prob p_legit (want close to 1):", p_legit)

# A) Shuffled OS (time shuffle)
idxs = np.arange(SEQ_LEN)
np.random.shuffle(idxs)
toks_shuf = toks_legit[idxs]
m_shuf    = m_legit[idxs]
X_shuf = build_X(toks_shuf, m_shuf, window_local_id=w_eval)
hs_shuf, _ = gru_forward(X_shuf, params)
context_shuf, _, _ = attention_forward(hs_shuf, params)
ms_hat_shuf = ms_head_forward(context_shuf, params)
_, p_shuf   = beh_head_forward(context_shuf, params)

print("\n=== TAMPERED: SHUFFLED OS ===")
print("Reconst MS:          ", np.round(ms_hat_shuf, 3))
print("L2(MS_hat,MS):      ", l2(ms_hat_shuf, ms_eval))
print("Behavior prob p_shuf (want close to 0):", p_shuf)

# B) Wrong window (different identity / window)
other_global = (global_eval_wid + 7) % (N_IDENTITIES * N_WINDOWS_PER_ID)
other_id = other_global // N_WINDOWS_PER_ID
other_local_w = other_global % N_WINDOWS_PER_ID
ms_other = MS_all[other_id]
toks_wrong, m_wrong = generate_os_chain(ms_other, window_id=other_global)
X_wrong = build_X(toks_wrong, m_wrong, window_local_id=other_local_w)
hs_wrong, _ = gru_forward(X_wrong, params)
context_wrong, _, _ = attention_forward(hs_wrong, params)
ms_hat_wrong = ms_head_forward(context_wrong, params)
_, p_wrong   = beh_head_forward(context_wrong, params)

print("\n=== TAMPERED: WRONG WINDOW/IDENTITY ===")
print("Other identity:", other_id, "Other window:", other_local_w)
print("Reconst MS:          ", np.round(ms_hat_wrong, 3))
print("L2(MS_hat,MS_eval): ", l2(ms_hat_wrong, ms_eval))
print("Behavior prob p_wrong (want close to 0):", p_wrong)

# C) Truncated chain
trunc_len = SEQ_LEN // 2
X_trunc = X_legit[:trunc_len]
hs_trunc, _ = gru_forward(X_trunc, params)
context_trunc, _, _ = attention_forward(hs_trunc, params)
ms_hat_trunc = ms_head_forward(context_trunc, params)
_, p_trunc   = beh_head_forward(context_trunc, params)

print("\n=== TAMPERED: TRUNCATED CHAIN ===")
print("Used first", trunc_len, "steps.")
print("Reconst MS:          ", np.round(ms_hat_trunc, 3))
print("L2(MS_hat,MS):      ", l2(ms_hat_trunc, ms_eval))
print("Behavior prob p_trunc (want close to 0):", p_trunc)
