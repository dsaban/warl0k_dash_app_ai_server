import numpy as np

# ============================================================
# WARL0K PIM TOY (NumPy-only, C++ portable forward pass)
# Goals:
#  1) OS-chain -> MS reconstruction (learnable, stable)
#  2) Behavior gate p_valid:
#       - legit chain => high
#       - shuffled / truncated / wrong-window / wrong-identity => low
#
# Key fixes vs previous:
#  - Window/seed is STRUCTURALLY bound into the OS generator (not just as a feature):
#       A_t becomes A_t(window, t)
#  - More capacity + Adam + mild regularization
#  - Hard negatives: force wrong-identity != true identity
#  - Optional identity feature (claimed id) can be included
# ============================================================

np.random.seed(0)

# ------------------------------
# CONFIG
# ------------------------------
VOCAB_SIZE        = 16
MS_DIM            = 8
SEQ_LEN           = 20

N_IDENTITIES      = 6
N_WINDOWS_PER_ID  = 48

HIDDEN_DIM        = 48
ATTN_DIM          = 24

EPOCHS            = 300
LR                = 0.01
CLIP_NORM         = 5.0

LAMBDA_MS         = 0.15      # MS recon weight (aux)
WEIGHT_DECAY      = 1e-4      # tiny L2 regularization on weights
POS_WEIGHT        = 3.0       # can increase if you want more confident p_legit

USE_ID_FEATURE    = True      # include claimed id feature

# ------------------------------
# Utility
# ------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def l2(a, b):
    return float(np.linalg.norm(a - b))

# ============================================================
# 1) DATA / OS GENERATOR
#   We bind window_id and t into measurement direction a_t(window,t):
#     z_t = (A_base[t] + delta(window,t)) @ MS + noise
#   This makes "wrong window" structurally different.
# ============================================================

MS_all = np.random.uniform(-1.0, 1.0, size=(N_IDENTITIES, MS_DIM)).astype(np.float32)
A_base = np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32) * 0.8

def window_delta(window_id, t, ms_dim=MS_DIM):
    # deterministic small delta, depends on (window_id, t)
    seed = (window_id * 10007 + t * 97) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return (0.15 * rng.randn(ms_dim)).astype(np.float32)

def generate_os_chain(ms_vec, window_id, seq_len=SEQ_LEN):
    # structured measurement that depends on window + time step
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_id, t)
        zs[t] = float(a_t @ ms_vec)

    # noise tied to window + ms (deterministic per window for reproducibility)
    noise_seed = (window_id * 1337 + int((ms_vec * 1000).sum())) & 0xFFFFFFFF
    rng = np.random.RandomState(noise_seed)
    zs = zs + rng.normal(scale=0.04, size=seq_len).astype(np.float32)

    # normalize per-chain
    m = (zs - zs.mean()) / (zs.std() + 1e-6)

    # quantize to token bins
    scaled = np.clip((m + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

def build_X(tokens, m, window_local_id, id_claimed, seq_len_ref=SEQ_LEN):
    """
    Features per timestep:
      - one-hot token [VOCAB_SIZE]
      - measurement m[t]
      - position t/(T-1)
      - window_local_id normalized (0..N_WINDOWS_PER_ID-1)
      - (optional) claimed identity normalized (0..N_IDENTITIES-1)
    """
    T = len(tokens)
    base_dim = VOCAB_SIZE + 3
    input_dim = base_dim + (1 if USE_ID_FEATURE else 0)

    X = np.zeros((T, input_dim), dtype=np.float32)
    max_window = max(1, N_WINDOWS_PER_ID - 1)
    max_id = max(1, N_IDENTITIES - 1)

    for t in range(T):
        X[t, tokens[t]] = 1.0
        X[t, VOCAB_SIZE] = m[t]
        X[t, VOCAB_SIZE + 1] = t / max(1, (T - 1))
        X[t, VOCAB_SIZE + 2] = window_local_id / max_window
        if USE_ID_FEATURE:
            X[t, VOCAB_SIZE + 3] = id_claimed / max_id

    return X

def build_dataset_with_negatives():
    """
    For each (id, window):
      Positive:
        - legit chain with correct claimed id
      Negatives:
        - shuffled (order broken)
        - truncated (missing motion)
        - wrong-window (same identity, different window)  [now structurally different]
        - wrong-identity (different identity, different chain) [forced !=]
    """
    X_all, y_ms_all, y_cls_all, meta_all = [], [], [], []

    for id_idx in range(N_IDENTITIES):
        ms = MS_all[id_idx]
        for w in range(N_WINDOWS_PER_ID):
            global_wid = id_idx * N_WINDOWS_PER_ID + w

            # ---- POS legit ----
            toks, m = generate_os_chain(ms, window_id=global_wid)
            X_pos = build_X(toks, m, window_local_id=w, id_claimed=id_idx)
            X_all.append(X_pos); y_ms_all.append(ms); y_cls_all.append(1)
            meta_all.append((id_idx, w, "legit"))

            # ---- NEG shuffled ----
            idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
            X_shuf = build_X(toks[idxs], m[idxs], window_local_id=w, id_claimed=id_idx)
            X_all.append(X_shuf); y_ms_all.append(ms); y_cls_all.append(0)
            meta_all.append((id_idx, w, "shuffled"))

            # ---- NEG truncated ----
            trunc_len = SEQ_LEN // 2
            X_trunc = X_pos[:trunc_len]
            X_all.append(X_trunc); y_ms_all.append(ms); y_cls_all.append(0)
            meta_all.append((id_idx, w, "truncated"))

            # ---- NEG wrong-window (same identity, different window) ----
            wrong_w = (w + 7) % N_WINDOWS_PER_ID
            wrong_global = id_idx * N_WINDOWS_PER_ID + wrong_w
            toks_w, m_w = generate_os_chain(ms, window_id=wrong_global)
            # claimed id stays id_idx (attacker replays a different window's chain)
            X_wrong_w = build_X(toks_w, m_w, window_local_id=wrong_w, id_claimed=id_idx)
            X_all.append(X_wrong_w); y_ms_all.append(ms); y_cls_all.append(0)
            meta_all.append((id_idx, wrong_w, "wrong_window_same_id"))

            # ---- NEG wrong-identity (forced different identity) ----
            other_id = (id_idx + np.random.randint(1, N_IDENTITIES)) % N_IDENTITIES
            other_w  = np.random.randint(0, N_WINDOWS_PER_ID)
            other_global = other_id * N_WINDOWS_PER_ID + other_w
            ms_other = MS_all[other_id]
            toks_o, m_o = generate_os_chain(ms_other, window_id=other_global)
            # attacker CLAIMS they are id_idx, but chain is from other_id
            X_wrong_id = build_X(toks_o, m_o, window_local_id=other_w, id_claimed=id_idx)
            X_all.append(X_wrong_id); y_ms_all.append(ms); y_cls_all.append(0)
            meta_all.append((other_id, other_w, "wrong_identity_claimed_as_true"))

    return X_all, y_ms_all, y_cls_all, meta_all

X_all, y_ms_all, y_cls_all, meta_all = build_dataset_with_negatives()
print("Num samples:", len(X_all), " X[0].shape:", X_all[0].shape)

# ============================================================
# 2) MODEL: GRU + Attention + (MS head) + (Behavior head)
# ============================================================

def init_model(input_dim, hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM, ms_dim=MS_DIM):
    p = {}
    # GRU
    p["W_z"] = np.random.randn(hidden_dim, input_dim) * 0.08
    p["U_z"] = np.random.randn(hidden_dim, hidden_dim) * 0.08
    p["b_z"] = np.zeros((hidden_dim,), dtype=np.float32)

    p["W_r"] = np.random.randn(hidden_dim, input_dim) * 0.08
    p["U_r"] = np.random.randn(hidden_dim, hidden_dim) * 0.08
    p["b_r"] = np.zeros((hidden_dim,), dtype=np.float32)

    p["W_h"] = np.random.randn(hidden_dim, input_dim) * 0.08
    p["U_h"] = np.random.randn(hidden_dim, hidden_dim) * 0.08
    p["b_h"] = np.zeros((hidden_dim,), dtype=np.float32)

    # Attention
    p["W_att"] = np.random.randn(attn_dim, hidden_dim) * 0.08
    p["v_att"] = np.random.randn(attn_dim) * 0.08

    # Heads
    p["W_ms"]  = np.random.randn(ms_dim, hidden_dim) * 0.08
    p["b_ms"]  = np.zeros((ms_dim,), dtype=np.float32)

    p["W_beh"] = np.random.randn(1, hidden_dim) * 0.08
    p["b_beh"] = np.zeros((1,), dtype=np.float32)
    return p

def gru_forward(X, p):
    W_z, U_z, b_z = p["W_z"], p["U_z"], p["b_z"]
    W_r, U_r, b_r = p["W_r"], p["U_r"], p["b_r"]
    W_h, U_h, b_h = p["W_h"], p["U_h"], p["b_h"]

    T, _ = X.shape
    H = W_z.shape[0]
    hs = np.zeros((T, H), dtype=np.float32)

    z_list, r_list, htil_list = [], [], []
    a_z_list, a_r_list, a_h_list = [], [], []

    h_prev = np.zeros((H,), dtype=np.float32)
    for t in range(T):
        x = X[t]
        a_z = W_z @ x + U_z @ h_prev + b_z
        a_r = W_r @ x + U_r @ h_prev + b_r
        z = sigmoid(a_z); r = sigmoid(a_r)

        a_h = W_h @ x + U_h @ (r * h_prev) + b_h
        htil = np.tanh(a_h)

        h = (1 - z) * h_prev + z * htil

        hs[t] = h
        z_list.append(z); r_list.append(r); htil_list.append(htil)
        a_z_list.append(a_z); a_r_list.append(a_r); a_h_list.append(a_h)
        h_prev = h

    cache = {
        "X": X, "hs": hs,
        "z": z_list, "r": r_list, "htil": htil_list,
        "a_z": a_z_list, "a_r": a_r_list, "a_h": a_h_list
    }
    return hs, cache

def attention_forward(hs, p):
    W_att, v_att = p["W_att"], p["v_att"]
    T, H = hs.shape

    u_list = []
    scores = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        a = W_att @ hs[t]
        u = np.tanh(a)
        scores[t] = v_att @ u
        u_list.append(u)

    alphas = softmax(scores)
    context = np.sum(alphas[:, None] * hs, axis=0)

    cache = {"hs": hs, "u": u_list, "scores": scores, "alphas": alphas}
    return context, alphas, cache

def ms_head_forward(context, p):
    return p["W_ms"] @ context + p["b_ms"]

def beh_head_forward(context, p):
    z = p["W_beh"] @ context + p["b_beh"]   # (1,)
    logit = np.squeeze(z)
    prob = sigmoid(logit)
    return logit, prob

# ============================================================
# 3) BACKPROP
# ============================================================

def init_grads(p):
    return {k: np.zeros_like(v) for k, v in p.items()}

def attention_backward(dcontext, cache, p):
    W_att, v_att = p["W_att"], p["v_att"]
    hs = cache["hs"]
    u_list = cache["u"]
    alphas = cache["alphas"]
    T, H = hs.shape

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
        ds = dscores[t]
        u = u_list[t]
        h = hs[t]

        dv_att += ds * u
        du = ds * v_att
        da = du * (1 - u**2)

        dW_att += da[:, None] @ h[None, :]
        dhs[t] += W_att.T @ da

    return dhs, dW_att, dv_att

def gru_backward(dhs_total, cache, p):
    W_z, U_z, b_z = p["W_z"], p["U_z"], p["b_z"]
    W_r, U_r, b_r = p["W_r"], p["U_r"], p["b_r"]
    W_h, U_h, b_h = p["W_h"], p["U_h"], p["b_h"]

    X = cache["X"]
    hs = cache["hs"]
    z_list = cache["z"]
    r_list = cache["r"]
    htil_list = cache["htil"]

    T, D = X.shape
    H = hs.shape[1]

    dW_z = np.zeros_like(W_z); dU_z = np.zeros_like(U_z); db_z = np.zeros_like(b_z)
    dW_r = np.zeros_like(W_r); dU_r = np.zeros_like(U_r); db_r = np.zeros_like(b_r)
    dW_h = np.zeros_like(W_h); dU_h = np.zeros_like(U_h); db_h = np.zeros_like(b_h)

    dh_next = np.zeros((H,), dtype=np.float32)

    for t in reversed(range(T)):
        x = X[t]
        h = hs[t]
        h_prev = np.zeros((H,), dtype=np.float32) if t == 0 else hs[t-1]
        z = z_list[t]; r = r_list[t]; htil = htil_list[t]

        dh = dh_next + dhs_total[t]

        dh_til = dh * z
        dz = dh * (htil - h_prev)
        dh_prev = dh * (1 - z)

        da_h = dh_til * (1 - htil**2)
        dW_h += da_h[:, None] @ x[None, :]
        dU_h += da_h[:, None] @ (r * h_prev)[None, :]
        db_h += da_h

        dh_prev += (U_h.T @ da_h) * r
        dr = (U_h.T @ da_h) * h_prev

        da_r = dr * r * (1 - r)
        dW_r += da_r[:, None] @ x[None, :]
        dU_r += da_r[:, None] @ h_prev[None, :]
        db_r += da_r
        dh_prev += U_r.T @ da_r

        da_z = dz * z * (1 - z)
        dW_z += da_z[:, None] @ x[None, :]
        dU_z += da_z[:, None] @ h_prev[None, :]
        db_z += da_z
        dh_prev += U_z.T @ da_z

        dh_next = dh_prev

    return {
        "W_z": dW_z, "U_z": dU_z, "b_z": db_z,
        "W_r": dW_r, "U_r": dU_r, "b_r": db_r,
        "W_h": dW_h, "U_h": dU_h, "b_h": db_h
    }

def bce_loss_and_grad(logit, y, pos_weight=1.0):
    # Weighted BCE for positives (optional, encourages higher p_legit confidence)
    p = sigmoid(logit)
    eps = 1e-8
    if y == 1:
        loss = -pos_weight * np.log(p + eps)
        dlogit = pos_weight * (p - 1.0)
    else:
        loss = -np.log(1.0 - p + eps)
        dlogit = p
    return loss, dlogit

# ============================================================
# 4) TRAIN (Adam + weight decay)
# ============================================================

def train(X_all, y_ms_all, y_cls_all, p, epochs=EPOCHS, lr=LR):
    n = len(X_all)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8

    for ep in range(1, epochs+1):
        grads_sum = init_grads(p)
        total = 0.0

        # shuffle samples each epoch
        idx = np.arange(n)
        np.random.shuffle(idx)

        for ii in idx:
            X = X_all[ii]
            y_ms = y_ms_all[ii]
            y_cls = y_cls_all[ii]

            hs, cache_gru = gru_forward(X, p)
            context, alphas, cache_att = attention_forward(hs, p)

            ms_hat = ms_head_forward(context, p)
            logit, prob = beh_head_forward(context, p)

            # cls loss
            loss_cls, dlogit = bce_loss_and_grad(logit, y_cls, pos_weight=POS_WEIGHT)

            # ms loss (only on positives)
            if y_cls == 1:
                diff = ms_hat - y_ms
                loss_ms = 0.5 * np.mean(diff**2)
                dms_hat = diff / MS_DIM
            else:
                loss_ms = 0.0
                dms_hat = np.zeros_like(ms_hat)

            # total loss
            loss = loss_cls + LAMBDA_MS * loss_ms
            total += loss

            # heads grads
            dW_ms = np.outer(dms_hat, context)
            db_ms = dms_hat
            dcontext_ms = p["W_ms"].T @ dms_hat

            dW_beh = dlogit * context[None, :]
            db_beh = np.array([dlogit], dtype=np.float32)
            dcontext_beh = p["W_beh"][0] * dlogit

            dcontext = dcontext_ms + dcontext_beh

            # backprop attention + gru
            dhs_att, dW_att, dv_att = attention_backward(dcontext, cache_att, p)
            cache_gru["X"] = X
            g_gru = gru_backward(dhs_att, cache_gru, p)

            # accumulate
            grads_sum["W_ms"] += dW_ms
            grads_sum["b_ms"] += db_ms
            grads_sum["W_beh"] += dW_beh
            grads_sum["b_beh"] += db_beh
            grads_sum["W_att"] += dW_att
            grads_sum["v_att"] += dv_att
            for k in g_gru:
                grads_sum[k] += g_gru[k]

        # average
        for k in grads_sum:
            grads_sum[k] /= n

        # weight decay (L2) on matrices (not biases)
        for k in p:
            if k.startswith("b_"):
                continue
            grads_sum[k] += WEIGHT_DECAY * p[k]

        # clip
        norm = np.sqrt(sum(np.sum(g*g) for g in grads_sum.values()))
        if norm > CLIP_NORM:
            s = CLIP_NORM / (norm + 1e-8)
            for k in grads_sum:
                grads_sum[k] *= s

        # Adam update
        for k in p:
            g = grads_sum[k]
            m[k] = b1*m[k] + (1-b1)*g
            v[k] = b2*v[k] + (1-b2)*(g*g)
            mhat = m[k] / (1 - b1**ep)
            vhat = v[k] / (1 - b2**ep)
            p[k] -= lr * mhat / (np.sqrt(vhat) + eps)

        if ep in (2,) or ep % max(1, epochs//10) == 0:
            print(f"Epoch {ep}/{epochs}  avg_loss={total/n:.4f}")

    return p

# ============================================================
# 5) EVAL: One window + tamper tests
# ============================================================

def run_chain(id_claimed, id_true, w_true, p):
    ms_true = MS_all[id_true]
    global_wid = id_true * N_WINDOWS_PER_ID + w_true
    toks, m = generate_os_chain(ms_true, window_id=global_wid)
    X = build_X(toks, m, window_local_id=w_true, id_claimed=id_claimed)
    hs, _ = gru_forward(X, p)
    context, _, _ = attention_forward(hs, p)
    ms_hat = ms_head_forward(context, p)
    _, prob = beh_head_forward(context, p)
    return X, toks, m, ms_true, ms_hat, prob

def eval_demo(p):
    id_eval = 0
    w_eval = 5

    # LEGIT: claimed id == true id, window correct
    X_legit, toks_legit, m_legit, ms_true, ms_hat, p_legit = run_chain(id_eval, id_eval, w_eval, p)

    print("\n=== LEGIT WINDOW VALIDATION ===")
    print("Claimed ID:", id_eval, "True ID:", id_eval, "Window:", w_eval)
    print("True MS:     ", np.round(ms_true, 3))
    print("Reconst MS:  ", np.round(ms_hat, 3))
    print("L2(MS_hat,MS):", l2(ms_hat, ms_true))
    print("p_legit:", float(p_legit))

    # SHUFFLED
    idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
    X_shuf = build_X(toks_legit[idxs], m_legit[idxs], window_local_id=w_eval, id_claimed=id_eval)
    hs, _ = gru_forward(X_shuf, p)
    ctx, _, _ = attention_forward(hs, p)
    ms_hat_shuf = ms_head_forward(ctx, p)
    _, p_shuf = beh_head_forward(ctx, p)

    print("\n=== TAMPERED: SHUFFLED ===")
    print("L2(MS_hat,MS):", l2(ms_hat_shuf, ms_true))
    print("p_shuf:", float(p_shuf))

    # TRUNCATED
    trunc_len = SEQ_LEN // 2
    X_trunc = X_legit[:trunc_len]
    hs, _ = gru_forward(X_trunc, p)
    ctx, _, _ = attention_forward(hs, p)
    ms_hat_trunc = ms_head_forward(ctx, p)
    _, p_trunc = beh_head_forward(ctx, p)

    print("\n=== TAMPERED: TRUNCATED ===")
    print("L2(MS_hat,MS):", l2(ms_hat_trunc, ms_true))
    print("p_trunc:", float(p_trunc))

    # WRONG WINDOW (same identity, different window)
    wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID
    X_w, toks_w, m_w, _, ms_hat_w, p_wrong_w = run_chain(id_eval, id_eval, wrong_w, p)

    print("\n=== TAMPERED: WRONG WINDOW (same id) ===")
    print("Wrong window:", wrong_w)
    print("L2(MS_hat,MS):", l2(ms_hat_w, ms_true))
    print("p_wrong_window:", float(p_wrong_w))

    # WRONG IDENTITY (forced different id), attacker still claims id_eval
    other_id = (id_eval + 1) % N_IDENTITIES
    X_o, toks_o, m_o, ms_other, ms_hat_o, p_wrong_id = run_chain(id_eval, other_id, np.random.randint(0, N_WINDOWS_PER_ID), p)

    print("\n=== TAMPERED: WRONG IDENTITY (claim id0, real id!=0) ===")
    print("Real other id:", other_id)
    print("L2(MS_hat,MS_eval):", l2(ms_hat_o, ms_true))
    print("p_wrong_id:", float(p_wrong_id))

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    input_dim = X_all[0].shape[1]
    p = init_model(input_dim)
    p = train(X_all, y_ms_all, y_cls_all, p, epochs=EPOCHS, lr=LR)
    eval_demo(p)
