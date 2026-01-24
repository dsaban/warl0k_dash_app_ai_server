import numpy as np

# ============================================================
# FAST WARL0K PIM TRAINING (NumPy-only)
# - Batched GRU + masked attention
# - Two-phase training:
#     Phase 1: Train backbone + MS head on positives only (batched BPTT)
#     Phase 2: Freeze backbone, precompute contexts, train behavior head only
# - Correct "wrong window" = claim/expected window mismatch
# ============================================================

np.random.seed(0)

# ------------------------------
# CONFIG
# ------------------------------
VOCAB_SIZE         = 16
MS_DIM             = 8
SEQ_LEN            = 20

N_IDENTITIES       = 6
N_WINDOWS_PER_ID   = 48

HIDDEN_DIM         = 48
ATTN_DIM           = 24

BATCH_SIZE         = 64

EPOCHS_RECON       = 1200
EPOCHS_BEH         = 80
LR_RECON           = 0.006
LR_BEH             = 0.01
CLIP_NORM          = 5.0

POS_WEIGHT         = 3.0
WEIGHT_DECAY       = 1e-4

USE_ID_FEATURE      = True
USE_WINDOW_FEATURE  = True

SAMPLE_NEGATIVES_EACH_EPOCH = False  # keep False for stable comparisons
NEG_PER_POS = 4

# ------------------------------
# Helpers
# ------------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def softmax_masked(scores, mask):
    huge_neg = -1e9
    s = np.where(mask > 0, scores, huge_neg)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s) * mask
    denom = np.sum(e, axis=1, keepdims=True) + 1e-12
    return e / denom

def l2(a, b):
    return float(np.linalg.norm(a - b))

def clip_grads(grads, max_norm=CLIP_NORM):
    norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return grads

# ============================================================
# 1) OS GENERATOR (window-bound)
# ============================================================
MS_all = np.random.uniform(-1.0, 1.0, size=(N_IDENTITIES, MS_DIM)).astype(np.float32)
A_base = (np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32) * 0.8)

def window_delta(window_global_id, t, ms_dim=MS_DIM):
    seed = (window_global_id * 10007 + t * 97) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return (0.15 * rng.randn(ms_dim)).astype(np.float32)

def generate_os_chain(ms_vec, window_global_id, seq_len=SEQ_LEN):
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_global_id, t)
        zs[t] = float(a_t @ ms_vec)

    noise_seed = (window_global_id * 1337 + int((ms_vec * 1000).sum())) & 0xFFFFFFFF
    rng = np.random.RandomState(noise_seed)
    zs = zs + rng.normal(scale=0.04, size=seq_len).astype(np.float32)

    m = (zs - zs.mean()) / (zs.std() + 1e-6)
    scaled = np.clip((m + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

def build_X(tokens, m, expected_window_local, claimed_id):
    T = len(tokens)
    base_dim = VOCAB_SIZE + 2
    extra = (1 if USE_WINDOW_FEATURE else 0) + (1 if USE_ID_FEATURE else 0)
    D = base_dim + extra

    X = np.zeros((T, D), dtype=np.float32)
    max_w = max(1, N_WINDOWS_PER_ID - 1)
    max_id = max(1, N_IDENTITIES - 1)

    for t in range(T):
        X[t, tokens[t]] = 1.0
        X[t, VOCAB_SIZE] = m[t]
        X[t, VOCAB_SIZE + 1] = t / max(1, (T - 1))
        k = VOCAB_SIZE + 2
        if USE_WINDOW_FEATURE:
            X[t, k] = expected_window_local / max_w
            k += 1
        if USE_ID_FEATURE:
            X[t, k] = claimed_id / max_id

    return X

def pad_to_T(X, T=SEQ_LEN):
    D = X.shape[1]
    out = np.zeros((T, D), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    L = min(T, X.shape[0])
    out[:L] = X[:L]
    mask[:L] = 1.0
    return out, mask

# ============================================================
# 2) POSITIVES + NEGATIVE SAMPLER
# ============================================================
def build_positives():
    X_pos, M_pos, y_ms, ids, ws = [], [], [], [], []
    for id_true in range(N_IDENTITIES):
        ms_true = MS_all[id_true]
        for w in range(N_WINDOWS_PER_ID):
            g = id_true * N_WINDOWS_PER_ID + w
            toks, meas = generate_os_chain(ms_true, g)
            X = build_X(toks, meas, expected_window_local=w, claimed_id=id_true)
            Xp, mk = pad_to_T(X, SEQ_LEN)
            X_pos.append(Xp); M_pos.append(mk); y_ms.append(ms_true)
            ids.append(id_true); ws.append(w)
    return np.stack(X_pos), np.stack(M_pos), np.stack(y_ms), np.array(ids), np.array(ws)

X_POS, MASK_POS, Y_MS_POS, IDS_POS, WS_POS = build_positives()
N_POS, T, D = X_POS.shape
print("Positives:", N_POS, "X shape:", X_POS.shape)

def sample_negatives():
    Xs, Ms = [], []
    for i in range(N_POS):
        id_true = int(IDS_POS[i])
        w_true  = int(WS_POS[i])
        ms_true = MS_all[id_true]
        g_true  = id_true * N_WINDOWS_PER_ID + w_true

        toks, meas = generate_os_chain(ms_true, g_true)

        kinds = ["shuf", "trunc", "wrong_w", "wrong_id"]
        if NEG_PER_POS < 4:
            kinds = list(np.random.choice(kinds, size=NEG_PER_POS, replace=False))

        for kind in kinds:
            if kind == "shuf":
                idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
                X = build_X(toks[idxs], meas[idxs], expected_window_local=w_true, claimed_id=id_true)
                Xp, mk = pad_to_T(X)
                Xs.append(Xp); Ms.append(mk)

            elif kind == "trunc":
                L = SEQ_LEN // 2
                X = build_X(toks, meas, expected_window_local=w_true, claimed_id=id_true)[:L]
                Xp, mk = pad_to_T(X)
                Xs.append(Xp); Ms.append(mk)

            elif kind == "wrong_w":
                wrong_w = (w_true + 7) % N_WINDOWS_PER_ID
                g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w
                toks_w, meas_w = generate_os_chain(ms_true, g_wrong)
                # CLAIM/EXPECT remains w_true -> mismatch
                X = build_X(toks_w, meas_w, expected_window_local=w_true, claimed_id=id_true)
                Xp, mk = pad_to_T(X)
                Xs.append(Xp); Ms.append(mk)

            elif kind == "wrong_id":
                other_id = (id_true + np.random.randint(1, N_IDENTITIES)) % N_IDENTITIES
                other_w  = np.random.randint(0, N_WINDOWS_PER_ID)
                g_other  = other_id * N_WINDOWS_PER_ID + other_w
                ms_other = MS_all[other_id]
                toks_o, meas_o = generate_os_chain(ms_other, g_other)
                # attacker CLAIMS id_true + expects w_true
                X = build_X(toks_o, meas_o, expected_window_local=w_true, claimed_id=id_true)
                Xp, mk = pad_to_T(X)
                Xs.append(Xp); Ms.append(mk)

    X_neg = np.stack(Xs).astype(np.float32)
    M_neg = np.stack(Ms).astype(np.float32)
    y_neg = np.zeros((X_neg.shape[0],), dtype=np.float32)
    return X_neg, M_neg, y_neg

# ============================================================
# 3) MODEL (batched GRU + attention)
# ============================================================
def init_model(input_dim, hidden_dim=HIDDEN_DIM, attn_dim=ATTN_DIM, ms_dim=MS_DIM):
    p = {}
    s = 0.08
    p["W_z"] = np.random.randn(hidden_dim, input_dim) * s
    p["U_z"] = np.random.randn(hidden_dim, hidden_dim) * s
    p["b_z"] = np.zeros((hidden_dim,), dtype=np.float32)

    p["W_r"] = np.random.randn(hidden_dim, input_dim) * s
    p["U_r"] = np.random.randn(hidden_dim, hidden_dim) * s
    p["b_r"] = np.zeros((hidden_dim,), dtype=np.float32)

    p["W_h"] = np.random.randn(hidden_dim, input_dim) * s
    p["U_h"] = np.random.randn(hidden_dim, hidden_dim) * s
    p["b_h"] = np.zeros((hidden_dim,), dtype=np.float32)

    p["W_att"] = np.random.randn(attn_dim, hidden_dim) * s
    p["v_att"] = np.random.randn(attn_dim) * s

    p["W_ms"] = np.random.randn(ms_dim, hidden_dim) * s
    p["b_ms"] = np.zeros((ms_dim,), dtype=np.float32)

    p["W_beh"] = np.random.randn(1, hidden_dim) * s
    p["b_beh"] = np.zeros((1,), dtype=np.float32)
    return p

def gru_forward_batch(Xb, Mb, p):
    W_z, U_z, b_z = p["W_z"], p["U_z"], p["b_z"]
    W_r, U_r, b_r = p["W_r"], p["U_r"], p["b_r"]
    W_h, U_h, b_h = p["W_h"], p["U_h"], p["b_h"]

    B, T, D = Xb.shape
    H = W_z.shape[0]

    hs = np.zeros((B, T, H), dtype=np.float32)
    z_list, r_list, htil_list = [], [], []

    h_prev = np.zeros((B, H), dtype=np.float32)

    for t in range(T):
        x = Xb[:, t, :]
        mt = Mb[:, t:t+1]

        a_z = x @ W_z.T + h_prev @ U_z.T + b_z
        a_r = x @ W_r.T + h_prev @ U_r.T + b_r
        z = sigmoid(a_z); r = sigmoid(a_r)

        a_h = x @ W_h.T + (r * h_prev) @ U_h.T + b_h
        htil = np.tanh(a_h)

        h = (1 - z) * h_prev + z * htil
        h = mt * h + (1 - mt) * h_prev

        hs[:, t, :] = h
        z_list.append(z); r_list.append(r); htil_list.append(htil)
        h_prev = h

    cache = {"Xb": Xb, "Mb": Mb, "hs": hs, "z": z_list, "r": r_list, "htil": htil_list}
    return hs, cache

def attention_forward_batch(hs, Mb, p):
    W_att, v_att = p["W_att"], p["v_att"]
    u = np.tanh(hs @ W_att.T)      # (B,T,A)
    scores = u @ v_att             # (B,T)
    alphas = softmax_masked(scores, Mb)
    context = np.sum(hs * alphas[:, :, None], axis=1)  # (B,H)
    cache = {"hs": hs, "Mb": Mb, "u": u, "alphas": alphas}
    return context, alphas, cache

def ms_head(ctx, p):
    return ctx @ p["W_ms"].T + p["b_ms"]

def beh_logits(ctx, p):
    return np.squeeze(ctx @ p["W_beh"].T + p["b_beh"], axis=1)

# ============================================================
# 4) OPTIMIZER
# ============================================================
class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

    def step(self, params, grads, weight_decay=0.0, freeze_keys=None):
        self.t += 1
        if freeze_keys is None:
            freeze_keys = set()

        for k in params:
            if k in freeze_keys:
                continue

            g = grads[k]
            if weight_decay > 0 and (not k.startswith("b_")):
                g = g + weight_decay * params[k]

            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)

            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)

            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

# ============================================================
# 5) TRAIN PHASE 1 (RECON, positives only)
# ============================================================
def train_recon_phase(p):
    opt = Adam(p, lr=LR_RECON)

    X_full = X_POS
    M_full = MASK_POS
    Y_full = Y_MS_POS
    n = X_full.shape[0]

    for ep in range(1, EPOCHS_RECON + 1):
        idx = np.arange(n)
        np.random.shuffle(idx)

        total = 0.0
        for s in range(0, n, BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            Xb, Mb, Yb = X_full[b], M_full[b], Y_full[b]
            B = Xb.shape[0]

            # forward
            hs, cache_gru = gru_forward_batch(Xb, Mb, p)
            ctx, alphas, cache_att = attention_forward_batch(hs, Mb, p)
            ms_hat = ms_head(ctx, p)

            diff = (ms_hat - Yb)
            loss = 0.5 * np.mean(diff * diff)
            total += loss * B

            # grads init
            grads = {k: np.zeros_like(v) for k, v in p.items()}

            # MS head
            dms_hat = diff / MS_DIM / B
            grads["W_ms"] = dms_hat.T @ ctx
            grads["b_ms"] = np.sum(dms_hat, axis=0)
            dctx = dms_hat @ p["W_ms"]  # (B,H)

            # attention backward (batched)
            hs2 = cache_att["hs"]
            u = cache_att["u"]
            al = cache_att["alphas"]
            mask = cache_att["Mb"]
            W_att = p["W_att"]
            v_att = p["v_att"]

            dhs = al[:, :, None] * dctx[:, None, :]
            d_alpha = np.sum(dctx[:, None, :] * hs2, axis=2)
            sum_term = np.sum(al * d_alpha, axis=1, keepdims=True)
            dscores = (al * (d_alpha - sum_term)) * mask

            grads["v_att"] = np.sum(dscores[:, :, None] * u, axis=(0, 1))
            du = dscores[:, :, None] * v_att[None, None, :]
            da = du * (1 - u*u)

            grads["W_att"] = np.einsum("bta,bth->ah", da, hs2)
            dhs += np.einsum("bta,ah->bth", da, W_att)

            # GRU backward (batched)
            Xb2 = cache_gru["Xb"]
            Mb2 = cache_gru["Mb"]
            hs_all = cache_gru["hs"]
            z_list = cache_gru["z"]
            r_list = cache_gru["r"]
            htil_list = cache_gru["htil"]

            W_z, U_z = p["W_z"], p["U_z"]
            W_r, U_r = p["W_r"], p["U_r"]
            W_h, U_h = p["W_h"], p["U_h"]

            dW_z = np.zeros_like(W_z); dU_z = np.zeros_like(U_z); db_z = np.zeros_like(p["b_z"])
            dW_r = np.zeros_like(W_r); dU_r = np.zeros_like(U_r); db_r = np.zeros_like(p["b_r"])
            dW_h = np.zeros_like(W_h); dU_h = np.zeros_like(U_h); db_h = np.zeros_like(p["b_h"])

            dh_next = np.zeros((B, HIDDEN_DIM), dtype=np.float32)

            for t in reversed(range(SEQ_LEN)):
                x = Xb2[:, t, :]
                mt = Mb2[:, t:t+1]

                h = hs_all[:, t, :]
                h_prev = np.zeros_like(h) if t == 0 else hs_all[:, t-1, :]

                z = z_list[t]
                r = r_list[t]
                htil = htil_list[t]

                dh = (dh_next + dhs[:, t, :]) * mt

                dh_til = dh * z
                dz = dh * (htil - h_prev)
                dh_prev = dh * (1 - z)

                da_h = dh_til * (1 - htil*htil)

                dW_h += da_h.T @ x
                dU_h += da_h.T @ (r * h_prev)
                db_h += np.sum(da_h, axis=0)

                dh_prev += (da_h @ U_h) * r
                dr = (da_h @ U_h) * h_prev

                da_r = dr * r * (1 - r)
                dW_r += da_r.T @ x
                dU_r += da_r.T @ h_prev
                db_r += np.sum(da_r, axis=0)
                dh_prev += da_r @ U_r

                da_z = dz * z * (1 - z)
                dW_z += da_z.T @ x
                dU_z += da_z.T @ h_prev
                db_z += np.sum(da_z, axis=0)
                dh_prev += da_z @ U_z

                dh_next = dh_prev

            grads["W_z"] = dW_z; grads["U_z"] = dU_z; grads["b_z"] = db_z
            grads["W_r"] = dW_r; grads["U_r"] = dU_r; grads["b_r"] = db_r
            grads["W_h"] = dW_h; grads["U_h"] = dU_h; grads["b_h"] = db_h

            grads = clip_grads(grads, CLIP_NORM)
            opt.step(p, grads, weight_decay=WEIGHT_DECAY)

        if ep == 2 or ep % max(1, EPOCHS_RECON // 10) == 0:
            print(f"[Recon] Epoch {ep}/{EPOCHS_RECON}  avg_loss={total/n:.6f}")

    return p

# ============================================================
# 6) PHASE 2 (behavior head only, no BPTT)
# ============================================================
def compute_contexts(p, X, M):
    hs, _ = gru_forward_batch(X, M, p)
    ctx, _, _ = attention_forward_batch(hs, M, p)
    return ctx

def train_behavior_phase(p):
    # build negatives once (or every epoch if you want)
    X_neg, M_neg, y_neg = sample_negatives()
    y_pos = np.ones((X_POS.shape[0],), dtype=np.float32)

    X_all = np.concatenate([X_POS, X_neg], axis=0)
    M_all = np.concatenate([MASK_POS, M_neg], axis=0)
    y_all = np.concatenate([y_pos, y_neg], axis=0)

    # precompute contexts once -> very fast phase
    ctx_all = compute_contexts(p, X_all, M_all)

    opt = Adam(p, lr=LR_BEH)
    freeze = set([k for k in p.keys() if k not in ("W_beh", "b_beh")])

    n = ctx_all.shape[0]
    for ep in range(1, EPOCHS_BEH + 1):
        idx = np.arange(n)
        np.random.shuffle(idx)

        total = 0.0
        for s in range(0, n, BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            cb = ctx_all[b]
            yb = y_all[b]

            logits = np.squeeze(cb @ p["W_beh"].T + p["b_beh"], axis=1)
            probs = sigmoid(logits)

            eps = 1e-8
            loss = -(POS_WEIGHT * yb * np.log(probs + eps) + (1 - yb) * np.log(1 - probs + eps))
            total += float(np.mean(loss)) * len(b)

            dlogit = (probs - yb)
            dlogit = np.where(yb > 0.5, POS_WEIGHT * dlogit, dlogit)

            grads = {k: np.zeros_like(v) for k, v in p.items()}
            grads["W_beh"] = (dlogit[:, None] * cb).mean(axis=0, keepdims=True)
            grads["b_beh"] = np.array([dlogit.mean()], dtype=np.float32)

            opt.step(p, grads, weight_decay=WEIGHT_DECAY, freeze_keys=freeze)

        if ep == 2 or ep % max(1, EPOCHS_BEH // 10) == 0:
            print(f"[Beh] Epoch {ep}/{EPOCHS_BEH}  avg_loss={total/n:.6f}")

    return p

# ============================================================
# 7) EVAL
# ============================================================
def eval_demo(p, id_eval=0, w_eval=5):
    ms_eval = MS_all[id_eval]
    g_true = id_eval * N_WINDOWS_PER_ID + w_eval
    toks, meas = generate_os_chain(ms_eval, g_true)

    X = build_X(toks, meas, expected_window_local=w_eval, claimed_id=id_eval)
    Xp, Mp = pad_to_T(X)
    ctx = compute_contexts(p, Xp[None, :, :], Mp[None, :])
    ms_hat = ms_head(ctx, p)[0]
    p_legit = sigmoid(beh_logits(ctx, p)[0])

    print("\n=== LEGIT ===")
    print("L2(MS_hat,MS):", l2(ms_hat, ms_eval))
    print("p_legit:", float(p_legit))

    # shuffled
    idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
    Xs = build_X(toks[idxs], meas[idxs], expected_window_local=w_eval, claimed_id=id_eval)
    Xs, Ms = pad_to_T(Xs)
    ctxs = compute_contexts(p, Xs[None, :, :], Ms[None, :])
    print("\n=== SHUFFLED ===")
    print("p_shuf:", float(sigmoid(beh_logits(ctxs, p)[0])))

    # truncated
    L = SEQ_LEN // 2
    Xt = build_X(toks, meas, expected_window_local=w_eval, claimed_id=id_eval)[:L]
    Xt, Mt = pad_to_T(Xt)
    ctxt = compute_contexts(p, Xt[None, :, :], Mt[None, :])
    print("\n=== TRUNCATED ===")
    print("p_trunc:", float(sigmoid(beh_logits(ctxt, p)[0])))

    # wrong-window claim mismatch
    wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID
    g_wrong = id_eval * N_WINDOWS_PER_ID + wrong_w
    toks_w, meas_w = generate_os_chain(ms_eval, g_wrong)
    Xw = build_X(toks_w, meas_w, expected_window_local=w_eval, claimed_id=id_eval)  # expected stays w_eval!
    Xw, Mw = pad_to_T(Xw)
    ctxw = compute_contexts(p, Xw[None, :, :], Mw[None, :])
    print("\n=== WRONG WINDOW CLAIM (same id) ===")
    print("p_wrong_window:", float(sigmoid(beh_logits(ctxw, p)[0])))

    # wrong-identity
    other_id = (id_eval + 1) % N_IDENTITIES
    other_w = np.random.randint(0, N_WINDOWS_PER_ID)
    g_other = other_id * N_WINDOWS_PER_ID + other_w
    ms_other = MS_all[other_id]
    toks_o, meas_o = generate_os_chain(ms_other, g_other)
    Xo = build_X(toks_o, meas_o, expected_window_local=w_eval, claimed_id=id_eval)
    Xo, Mo = pad_to_T(Xo)
    ctxo = compute_contexts(p, Xo[None, :, :], Mo[None, :])
    print("\n=== WRONG IDENTITY (claim id0) ===")
    print("p_wrong_id:", float(sigmoid(beh_logits(ctxo, p)[0])))

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    p = init_model(D)
    p = train_recon_phase(p)
    p = train_behavior_phase(p)
    eval_demo(p, id_eval=0, w_eval=5)
