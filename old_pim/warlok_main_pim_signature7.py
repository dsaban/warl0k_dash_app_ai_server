import numpy as np
np.random.seed(0)

# =========================
# CONFIG
# =========================
VOCAB_SIZE = 16
MS_DIM     = 8
SEQ_LEN    = 20

N_IDENTITIES     = 6
N_WINDOWS_PER_ID = 48

HIDDEN_DIM = 48
ATTN_DIM   = 24
MS_HID     = 32

BATCH_SIZE = 64

EPOCHS_PHASE1 = 600
EPOCHS_PHASE2 = 420

LR_PHASE1 = 0.006
LR_PHASE2 = 0.01   # lower than before (more stable)

CLIP_NORM    = 5.0
WEIGHT_DECAY = 1e-4

# Phase1 losses
LAMBDA_MS  = 1.0
LAMBDA_TOK = 0.35
TOK_STOP_EPS = 0.12
TOK_ONLY_POS = True

# Phase2 losses
LAMBDA_ID  = 1.0
LAMBDA_W   = 1.0
LAMBDA_BCE = 1.0
POS_WEIGHT = 10.0


# Strong accept-rule (claim consistency)
PID_MIN = 0.80  # min softmax prob for claimed identity
PW_MIN  = 0.80  # min softmax prob for expected window
# =========================
# Helpers
# =========================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def softmax1d(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def softmax_masked(scores, mask):
    huge_neg = -1e9
    s = np.where(mask > 0, scores, huge_neg)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s) * mask
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def clip_grads(grads, max_norm=CLIP_NORM):
    norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return grads

def l2(a,b): return float(np.linalg.norm(a-b))

# =========================
# 1) DATA GENERATOR
# =========================
MS_all = np.random.uniform(-1.0, 1.0, size=(N_IDENTITIES, MS_DIM)).astype(np.float32)
A_base = (np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32) * 0.8)

def window_delta(window_global_id, t, ms_dim=MS_DIM):
    # stronger window fingerprint
    seed = (window_global_id * 10007 + t * 97) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return (0.25 * rng.randn(ms_dim)).astype(np.float32)   # 0.15 -> 0.25

def generate_os_chain(ms_vec, window_global_id, seq_len=SEQ_LEN):
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_global_id, t)
        zs[t] = float(a_t @ ms_vec)

    # slightly lower noise
    noise_seed = (window_global_id * 1337 + int((ms_vec * 1000).sum())) & 0xFFFFFFFF
    rng = np.random.RandomState(noise_seed)
    zs = zs + rng.normal(scale=0.02, size=seq_len).astype(np.float32)  # 0.04 -> 0.02

    m = (zs - zs.mean()) / (zs.std() + 1e-6)
    scaled = np.clip((m + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

# =========================
# 2) Backbone input (NO CLAIM FEATURES)
# =========================
def build_X_backbone(tokens, m):
    # Only: token onehot + measurement + position
    T = len(tokens)
    D = VOCAB_SIZE + 2
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        X[t, tokens[t]] = 1.0
        X[t, VOCAB_SIZE] = m[t]
        X[t, VOCAB_SIZE + 1] = t / max(1, (T - 1))
    return X

def pad_to_T(X, tokens, T=SEQ_LEN):
    D = X.shape[1]
    out = np.zeros((T, D), dtype=np.float32)
    mask = np.zeros((T,), dtype=np.float32)
    tok_pad = np.zeros((T,), dtype=np.int32)
    L = min(T, X.shape[0])
    out[:L] = X[:L]
    mask[:L] = 1.0
    tok_pad[:L] = tokens[:L]
    return out, mask, tok_pad

# =========================
# 3) Dataset (store TRUE + CLAIM separately)
# =========================
def build_dataset():
    Xs, Ms, Tok = [], [], []
    y_ms, y_cls = [], []
    true_id, true_w = [], []
    claim_id, expect_w = [], []

    for id_true in range(N_IDENTITIES):
        ms_true = MS_all[id_true]
        for w_true in range(N_WINDOWS_PER_ID):
            g_true = id_true * N_WINDOWS_PER_ID + w_true
            toks, meas = generate_os_chain(ms_true, g_true)

            # POS legit: true==claim
            X = build_X_backbone(toks, meas)
            Xp, Mp, Tp = pad_to_T(X, toks)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(1)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG shuffled
            idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
            toks_s, meas_s = toks[idxs], meas[idxs]
            X = build_X_backbone(toks_s, meas_s)
            Xp, Mp, Tp = pad_to_T(X, toks_s)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG truncated
            Ltr = SEQ_LEN // 2
            toks_t, meas_t = toks[:Ltr], meas[:Ltr]
            X = build_X_backbone(toks_t, meas_t)
            Xp, Mp, Tp = pad_to_T(X, toks_t)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG wrong-window chain but verifier expects w_true
            wrong_w = (w_true + 7) % N_WINDOWS_PER_ID
            g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w
            toks_w, meas_w = generate_os_chain(ms_true, g_wrong)
            X = build_X_backbone(toks_w, meas_w)
            Xp, Mp, Tp = pad_to_T(X, toks_w)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(wrong_w)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG wrong-identity chain but claim id_true
            other_id = (id_true + np.random.randint(1, N_IDENTITIES)) % N_IDENTITIES
            other_w  = np.random.randint(0, N_WINDOWS_PER_ID)
            g_other  = other_id * N_WINDOWS_PER_ID + other_w
            ms_other = MS_all[other_id]
            toks_o, meas_o = generate_os_chain(ms_other, g_other)
            X = build_X_backbone(toks_o, meas_o)
            Xp, Mp, Tp = pad_to_T(X, toks_o)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(other_id); true_w.append(other_w)
            claim_id.append(id_true); expect_w.append(w_true)

    return (
        np.stack(Xs).astype(np.float32),
        np.stack(Ms).astype(np.float32),
        np.stack(Tok).astype(np.int32),
        np.stack(y_ms).astype(np.float32),
        np.array(y_cls, dtype=np.float32),
        np.array(true_id, dtype=np.int32),
        np.array(true_w, dtype=np.int32),
        np.array(claim_id, dtype=np.int32),
        np.array(expect_w, dtype=np.int32),
    )

X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W = build_dataset()
N, T, D = X_ALL.shape
print("All samples:", N, "X shape:", X_ALL.shape, "Positives:", int(np.sum(Y_CLS_ALL)))

# =========================
# 4) Model init
# =========================
def init_model(input_dim):
    p = {}
    s = 0.08

    # GRU
    p["W_z"] = np.random.randn(HIDDEN_DIM, input_dim) * s
    p["U_z"] = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_z"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    p["W_r"] = np.random.randn(HIDDEN_DIM, input_dim) * s
    p["U_r"] = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_r"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    p["W_h"] = np.random.randn(HIDDEN_DIM, input_dim) * s
    p["U_h"] = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_h"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    # attention
    p["W_att"] = np.random.randn(ATTN_DIM, HIDDEN_DIM) * s
    p["v_att"] = np.random.randn(ATTN_DIM) * s

    # MS MLP head
    p["W_ms1"] = np.random.randn(MS_HID, HIDDEN_DIM) * s
    p["b_ms1"] = np.zeros((MS_HID,), dtype=np.float32)
    p["W_ms2"] = np.random.randn(MS_DIM, MS_HID) * s
    p["b_ms2"] = np.zeros((MS_DIM,), dtype=np.float32)

    # token scaffold
    p["W_tok"] = np.random.randn(VOCAB_SIZE, HIDDEN_DIM) * s
    p["b_tok"] = np.zeros((VOCAB_SIZE,), dtype=np.float32)

    # Phase2 heads
    p["W_id"] = np.random.randn(N_IDENTITIES, HIDDEN_DIM) * s
    p["b_id"] = np.zeros((N_IDENTITIES,), dtype=np.float32)

    # window head uses 3H: [ctx, h_last, h_mean]
    p["W_w"]  = np.random.randn(N_WINDOWS_PER_ID, 3*HIDDEN_DIM) * s
    p["b_w"]  = np.zeros((N_WINDOWS_PER_ID,), dtype=np.float32)

    # validity head uses claim-consistency residuals too:
    # input = [ctx (H), claim_id_norm, expected_w_norm, p_id_claimed, p_w_expected]
    p["W_beh"] = np.random.randn(1, HIDDEN_DIM + 4) * s
    p["b_beh"] = np.zeros((1,), dtype=np.float32)

    return p

# =========================
# 5) Forward passes
# =========================
def gru_forward_batch(Xb, Mb, p):
    B, T, _ = Xb.shape
    hs = np.zeros((B, T, HIDDEN_DIM), dtype=np.float32)

    z_list, r_list, htil_list = [], [], []
    h_prev = np.zeros((B, HIDDEN_DIM), dtype=np.float32)

    for t in range(T):
        x = Xb[:, t, :]
        mt = Mb[:, t:t+1]

        a_z = x @ p["W_z"].T + h_prev @ p["U_z"].T + p["b_z"]
        a_r = x @ p["W_r"].T + h_prev @ p["U_r"].T + p["b_r"]
        z = sigmoid(a_z); r = sigmoid(a_r)

        a_h = x @ p["W_h"].T + (r * h_prev) @ p["U_h"].T + p["b_h"]
        htil = np.tanh(a_h)

        h = (1 - z) * h_prev + z * htil
        h = mt * h + (1 - mt) * h_prev

        hs[:, t, :] = h
        z_list.append(z); r_list.append(r); htil_list.append(htil)
        h_prev = h

    cache = {"Xb": Xb, "Mb": Mb, "hs": hs, "z": z_list, "r": r_list, "htil": htil_list}
    return hs, cache

def attention_forward_batch(hs, Mb, p):
    u = np.tanh(hs @ p["W_att"].T)
    scores = u @ p["v_att"]
    alphas = softmax_masked(scores, Mb)
    ctx = np.sum(hs * alphas[:, :, None], axis=1)
    cache = {"hs": hs, "Mb": Mb, "u": u, "alphas": alphas}
    return ctx, alphas, cache

def ms_head(ctx, p):
    h = np.tanh(ctx @ p["W_ms1"].T + p["b_ms1"])
    ms_hat = h @ p["W_ms2"].T + p["b_ms2"]
    return ms_hat, h

# =========================
# 6) Adam
# =========================
class Adam:
    def __init__(self, params, lr):
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k,v in params.items()}
        self.v = {k: np.zeros_like(v) for k,v in params.items()}
        self.t = 0
        self.b1 = 0.9
        self.b2 = 0.999
        self.eps = 1e-8

    def step(self, params, grads, weight_decay=0.0, freeze_keys=None):
        self.t += 1
        if freeze_keys is None: freeze_keys = set()
        for k in params:
            if k in freeze_keys:
                continue
            g = grads[k]
            if weight_decay > 0 and (not k.startswith("b_")):
                g = g + weight_decay * params[k]
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*(g*g)
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

# =========================
# 7) Phase1 training
# =========================
def token_ce_one(logits, target):
    p = softmax1d(logits)
    loss = -np.log(p[target] + 1e-12)
    dlog = p
    dlog[target] -= 1.0
    return loss, dlog

def train_phase1(p):
    opt = Adam(p, lr=LR_PHASE1)
    tok_enabled = True

    for ep in range(1, EPOCHS_PHASE1 + 1):
        idx = np.arange(N); np.random.shuffle(idx)
        total = 0.0

        for s in range(0, N, BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            Xb, Mb, Tb = X_ALL[b], M_ALL[b], TOK_ALL[b]
            yms, ycls = Y_MS_ALL[b], Y_CLS_ALL[b]
            B = Xb.shape[0]

            hs, cache_gru = gru_forward_batch(Xb, Mb, p)
            ctx, _, cache_att = attention_forward_batch(hs, Mb, p)
            ms_hat, ms_hid = ms_head(ctx, p)

            # MS loss on positives
            pos = (ycls > 0.5).astype(np.float32)[:, None]
            diff = (ms_hat - yms) * pos
            loss_ms = 0.5 * np.mean(diff * diff)

            # token scaffold on positives only
            loss_tok = 0.0
            dH_tok = np.zeros_like(hs)
            dW_tok = np.zeros_like(p["W_tok"])
            db_tok = np.zeros_like(p["b_tok"])

            if tok_enabled:
                valid_pair = (Mb[:, :-1] * Mb[:, 1:]) * (ycls[:, None] > 0.5)
                denom = float(np.sum(valid_pair) + 1e-6)

                for t in range(SEQ_LEN - 1):
                    valid = Mb[:, t] * Mb[:, t+1] * (ycls > 0.5)
                    if np.sum(valid) < 1:
                        continue
                    h_t = hs[:, t, :]
                    logits_bt = h_t @ p["W_tok"].T + p["b_tok"]
                    for i in range(B):
                        if valid[i] <= 0:
                            continue
                        target = int(Tb[i, t+1])
                        li, dlog = token_ce_one(logits_bt[i], target)
                        loss_tok += li
                        dW_tok += dlog[:, None] @ h_t[i:i+1, :]
                        db_tok += dlog
                        dH_tok[i, t, :] += dlog @ p["W_tok"]

                loss_tok /= denom
                dW_tok /= denom
                db_tok /= denom
                dH_tok /= denom

                if loss_tok < TOK_STOP_EPS:
                    tok_enabled = False

            loss = LAMBDA_MS*loss_ms + (LAMBDA_TOK*loss_tok if tok_enabled else 0.0)
            total += loss * B

            grads = {k: np.zeros_like(v) for k,v in p.items()}
            grads["W_tok"] = dW_tok
            grads["b_tok"] = db_tok

            # MS MLP backprop
            dms = (diff / MS_DIM) / B
            grads["W_ms2"] = dms.T @ ms_hid
            grads["b_ms2"] = np.sum(dms, axis=0)
            dms_hid = dms @ p["W_ms2"]
            dpre = dms_hid * (1 - ms_hid*ms_hid)
            grads["W_ms1"] = dpre.T @ ctx
            grads["b_ms1"] = np.sum(dpre, axis=0)
            dctx = dpre @ p["W_ms1"]

            # attention backprop
            hs2 = cache_att["hs"]; u = cache_att["u"]; al = cache_att["alphas"]; mask = cache_att["Mb"]
            dhs = al[:, :, None] * dctx[:, None, :]
            d_alpha = np.sum(dctx[:, None, :] * hs2, axis=2)
            sum_term = np.sum(al * d_alpha, axis=1, keepdims=True)
            dscores = (al * (d_alpha - sum_term)) * mask

            grads["v_att"] += np.sum(dscores[:, :, None] * u, axis=(0,1))
            du = dscores[:, :, None] * p["v_att"][None,None,:]
            da = du * (1 - u*u)
            grads["W_att"] += np.einsum("bta,bth->ah", da, hs2)
            dhs += np.einsum("bta,ah->bth", da, p["W_att"])

            dhs += dH_tok

            # GRU BPTT
            Xb2, Mb2, hs_all = cache_gru["Xb"], cache_gru["Mb"], cache_gru["hs"]
            z_list, r_list, htil_list = cache_gru["z"], cache_gru["r"], cache_gru["htil"]

            dW_z = np.zeros_like(p["W_z"]); dU_z = np.zeros_like(p["U_z"]); db_z = np.zeros_like(p["b_z"])
            dW_r = np.zeros_like(p["W_r"]); dU_r = np.zeros_like(p["U_r"]); db_r = np.zeros_like(p["b_r"])
            dW_h = np.zeros_like(p["W_h"]); dU_h = np.zeros_like(p["U_h"]); db_h = np.zeros_like(p["b_h"])
            dh_next = np.zeros((B, HIDDEN_DIM), dtype=np.float32)

            for t in reversed(range(SEQ_LEN)):
                x = Xb2[:, t, :]
                mt = Mb2[:, t:t+1]
                h_prev = np.zeros((B,HIDDEN_DIM),dtype=np.float32) if t==0 else hs_all[:,t-1,:]
                z, r, htil = z_list[t], r_list[t], htil_list[t]

                dh = (dh_next + dhs[:,t,:]) * mt

                dh_til = dh * z
                dz = dh * (htil - h_prev)
                dh_prev = dh * (1 - z)

                da_h = dh_til * (1 - htil*htil)
                dW_h += da_h.T @ x
                dU_h += da_h.T @ (r * h_prev)
                db_h += np.sum(da_h, axis=0)

                dh_prev += (da_h @ p["U_h"]) * r
                dr = (da_h @ p["U_h"]) * h_prev

                da_r = dr * r * (1 - r)
                dW_r += da_r.T @ x
                dU_r += da_r.T @ h_prev
                db_r += np.sum(da_r, axis=0)
                dh_prev += da_r @ p["U_r"]

                da_z = dz * z * (1 - z)
                dW_z += da_z.T @ x
                dU_z += da_z.T @ h_prev
                db_z += np.sum(da_z, axis=0)
                dh_prev += da_z @ p["U_z"]

                dh_next = dh_prev

            grads["W_z"] += dW_z; grads["U_z"] += dU_z; grads["b_z"] += db_z
            grads["W_r"] += dW_r; grads["U_r"] += dU_r; grads["b_r"] += db_r
            grads["W_h"] += dW_h; grads["U_h"] += dU_h; grads["b_h"] += db_h

            grads = clip_grads(grads)
            opt.step(p, grads, weight_decay=WEIGHT_DECAY)

        if ep == 2 or ep % max(1, EPOCHS_PHASE1 // 10) == 0:
            print(f"[Phase1] Epoch {ep}/{EPOCHS_PHASE1} avg_loss={total/N:.6f} tok_enabled={tok_enabled}")

    return p

# =========================
# 8) Embeddings for Phase2
# =========================
def compute_embeddings(p, X, M):
    hs, _ = gru_forward_batch(X, M, p)
    ctx, _, _ = attention_forward_batch(hs, M, p)

    B, T, H = hs.shape
    denom = np.sum(M, axis=1, keepdims=True) + 1e-6
    h_mean = np.sum(hs * M[:, :, None], axis=1) / denom

    last_idx = np.maximum(0, np.sum(M, axis=1).astype(np.int32) - 1)
    h_last = np.zeros((B, H), dtype=np.float32)
    for i in range(B):
        h_last[i] = hs[i, last_idx[i], :]

    return ctx, h_last, h_mean

# =========================
# 9) Phase2 training (heads only)
# =========================
def ce_loss_batch(logits, targets):
    B, C = logits.shape
    loss = 0.0
    dlog = np.zeros_like(logits)
    for i in range(B):
        p = softmax1d(logits[i])
        loss += -np.log(p[targets[i]] + 1e-12)
        d = p
        d[targets[i]] -= 1.0
        dlog[i] = d
    return loss / B, dlog / B

def bce_loss_batch(logits, y):
    p = sigmoid(logits)
    eps = 1e-8
    loss = -(POS_WEIGHT*y*np.log(p+eps) + (1-y)*np.log(1-p+eps))
    loss = float(np.mean(loss))
    dlog = (p - y)
    dlog = np.where(y > 0.5, POS_WEIGHT*dlog, dlog)
    dlog = dlog / len(y)
    return loss, dlog

def train_phase2(p):
    ctx_all, h_last_all, h_mean_all = compute_embeddings(p, X_ALL, M_ALL)
    opt = Adam(p, lr=LR_PHASE2)

    freeze = set([k for k in p.keys() if k not in ("W_id","b_id","W_w","b_w","W_beh","b_beh")])

    for ep in range(1, EPOCHS_PHASE2 + 1):
        idx = np.arange(N); np.random.shuffle(idx)
        total = 0.0

        for s in range(0, N, BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            cb = ctx_all[b]
            hl = h_last_all[b]
            hm = h_mean_all[b]
            yb  = Y_CLS_ALL[b]
            tid = TRUE_ID[b]
            tw  = TRUE_W[b]

            # ID head
            logits_id = cb @ p["W_id"].T + p["b_id"]

            # WINDOW head (sequence-derived)
            feat_w = np.concatenate([cb, hl, hm], axis=1)
            logits_w = feat_w @ p["W_w"].T + p["b_w"]

            # validity head uses claim-consistency residuals:
            #   p_id_claimed   = softmax(logits_id)[claimed_id]
            #   p_w_expected   = softmax(logits_w)[expected_w]
            cid = CLAIM_ID[b] / max(1, N_IDENTITIES-1)
            ew  = EXPECT_W[b] / max(1, N_WINDOWS_PER_ID-1)

            pid = np.zeros((len(b),), dtype=np.float32)
            pw  = np.zeros((len(b),), dtype=np.float32)
            for i in range(len(b)):
                pid[i] = softmax1d(logits_id[i])[CLAIM_ID[b[i]]]
                pw[i]  = softmax1d(logits_w[i])[EXPECT_W[b[i]]]

            vb_in = np.concatenate([
                cb,
                cid[:, None].astype(np.float32),
                ew[:, None].astype(np.float32),
                pid[:, None],
                pw[:, None]
            ], axis=1)
            logits_v = np.squeeze(vb_in @ p["W_beh"].T + p["b_beh"])

            loss_id, dlog_id = ce_loss_batch(logits_id, tid)
            loss_w,  dlog_w  = ce_loss_batch(logits_w, tw)
            loss_v,  dlog_v  = bce_loss_batch(logits_v, yb)

            loss = LAMBDA_ID*loss_id + LAMBDA_W*loss_w + LAMBDA_BCE*loss_v
            total += loss * len(b)

            grads = {k: np.zeros_like(v) for k,v in p.items()}
            grads["W_id"] = dlog_id.T @ cb
            grads["b_id"] = np.sum(dlog_id, axis=0)

            grads["W_w"]  = dlog_w.T @ feat_w
            grads["b_w"]  = np.sum(dlog_w, axis=0)

            grads["W_beh"] = (dlog_v[:,None] * vb_in).sum(axis=0, keepdims=True)
            grads["b_beh"] = np.array([np.sum(dlog_v)], dtype=np.float32)

            grads = clip_grads(grads)
            opt.step(p, grads, weight_decay=WEIGHT_DECAY, freeze_keys=freeze)

        if ep == 2 or ep % max(1, EPOCHS_PHASE2 // 10) == 0:
            print(f"[Phase2] Epoch {ep}/{EPOCHS_PHASE2} avg_loss={total/N:.6f}")

    return p

# =========================
# 10) Verification
# =========================
def verify_chain(p, tokens, meas, claimed_id, expected_w, true_ms=None, threshold=0.7):
    X = build_X_backbone(tokens, meas)
    Xp, Mp, _ = pad_to_T(X, tokens)

    ctx, h_last, h_mean = compute_embeddings(p, Xp[None,:,:], Mp[None,:])
    ctx1 = ctx[0]
    ms_hat, _ = ms_head(ctx, p)
    ms_hat = ms_hat[0]

    # ID pred
    logits_id = (ctx @ p["W_id"].T + p["b_id"])[0]
    id_pred = int(np.argmax(logits_id))

    # Window pred (sequence-derived)
    feat_w = np.concatenate([ctx1, h_last[0], h_mean[0]], axis=0)[None,:]
    logits_w = (feat_w @ p["W_w"].T + p["b_w"])[0]
    w_pred = int(np.argmax(logits_w))

    # validity pred (claim-residual aware)
    p_id_claimed = softmax1d(logits_id)[claimed_id]
    p_w_expected = softmax1d(logits_w)[expected_w]

    cid = claimed_id / max(1, N_IDENTITIES-1)
    ew  = expected_w / max(1, N_WINDOWS_PER_ID-1)

    vb_in = np.concatenate([
        ctx1,
        np.array([cid, ew, p_id_claimed, p_w_expected], dtype=np.float32)
    ], axis=0)[None, :]
    logit_v = float(np.squeeze(vb_in @ p["W_beh"].T + p["b_beh"]))
    p_valid = float(sigmoid(logit_v))

    ok = (p_valid >= threshold) and (id_pred == claimed_id) and (w_pred == expected_w) and (p_id_claimed >= PID_MIN) and (p_w_expected >= PW_MIN) and (p_id_claimed >= PID_MIN) and (p_w_expected >= PW_MIN)
    out = {"ok": ok, "p_valid": p_valid, "id_pred": id_pred, "w_pred": w_pred, "p_id_claimed": float(p_id_claimed), "p_w_expected": float(p_w_expected), "ms_hat": ms_hat}
    if true_ms is not None:
        out["l2_ms"] = l2(ms_hat, true_ms)
    return out

def eval_demo(p, id_eval=0, w_eval=5):
    ms_eval = MS_all[id_eval]
    g_true = id_eval * N_WINDOWS_PER_ID + w_eval
    toks, meas = generate_os_chain(ms_eval, g_true)

    print("\n=== LEGIT ===")
    r = verify_chain(p, toks, meas, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print("L2(MS_hat,MS):", r["l2_ms"])
    print("p_valid:", r["p_valid"], "id_pred:", r["id_pred"], "w_pred:", r["w_pred"], "OK:", r["ok"])

    print("\n=== SHUFFLED ===")
    idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
    r = verify_chain(p, toks[idxs], meas[idxs], claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print("p_valid:", r["p_valid"], "id_pred:", r["id_pred"], "w_pred:", r["w_pred"], "OK:", r["ok"])

    print("\n=== TRUNCATED ===")
    Ltr = SEQ_LEN//2
    r = verify_chain(p, toks[:Ltr], meas[:Ltr], claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print("p_valid:", r["p_valid"], "id_pred:", r["id_pred"], "w_pred:", r["w_pred"], "OK:", r["ok"])

    print("\n=== WRONG WINDOW CLAIM ===")
    wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID
    g_wrong = id_eval * N_WINDOWS_PER_ID + wrong_w
    toks_w, meas_w = generate_os_chain(ms_eval, g_wrong)
    r = verify_chain(p, toks_w, meas_w, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print("p_valid:", r["p_valid"], "id_pred:", r["id_pred"], "w_pred:", r["w_pred"], "OK:", r["ok"])

    print("\n=== WRONG IDENTITY (claim id0) ===")
    other_id = (id_eval + 1) % N_IDENTITIES
    other_w = np.random.randint(0, N_WINDOWS_PER_ID)
    g_other = other_id * N_WINDOWS_PER_ID + other_w
    ms_other = MS_all[other_id]
    toks_o, meas_o = generate_os_chain(ms_other, g_other)
    r = verify_chain(p, toks_o, meas_o, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print("p_valid:", r["p_valid"], "id_pred:", r["id_pred"], "w_pred:", r["w_pred"], "OK:", r["ok"])

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    p = init_model(D)
    p = train_phase1(p)
    p = train_phase2(p)
    eval_demo(p, id_eval=0, w_eval=5)
