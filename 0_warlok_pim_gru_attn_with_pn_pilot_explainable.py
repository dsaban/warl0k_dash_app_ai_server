
import numpy as np
np.random.seed(0)

# ============================================================
# WARL0K PIM (NumPy-only) — GRU + Attention with Window "Pilot"
#
# Goal:
# - Learn to reconstruct MS from a CHAIN of OS tokens/measurements (Phase1)
# - Learn to validate behavioral chain continuity + claim binding (Phase2)
# - Make WINDOW decoding reliable by embedding a deterministic window pilot
#
# What you will see in prints:
# - Reconstruction quality: L2(MS_hat, MS)
# - Identity head: id_pred and prob of claimed id (pid)
# - Window head: w_pred and prob of expected window (pw)
# - Validity head: p_valid
# - Final decision: OK only if ALL thresholds & exact matches pass
# ============================================================

# ------------------------------
# CONFIG
# ------------------------------
VOCAB_SIZE         = 16
MS_DIM             = 8
SEQ_LEN            = 20

N_IDENTITIES       = 2
N_WINDOWS_PER_ID   = 48

HIDDEN_DIM         = 64
ATTN_DIM           = 32
MS_HID             = 32

BATCH_SIZE         = 32

EPOCHS_PHASE1      = 200
EPOCHS_PHASE2      = 500

LR_PHASE1          = 0.006
LR_PHASE2_BASE     = 0.03

CLIP_NORM          = 5.0
WEIGHT_DECAY       = 1e-4

# Phase1 losses
LAMBDA_MS          = 1.0
LAMBDA_TOK         = 0.10
TOK_STOP_EPS       = 0.25
TOK_WARMUP_EPOCHS  = 60

# Phase2 losses
LAMBDA_ID          = 1.0
LAMBDA_W           = 1.0
LAMBDA_BCE         = 1.0
POS_WEIGHT         = 10.0

# Strong accept thresholds
THRESH_P_VALID     = 0.80
PID_MIN            = 0.70
PW_MIN             = 0.70

# WINDOW PILOT (key change)
# A deterministic PN watermark-like pattern that survives normalization.
PILOT_AMP          = 0.55     # increase if window still weak (0.35-0.75 reasonable)
PILOT_PN_CHIPS     = SEQ_LEN   # PN pilot length (chips) equals sequence length
# PN pilot uses a seeded PRNG to generate +/-1 chips per window. High separability.

# ============================================================
# Helpers
# ============================================================
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

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

def l2(a,b): return float(np.linalg.norm(a-b))

def clip_grads(grads, max_norm=CLIP_NORM):
    norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return grads

# ============================================================
# 1) OS GENERATOR (identity mapping + window pilot)
# ============================================================
MS_all = np.random.uniform(-1.0, 1.0, size=(N_IDENTITIES, MS_DIM)).astype(np.float32)
A_base = (np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32) * 0.8)

def window_delta(window_global_id, t, ms_dim=MS_DIM):
    seed = (window_global_id * 10007 + t * 97) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return (0.25 * rng.randn(ms_dim)).astype(np.float32)

def window_pilot(window_global_id, seq_len=SEQ_LEN):
    """
    Deterministic PN (pseudo-noise) pilot keyed by window_global_id.

    Why PN works better than a sinusoid here:
    - Each window gets a near-unique +/-1 chip pattern (high entropy)
    - Survives normalization and coarse quantization better than phase-based sinusoids
    - Easy to generate on MCU: xorshift/LCG-based PRNG -> sign bits
    """
    rng = np.random.RandomState((window_global_id * 9176 + 11) & 0xFFFFFFFF)
    chips = rng.randint(0, 2, size=seq_len).astype(np.float32)  # 0/1
    chips = 2.0 * chips - 1.0                                  # -> -1/+1
    pilot = PILOT_AMP * chips
    pilot = pilot - pilot.mean()  # keep roughly zero-mean
    return pilot

def generate_os_chain(ms_vec, window_global_id, seq_len=SEQ_LEN):
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_global_id, t)
        zs[t] = float(a_t @ ms_vec)

    # Add window pilot watermark
    zs = zs + window_pilot(window_global_id, seq_len)

    # Add small noise
    noise_seed = (window_global_id * 1337 + int((ms_vec * 1000).sum())) & 0xFFFFFFFF
    rng = np.random.RandomState(noise_seed)
    zs = zs + rng.normal(scale=0.02, size=seq_len).astype(np.float32)

    # Normalize then quantize
    m = (zs - zs.mean()) / (zs.std() + 1e-6)
    scaled = np.clip((m + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

# ============================================================
# 2) Backbone input (NO CLAIM FEATURES)
# ============================================================
def build_X_backbone(tokens, m):
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

# ============================================================
# 3) DATASET
# ============================================================
def build_dataset():
    Xs, Ms, Tok = [], [], []
    y_ms, y_cls = [], []
    true_id, true_w, claim_id, expect_w = [], [], [], []

    for id_true in range(N_IDENTITIES):
        ms_true = MS_all[id_true]
        for w_true in range(N_WINDOWS_PER_ID):
            g_true = id_true * N_WINDOWS_PER_ID + w_true
            toks, meas = generate_os_chain(ms_true, g_true)

            # POS legit
            X = build_X_backbone(toks, meas)
            Xp, Mp, Tp = pad_to_T(X, toks)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(1)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG shuffled
            idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
            X = build_X_backbone(toks[idxs], meas[idxs])
            Xp, Mp, Tp = pad_to_T(X, toks[idxs])
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG truncated
            Ltr = SEQ_LEN//2
            X = build_X_backbone(toks[:Ltr], meas[:Ltr])
            Xp, Mp, Tp = pad_to_T(X, toks[:Ltr])
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(w_true)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG wrong-window chain (claim expects w_true)
            wrong_w = (w_true + 7) % N_WINDOWS_PER_ID
            g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w
            toks_w, meas_w = generate_os_chain(ms_true, g_wrong)
            X = build_X_backbone(toks_w, meas_w)
            Xp, Mp, Tp = pad_to_T(X, toks_w)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(wrong_w)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG wrong identity chain (claim id_true)
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

    return (np.stack(Xs).astype(np.float32),
            np.stack(Ms).astype(np.float32),
            np.stack(Tok).astype(np.int32),
            np.stack(y_ms).astype(np.float32),
            np.array(y_cls, dtype=np.float32),
            np.array(true_id, dtype=np.int32),
            np.array(true_w, dtype=np.int32),
            np.array(claim_id, dtype=np.int32),
            np.array(expect_w, dtype=np.int32))

# ============================================================
# 4) MODEL
# ============================================================
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

    # Attention
    p["W_att"] = np.random.randn(ATTN_DIM, HIDDEN_DIM) * s
    p["v_att"] = np.random.randn(ATTN_DIM) * s

    # MS head
    p["W_ms1"] = np.random.randn(MS_HID, HIDDEN_DIM) * s
    p["b_ms1"] = np.zeros((MS_HID,), dtype=np.float32)
    p["W_ms2"] = np.random.randn(MS_DIM, MS_HID) * s
    p["b_ms2"] = np.zeros((MS_DIM,), dtype=np.float32)

    # Token scaffold
    p["W_tok"] = np.random.randn(VOCAB_SIZE, HIDDEN_DIM) * s
    p["b_tok"] = np.zeros((VOCAB_SIZE,), dtype=np.float32)

    # Heads
    p["W_id"]  = np.random.randn(N_IDENTITIES, HIDDEN_DIM) * s
    p["b_id"]  = np.zeros((N_IDENTITIES,), dtype=np.float32)

    p["W_w"]   = np.random.randn(N_WINDOWS_PER_ID, 3*HIDDEN_DIM) * s
    p["b_w"]   = np.zeros((N_WINDOWS_PER_ID,), dtype=np.float32)

    p["W_beh"] = np.random.randn(1, HIDDEN_DIM + 4) * s
    p["b_beh"] = np.zeros((1,), dtype=np.float32)
    return p

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

# ============================================================
# 5) Adam
# ============================================================
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
        if freeze_keys is None:
            freeze_keys = set()
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

class AdamLite(Adam):
    def set_lr(self, lr): self.lr = lr

# ============================================================
# 6) Phase1 training
# ============================================================
def token_ce_one(logits, target):
    p_ = softmax1d(logits)
    loss = -np.log(p_[target] + 1e-12)
    dlog = p_
    dlog[target] -= 1.0
    return loss, dlog

def train_phase1(p, X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL):
    opt = Adam(p, lr=LR_PHASE1)
    tok_enabled = True
    N = X_ALL.shape[0]

    for ep in range(1, EPOCHS_PHASE1+1):
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

            pos_mask = (ycls > 0.5).astype(np.float32)[:, None]
            pos_count = float(np.sum(pos_mask) + 1e-6)

            diff = (ms_hat - yms) * pos_mask
            loss_ms = 0.5 * np.sum(diff * diff) / (pos_count * MS_DIM)

            loss_tok = 0.0
            dH_tok = np.zeros_like(hs)
            dW_tok = np.zeros_like(p["W_tok"])
            db_tok = np.zeros_like(p["b_tok"])

            if tok_enabled and ep <= TOK_WARMUP_EPOCHS:
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
            else:
                tok_enabled = False

            loss = LAMBDA_MS*loss_ms + (LAMBDA_TOK*loss_tok if tok_enabled else 0.0)
            total += loss * B

            grads = {k: np.zeros_like(v) for k,v in p.items()}
            grads["W_tok"] = dW_tok
            grads["b_tok"] = db_tok

            dms = diff / (pos_count * MS_DIM)
            grads["W_ms2"] = dms.T @ ms_hid
            grads["b_ms2"] = np.sum(dms, axis=0)

            dms_hid = dms @ p["W_ms2"]
            dpre = dms_hid * (1 - ms_hid*ms_hid)
            grads["W_ms1"] = dpre.T @ ctx
            grads["b_ms1"] = np.sum(dpre, axis=0)
            dctx = dpre @ p["W_ms1"]

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
                dU_h += da_h.T @ (r*h_prev)
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

        if ep == 2 or ep % max(1, EPOCHS_PHASE1//10) == 0:
            print(f"[Phase1] Epoch {ep}/{EPOCHS_PHASE1} avg_loss={total/N:.6f} tok_enabled={tok_enabled}")

    return p

# ============================================================
# 7) Embeddings
# ============================================================
def compute_embeddings(p, X, M):
    hs, _ = gru_forward_batch(X, M, p)
    ctx, _, _ = attention_forward_batch(hs, M, p)
    B, T, H = hs.shape
    denom = np.sum(M, axis=1, keepdims=True) + 1e-6
    h_mean = np.sum(hs * M[:,:,None], axis=1) / denom
    last_idx = np.maximum(0, np.sum(M, axis=1).astype(np.int32) - 1)
    h_last = np.zeros((B,H), dtype=np.float32)
    for i in range(B):
        h_last[i] = hs[i, last_idx[i], :]
    return ctx, h_last, h_mean

# ============================================================
# 8) Phase2 training (Heads only)
# ============================================================
def ce_loss_batch_masked(logits, targets, mask_01):
    idxs = np.where(mask_01 > 0.5)[0]
    if len(idxs) == 0:
        return 0.0, np.zeros_like(logits)
    L = 0.0
    dlog = np.zeros_like(logits)
    for i in idxs:
        p_ = softmax1d(logits[i])
        L += -np.log(p_[targets[i]] + 1e-12)
        d = p_; d[targets[i]] -= 1.0
        dlog[i] = d
    L /= len(idxs)
    dlog /= len(idxs)
    return L, dlog

def bce_loss_batch(logits, y):
    p_ = sigmoid(logits)
    eps = 1e-8
    loss = -(POS_WEIGHT*y*np.log(p_+eps) + (1-y)*np.log(1-p_+eps))
    loss = float(np.mean(loss))
    dlog = (p_ - y)
    dlog = np.where(y > 0.5, POS_WEIGHT*dlog, dlog)
    dlog = dlog / len(y)
    return loss, dlog

def train_phase2(p, X_ALL, M_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W):
    ctx_all, h_last_all, h_mean_all = compute_embeddings(p, X_ALL, M_ALL)
    opt = AdamLite(p, lr=LR_PHASE2_BASE)

    trainable = {"W_id","b_id","W_w","b_w","W_beh","b_beh"}
    freeze = set([k for k in p.keys() if k not in trainable])

    N = X_ALL.shape[0]
    for ep in range(1, EPOCHS_PHASE2+1):
        lr = LR_PHASE2_BASE * (0.98 ** (ep / 30.0))
        opt.set_lr(lr)

        idx = np.arange(N); np.random.shuffle(idx)
        total = 0.0

        for s in range(0, N, BATCH_SIZE):
            b = idx[s:s+BATCH_SIZE]
            cb = ctx_all[b]
            hl = h_last_all[b]
            hm = h_mean_all[b]
            yb = Y_CLS_ALL[b]
            pos = (yb > 0.5).astype(np.float32)

            tid = TRUE_ID[b]
            tw  = TRUE_W[b]

            logits_id = cb @ p["W_id"].T + p["b_id"]
            feat_w = np.concatenate([cb, hl, hm], axis=1)
            logits_w  = feat_w @ p["W_w"].T + p["b_w"]

            prob_id = softmax(logits_id, axis=1)
            prob_w  = softmax(logits_w, axis=1)

            claim = CLAIM_ID[b]
            expw  = EXPECT_W[b]
            rows = np.arange(len(b))

            p_id_claimed = prob_id[rows, claim]
            p_w_expected = prob_w[rows, expw]

            cid = claim / max(1, N_IDENTITIES-1)
            ew  = expw  / max(1, N_WINDOWS_PER_ID-1)

            vb_in = np.concatenate([
                cb,
                cid[:,None].astype(np.float32),
                ew[:,None].astype(np.float32),
                p_id_claimed[:,None].astype(np.float32),
                p_w_expected[:,None].astype(np.float32)
            ], axis=1)

            logits_v = np.squeeze(vb_in @ p["W_beh"].T + p["b_beh"])

            # Train ID/W only on positives
            loss_id, dlog_id = ce_loss_batch_masked(logits_id, tid, pos)
            loss_w,  dlog_w  = ce_loss_batch_masked(logits_w, tw, pos)

            # Validity on all
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

        if ep == 2 or ep % max(1, EPOCHS_PHASE2//10) == 0:
            print(f"[Phase2] Epoch {ep}/{EPOCHS_PHASE2} avg_loss={total/N:.6f} lr={lr:.5f}")

    return p

# ============================================================
# 9) Explainable Verification
# ============================================================
def verify_chain(p, tokens, meas, claimed_id, expected_w, true_ms=None):
    X = build_X_backbone(tokens, meas)
    Xp, Mp, _ = pad_to_T(X, tokens)

    ctx, h_last, h_mean = compute_embeddings(p, Xp[None,:,:], Mp[None,:])
    ctx1 = ctx[0]

    ms_hat, _ = ms_head(ctx, p)
    ms_hat = ms_hat[0]

    logits_id = (ctx @ p["W_id"].T + p["b_id"])[0]
    id_pred = int(np.argmax(logits_id))
    pid = float(softmax1d(logits_id)[claimed_id])

    feat_w = np.concatenate([ctx1, h_last[0], h_mean[0]], axis=0)[None,:]
    logits_w = (feat_w @ p["W_w"].T + p["b_w"])[0]
    w_pred = int(np.argmax(logits_w))
    pw = float(softmax1d(logits_w)[expected_w])

    cid = claimed_id / max(1, N_IDENTITIES-1)
    ew  = expected_w / max(1, N_WINDOWS_PER_ID-1)
    vb_in = np.concatenate([ctx1, np.array([cid, ew, pid, pw], dtype=np.float32)], axis=0)[None,:]
    logit_v = float(np.squeeze(vb_in @ p["W_beh"].T + p["b_beh"]))
    p_valid = float(sigmoid(logit_v))

    checks = {
        f"p_valid >= {THRESH_P_VALID}": (p_valid >= THRESH_P_VALID),
        "id_pred == claimed_id": (id_pred == claimed_id),
        "w_pred == expected_w": (w_pred == expected_w),
        f"pid >= {PID_MIN}": (pid >= PID_MIN),
        f"pw >= {PW_MIN}": (pw >= PW_MIN),
    }
    ok = all(checks.values())

    out = {
        "ok": ok, "p_valid": p_valid, "id_pred": id_pred, "w_pred": w_pred,
        "pid": pid, "pw": pw, "checks": checks, "ms_hat": ms_hat,
    }
    if true_ms is not None:
        out["l2_ms"] = l2(ms_hat, true_ms)
    return out

def print_case(title, r, claimed_id, expected_w):
    print(f"\n=== {title} ===")
    print(f"CLAIMED ENTITY:   PeerID={claimed_id}   ExpectedWindow={expected_w}")
    print(f"MODEL PREDICTION: id_pred={r['id_pred']}  w_pred={r['w_pred']}")
    print(f"SCORES: p_valid={r['p_valid']:.6f}  pid(claim)={r['pid']:.6f}  pw(expected)={r['pw']:.6f}")
    if "l2_ms" in r:
        print(f"RECON:  L2(MS_hat, MS_true)={r['l2_ms']:.6f}")
    print("THRESHOLD GATES (ALL must pass):")
    for k,v in r["checks"].items():
        print(f"  - {k:<20} -> {v}")
    print(f"FINAL DECISION: OK={r['ok']}")

def eval_demo(p, id_eval=0, w_eval=5):
    ms_eval = MS_all[id_eval]
    g_true = id_eval * N_WINDOWS_PER_ID + w_eval
    toks, meas = generate_os_chain(ms_eval, g_true)

    r = verify_chain(p, toks, meas, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print_case("LEGIT (intact chain)", r, id_eval, w_eval)

    idxs = np.arange(SEQ_LEN); np.random.shuffle(idxs)
    r = verify_chain(p, toks[idxs], meas[idxs], claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print_case("TAMPER: SHUFFLED (replay-ish)", r, id_eval, w_eval)

    Ltr = SEQ_LEN//2
    r = verify_chain(p, toks[:Ltr], meas[:Ltr], claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print_case("TAMPER: TRUNCATED (dropped steps)", r, id_eval, w_eval)

    wrong_w = (w_eval + 7) % N_WINDOWS_PER_ID
    g_wrong = id_eval * N_WINDOWS_PER_ID + wrong_w
    toks_w, meas_w = generate_os_chain(ms_eval, g_wrong)
    r = verify_chain(p, toks_w, meas_w, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print_case("TAMPER: WRONG WINDOW (counter drift)", r, id_eval, w_eval)

    other_id = (id_eval + 1) % N_IDENTITIES
    other_w  = np.random.randint(0, N_WINDOWS_PER_ID)
    g_other  = other_id * N_WINDOWS_PER_ID + other_w
    ms_other = MS_all[other_id]
    toks_o, meas_o = generate_os_chain(ms_other, g_other)
    r = verify_chain(p, toks_o, meas_o, claimed_id=id_eval, expected_w=w_eval, true_ms=ms_eval)
    print_case("TAMPER: WRONG IDENTITY (impersonation)", r, id_eval, w_eval)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W = build_dataset()
    N, T, D = X_ALL.shape
    print("All samples:", N, "X shape:", X_ALL.shape, "Positives:", int(np.sum(Y_CLS_ALL)))
    print(f"PN PILOT ENABLED: amp={PILOT_AMP}  chips={PILOT_PN_CHIPS}  (window-bound watermark)\n")

    p = init_model(D)
    p = train_phase1(p, X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL)
    p = train_phase2(p, X_ALL, M_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W)
    eval_demo(p, id_eval=0, w_eval=5)
