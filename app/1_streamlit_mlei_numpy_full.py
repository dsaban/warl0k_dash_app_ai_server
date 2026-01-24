# streamlit_mlei_numpy_full.py
# Run:
#   streamlit run streamlit_mlei_numpy_full.py
#
# Full end-to-end Streamlit wrapper for the NumPy-only WARL0K PIM/MLEI:
# - Dataset generation (pos + attack negatives)
# - Phase1 training (MS reconstruction + optional token scaffold)
# - Phase2 training (ID head + Window head + Behavioral validity head)
# - Interactive verification for legit + attacks, with dashboards
#
# Source adapted from the user's uploaded NumPy code.  :contentReference[oaicite:1]{index=1}

import time
import numpy as np
import streamlit as st

# ============================================================
# CONFIG (defaults match your file; UI lets you shorten epochs)
# ============================================================

VOCAB_SIZE = 16
MS_DIM     = 8
SEQ_LEN    = 20

N_IDENTITIES       = 2
N_WINDOWS_PER_ID   = 48

HIDDEN_DIM = 64
ATTN_DIM   = 32
MS_HID     = 32

BATCH_SIZE = 32

# Default "full" epochs from your file (UI can reduce for faster runs)
EPOCHS_PHASE1_DEFAULT = 200
EPOCHS_PHASE2_DEFAULT = 500

LR_PHASE1      = 0.006
LR_PHASE2_BASE = 0.03

CLIP_NORM     = 5.0
WEIGHT_DECAY  = 1e-4

# Phase1 losses
LAMBDA_MS         = 1.0
LAMBDA_TOK        = 0.10
TOK_STOP_EPS      = 0.25
TOK_WARMUP_EPOCHS = 60

# Phase2 losses
LAMBDA_ID   = 1.0
LAMBDA_W    = 1.0
LAMBDA_BCE  = 1.0
POS_WEIGHT  = 10.0

# Strong accept thresholds
THRESH_P_VALID = 0.80
PID_MIN        = 0.70
PW_MIN         = 0.70

# WINDOW PILOT
PILOT_AMP      = 0.55
PILOT_PN_CHIPS = SEQ_LEN


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

@st.cache_resource(show_spinner=False)
def init_world(seed: int = 0):
    np.random.seed(seed)
    MS_all = np.random.uniform(-1.0, 1.0, size=(N_IDENTITIES, MS_DIM)).astype(np.float32)
    A_base = (np.random.randn(SEQ_LEN, MS_DIM).astype(np.float32) * 0.8)
    return MS_all, A_base

def window_delta(window_global_id, t, ms_dim=MS_DIM):
    seed = (window_global_id * 10007 + t * 97) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    return (0.25 * rng.randn(ms_dim)).astype(np.float32)

def window_pilot(window_global_id, seq_len=SEQ_LEN, pilot_amp=PILOT_AMP):
    rng = np.random.RandomState((window_global_id * 9176 + 11) & 0xFFFFFFFF)
    chips = rng.randint(0, 2, size=seq_len).astype(np.float32)
    chips = 2.0 * chips - 1.0
    pilot = pilot_amp * chips
    pilot = pilot - pilot.mean()
    return pilot

def generate_os_chain(ms_vec, window_global_id, A_base, seq_len=SEQ_LEN, pilot_amp=PILOT_AMP):
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_global_id, t)
        zs[t] = float(a_t @ ms_vec)

    # Add PN pilot watermark
    zs = zs + window_pilot(window_global_id, seq_len, pilot_amp=pilot_amp)

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
# 3) DATASET (pos + multiple negatives)
# ============================================================

@st.cache_data(show_spinner=False)
def build_dataset_cached(seed: int, pilot_amp: float):
    np.random.seed(seed)
    MS_all, A_base = init_world(seed)

    Xs, Ms, Tok = [], [], []
    y_ms, y_cls = [], []
    true_id, true_w, claim_id, expect_w = [], [], [], []

    for id_true in range(N_IDENTITIES):
        ms_true = MS_all[id_true]
        for w_true in range(N_WINDOWS_PER_ID):
            g_true = id_true * N_WINDOWS_PER_ID + w_true
            toks, meas = generate_os_chain(ms_true, g_true, A_base, pilot_amp=pilot_amp)

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

            # NEG wrong-window (claim expects w_true)
            wrong_w = (w_true + 7) % N_WINDOWS_PER_ID
            g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w
            toks_w, meas_w = generate_os_chain(ms_true, g_wrong, A_base, pilot_amp=pilot_amp)
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
            toks_o, meas_o = generate_os_chain(ms_other, g_other, A_base, pilot_amp=pilot_amp)
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

def init_model(input_dim, seed=0):
    rng = np.random.RandomState(seed)
    p = {}
    s = 0.08

    # GRU
    p["W_z"] = rng.randn(HIDDEN_DIM, input_dim) * s
    p["U_z"] = rng.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_z"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    p["W_r"] = rng.randn(HIDDEN_DIM, input_dim) * s
    p["U_r"] = rng.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_r"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    p["W_h"] = rng.randn(HIDDEN_DIM, input_dim) * s
    p["U_h"] = rng.randn(HIDDEN_DIM, HIDDEN_DIM) * s
    p["b_h"] = np.zeros((HIDDEN_DIM,), dtype=np.float32)

    # Attention
    p["W_att"] = rng.randn(ATTN_DIM, HIDDEN_DIM) * s
    p["v_att"] = rng.randn(ATTN_DIM) * s

    # MS head
    p["W_ms1"] = rng.randn(MS_HID, HIDDEN_DIM) * s
    p["b_ms1"] = np.zeros((MS_HID,), dtype=np.float32)
    p["W_ms2"] = rng.randn(MS_DIM, MS_HID) * s
    p["b_ms2"] = np.zeros((MS_DIM,), dtype=np.float32)

    # Token scaffold
    p["W_tok"] = rng.randn(VOCAB_SIZE, HIDDEN_DIM) * s
    p["b_tok"] = np.zeros((VOCAB_SIZE,), dtype=np.float32)

    # Heads
    p["W_id"]  = rng.randn(N_IDENTITIES, HIDDEN_DIM) * s
    p["b_id"]  = np.zeros((N_IDENTITIES,), dtype=np.float32)

    p["W_w"]   = rng.randn(N_WINDOWS_PER_ID, 3*HIDDEN_DIM) * s
    p["b_w"]   = np.zeros((N_WINDOWS_PER_ID,), dtype=np.float32)

    p["W_beh"] = rng.randn(1, HIDDEN_DIM + 4) * s
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

def train_phase1(p, X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, epochs, log_cb=None):
    opt = Adam(p, lr=LR_PHASE1)
    tok_enabled = True
    N = X_ALL.shape[0]
    losses = []

    for ep in range(1, epochs+1):
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
                denom = float(np.sum((Mb[:, :-1] * Mb[:, 1:]) * (ycls[:, None] > 0.5)) + 1e-6)

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

        avg = total / N
        losses.append(avg)

        if log_cb and (ep == 1 or ep == 2 or ep % max(1, epochs//10) == 0):
            log_cb(f"[Phase1] Epoch {ep}/{epochs} avg_loss={avg:.6f} tok_enabled={tok_enabled}")

    return p, losses


# ============================================================
# 7) Embeddings
# ============================================================

def compute_embeddings(p, X, M):
    hs, _ = gru_forward_batch(X, M, p)
    ctx, alphas, _ = attention_forward_batch(hs, M, p)
    B, T, H = hs.shape
    denom = np.sum(M, axis=1, keepdims=True) + 1e-6
    h_mean = np.sum(hs * M[:,:,None], axis=1) / denom
    last_idx = np.maximum(0, np.sum(M, axis=1).astype(np.int32) - 1)
    h_last = np.zeros((B,H), dtype=np.float32)
    for i in range(B):
        h_last[i] = hs[i, last_idx[i], :]
    return ctx, h_last, h_mean, alphas


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

def train_phase2(p, X_ALL, M_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W, epochs, log_cb=None):
    ctx_all, h_last_all, h_mean_all, _ = compute_embeddings(p, X_ALL, M_ALL)
    opt = AdamLite(p, lr=LR_PHASE2_BASE)

    trainable = {"W_id","b_id","W_w","b_w","W_beh","b_beh"}
    freeze = set([k for k in p.keys() if k not in trainable])

    N = X_ALL.shape[0]
    losses = []

    for ep in range(1, epochs+1):
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

            loss_id, dlog_id = ce_loss_batch_masked(logits_id, tid, pos)
            loss_w,  dlog_w  = ce_loss_batch_masked(logits_w, tw, pos)
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

        avg = total / N
        losses.append(avg)

        if log_cb and (ep == 1 or ep == 2 or ep % max(1, epochs//10) == 0):
            log_cb(f"[Phase2] Epoch {ep}/{epochs} avg_loss={avg:.6f} lr={lr:.5f}")

    return p, losses


# ============================================================
# 9) Explainable Verification (gated decision)
# ============================================================

def verify_chain(p, tokens, meas, claimed_id, expected_w, true_ms=None):
    X = build_X_backbone(tokens, meas)
    Xp, Mp, _ = pad_to_T(X, tokens)

    ctx, h_last, h_mean, alphas = compute_embeddings(p, Xp[None,:,:], Mp[None,:])
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
        "ok": ok,
        "p_valid": p_valid,
        "id_pred": id_pred,
        "w_pred": w_pred,
        "pid": pid,
        "pw": pw,
        "checks": checks,
        "ms_hat": ms_hat,
        "alphas": alphas[0],  # attention weights for the single sample
    }
    if true_ms is not None:
        out["l2_ms"] = l2(ms_hat, true_ms)
    return out


# ============================================================
# Attacks for interactive verification (same spirit as your eval_demo)
# ============================================================

def attack_shuffled(tokens, meas, seed=0):
    rng = np.random.RandomState(seed)
    idxs = np.arange(len(tokens))
    rng.shuffle(idxs)
    return tokens[idxs], meas[idxs]

def attack_truncated(tokens, meas):
    Ltr = len(tokens)//2
    return tokens[:Ltr], meas[:Ltr]


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="WARL0K MLEI NumPy Nano-Model (Full)", layout="wide")

st.title("WARL0K MLEI — NumPy Nano-Model (GRU+Attention) Full Pipeline")
st.caption("End-to-end: dataset → Phase1 → Phase2 → verify (legit + attacks) with gated decision + dashboards. :contentReference[oaicite:2]{index=2}")

with st.sidebar:
    st.subheader("Experiment")
    seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1)
    pilot_amp = st.slider("PN Pilot amplitude", 0.10, 0.90, float(PILOT_AMP), 0.05)

    st.divider()
    st.subheader("Training (can be full or fast)")
    epochs1 = st.slider("Phase1 epochs", 1, EPOCHS_PHASE1_DEFAULT, EPOCHS_PHASE1_DEFAULT, 1)
    epochs2 = st.slider("Phase2 epochs", 1, EPOCHS_PHASE2_DEFAULT, EPOCHS_PHASE2_DEFAULT, 1)

    st.caption("Tip: if you want quick iterations, set Phase1≈30 and Phase2≈60.")
    show_attention = st.checkbox("Show attention weights", value=True)

    run_train = st.button("Train model", type="primary")

# --- Load dataset
X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W = build_dataset_cached(seed, pilot_amp)
N, T, D = X_ALL.shape

top = st.columns([1.2, 1.2, 1.6, 1.0])
with top[0]:
    st.metric("Samples", N)
with top[1]:
    st.metric("Positives", int(np.sum(Y_CLS_ALL)))
with top[2]:
    st.write("PN pilot:", f"amp={pilot_amp:.2f}, chips={PILOT_PN_CHIPS}, seq_len={SEQ_LEN}")
with top[3]:
    st.write("Shapes:", f"X={X_ALL.shape}, M={M_ALL.shape}")

st.divider()

# --- Train or reuse from session
if "trained" not in st.session_state:
    st.session_state.trained = False
    st.session_state.p = None
    st.session_state.loss1 = []
    st.session_state.loss2 = []
    st.session_state.logs = []

def log_append(msg: str):
    st.session_state.logs.append(msg)

if run_train:
    st.session_state.trained = False
    st.session_state.logs = []
    st.session_state.loss1 = []
    st.session_state.loss2 = []

    log_box = st.empty()
    prog = st.progress(0.0)

    p = init_model(D, seed=seed)

    t0 = time.time()
    # Phase1
    def cb1(m):
        log_append(m)
        log_box.code("\n".join(st.session_state.logs[-12:]))
    p, loss1 = train_phase1(p, X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, epochs=epochs1, log_cb=cb1)
    st.session_state.loss1 = loss1
    prog.progress(0.45)

    # Phase2
    def cb2(m):
        log_append(m)
        log_box.code("\n".join(st.session_state.logs[-12:]))
    p, loss2 = train_phase2(p, X_ALL, M_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W, epochs=epochs2, log_cb=cb2)
    st.session_state.loss2 = loss2
    prog.progress(1.0)

    st.session_state.p = p
    st.session_state.trained = True

    dt = time.time() - t0
    st.success(f"Training complete in {dt:.2f}s")

# --- Show training panel
left, right = st.columns([1.25, 1.75])

with left:
    st.markdown("## Process (Start → Train → Verify)")
    st.markdown(
        """
**1) Dataset**
- Generates chains per (identity, window) with PN pilot watermark.
- Adds **negatives**: shuffled / truncated / wrong-window / wrong-identity.

**2) Phase1 (Reconstruction)**
- GRU+Attention produces a context vector.
- MS head reconstructs the master secret **MS** (trained mainly on positives).
- Optional next-token scaffold improves sequence understanding early.

**3) Phase2 (MLEI gates)**
- **Identity head**: predicts identity from context.
- **Window head**: predicts window from (ctx, last, mean).
- **Validity head**: outputs p_valid using (ctx + claim features + pid + pw).

**4) Verification (Explainable)**
- Final OK only if all gates pass: p_valid, id match, window match, pid>=min, pw>=min.
"""
    )

with right:
    st.markdown("## Training dashboard")
    if st.session_state.loss1:
        st.write("Phase1 loss curve")
        st.line_chart(st.session_state.loss1)
    if st.session_state.loss2:
        st.write("Phase2 loss curve")
        st.line_chart(st.session_state.loss2)
    if st.session_state.logs:
        st.write("Training logs (latest)")
        st.code("\n".join(st.session_state.logs[-16:]))

st.divider()

# --- Verification UI
st.markdown("## Verification + Attacks (B-side validation)")
if not st.session_state.trained:
    st.info("Click **Train model** in the sidebar to enable verification.")
    st.stop()

p = st.session_state.p
MS_all, A_base = init_world(seed)

v1, v2, v3, v4 = st.columns([1.0, 1.0, 1.4, 1.2])
with v1:
    claimed_id = st.selectbox("Claimed identity", list(range(N_IDENTITIES)), index=0)
with v2:
    expected_w = st.selectbox("Expected window", list(range(N_WINDOWS_PER_ID)), index=5)
with v3:
    scenario = st.selectbox(
        "Scenario",
        ["LEGIT (intact chain)", "ATTACK: SHUFFLED (reorder/replay-ish)", "ATTACK: TRUNCATED (drop steps)",
         "ATTACK: WRONG WINDOW (drift)", "ATTACK: WRONG IDENTITY (impersonation)"],
        index=0
    )
with v4:
    attack_seed = st.number_input("Attack seed (shuffle)", min_value=0, max_value=10_000, value=0, step=1)

# Build the base chain
ms_true = MS_all[claimed_id]
g_true = claimed_id * N_WINDOWS_PER_ID + expected_w
toks, meas = generate_os_chain(ms_true, g_true, A_base, pilot_amp=pilot_amp)

# Apply scenarios
true_ms_for_l2 = ms_true
true_id_for_label = claimed_id
true_w_for_label = expected_w

if scenario.startswith("ATTACK: SHUFFLED"):
    toks2, meas2 = attack_shuffled(toks, meas, seed=int(attack_seed))
elif scenario.startswith("ATTACK: TRUNCATED"):
    toks2, meas2 = attack_truncated(toks, meas)
elif scenario.startswith("ATTACK: WRONG WINDOW"):
    wrong_w = (expected_w + 7) % N_WINDOWS_PER_ID
    g_wrong = claimed_id * N_WINDOWS_PER_ID + wrong_w
    toks2, meas2 = generate_os_chain(ms_true, g_wrong, A_base, pilot_amp=pilot_amp)
    true_w_for_label = wrong_w
elif scenario.startswith("ATTACK: WRONG IDENTITY"):
    other_id = (claimed_id + 1) % N_IDENTITIES
    other_w  = int((expected_w + 13) % N_WINDOWS_PER_ID)
    ms_other = MS_all[other_id]
    g_other  = other_id * N_WINDOWS_PER_ID + other_w
    toks2, meas2 = generate_os_chain(ms_other, g_other, A_base, pilot_amp=pilot_amp)
    true_id_for_label = other_id
    true_w_for_label = other_w
else:
    toks2, meas2 = toks, meas

# Verify
r = verify_chain(p, toks2, meas2, claimed_id=claimed_id, expected_w=expected_w, true_ms=true_ms_for_l2)

# --- Results cards
cA, cB, cC, cD = st.columns([1.1, 1.1, 1.1, 1.7])
with cA:
    st.metric("p_valid", f"{r['p_valid']:.4f}")
with cB:
    st.metric("pid (claim prob)", f"{r['pid']:.4f}")
with cC:
    st.metric("pw (expected window prob)", f"{r['pw']:.4f}")
with cD:
    st.metric("Decision", "OK ✅" if r["ok"] else "REJECT ❌")

# --- Gate breakdown
st.markdown("### Gate-by-gate validation (must ALL pass)")
gate_rows = []
for k, v in r["checks"].items():
    gate_rows.append({"gate": k, "pass": bool(v)})

gate_rows.append({"gate": f"l2(MS_hat, MS_true) (info)", "pass": True})
st.dataframe(gate_rows, use_container_width=True, height=220)

# --- Predictions vs truth
st.markdown("### Predicted vs actual (context)")
m1, m2, m3 = st.columns([1.0, 1.0, 2.0])
with m1:
    st.write("Claim:", f"id={claimed_id}, expected_w={expected_w}")
with m2:
    st.write("True source:", f"id={true_id_for_label}, window={true_w_for_label}")
with m3:
    st.write("Model preds:", f"id_pred={r['id_pred']}, w_pred={r['w_pred']}, L2={r.get('l2_ms', 0.0):.4f}")

# --- Signal visuals: tokens + meas
st.divider()
st.markdown("## Chain visuals")
p1, p2 = st.columns(2)
with p1:
    st.markdown("### Tokens (quantized)")
    st.line_chart([int(x) for x in toks2])
with p2:
    st.markdown("### Measurements (normalized m[t])")
    st.line_chart([float(x) for x in meas2])

# --- Attention weights (optional)
if show_attention:
    st.divider()
    st.markdown("## Attention weights (where the model “looked”)")
    att = r["alphas"]
    # pad to SEQ_LEN for consistent display
    if att.shape[0] < SEQ_LEN:
        att = np.concatenate([att, np.zeros((SEQ_LEN - att.shape[0],), dtype=np.float32)])
    st.line_chart([float(x) for x in att])

# --- Attack explanation panel
st.divider()
st.markdown("## What each attack represents (MLEI interpretation)")
st.markdown(
    """
- **LEGIT**: intact chain from the claimed identity + expected window.
- **SHUFFLED**: steps reordered → continuity broken (replay-ish / reorder).
- **TRUNCATED**: missing steps → drop / loss / truncated proof window.
- **WRONG WINDOW**: chain belongs to a different window than the claim expects → drift / counter-window mismatch.
- **WRONG IDENTITY**: chain originates from a different identity → impersonation / stolen-session-but-wrong-runtime.

The model rejects when the combined gates fail: `p_valid`, identity consistency, window consistency, and confidence (`pid`, `pw`).
"""
)
