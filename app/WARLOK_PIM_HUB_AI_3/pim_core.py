"""
WARL0K PIM Core Engine — Python/NumPy port of the C++ GRU+Attention system
Adds: AES-256-GCM encryption layer, HMAC chain-proof validation, SHA3-256 state hashing
"""

import numpy as np
import hashlib
import hmac
import struct
import time
import json
import os
from typing import Optional, Tuple, List, Dict, Any

# ── Try fast crypto; fall back to pure-Python alternatives ──────────────────
try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes as crypto_hashes
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# ═══════════════════════════════════════════════════════════════════════════
# CONFIG  (matches final C++ constants)
# ═══════════════════════════════════════════════════════════════════════════
CFG = dict(
    VOCAB_SIZE       = 16,
    MS_DIM           = 8,
    SEQ_LEN          = 20,
    N_IDENTITIES     = 2,
    N_WINDOWS_PER_ID = 48,
    HIDDEN_DIM       = 64,
    ATTN_DIM         = 32,
    MS_HID           = 32,
    BATCH_SIZE       = 32,
    EPOCHS_PHASE1    = 90,
    EPOCHS_PHASE2    = 100,
    LR_PHASE1        = 0.006,
    LR_PHASE2_BASE   = 0.008,
    CLIP_NORM        = 5.0,
    WEIGHT_DECAY     = 1e-4,
    LAMBDA_MS        = 1.0,
    LAMBDA_TOK       = 0.10,
    TOK_STOP_EPS     = 0.25,
    TOK_WARMUP_EPOCHS= 60,
    LAMBDA_ID        = 1.0,
    LAMBDA_W         = 1.0,
    LAMBDA_BCE       = 1.0,
    POS_WEIGHT       = 10.0,
    THRESH_P_VALID   = 0.80,
    PID_MIN          = 0.70,
    PW_MIN           = 0.40,
    PILOT_AMP        = 0.55,
    PILOT_CORR_MIN   = 0.02,
)
CFG['INPUT_DIM'] = CFG['VOCAB_SIZE'] + 2

# ═══════════════════════════════════════════════════════════════════════════
# XorShift32 — deterministic PRNG matching C++
# ═══════════════════════════════════════════════════════════════════════════
class XorShift32:
    def __init__(self, seed: int = 0x12345678):
        self.s = int(seed) & 0xFFFFFFFF
        if self.s == 0:
            self.s = 0x12345678

    def next_u32(self) -> int:
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5)  & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s

    def next_f01(self) -> float:
        return (self.next_u32() >> 8) * (1.0 / 16777216.0)

    def next_int(self, lo: int, hi: int) -> int:
        return lo + int(self.next_u32() % (hi - lo))

    def next_norm(self) -> float:
        u1 = max(1e-7, self.next_f01())
        u2 = self.next_f01()
        r = np.sqrt(-2.0 * np.log(u1))
        t = 2.0 * np.pi * u2
        return float(r * np.cos(t))

    def shuffle(self, v: list) -> list:
        v = list(v)
        for i in range(len(v) - 1, 0, -1):
            j = self.next_int(0, i + 1)
            v[i], v[j] = v[j], v[i]
        return v

# ═══════════════════════════════════════════════════════════════════════════
# Activation helpers
# ═══════════════════════════════════════════════════════════════════════════
def sigmoid(x): return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

# ═══════════════════════════════════════════════════════════════════════════
# Global deterministic data (MS_all, A_base)
# ═══════════════════════════════════════════════════════════════════════════
def _init_globals():
    rng = XorShift32(0xDEADBEEF)
    V = CFG['N_IDENTITIES']; K = CFG['MS_DIM']
    T = CFG['SEQ_LEN']
    MS_all = np.zeros((V, K), dtype=np.float32)
    for i in range(V):
        for k in range(K):
            MS_all[i, k] = 2.0 * rng.next_f01() - 1.0
    A_base = np.zeros((T, K), dtype=np.float32)
    for t in range(T):
        for k in range(K):
            A_base[t, k] = 0.8 * rng.next_norm()
    return MS_all, A_base

MS_ALL, A_BASE = _init_globals()

# ═══════════════════════════════════════════════════════════════════════════
# PN pilot & OS chain generation  (deterministic, matches C++)
# ═══════════════════════════════════════════════════════════════════════════
def window_delta(g: int, t: int) -> np.ndarray:
    seed = ((g * 10007 + t * 97) & 0xFFFFFFFF)
    rng = XorShift32(seed or 0xA5A5A5A5)
    return np.array([0.25 * rng.next_norm() for _ in range(CFG['MS_DIM'])], dtype=np.float32)

def window_pilot(g: int) -> np.ndarray:
    seed = ((g * 9176 + 11) & 0xFFFFFFFF)
    rng = XorShift32(seed or 0xBEEFBEEF)
    T = CFG['SEQ_LEN']
    pilot = np.array([(rng.next_int(0,2)*2-1) * CFG['PILOT_AMP'] for _ in range(T)], dtype=np.float32)
    return pilot - pilot.mean()

def pilot_corr(meas: np.ndarray, g: int) -> float:
    p = window_pilot(g)
    num = float(meas @ p)
    denom = float(np.sqrt((meas**2).sum() * (p**2).sum())) + 1e-9
    return num / denom

def generate_os_chain(ms: np.ndarray, g: int) -> Tuple[np.ndarray, np.ndarray]:
    T, K = CFG['SEQ_LEN'], CFG['MS_DIM']
    zs = np.zeros(T, dtype=np.float32)
    for t in range(T):
        d = window_delta(g, t)
        a = A_BASE[t] + d
        zs[t] = float(a @ ms)
    zs += window_pilot(g)
    ms_sum = int(ms.sum() * 1000)
    seed = ((g * 1337 + ms_sum) & 0xFFFFFFFF)
    rng = XorShift32(seed or 0xCAFE1234)
    noise = np.array([0.02 * rng.next_norm() for _ in range(T)], dtype=np.float32)
    zs += noise
    mu, st = zs.mean(), zs.std() + 1e-6
    meas = (zs - mu) / st
    scaled = np.clip((meas + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * CFG['VOCAB_SIZE']).astype(np.int32)
    tokens = np.clip(tokens, 0, CFG['VOCAB_SIZE'] - 1)
    return tokens, meas

def build_X(tokens: np.ndarray, meas: np.ndarray) -> np.ndarray:
    T, D = CFG['SEQ_LEN'], CFG['INPUT_DIM']
    V = CFG['VOCAB_SIZE']
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        X[t, tokens[t]] = 1.0
        X[t, V] = meas[t]
        X[t, V+1] = t / (T-1) if T > 1 else 0.0
    return X

# ═══════════════════════════════════════════════════════════════════════════
# Model parameters  (flat NumPy arrays, Adam state bundled)
# ═══════════════════════════════════════════════════════════════════════════
class Params:
    __slots__ = ['W_z','U_z','b_z','W_r','U_r','b_r','W_h','U_h','b_h',
                 'W_att','v_att','W_ms1','b_ms1','W_ms2','b_ms2',
                 'W_tok','b_tok','W_id','b_id','W_w','b_w','W_beh','b_beh']

    def arrays(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in self.__slots__}

    def param_bytes(self) -> int:
        return sum(a.nbytes for a in self.arrays().values())

def init_params(seed: int = 0xC0FFEE) -> Params:
    rng = XorShift32(seed)
    H  = CFG['HIDDEN_DIM']; D = CFG['INPUT_DIM']
    AH = CFG['ATTN_DIM']; MH = CFG['MS_HID']
    MS = CFG['MS_DIM']; V  = CFG['VOCAB_SIZE']
    NI = CFG['N_IDENTITIES']; NW = CFG['N_WINDOWS_PER_ID']

    def rm(r, c, s=0.08):
        return np.array([s * rng.next_norm() for _ in range(r*c)], dtype=np.float32).reshape(r, c)
    def zv(n): return np.zeros(n, dtype=np.float32)

    p = Params()
    p.W_z = rm(H,D); p.U_z = rm(H,H); p.b_z = zv(H)
    p.W_r = rm(H,D); p.U_r = rm(H,H); p.b_r = zv(H)
    p.W_h = rm(H,D); p.U_h = rm(H,H); p.b_h = zv(H)
    p.W_att = rm(AH,H)
    p.v_att = np.array([0.08 * rng.next_norm() for _ in range(AH)], dtype=np.float32)
    p.W_ms1 = rm(MH,H); p.b_ms1 = zv(MH)
    p.W_ms2 = rm(MS,MH); p.b_ms2 = zv(MS)
    p.W_tok = rm(V,H);  p.b_tok = zv(V)
    p.W_id  = rm(NI,H); p.b_id  = zv(NI)
    p.W_w   = rm(NW,3*H); p.b_w  = zv(NW)
    p.W_beh = rm(1,H+4);  p.b_beh= zv(1)
    return p

def zeros_like(p: Params) -> Params:
    g = Params()
    for k in p.__slots__:
        setattr(g, k, np.zeros_like(getattr(p, k)))
    return g

# ═══════════════════════════════════════════════════════════════════════════
# Adam optimiser
# ═══════════════════════════════════════════════════════════════════════════
class Adam:
    def __init__(self, p: Params, lr: float, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps; self.t = 0
        self.m = zeros_like(p); self.v = zeros_like(p)

    def step(self, p: Params, g: Params, wd: float, freeze: set):
        self.t += 1
        b1t = self.b1 ** self.t; b2t = self.b2 ** self.t
        for k in p.__slots__:
            if k in freeze: continue
            gr = getattr(g, k).copy()
            is_bias = k.startswith('b_') or k == 'v_att'
            if wd > 0 and not is_bias:
                gr += wd * getattr(p, k)
            m = getattr(self.m, k); v = getattr(self.v, k)
            m[:] = self.b1 * m + (1-self.b1) * gr
            v[:] = self.b2 * v + (1-self.b2) * gr**2
            mh = m / (1 - b1t); vh = v / (1 - b2t)
            getattr(p, k)[:] -= self.lr * mh / (np.sqrt(vh) + self.eps)

# ═══════════════════════════════════════════════════════════════════════════
# GRU forward (batch)
# ═══════════════════════════════════════════════════════════════════════════
def gru_forward(p: Params, X: np.ndarray, M: np.ndarray):
    """
    X: [B, T, D]  M: [B, T]
    Returns H[B,T,H], cache dict
    """
    B, T, D = X.shape; H = CFG['HIDDEN_DIM']
    Hout = np.zeros((B, T, H), dtype=np.float32)
    Z = np.zeros_like(Hout); R = np.zeros_like(Hout); HT = np.zeros_like(Hout)
    h = np.zeros((B, H), dtype=np.float32)
    for t in range(T):
        x = X[:, t, :]               # [B, D]
        az = x @ p.W_z.T + h @ p.U_z.T + p.b_z
        ar = x @ p.W_r.T + h @ p.U_r.T + p.b_r
        z = sigmoid(az); r = sigmoid(ar)
        ah = x @ p.W_h.T + (r * h) @ p.U_h.T + p.b_h
        ht = np.tanh(ah)
        hn = (1-z)*h + z*ht
        mt = M[:, t:t+1]
        hn = mt * hn + (1-mt) * h
        Hout[:, t] = hn; Z[:, t] = z; R[:, t] = r; HT[:, t] = ht
        h = hn
    cache = dict(X=X, M=M, H=Hout, Z=Z, R=R, HT=HT)
    return Hout, cache

def gru_backward(p: Params, cache: dict, dH: np.ndarray):
    X, M, H, Z, R, HT = cache['X'], cache['M'], cache['H'], cache['Z'], cache['R'], cache['HT']
    B, T, Hd = H.shape; D = X.shape[2]
    gp = zeros_like(p)
    dh_next = np.zeros((B, Hd), dtype=np.float32)
    for t in reversed(range(T)):
        mt = M[:, t:t+1]
        hprev = H[:, t-1] if t > 0 else np.zeros((B, Hd), dtype=np.float32)
        z = Z[:, t]; r = R[:, t]; ht_val = HT[:, t]
        x = X[:, t]
        dh = (dh_next + dH[:, t]) * mt
        dht = dh * z
        dz_raw = dh * (ht_val - hprev)
        dh_prev = dh * (1-z)
        da_h = dht * (1 - ht_val**2)
        gp.b_h += da_h.sum(0)
        gp.W_h += da_h.T @ x
        gp.U_h += da_h.T @ (r * hprev)
        tmp = da_h @ p.U_h        # [B, H]
        dh_prev += tmp * r
        dr_raw = tmp * hprev
        da_r = dr_raw * r * (1-r)
        gp.b_r += da_r.sum(0)
        gp.W_r += da_r.T @ x
        gp.U_r += da_r.T @ hprev
        dh_prev += da_r @ p.U_r
        da_z = dz_raw * z * (1-z)
        gp.b_z += da_z.sum(0)
        gp.W_z += da_z.T @ x
        gp.U_z += da_z.T @ hprev
        dh_prev += da_z @ p.U_z
        dh_next = dh_prev
    return gp

# ═══════════════════════════════════════════════════════════════════════════
# Attention forward + backward
# ═══════════════════════════════════════════════════════════════════════════
def attention_forward(p: Params, H: np.ndarray, M: np.ndarray):
    B, T, Hd = H.shape
    U = np.tanh(H @ p.W_att.T)         # [B,T,ATTN]
    scores = U @ p.v_att                 # [B,T]
    scores = np.where(M > 0, scores, -1e30)
    alpha = softmax(scores)              # [B,T]
    ctx = (alpha[:, :, None] * H).sum(1) # [B,H]
    cache = dict(H=H, M=M, U=U, alpha=alpha, scores=scores)
    return ctx, cache

def attention_backward(p: Params, cache: dict, dctx: np.ndarray):
    H, M, U, alpha = cache['H'], cache['M'], cache['U'], cache['alpha']
    B, T, Hd = H.shape
    gp = zeros_like(p)
    # dctx -> dalpha, dH_from_ctx
    dH = alpha[:, :, None] * dctx[:, None, :]     # [B,T,H]
    d_alpha = (dctx[:, None, :] * H).sum(-1)       # [B,T]
    # softmax backward
    s_term = (alpha * d_alpha).sum(-1, keepdims=True)
    dscores = alpha * (d_alpha - s_term) * (M > 0)  # [B,T]
    gp.v_att += (dscores[:, :, None] * U).sum((0,1))
    dU = dscores[:, :, None] * p.v_att[None, None, :]
    da_pre = dU * (1 - U**2)                        # [B,T,ATTN]
    gp.W_att += da_pre.reshape(B*T, -1).T @ H.reshape(B*T, Hd)
    dH += da_pre @ p.W_att                           # [B,T,H]
    return gp, dH

# ═══════════════════════════════════════════════════════════════════════════
# MS head forward + backward
# ═══════════════════════════════════════════════════════════════════════════
def ms_head_forward(p: Params, ctx: np.ndarray):
    hid = np.tanh(ctx @ p.W_ms1.T + p.b_ms1)    # [B, MS_HID]
    out = hid @ p.W_ms2.T + p.b_ms2              # [B, MS_DIM]
    return out, hid

def ms_head_backward(p: Params, ctx: np.ndarray, hid: np.ndarray, dout: np.ndarray):
    gp = zeros_like(p)
    gp.b_ms2 += dout.sum(0)
    gp.W_ms2 += dout.T @ hid
    dhid = dout @ p.W_ms2
    dpre = dhid * (1 - hid**2)
    gp.b_ms1 += dpre.sum(0)
    gp.W_ms1 += dpre.T @ ctx
    dctx = dpre @ p.W_ms1
    return gp, dctx

# ═══════════════════════════════════════════════════════════════════════════
# Gradient clipping
# ═══════════════════════════════════════════════════════════════════════════
def global_norm(g: Params) -> float:
    return float(np.sqrt(sum((getattr(g,k)**2).sum() for k in g.__slots__)))

def clip_grads(g: Params, max_norm: float):
    n = global_norm(g)
    if n > max_norm:
        s = max_norm / (n + 1e-12)
        for k in g.__slots__:
            getattr(g, k)[:] *= s

def add_grads(dst: Params, src: Params, scale=1.0):
    for k in dst.__slots__:
        getattr(dst, k)[:] += scale * getattr(src, k)

# ═══════════════════════════════════════════════════════════════════════════
# Dataset builder
# ═══════════════════════════════════════════════════════════════════════════
def build_dataset():
    NI, NW = CFG['N_IDENTITIES'], CFG['N_WINDOWS_PER_ID']
    T, D, MS = CFG['SEQ_LEN'], CFG['INPUT_DIM'], CFG['MS_DIM']
    spp = 5
    N = NI * NW * spp
    ds = dict(
        X=np.zeros((N,T,D), np.float32), M=np.ones((N,T), np.float32),
        TOK=np.zeros((N,T), np.int32), Y_MS=np.zeros((N,MS), np.float32),
        Y_CLS=np.zeros(N, np.float32),
        TRUE_ID=np.zeros(N,np.int32), TRUE_W=np.zeros(N,np.int32),
        CLAIM_ID=np.zeros(N,np.int32), EXPECT_W=np.zeros(N,np.int32),
    )
    rng = XorShift32(0xBADC0DE)
    n = 0

    def push(toks, meas, ycls, true_id, true_w, claim_id, expect_w, yms, mask=None):
        nonlocal n
        ds['X'][n] = build_X(toks, meas)
        ds['M'][n] = mask if mask is not None else np.ones(T, np.float32)
        ds['TOK'][n] = toks
        ds['Y_MS'][n] = yms
        ds['Y_CLS'][n] = ycls
        ds['TRUE_ID'][n] = true_id; ds['TRUE_W'][n] = true_w
        ds['CLAIM_ID'][n] = claim_id; ds['EXPECT_W'][n] = expect_w
        n += 1

    for id_t in range(NI):
        ms_true = MS_ALL[id_t]
        for w_t in range(NW):
            g_t = id_t * NW + w_t
            toks, meas = generate_os_chain(ms_true, g_t)
            push(toks, meas, 1.0, id_t, w_t, id_t, w_t, ms_true)

            # NEG shuffled
            rsh = XorShift32(0x1234 + g_t)
            idx = rsh.shuffle(list(range(T)))
            push(toks[idx], meas[idx], 0.0, id_t, w_t, id_t, w_t, ms_true)

            # NEG truncated
            Ltr = T // 2
            toks2 = np.zeros(T, np.int32); meas2 = np.zeros(T, np.float32)
            toks2[:Ltr] = toks[:Ltr]; meas2[:Ltr] = meas[:Ltr]
            mask2 = np.zeros(T, np.float32); mask2[:Ltr] = 1.0
            push(toks2, meas2, 0.0, id_t, w_t, id_t, w_t, ms_true, mask2)

            # NEG wrong window
            ww = (w_t + 7) % NW
            tw, mw = generate_os_chain(ms_true, id_t*NW+ww)
            push(tw, mw, 0.0, id_t, ww, id_t, w_t, ms_true)

            # NEG wrong identity
            oid = (id_t + rng.next_int(1, NI)) % NI
            ow  = rng.next_int(0, NW)
            to, mo = generate_os_chain(MS_ALL[oid], oid*NW+ow)
            push(to, mo, 0.0, oid, ow, id_t, w_t, ms_true)

    return ds

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1 training: MS reconstruction + token scaffold
# ═══════════════════════════════════════════════════════════════════════════
def train_phase1(p: Params, ds: dict, cb=None) -> Params:
    N = ds['X'].shape[0]; BS = CFG['BATCH_SIZE']
    opt = Adam(p, CFG['LR_PHASE1'])
    tok_enabled = True
    idx = np.arange(N)

    for ep in range(1, CFG['EPOCHS_PHASE1']+1):
        rsh = XorShift32(0x1000 + ep)
        idx = np.array(rsh.shuffle(list(idx)))
        ep_loss = 0.0

        for s in range(0, N, BS):
            bi = idx[s:s+BS]
            Xb = ds['X'][bi]; Mb = ds['M'][bi]; Tb = ds['TOK'][bi]
            yms = ds['Y_MS'][bi]; ycls = ds['Y_CLS'][bi]
            B = len(bi)

            H, gc = gru_forward(p, Xb, Mb)
            ctx, ac = attention_forward(p, H, Mb)
            ms_hat, ms_hid = ms_head_forward(p, ctx)

            pos = (ycls > 0.5).astype(np.float32)
            pc = pos.sum() + 1e-6
            diff = (ms_hat - yms) * pos[:, None]
            loss_ms = float(0.5*(diff**2).sum() / (pc * CFG['MS_DIM']))

            # token scaffold
            loss_tok = 0.0
            dH_tok = np.zeros_like(H)
            gW_tok = np.zeros_like(p.W_tok); gb_tok = np.zeros_like(p.b_tok)

            if tok_enabled and ep <= CFG['TOK_WARMUP_EPOCHS']:
                denom = 0.0
                for i in range(B):
                    if ycls[i] <= 0.5: continue
                    for t in range(CFG['SEQ_LEN']-1):
                        if Mb[i,t]>0.5 and Mb[i,t+1]>0.5: denom += 1.0
                denom += 1e-6

                for i in range(B):
                    if ycls[i] <= 0.5: continue
                    for t in range(CFG['SEQ_LEN']-1):
                        if not (Mb[i,t]>0.5 and Mb[i,t+1]>0.5): continue
                        logits = H[i,t] @ p.W_tok.T + p.b_tok
                        probs = softmax(logits)
                        tgt = int(Tb[i,t+1])
                        loss_tok += -np.log(probs[tgt]+1e-12)
                        dl = probs.copy(); dl[tgt] -= 1.0
                        gb_tok += dl
                        gW_tok += np.outer(dl, H[i,t])
                        dH_tok[i,t] += dl @ p.W_tok
                loss_tok /= denom
                gW_tok /= denom; gb_tok /= denom; dH_tok /= denom
                if loss_tok < CFG['TOK_STOP_EPS']: tok_enabled = False

            loss = CFG['LAMBDA_MS']*loss_ms + (CFG['LAMBDA_TOK']*loss_tok if tok_enabled else 0.0)
            ep_loss += loss * B

            # backward MS
            dms = diff / (pc * CFG['MS_DIM'])
            gms, gctx = ms_head_backward(p, ctx, ms_hid, dms)
            gatt, dH = attention_backward(p, ac, gctx)
            dH += dH_tok
            ggru = gru_backward(p, gc, dH)

            g_total = zeros_like(p)
            for k in p.__slots__:
                getattr(g_total, k)[:] = (getattr(gms, k) + getattr(gatt, k) + getattr(ggru, k))
            g_total.W_tok[:] += gW_tok; g_total.b_tok[:] += gb_tok

            clip_grads(g_total, CFG['CLIP_NORM'])
            opt.step(p, g_total, CFG['WEIGHT_DECAY'], set())

        avg = ep_loss / N
        if ep == 2 or ep % max(1, CFG['EPOCHS_PHASE1']//10) == 0:
            msg = f"[Phase1] Epoch {ep}/{CFG['EPOCHS_PHASE1']} avg_loss={avg:.4f} tok={tok_enabled}"
            print(msg)
            if cb: cb({'phase':1,'epoch':ep,'loss':avg,'tok':tok_enabled})
    return p

# ═══════════════════════════════════════════════════════════════════════════
# Compute embeddings (for Phase2)
# ═══════════════════════════════════════════════════════════════════════════
def compute_embeddings(p: Params, ds: dict):
    N = ds['X'].shape[0]; BS = CFG['BATCH_SIZE']; H = CFG['HIDDEN_DIM']
    ctx_all = np.zeros((N, H), np.float32)
    hlast_all = np.zeros((N, H), np.float32)
    hmean_all = np.zeros((N, H), np.float32)
    for s in range(0, N, BS):
        bi = np.arange(s, min(s+BS, N))
        Xb = ds['X'][bi]; Mb = ds['M'][bi]; B = len(bi)
        Hb, _ = gru_forward(p, Xb, Mb)
        ctx, _ = attention_forward(p, Hb, Mb)
        denom = Mb.sum(1, keepdims=True) + 1e-6
        hmean_all[bi] = (Hb * Mb[:,:,None]).sum(1) / denom
        for ii, gi in enumerate(bi):
            valid_ts = np.where(Mb[ii] > 0.5)[0]
            lt = int(valid_ts[-1]) if len(valid_ts) > 0 else CFG['SEQ_LEN'] - 1
            hlast_all[gi] = Hb[ii, lt]
        ctx_all[bi] = ctx
    return ctx_all, hlast_all, hmean_all

# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2 training: ID/Window/Validity heads
# ═══════════════════════════════════════════════════════════════════════════
def train_phase2(p: Params, ds: dict, cb=None) -> Params:
    NI, NW = CFG['N_IDENTITIES'], CFG['N_WINDOWS_PER_ID']
    H = CFG['HIDDEN_DIM']; N = ds['X'].shape[0]; BS = CFG['BATCH_SIZE']
    freeze = {k for k in p.__slots__ if k not in ('W_id','b_id','W_w','b_w','W_beh','b_beh')}
    opt = Adam(p, CFG['LR_PHASE2_BASE'])
    idx = np.arange(N)

    for ep in range(1, CFG['EPOCHS_PHASE2']+1):
        ctx_all, hlast_all, hmean_all = compute_embeddings(p, ds)
        lr = CFG['LR_PHASE2_BASE'] * (0.98 ** (ep/30.0))
        opt.lr = lr
        rsh = XorShift32(0x9000 + ep)
        idx = np.array(rsh.shuffle(list(idx)))
        ep_loss = 0.0

        for s in range(0, N, BS):
            bi = idx[s:s+BS]; B = len(bi)
            cb_b = ctx_all[bi]; hl_b = hlast_all[bi]; hm_b = hmean_all[bi]
            ycls = ds['Y_CLS'][bi]; pos = (ycls > 0.5).astype(np.float32)
            tid = ds['TRUE_ID'][bi]; tw = ds['TRUE_W'][bi]
            claim = ds['CLAIM_ID'][bi]; expw = ds['EXPECT_W'][bi]

            logits_id = cb_b @ p.W_id.T + p.b_id           # [B, NI]
            feat_w = np.concatenate([cb_b, hl_b, hm_b], 1)  # [B, 3H]
            logits_w = feat_w @ p.W_w.T + p.b_w             # [B, NW]

            prob_id = softmax(logits_id)
            prob_w  = softmax(logits_w)
            p_id_cl = prob_id[np.arange(B), claim]
            p_w_ex  = prob_w[np.arange(B), expw]

            cid_n = claim / max(1, NI-1)
            ew_n  = expw  / max(1, NW-1)
            vb = np.concatenate([cb_b,
                                  cid_n[:, None].astype(np.float32),
                                  ew_n[:, None].astype(np.float32),
                                  p_id_cl[:, None],
                                  p_w_ex[:, None]], 1)      # [B, H+4]
            logits_v = (vb @ p.W_beh.T).squeeze(1) + p.b_beh[0]

            # ID CE on positives
            loss_id, dlog_id = 0.0, np.zeros_like(logits_id)
            cnt = 0
            for i in range(B):
                if pos[i] < 0.5: continue
                cnt += 1; probs = prob_id[i].copy()
                loss_id += -np.log(probs[tid[i]]+1e-12)
                probs[tid[i]] -= 1.0; dlog_id[i] = probs
            if cnt: loss_id /= cnt; dlog_id /= cnt

            # W CE on positives
            loss_w, dlog_w = 0.0, np.zeros_like(logits_w)
            cnt = 0
            for i in range(B):
                if pos[i] < 0.5: continue
                cnt += 1; probs = prob_w[i].copy()
                loss_w += -np.log(probs[tw[i]]+1e-12)
                probs[tw[i]] -= 1.0; dlog_w[i] = probs
            if cnt: loss_w /= cnt; dlog_w /= cnt

            # Validity BCE
            pv = sigmoid(logits_v)
            loss_v = -(CFG['POS_WEIGHT'] * ycls * np.log(pv+1e-8) +
                       (1-ycls) * np.log(1-pv+1e-8)).mean()
            dlog_v = (pv - ycls); dlog_v[ycls>0.5] *= CFG['POS_WEIGHT']
            dlog_v /= B

            loss = CFG['LAMBDA_ID']*loss_id + CFG['LAMBDA_W']*loss_w + CFG['LAMBDA_BCE']*loss_v
            ep_loss += loss * B

            g_total = zeros_like(p)
            g_total.b_id[:] = dlog_id.sum(0)
            g_total.W_id[:] = dlog_id.T @ cb_b
            g_total.b_w[:] = dlog_w.sum(0)
            g_total.W_w[:] = dlog_w.T @ feat_w
            g_total.b_beh[0] = dlog_v.sum()
            g_total.W_beh[0] = dlog_v @ vb

            clip_grads(g_total, CFG['CLIP_NORM'])
            opt.step(p, g_total, CFG['WEIGHT_DECAY'], freeze)

        avg = ep_loss / N
        if ep == 2 or ep % max(1, CFG['EPOCHS_PHASE2']//10) == 0:
            msg = f"[Phase2] Epoch {ep}/{CFG['EPOCHS_PHASE2']} avg_loss={avg:.4f} lr={lr:.6f}"
            print(msg)
            if cb: cb({'phase':2,'epoch':ep,'loss':avg,'lr':lr})
    return p

# ═══════════════════════════════════════════════════════════════════════════
# Single-sample verify (inference path)
# ═══════════════════════════════════════════════════════════════════════════
def verify_chain(p: Params, tokens: np.ndarray, meas: np.ndarray,
                 claimed_id: int, expected_w: int,
                 true_ms: Optional[np.ndarray] = None) -> dict:
    NI, NW = CFG['N_IDENTITIES'], CFG['N_WINDOWS_PER_ID']
    result = dict(ok=False, p_valid=0.0, id_pred=-1, w_pred=-1,
                  pid=0.0, pw=0.0, l2ms=-1.0, pilot_corr=0.0,
                  gates={})

    if not (0 <= claimed_id < NI) or not (0 <= expected_w < NW):
        result['gates']['range'] = False
        return result

    g_exp = claimed_id * NW + expected_w
    pc = pilot_corr(meas, g_exp)
    result['pilot_corr'] = float(pc)
    pn_ok = pc >= CFG['PILOT_CORR_MIN']

    X = build_X(tokens, meas)[None]  # [1, T, D]
    M = np.ones((1, CFG['SEQ_LEN']), np.float32)

    H, _ = gru_forward(p, X, M)
    ctx, _ = attention_forward(p, H, M)
    ms_hat, _ = ms_head_forward(p, ctx)

    logits_id = ctx[0] @ p.W_id.T + p.b_id
    prob_id = softmax(logits_id)
    id_pred = int(prob_id.argmax())
    pid = float(prob_id[claimed_id])

    denom = M[0].sum() + 1e-6
    hmean = (H[0] * M[0, :, None]).sum(0) / denom
    hlast = H[0, -1]
    feat_w = np.concatenate([ctx[0], hlast, hmean])
    logits_w = feat_w @ p.W_w.T + p.b_w
    prob_w = softmax(logits_w)
    w_pred = int(prob_w.argmax())
    pw = float(prob_w[expected_w])

    cid_n = claimed_id / max(1, NI-1)
    ew_n  = expected_w / max(1, NW-1)
    vb = np.concatenate([ctx[0], [cid_n, ew_n, pid, pw]])
    logit_v = float(vb @ p.W_beh[0] + p.b_beh[0])
    p_valid = float(sigmoid(np.array([logit_v]))[0])

    if true_ms is not None:
        result['l2ms'] = float(np.linalg.norm(ms_hat[0] - true_ms))

    c1 = p_valid >= CFG['THRESH_P_VALID']
    c2 = id_pred == claimed_id
    c3 = w_pred == expected_w
    c4 = pid >= CFG['PID_MIN']
    c5 = pw >= CFG['PW_MIN']

    result.update(dict(
        ok=bool(pn_ok and c1 and c2 and c3 and c4 and c5),
        p_valid=p_valid, id_pred=id_pred, w_pred=w_pred, pid=pid, pw=pw,
        gates=dict(pn=pn_ok, p_valid=c1, id_match=c2, w_match=c3, pid=c4, pw=c5)
    ))
    return result

# ═══════════════════════════════════════════════════════════════════════════
# Model save / load (NumPy .npz)
# ═══════════════════════════════════════════════════════════════════════════
MAGIC = b'WARL0K1'

def save_model(path: str, p: Params, meta: dict = None):
    arrays = p.arrays()
    header = json.dumps({**(meta or {}), 'cfg': CFG}).encode()
    np.savez_compressed(path, **arrays, _header_=np.frombuffer(header, dtype=np.uint8))

def load_model(path: str) -> Tuple['Params', dict]:
    data = np.load(path, allow_pickle=False)
    p = Params()
    for k in p.__slots__:
        setattr(p, k, data[k].astype(np.float32))
    try:
        meta = json.loads(bytes(data['_header_'].astype(np.uint8)))
    except Exception:
        meta = {}
    return p, meta

# ═══════════════════════════════════════════════════════════════════════════
# ENCRYPTION LAYER  (AES-256-GCM or fallback XOR+HMAC)
# ═══════════════════════════════════════════════════════════════════════════
def derive_key(master_key: bytes, salt: bytes, info: bytes = b'WARL0K_PIM') -> bytes:
    if HAS_CRYPTO:
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        from cryptography.hazmat.primitives import hashes as ch
        hkdf = HKDF(algorithm=ch.SHA256(), length=32, salt=salt, info=info)
        return hkdf.derive(master_key)
    else:
        return hashlib.sha256(master_key + salt + info).digest()

def encrypt_chain(data: bytes, key: bytes) -> dict:
    salt = os.urandom(16)
    dk = derive_key(key, salt)
    nonce = os.urandom(12)
    if HAS_CRYPTO:
        aesgcm = AESGCM(dk)
        ct = aesgcm.encrypt(nonce, data, None)
        tag = ct[-16:]; ciphertext = ct[:-16]
    else:
        # XOR stream cipher (demo fallback — not production secure)
        ks = hashlib.sha256(dk + nonce).digest()
        ciphertext = bytes(b ^ ks[i % 32] for i, b in enumerate(data))
        tag = hmac.new(dk, nonce + ciphertext, hashlib.sha256).digest()[:16]
    chain_hash = hashlib.sha3_256(data).digest()
    return dict(
        salt=salt.hex(), nonce=nonce.hex(),
        ciphertext=ciphertext.hex(), tag=tag.hex(),
        chain_hash=chain_hash.hex(),
        has_crypto=HAS_CRYPTO,
    )

def decrypt_chain(pkg: dict, key: bytes) -> bytes:
    salt = bytes.fromhex(pkg['salt']); nonce = bytes.fromhex(pkg['nonce'])
    ct = bytes.fromhex(pkg['ciphertext']); tag = bytes.fromhex(pkg['tag'])
    dk = derive_key(key, salt)
    if HAS_CRYPTO:
        aesgcm = AESGCM(dk)
        data = aesgcm.decrypt(nonce, ct + tag, None)
    else:
        ks = hashlib.sha256(dk + nonce).digest()
        data = bytes(b ^ ks[i % 32] for i, b in enumerate(ct))
        expected_tag = hmac.new(dk, nonce + ct, hashlib.sha256).digest()[:16]
        if not hmac.compare_digest(tag, expected_tag):
            raise ValueError("HMAC tag mismatch — tamper detected")
    return data

# ═══════════════════════════════════════════════════════════════════════════
# CHAIN PROOF — sequential HMAC chain linking PIM sessions
# ═══════════════════════════════════════════════════════════════════════════
class ChainProof:
    """
    Each verification event is linked into an HMAC chain:
       state_n = SHA3-256(state_{n-1} || event_n)
    The chain is tamper-evident: any modification breaks all subsequent proofs.
    """
    def __init__(self, genesis: bytes = b'WARL0K_GENESIS'):
        self.state = hashlib.sha3_256(genesis).digest()
        self.events: List[dict] = []
        self.seq = 0

    def append(self, event: dict) -> str:
        blob = json.dumps(event, sort_keys=True, default=str).encode()
        self.state = hashlib.sha3_256(self.state + blob).digest()
        proof = hmac.new(self.state, blob, hashlib.sha256).hexdigest()
        self.seq += 1
        entry = {'seq': self.seq, 'event': event, 'proof': proof,
                 'chain_state': self.state.hex()}
        self.events.append(entry)
        return proof

    def verify_chain(self) -> Tuple[bool, int]:
        state = hashlib.sha3_256(b'WARL0K_GENESIS').digest()
        for i, entry in enumerate(self.events):
            blob = json.dumps(entry['event'], sort_keys=True, default=str).encode()
            state = hashlib.sha3_256(state + blob).digest()
            expected = hmac.new(state, blob, hashlib.sha256).hexdigest()
            if state.hex() != entry['chain_state'] or expected != entry['proof']:
                return False, i
        return True, len(self.events)

    def to_dict(self) -> dict:
        return {'seq': self.seq, 'state': self.state.hex(), 'events': self.events}

# ═══════════════════════════════════════════════════════════════════════════
# High-level PIM session
# ═══════════════════════════════════════════════════════════════════════════
class PIMSession:
    def __init__(self, params: Params, session_key: bytes = None):
        self.p = params
        self.key = session_key or os.urandom(32)
        self.chain = ChainProof()
        self.model_hash = self._hash_model()

    def _hash_model(self) -> str:
        h = hashlib.sha3_256()
        for k in self.p.__slots__:
            h.update(getattr(self.p, k).tobytes())
        return h.hexdigest()

    def verify(self, claimed_id: int, expected_w: int,
               tokens: np.ndarray = None, meas: np.ndarray = None,
               true_ms: np.ndarray = None) -> dict:
        if tokens is None or meas is None:
            g = claimed_id * CFG['N_WINDOWS_PER_ID'] + expected_w
            ms = MS_ALL[claimed_id] if 0 <= claimed_id < CFG['N_IDENTITIES'] else np.zeros(CFG['MS_DIM'])
            tokens, meas = generate_os_chain(ms, g)

        t0 = time.perf_counter()
        result = verify_chain(self.p, tokens, meas, claimed_id, expected_w, true_ms)
        latency_us = (time.perf_counter() - t0) * 1e6

        event = {
            'claimed_id': claimed_id, 'expected_w': expected_w,
            'ok': result['ok'], 'p_valid': round(result['p_valid'], 4),
            'pid': round(result['pid'], 4), 'pw': round(result['pw'], 4),
            'pilot_corr': round(result['pilot_corr'], 4),
            'latency_us': round(latency_us, 2),
            'model_hash': self.model_hash[:16],
            'ts': time.time(),
        }
        proof = self.chain.append(event)
        result['proof'] = proof
        result['seq']   = self.chain.seq
        result['latency_us'] = latency_us
        return result

    def encrypt_tokens(self, tokens: np.ndarray, meas: np.ndarray) -> dict:
        payload = json.dumps({'tokens': tokens.tolist(), 'meas': meas.tolist()}).encode()
        return encrypt_chain(payload, self.key)

    def decrypt_tokens(self, pkg: dict) -> Tuple[np.ndarray, np.ndarray]:
        raw = decrypt_chain(pkg, self.key)
        d = json.loads(raw)
        return np.array(d['tokens'], dtype=np.int32), np.array(d['meas'], dtype=np.float32)

    def chain_status(self) -> dict:
        ok, count = self.chain.verify_chain()
        return {'valid': ok, 'events': count, 'state': self.chain.state.hex()[:16]}

# ═══════════════════════════════════════════════════════════════════════════
# FULL TRAIN + EVALUATE pipeline
# ═══════════════════════════════════════════════════════════════════════════
def full_pipeline(cb=None) -> dict:
    print("=== WARL0K PIM Python Engine ===")
    print(f"Crypto backend: {'AES-256-GCM (cryptography)' if HAS_CRYPTO else 'HMAC-SHA256 (fallback)'}")
    ds = build_dataset()
    pos = int((ds['Y_CLS'] > 0.5).sum())
    print(f"Dataset: {ds['X'].shape[0]} samples, {pos} positives")

    p = init_params()
    print(f"Model params: {p.param_bytes()//1024:.1f} KB")

    t0 = time.time()
    train_phase1(p, ds, cb=cb)
    train_phase2(p, ds, cb=cb)
    train_s = time.time() - t0
    print(f"\nTraining: {train_s:.2f}s")

    session = PIMSession(p)

    # Evaluate
    results = {}
    id_e = 0; w_e = 3
    ms_e = MS_ALL[id_e]
    g_e  = id_e * CFG['N_WINDOWS_PER_ID'] + w_e
    toks, meas = generate_os_chain(ms_e, g_e)

    results['legit']     = session.verify(id_e, w_e, toks, meas, ms_e)
    rsh = XorShift32(0x7777)
    idx = rsh.shuffle(list(range(CFG['SEQ_LEN'])))
    results['shuffled']  = session.verify(id_e, w_e, toks[idx], meas[idx], ms_e)
    t2 = np.zeros(CFG['SEQ_LEN'], np.int32); m2 = np.zeros(CFG['SEQ_LEN'], np.float32)
    t2[:10]=toks[:10]; m2[:10]=meas[:10]
    results['truncated'] = session.verify(id_e, w_e, t2, m2, ms_e)
    ww = (w_e+7)%CFG['N_WINDOWS_PER_ID']
    tw, mw = generate_os_chain(ms_e, id_e*CFG['N_WINDOWS_PER_ID']+ww)
    results['wrong_win'] = session.verify(id_e, w_e, tw, mw, ms_e)
    oid = (id_e+1)%CFG['N_IDENTITIES']
    to, mo = generate_os_chain(MS_ALL[oid], oid*CFG['N_WINDOWS_PER_ID']+13)
    results['wrong_id']  = session.verify(id_e, w_e, to, mo, ms_e)
    results['out_range'] = session.verify(id_e, 999, toks, meas, ms_e)

    chain_ok = session.chain_status()
    print(f"\nChain proof: {chain_ok}")
    for name, r in results.items():
        ok_str = "✓ OK" if r['ok'] else "✗ FAIL"
        print(f"  {name:12s}: {ok_str}  p_valid={r['p_valid']:.3f}  pid={r['pid']:.3f}  pw={r['pw']:.3f}")

    return {'params': p, 'session': session, 'results': results,
            'train_s': train_s, 'chain': chain_ok, 'has_crypto': HAS_CRYPTO}


if __name__ == '__main__':
    full_pipeline()
