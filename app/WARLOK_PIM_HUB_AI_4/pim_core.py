"""
WARL0K PIM Core Engine v2 — GRU+Attention + Dual-Peer Hub Protocol
New in v2:
  - PeerSession  : full mutual handshake lifecycle per peer
  - AnchorProof  : AES-encrypted MS commitment for anchor transfer
  - 48-OS recon  : reconstruct_ms_from_48() — median MS estimate across all windows
  - ChainProof v2: time-delta + monotonic counter + replay detection
  - aes_encrypt/decrypt: AES-256-GCM with HKDF-SHA256
"""

import numpy as np
import hashlib
import hmac as hmac_mod
import time
import json
import os
import secrets
from typing import Optional, Tuple, List

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes as _ch
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════
CFG = dict(
    VOCAB_SIZE        = 16,
    MS_DIM            = 8,
    SEQ_LEN           = 20,
    N_IDENTITIES      = 2,
    N_WINDOWS_PER_ID  = 48,
    HIDDEN_DIM        = 64,
    ATTN_DIM          = 32,
    MS_HID            = 32,
    BATCH_SIZE        = 32,
    EPOCHS_PHASE1     = 90,
    EPOCHS_PHASE2     = 100,
    LR_PHASE1         = 0.006,
    LR_PHASE2_BASE    = 0.008,
    CLIP_NORM         = 5.0,
    WEIGHT_DECAY      = 1e-4,
    LAMBDA_MS         = 1.0,
    LAMBDA_TOK        = 0.10,
    TOK_STOP_EPS      = 0.25,
    TOK_WARMUP_EPOCHS = 60,
    LAMBDA_ID         = 1.0,
    LAMBDA_W          = 1.0,
    LAMBDA_BCE        = 1.0,
    POS_WEIGHT        = 10.0,
    THRESH_P_VALID    = 0.80,
    PID_MIN           = 0.70,
    PW_MIN            = 0.40,
    PILOT_AMP         = 0.55,
    PILOT_CORR_MIN    = 0.02,
    # Peer hub protocol
    MS_RECON_TOL      = 0.35,   # max L2 for MS acceptance
    DELTA_MAX_MS      = 30_000, # max allowed time-delta between chain steps
)
CFG['INPUT_DIM'] = CFG['VOCAB_SIZE'] + 2

# ═══════════════════════════════════════════════════════════════════
# XorShift32
# ═══════════════════════════════════════════════════════════════════
class XorShift32:
    def __init__(self, seed=0x12345678):
        self.s = int(seed) & 0xFFFFFFFF or 0x12345678

    def next_u32(self):
        x = self.s
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5)  & 0xFFFFFFFF
        self.s = x & 0xFFFFFFFF
        return self.s

    def next_f01(self): return (self.next_u32() >> 8) * (1.0 / 16777216.0)
    def next_int(self, lo, hi): return lo + int(self.next_u32() % (hi - lo))
    def next_norm(self):
        u1 = max(1e-7, self.next_f01()); u2 = self.next_f01()
        return float(np.sqrt(-2*np.log(u1)) * np.cos(2*np.pi*u2))
    def shuffle(self, v):
        v = list(v)
        for i in range(len(v)-1, 0, -1):
            j = self.next_int(0, i+1); v[i], v[j] = v[j], v[i]
        return v

# ═══════════════════════════════════════════════════════════════════
# Activations
# ═══════════════════════════════════════════════════════════════════
def sigmoid(x): return np.where(x>=0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-12)

# ═══════════════════════════════════════════════════════════════════
# Global data (MS_all, A_base)
# ═══════════════════════════════════════════════════════════════════
def _init_globals():
    rng = XorShift32(0xDEADBEEF)
    V = CFG['N_IDENTITIES']; K = CFG['MS_DIM']; T = CFG['SEQ_LEN']
    MS = np.array([[2*rng.next_f01()-1 for _ in range(K)] for _ in range(V)], dtype=np.float32)
    A  = np.array([[0.8*rng.next_norm() for _ in range(K)] for _ in range(T)], dtype=np.float32)
    return MS, A

MS_ALL, A_BASE = _init_globals()

# ═══════════════════════════════════════════════════════════════════
# OS chain generation
# ═══════════════════════════════════════════════════════════════════
def window_delta(g, t):
    rng = XorShift32((g*10007 + t*97) & 0xFFFFFFFF or 0xA5A5A5A5)
    return np.array([0.25*rng.next_norm() for _ in range(CFG['MS_DIM'])], dtype=np.float32)

def window_pilot(g):
    rng = XorShift32((g*9176 + 11) & 0xFFFFFFFF or 0xBEEFBEEF)
    T = CFG['SEQ_LEN']
    p = np.array([(rng.next_int(0,2)*2-1)*CFG['PILOT_AMP'] for _ in range(T)], dtype=np.float32)
    return p - p.mean()

def pilot_corr(meas, g):
    p = window_pilot(g)
    num = float(meas @ p)
    return num / (float(np.sqrt((meas**2).sum() * (p**2).sum())) + 1e-9)

def generate_os_chain(ms, g):
    T = CFG['SEQ_LEN']
    zs = np.array([float((A_BASE[t]+window_delta(g,t)) @ ms) for t in range(T)], dtype=np.float32)
    zs += window_pilot(g)
    ms_sum = int(ms.sum()*1000)
    rng = XorShift32((g*1337 + ms_sum) & 0xFFFFFFFF or 0xCAFE1234)
    zs += np.array([0.02*rng.next_norm() for _ in range(T)], dtype=np.float32)
    mu, st = zs.mean(), zs.std() + 1e-6
    meas = (zs - mu) / st
    tokens = np.clip((np.clip((meas+3)/6, 0, 0.999999)*CFG['VOCAB_SIZE']).astype(np.int32),
                     0, CFG['VOCAB_SIZE']-1)
    return tokens, meas

def build_X(tokens, meas):
    T = CFG['SEQ_LEN']; D = CFG['INPUT_DIM']; V = CFG['VOCAB_SIZE']
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        X[t, tokens[t]] = 1.0; X[t, V] = meas[t]
        X[t, V+1] = t/(T-1) if T>1 else 0.0
    return X

# ═══════════════════════════════════════════════════════════════════
# Model params
# ═══════════════════════════════════════════════════════════════════
class Params:
    __slots__ = ['W_z','U_z','b_z','W_r','U_r','b_r','W_h','U_h','b_h',
                 'W_att','v_att','W_ms1','b_ms1','W_ms2','b_ms2',
                 'W_tok','b_tok','W_id','b_id','W_w','b_w','W_beh','b_beh']
    def arrays(self): return {k: getattr(self,k) for k in self.__slots__}
    def param_bytes(self): return sum(a.nbytes for a in self.arrays().values())

def init_params(seed=0xC0FFEE):
    rng = XorShift32(seed)
    H=CFG['HIDDEN_DIM']; D=CFG['INPUT_DIM']; AH=CFG['ATTN_DIM']
    MH=CFG['MS_HID']; MS=CFG['MS_DIM']; V=CFG['VOCAB_SIZE']
    NI=CFG['N_IDENTITIES']; NW=CFG['N_WINDOWS_PER_ID']
    def rm(r,c,s=0.08): return np.array([s*rng.next_norm() for _ in range(r*c)],dtype=np.float32).reshape(r,c)
    def zv(n): return np.zeros(n, dtype=np.float32)
    p = Params()
    p.W_z=rm(H,D); p.U_z=rm(H,H); p.b_z=zv(H)
    p.W_r=rm(H,D); p.U_r=rm(H,H); p.b_r=zv(H)
    p.W_h=rm(H,D); p.U_h=rm(H,H); p.b_h=zv(H)
    p.W_att=rm(AH,H); p.v_att=np.array([0.08*rng.next_norm() for _ in range(AH)],dtype=np.float32)
    p.W_ms1=rm(MH,H); p.b_ms1=zv(MH)
    p.W_ms2=rm(MS,MH); p.b_ms2=zv(MS)
    p.W_tok=rm(V,H);   p.b_tok=zv(V)
    p.W_id=rm(NI,H);   p.b_id=zv(NI)
    p.W_w=rm(NW,3*H);  p.b_w=zv(NW)
    p.W_beh=rm(1,H+4); p.b_beh=zv(1)
    return p

def zeros_like(p):
    g = Params()
    for k in p.__slots__: setattr(g, k, np.zeros_like(getattr(p,k)))
    return g

# ═══════════════════════════════════════════════════════════════════
# Adam
# ═══════════════════════════════════════════════════════════════════
class Adam:
    def __init__(self, p, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.t=0
        self.m=zeros_like(p); self.v=zeros_like(p)

    def step(self, p, g, wd, freeze):
        self.t += 1
        b1t = self.b1**self.t; b2t = self.b2**self.t
        for k in p.__slots__:
            if k in freeze: continue
            gr = getattr(g,k).copy()
            if wd > 0 and not (k.startswith('b_') or k=='v_att'): gr += wd*getattr(p,k)
            m=getattr(self.m,k); v=getattr(self.v,k)
            m[:] = self.b1*m + (1-self.b1)*gr
            v[:] = self.b2*v + (1-self.b2)*gr**2
            getattr(p,k)[:] -= self.lr * (m/(1-b1t)) / (np.sqrt(v/(1-b2t)) + self.eps)

# ═══════════════════════════════════════════════════════════════════
# GRU + Attention + MS head
# ═══════════════════════════════════════════════════════════════════
def gru_forward(p, X, M):
    B,T,D = X.shape; H = CFG['HIDDEN_DIM']
    Ho = np.zeros((B,T,H),dtype=np.float32)
    Z  = np.zeros_like(Ho); R = np.zeros_like(Ho); HT = np.zeros_like(Ho)
    h  = np.zeros((B,H), dtype=np.float32)
    for t in range(T):
        x = X[:,t,:]
        z = sigmoid(x@p.W_z.T + h@p.U_z.T + p.b_z)
        r = sigmoid(x@p.W_r.T + h@p.U_r.T + p.b_r)
        ht = np.tanh(x@p.W_h.T + (r*h)@p.U_h.T + p.b_h)
        hn = (1-z)*h + z*ht; mt = M[:,t:t+1]
        hn = mt*hn + (1-mt)*h
        Ho[:,t]=hn; Z[:,t]=z; R[:,t]=r; HT[:,t]=ht; h=hn
    return Ho, dict(X=X, M=M, H=Ho, Z=Z, R=R, HT=HT)

def gru_backward(p, cache, dH):
    X,M,H,Z,R,HT = cache['X'],cache['M'],cache['H'],cache['Z'],cache['R'],cache['HT']
    B,T,Hd = H.shape; gp = zeros_like(p)
    dn = np.zeros((B,Hd), dtype=np.float32)
    for t in reversed(range(T)):
        mt = M[:,t:t+1]; hp = H[:,t-1] if t>0 else np.zeros((B,Hd),dtype=np.float32)
        z=Z[:,t]; r=R[:,t]; ht=HT[:,t]; x=X[:,t]
        dh = (dn + dH[:,t]) * mt
        dht = dh*z; dz = dh*(ht-hp); dh_p = dh*(1-z)
        da_h = dht*(1-ht**2)
        gp.b_h+=da_h.sum(0); gp.W_h+=da_h.T@x; gp.U_h+=da_h.T@(r*hp)
        tmp = da_h@p.U_h; dh_p += tmp*r; dr = tmp*hp
        da_r = dr*r*(1-r)
        gp.b_r+=da_r.sum(0); gp.W_r+=da_r.T@x; gp.U_r+=da_r.T@hp; dh_p+=da_r@p.U_r
        da_z = dz*z*(1-z)
        gp.b_z+=da_z.sum(0); gp.W_z+=da_z.T@x; gp.U_z+=da_z.T@hp; dh_p+=da_z@p.U_z
        dn = dh_p
    return gp

def attention_forward(p, H, M):
    U = np.tanh(H @ p.W_att.T)
    sc = np.where(M>0, U@p.v_att, -1e30)
    alpha = softmax(sc); ctx = (alpha[:,:,None]*H).sum(1)
    return ctx, dict(H=H, M=M, U=U, alpha=alpha)

def attention_backward(p, cache, dctx):
    H,M,U,alpha = cache['H'],cache['M'],cache['U'],cache['alpha']
    B,T,Hd = H.shape; gp = zeros_like(p)
    dH = alpha[:,:,None] * dctx[:,None,:]
    da  = (dctx[:,None,:]*H).sum(-1)
    st  = (alpha*da).sum(-1, keepdims=True)
    dsc = alpha*(da-st)*(M>0)
    gp.v_att += (dsc[:,:,None]*U).sum((0,1))
    dU = dsc[:,:,None]*p.v_att[None,None,:]
    dUp = dU*(1-U**2)
    gp.W_att += dUp.reshape(B*T,-1).T @ H.reshape(B*T,Hd)
    dH += dUp @ p.W_att
    return gp, dH

def ms_head_forward(p, ctx):
    hid = np.tanh(ctx@p.W_ms1.T + p.b_ms1)
    out = hid@p.W_ms2.T + p.b_ms2
    return out, hid

def ms_head_backward(p, ctx, hid, dout):
    gp = zeros_like(p)
    gp.b_ms2 += dout.sum(0); gp.W_ms2 += dout.T@hid
    dhid = dout@p.W_ms2; dpre = dhid*(1-hid**2)
    gp.b_ms1 += dpre.sum(0); gp.W_ms1 += dpre.T@ctx
    return gp, dpre@p.W_ms1

def global_norm(g):
    return float(np.sqrt(sum((getattr(g,k)**2).sum() for k in g.__slots__)))

def clip_grads(g, mn):
    n = global_norm(g)
    if n > mn:
        s = mn/(n+1e-12)
        for k in g.__slots__: getattr(g,k).__imul__(s)

# ═══════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════
def build_dataset():
    NI,NW = CFG['N_IDENTITIES'], CFG['N_WINDOWS_PER_ID']
    T,D,MS = CFG['SEQ_LEN'], CFG['INPUT_DIM'], CFG['MS_DIM']
    N = NI*NW*5
    ds = dict(X=np.zeros((N,T,D),np.float32), M=np.ones((N,T),np.float32),
              TOK=np.zeros((N,T),np.int32),    Y_MS=np.zeros((N,MS),np.float32),
              Y_CLS=np.zeros(N,np.float32),
              TRUE_ID=np.zeros(N,np.int32),    TRUE_W=np.zeros(N,np.int32),
              CLAIM_ID=np.zeros(N,np.int32),   EXPECT_W=np.zeros(N,np.int32))
    rng=XorShift32(0xBADC0DE); n=0
    def push(toks,meas,ycls,tid,tw,cid,ew,yms,mask=None):
        nonlocal n
        ds['X'][n]=build_X(toks,meas)
        ds['M'][n]=mask if mask is not None else np.ones(T,np.float32)
        ds['TOK'][n]=toks; ds['Y_MS'][n]=yms; ds['Y_CLS'][n]=ycls
        ds['TRUE_ID'][n]=tid; ds['TRUE_W'][n]=tw
        ds['CLAIM_ID'][n]=cid; ds['EXPECT_W'][n]=ew; n+=1
    for id_t in range(NI):
        ms = MS_ALL[id_t]
        for w_t in range(NW):
            g=id_t*NW+w_t; toks,meas=generate_os_chain(ms,g)
            push(toks,meas,1.0,id_t,w_t,id_t,w_t,ms)
            rsh=XorShift32(0x1234+g); idx=np.array(rsh.shuffle(list(range(T))))
            push(toks[idx],meas[idx],0.0,id_t,w_t,id_t,w_t,ms)
            t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
            t2[:T//2]=toks[:T//2]; m2[:T//2]=meas[:T//2]
            mask=np.zeros(T,np.float32); mask[:T//2]=1.0
            push(t2,m2,0.0,id_t,w_t,id_t,w_t,ms,mask)
            ww=(w_t+7)%NW; tw2,mw=generate_os_chain(ms,id_t*NW+ww)
            push(tw2,mw,0.0,id_t,ww,id_t,w_t,ms)
            oid=(id_t+rng.next_int(1,NI))%NI; ow=rng.next_int(0,NW)
            to,mo=generate_os_chain(MS_ALL[oid],oid*NW+ow)
            push(to,mo,0.0,oid,ow,id_t,w_t,ms)
    return ds

# ═══════════════════════════════════════════════════════════════════
# Training — Phase 1
# ═══════════════════════════════════════════════════════════════════
def train_phase1(p, ds, cb=None):
    N=ds['X'].shape[0]; BS=CFG['BATCH_SIZE']; opt=Adam(p,CFG['LR_PHASE1'])
    tok_on=True; idx=np.arange(N)
    for ep in range(1, CFG['EPOCHS_PHASE1']+1):
        idx=np.array(XorShift32(0x1000+ep).shuffle(list(idx))); ep_loss=0.0
        for s in range(0,N,BS):
            bi=idx[s:s+BS]; B=len(bi)
            Xb=ds['X'][bi]; Mb=ds['M'][bi]; Tb=ds['TOK'][bi]
            yms=ds['Y_MS'][bi]; ycls=ds['Y_CLS'][bi]
            H,gc=gru_forward(p,Xb,Mb); ctx,ac=attention_forward(p,H,Mb)
            ms_hat,ms_hid=ms_head_forward(p,ctx)
            pc=(ycls>0.5).sum()+1e-6
            diff=(ms_hat-yms)*(ycls>0.5)[:,None]
            loss_ms=float(0.5*(diff**2).sum()/(pc*CFG['MS_DIM']))
            loss_tok=0.0; dH_tok=np.zeros_like(H)
            gWt=np.zeros_like(p.W_tok); gbt=np.zeros_like(p.b_tok)
            if tok_on and ep<=CFG['TOK_WARMUP_EPOCHS']:
                T=CFG['SEQ_LEN']
                denom=sum(1.0 for i in range(B) if ycls[i]>0.5
                          for t in range(T-1)
                          if Mb[i,t]>0.5 and Mb[i,t+1]>0.5)+1e-6
                for i in range(B):
                    if ycls[i]<=0.5: continue
                    for t in range(T-1):
                        if not(Mb[i,t]>0.5 and Mb[i,t+1]>0.5): continue
                        logits=H[i,t]@p.W_tok.T+p.b_tok; pr=softmax(logits)
                        tgt=int(Tb[i,t+1]); loss_tok+=-np.log(pr[tgt]+1e-12)
                        dl=pr.copy(); dl[tgt]-=1.0
                        gbt+=dl; gWt+=np.outer(dl,H[i,t]); dH_tok[i,t]+=dl@p.W_tok
                loss_tok/=denom; gWt/=denom; gbt/=denom; dH_tok/=denom
                if loss_tok<CFG['TOK_STOP_EPS']: tok_on=False
            else: tok_on=False
            loss=CFG['LAMBDA_MS']*loss_ms+(CFG['LAMBDA_TOK']*loss_tok if tok_on else 0)
            ep_loss+=loss*B
            dms=diff/(pc*CFG['MS_DIM'])
            gms,gctx=ms_head_backward(p,ctx,ms_hid,dms)
            gatt,dH=attention_backward(p,ac,gctx); dH+=dH_tok
            ggru=gru_backward(p,gc,dH)
            gt=zeros_like(p)
            for k in p.__slots__:
                getattr(gt,k)[:] = getattr(gms,k)+getattr(gatt,k)+getattr(ggru,k)
            gt.W_tok[:]=gWt; gt.b_tok[:]=gbt
            clip_grads(gt,CFG['CLIP_NORM']); opt.step(p,gt,CFG['WEIGHT_DECAY'],set())
        avg=ep_loss/N
        if ep==2 or ep%max(1,CFG['EPOCHS_PHASE1']//10)==0:
            if cb: cb({'phase':1,'epoch':ep,'loss':float(avg),'tok':tok_on})
            else:  print(f"[P1] ep{ep}/{CFG['EPOCHS_PHASE1']} loss={avg:.4f}")
    return p

# ═══════════════════════════════════════════════════════════════════
# Training — Phase 2
# ═══════════════════════════════════════════════════════════════════
def _embeddings(p, ds):
    N=ds['X'].shape[0]; H=CFG['HIDDEN_DIM']; BS=CFG['BATCH_SIZE']
    ca=np.zeros((N,H),np.float32); hla=np.zeros((N,H),np.float32); hma=np.zeros((N,H),np.float32)
    for s in range(0,N,BS):
        bi=np.arange(s,min(s+BS,N)); Xb=ds['X'][bi]; Mb=ds['M'][bi]; B=len(bi)
        Hb,_=gru_forward(p,Xb,Mb); ctx,_=attention_forward(p,Hb,Mb)
        ca[bi]=ctx; hma[bi]=(Hb*Mb[:,:,None]).sum(1)/(Mb.sum(1,keepdims=True)+1e-6)
        for ii,gi in enumerate(bi):
            vt=np.where(Mb[ii]>0.5)[0]; lt=int(vt[-1]) if len(vt) else CFG['SEQ_LEN']-1
            hla[gi]=Hb[ii,lt]
    return ca,hla,hma

def train_phase2(p, ds, cb=None):
    NI,NW=CFG['N_IDENTITIES'],CFG['N_WINDOWS_PER_ID']
    H=CFG['HIDDEN_DIM']; N=ds['X'].shape[0]; BS=CFG['BATCH_SIZE']
    freeze={k for k in p.__slots__ if k not in ('W_id','b_id','W_w','b_w','W_beh','b_beh')}
    opt=Adam(p,CFG['LR_PHASE2_BASE']); idx=np.arange(N)
    for ep in range(1,CFG['EPOCHS_PHASE2']+1):
        ca,hla,hma=_embeddings(p,ds)
        lr=CFG['LR_PHASE2_BASE']*(0.98**(ep/30.0)); opt.lr=lr
        idx=np.array(XorShift32(0x9000+ep).shuffle(list(idx))); ep_loss=0.0
        for s in range(0,N,BS):
            bi=idx[s:s+BS]; B=len(bi)
            cb_b=ca[bi]; hl_b=hla[bi]; hm_b=hma[bi]
            ycls=ds['Y_CLS'][bi]; pos=(ycls>0.5).astype(np.float32)
            tid=ds['TRUE_ID'][bi]; tw=ds['TRUE_W'][bi]
            claim=ds['CLAIM_ID'][bi]; expw=ds['EXPECT_W'][bi]
            lid=cb_b@p.W_id.T+p.b_id
            fw=np.concatenate([cb_b,hl_b,hm_b],1)
            lw=fw@p.W_w.T+p.b_w
            pid_p=softmax(lid); pw_p=softmax(lw)
            p_id=pid_p[np.arange(B),claim]; p_w=pw_p[np.arange(B),expw]
            vb=np.concatenate([cb_b,(claim/max(1,NI-1))[:,None].astype(np.float32),
                               (expw/max(1,NW-1))[:,None].astype(np.float32),
                               p_id[:,None],p_w[:,None]],1)
            lv=(vb@p.W_beh.T).squeeze(1)+p.b_beh[0]
            def ce(lg,C,tgt,mask):
                dl=np.zeros_like(lg); L=0.0; cnt=0
                for i in range(B):
                    if mask[i]<0.5: continue
                    cnt+=1; pr=softmax(lg[i]); L+=-np.log(pr[tgt[i]]+1e-12)
                    pr2=pr.copy(); pr2[tgt[i]]-=1.0; dl[i]=pr2
                return (L/cnt,dl/cnt) if cnt else (0.0,dl)
            li,dli=ce(lid,NI,tid,pos); lw2,dlw=ce(lw,NW,tw,pos)
            pv=sigmoid(lv)
            loss_v=-(CFG['POS_WEIGHT']*ycls*np.log(pv+1e-8)+(1-ycls)*np.log(1-pv+1e-8)).mean()
            dlv=(pv-ycls); dlv[ycls>0.5]*=CFG['POS_WEIGHT']; dlv/=B
            loss=CFG['LAMBDA_ID']*li+CFG['LAMBDA_W']*lw2+CFG['LAMBDA_BCE']*loss_v
            ep_loss+=loss*B
            gt=zeros_like(p)
            gt.b_id[:]=dli.sum(0); gt.W_id[:]=dli.T@cb_b
            gt.b_w[:]=dlw.sum(0);  gt.W_w[:]=dlw.T@fw
            gt.b_beh[0]=dlv.sum(); gt.W_beh[0]=dlv@vb
            clip_grads(gt,CFG['CLIP_NORM']); opt.step(p,gt,CFG['WEIGHT_DECAY'],freeze)
        avg=ep_loss/N
        if ep==2 or ep%max(1,CFG['EPOCHS_PHASE2']//10)==0:
            if cb: cb({'phase':2,'epoch':ep,'loss':float(avg),'lr':float(lr)})
            else:  print(f"[P2] ep{ep}/{CFG['EPOCHS_PHASE2']} loss={avg:.4f}")
    return p

# ═══════════════════════════════════════════════════════════════════
# Verify
# ═══════════════════════════════════════════════════════════════════
def verify_chain(p, tokens, meas, claimed_id, expected_w, true_ms=None):
    NI,NW=CFG['N_IDENTITIES'],CFG['N_WINDOWS_PER_ID']
    result=dict(ok=False,p_valid=0.0,id_pred=-1,w_pred=-1,
                pid=0.0,pw=0.0,l2ms=-1.0,pilot_corr=0.0,gates={})
    if not(0<=claimed_id<NI) or not(0<=expected_w<NW):
        result['gates']['range']=False; return result
    g=claimed_id*NW+expected_w; pc=pilot_corr(meas,g)
    result['pilot_corr']=float(pc); pn_ok=pc>=CFG['PILOT_CORR_MIN']
    X=build_X(tokens,meas)[None]; M=np.ones((1,CFG['SEQ_LEN']),np.float32)
    H,_=gru_forward(p,X,M); ctx,_=attention_forward(p,H,M)
    ms_hat,_=ms_head_forward(p,ctx)
    lid=ctx[0]@p.W_id.T+p.b_id; prob_id=softmax(lid)
    id_pred=int(prob_id.argmax()); pid=float(prob_id[claimed_id])
    d=M[0].sum()+1e-6; hm=(H[0]*M[0,:,None]).sum(0)/d; hl=H[0,-1]
    fw=np.concatenate([ctx[0],hl,hm])
    lw=fw@p.W_w.T+p.b_w; prob_w=softmax(lw)
    w_pred=int(prob_w.argmax()); pw=float(prob_w[expected_w])
    cid_n=claimed_id/max(1,NI-1); ew_n=expected_w/max(1,NW-1)
    vb=np.concatenate([ctx[0],[cid_n,ew_n,pid,pw]])
    p_valid=float(sigmoid(np.array([float(vb@p.W_beh[0]+p.b_beh[0])]))[0])
    if true_ms is not None: result['l2ms']=float(np.linalg.norm(ms_hat[0]-true_ms))
    c1=p_valid>=CFG['THRESH_P_VALID']; c2=id_pred==claimed_id
    c3=w_pred==expected_w; c4=pid>=CFG['PID_MIN']; c5=pw>=CFG['PW_MIN']
    result.update(dict(ok=bool(pn_ok and c1 and c2 and c3 and c4 and c5),
                       p_valid=p_valid,id_pred=id_pred,w_pred=w_pred,pid=pid,pw=pw,
                       gates=dict(pn=pn_ok,p_valid=c1,id_match=c2,w_match=c3,pid=c4,pw=c5)))
    return result

# ═══════════════════════════════════════════════════════════════════
# Save / Load
# ═══════════════════════════════════════════════════════════════════
def save_model(path, p, meta=None):
    hdr=json.dumps({**(meta or {}),'cfg':CFG}).encode()
    np.savez_compressed(path, **p.arrays(), _header_=np.frombuffer(hdr,dtype=np.uint8))

def load_model(path):
    data=np.load(path,allow_pickle=False); p=Params()
    for k in p.__slots__: setattr(p,k,data[k].astype(np.float32))
    try: meta=json.loads(bytes(data['_header_'].astype(np.uint8)))
    except: meta={}
    return p,meta

# ═══════════════════════════════════════════════════════════════════
# CRYPTO LAYER — AES-256-GCM + HKDF
# ═══════════════════════════════════════════════════════════════════
def derive_key(master_key, salt, info=b'WARL0K_PIM_V2'):
    if HAS_CRYPTO:
        return HKDF(algorithm=_ch.SHA256(),length=32,salt=salt,info=info).derive(master_key)
    return hashlib.sha256(master_key+salt+info).digest()

def aes_encrypt(data: bytes, key: bytes) -> dict:
    salt=os.urandom(16); dk=derive_key(key,salt); nonce=os.urandom(12)
    if HAS_CRYPTO:
        ct=AESGCM(dk).encrypt(nonce,data,None); tag=ct[-16:]; ct=ct[:-16]
    else:
        ks=hashlib.sha256(dk+nonce).digest()
        ct=bytes(b^ks[i%32] for i,b in enumerate(data))
        tag=hmac_mod.new(dk,nonce+ct,hashlib.sha256).digest()[:16]
    return dict(salt=salt.hex(),nonce=nonce.hex(),ct=ct.hex(),
                tag=tag.hex(),chain_hash=hashlib.sha3_256(data).hexdigest(),
                has_crypto=HAS_CRYPTO)

def aes_decrypt(pkg: dict, key: bytes) -> bytes:
    salt=bytes.fromhex(pkg['salt']); nonce=bytes.fromhex(pkg['nonce'])
    ct=bytes.fromhex(pkg['ct']);     tag=bytes.fromhex(pkg['tag'])
    dk=derive_key(key,salt)
    if HAS_CRYPTO:
        return AESGCM(dk).decrypt(nonce,ct+tag,None)
    ks=hashlib.sha256(dk+nonce).digest()
    data=bytes(b^ks[i%32] for i,b in enumerate(ct))
    expected=hmac_mod.new(dk,nonce+ct,hashlib.sha256).digest()[:16]
    if not hmac_mod.compare_digest(tag,expected): raise ValueError("Tag mismatch")
    return data

# ═══════════════════════════════════════════════════════════════════
# CHAIN PROOF v2 — time-delta + monotonic counter + replay guard
# ═══════════════════════════════════════════════════════════════════
class ChainProof:
    """
    state_n  = SHA3-256(state_{n-1} || event_json)
    proof_n  = HMAC-SHA256(state_n, event_json)
    Each event carries: seq, counter (monotonic), ts, delta_ms
    Replay detection: counter must strictly increase
    Time guard: delta_ms must be <= DELTA_MAX_MS
    """
    def __init__(self, genesis=b'WARL0K_GENESIS_V2'):
        self.state    = hashlib.sha3_256(genesis).digest()
        self.events: List[dict] = []
        self.seq      = 0
        self.counter  = 0
        self._last_ts = time.time()

    def append(self, event: dict, counter: int = None) -> dict:
        now      = time.time()
        delta_ms = round((now - self._last_ts)*1000, 2)
        self.counter = counter if counter is not None else self.counter+1
        self.seq    += 1
        aug = {**event, 'seq': self.seq, 'counter': self.counter,
               'ts': round(now,4), 'delta_ms': delta_ms}
        blob  = json.dumps(aug, sort_keys=True, default=str).encode()
        self.state = hashlib.sha3_256(self.state+blob).digest()
        proof = hmac_mod.new(self.state, blob, hashlib.sha256).hexdigest()
        entry = {'seq':self.seq,'event':aug,'proof':proof,'chain_state':self.state.hex()}
        self.events.append(entry)
        self._last_ts = now
        return entry

    def verify_integrity(self) -> Tuple[bool, int, str]:
        state = hashlib.sha3_256(b'WARL0K_GENESIS_V2').digest()
        prev_ctr = -1
        for i, entry in enumerate(self.events):
            blob  = json.dumps(entry['event'], sort_keys=True, default=str).encode()
            state = hashlib.sha3_256(state+blob).digest()
            exp   = hmac_mod.new(state, blob, hashlib.sha256).hexdigest()
            if state.hex() != entry['chain_state']: return False, i, 'state_mismatch'
            if exp           != entry['proof']:      return False, i, 'proof_mismatch'
            c = entry['event'].get('counter',0)
            if c <= prev_ctr:                        return False, i, 'counter_regression'
            if entry['event'].get('delta_ms',0) > CFG['DELTA_MAX_MS']:
                return False, i, 'time_violation'
            prev_ctr = c
        return True, len(self.events), 'ok'

    def status(self) -> dict:
        ok,n,reason = self.verify_integrity()
        return {'valid':ok,'events':n,'reason':reason,
                'state':self.state.hex()[:24],'counter':self.counter}

# ═══════════════════════════════════════════════════════════════════
# 48-OS MS RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════
def reconstruct_ms_from_48(p: Params, identity_id: int) -> dict:
    """
    Runs inference across all 48 OS windows for identity_id.
    Returns median MS estimate + per-window statistics.
    This is the core proof: 48 independent OS chains each reconstruct
    the same MS → median consensus is the anchor.
    """
    NW = CFG['N_WINDOWS_PER_ID']
    ms_true = MS_ALL[identity_id]
    recons = []
    l2s    = []
    pv_scores = []

    for w in range(NW):
        g = identity_id * NW + w
        toks, meas = generate_os_chain(ms_true, g)
        X = build_X(toks, meas)[None]
        M = np.ones((1, CFG['SEQ_LEN']), np.float32)
        H, _   = gru_forward(p, X, M)
        ctx, _ = attention_forward(p, H, M)
        ms_hat, _ = ms_head_forward(p, ctx)
        # Also get p_valid for this window
        vr = verify_chain(p, toks, meas, identity_id, w, ms_true)
        recons.append(ms_hat[0].copy())
        l2s.append(float(np.linalg.norm(ms_hat[0] - ms_true)))
        pv_scores.append(float(vr['p_valid']))

    recons = np.array(recons)            # [48, MS_DIM]
    ms_est  = np.median(recons, axis=0)  # robust consensus
    ms_mean = recons.mean(axis=0)
    consensus_l2 = float(np.linalg.norm(ms_est - ms_true))
    accepted = consensus_l2 <= CFG['MS_RECON_TOL']

    return {
        'ms_est':         ms_est.tolist(),
        'ms_mean':        ms_mean.tolist(),
        'ms_true':        ms_true.tolist(),
        'consensus_l2':   round(consensus_l2, 5),
        'per_window_l2':  [round(v,5) for v in l2s],
        'per_window_pv':  [round(v,4) for v in pv_scores],
        'mean_l2':        round(float(np.mean(l2s)), 5),
        'min_l2':         round(float(np.min(l2s)), 5),
        'max_l2':         round(float(np.max(l2s)), 5),
        'windows_ok':     int(sum(1 for v in pv_scores if v >= CFG['THRESH_P_VALID'])),
        'windows_used':   NW,
        'accepted':       accepted,
    }

# ═══════════════════════════════════════════════════════════════════
# ANCHOR PROOF — AES-encrypted MS commitment for peer transfer
# ═══════════════════════════════════════════════════════════════════
class AnchorProof:
    """
    Once a peer reconstructs MS via 48 OS chains, it creates an anchor:
      - SHA3-256 commitment = hash(nonce || ms_est_bytes)
      - AES-256-GCM encrypted payload containing ms_est
    The anchor is transferred to the remote peer for mutual verification.
    """
    def __init__(self, peer_id: str, ms_est: np.ndarray, session_key: bytes):
        self.peer_id = peer_id
        self.ms_est  = ms_est.astype(np.float32).copy()
        self.key     = session_key
        nonce_b      = secrets.token_bytes(16)
        self.nonce   = nonce_b.hex()
        self.ts      = time.time()
        commit_data  = nonce_b + self.ms_est.tobytes()
        self.commitment = hashlib.sha3_256(commit_data).hexdigest()
        payload = json.dumps({
            'peer_id':    peer_id,
            'ms_est':     self.ms_est.tolist(),
            'nonce':      self.nonce,
            'ts':         self.ts,
            'commitment': self.commitment,
        }).encode()
        self.encrypted = aes_encrypt(payload, session_key)

    def to_dict(self) -> dict:
        return {'peer_id':self.peer_id,'commitment':self.commitment,
                'nonce':self.nonce,'ts':self.ts,'encrypted':self.encrypted}

    @staticmethod
    def verify_remote(pkg: dict, session_key: bytes,
                      local_ms_est: np.ndarray) -> dict:
        try:
            raw     = aes_decrypt(pkg['encrypted'], session_key)
            payload = json.loads(raw)
        except Exception as e:
            return {'ok':False,'reason':f'decrypt_failed:{e}'}
        nonce_b  = bytes.fromhex(payload['nonce'])
        ms_bytes = np.array(payload['ms_est'],dtype=np.float32).tobytes()
        expected = hashlib.sha3_256(nonce_b+ms_bytes).hexdigest()
        if expected != payload.get('commitment',''):
            return {'ok':False,'reason':'commitment_mismatch'}
        remote_ms = np.array(payload['ms_est'], dtype=np.float32)
        l2 = float(np.linalg.norm(remote_ms - local_ms_est))
        accepted = l2 <= CFG['MS_RECON_TOL'] * 2
        return {'ok':accepted,'peer_id':payload.get('peer_id','?'),
                'l2_distance':round(l2,5),'accepted':accepted,
                'remote_ms':payload['ms_est'],
                'reason':'ok' if accepted else 'ms_distance_exceeds_tolerance'}

# ═══════════════════════════════════════════════════════════════════
# PEER SESSION — full mutual handshake lifecycle
# ═══════════════════════════════════════════════════════════════════
class PeerSession:
    """
    Full lifecycle for one side of the dual-peer PIM handshake:
      1. reconstruct()         → 48-OS MS reconstruction
      2. build_anchor()        → AES-encrypted anchor
      3. verify_remote_anchor()→ mutual proof of shared MS
      4. verify()              → regular PIM chain verification
    All events logged to ChainProof with counters + time-deltas.
    """
    def __init__(self, peer_id: str, params: Params,
                 identity_id: int, session_key: bytes = None):
        self.peer_id     = peer_id
        self.p           = params
        self.identity_id = identity_id
        self.key         = session_key or secrets.token_bytes(32)
        self.chain       = ChainProof()
        self.model_hash  = self._hash_model()
        self.ms_recon    = None
        self.anchor      = None
        self.remote_vfy  = None
        self.handshake_ok= False

    def _hash_model(self):
        h = hashlib.sha3_256()
        for k in self.p.__slots__: h.update(getattr(self.p,k).tobytes())
        return h.hexdigest()

    def reconstruct(self) -> dict:
        t0 = time.perf_counter()
        result = reconstruct_ms_from_48(self.p, self.identity_id)
        result['latency_ms'] = round((time.perf_counter()-t0)*1000, 2)
        self.ms_recon = result
        self.chain.append({
            'event':        'MS_RECONSTRUCT',
            'peer_id':      self.peer_id,
            'identity_id':  self.identity_id,
            'accepted':     result['accepted'],
            'consensus_l2': result['consensus_l2'],
            'windows_ok':   result['windows_ok'],
            'windows_used': result['windows_used'],
            'latency_ms':   result['latency_ms'],
            'model_hash':   self.model_hash[:16],
        })
        return result

    def build_anchor(self) -> dict:
        if self.ms_recon is None: raise RuntimeError("Call reconstruct() first")
        ms_est = np.array(self.ms_recon['ms_est'], dtype=np.float32)
        self.anchor = AnchorProof(self.peer_id, ms_est, self.key)
        pkg = self.anchor.to_dict()
        self.chain.append({
            'event':      'ANCHOR_BUILT',
            'peer_id':    self.peer_id,
            'commitment': pkg['commitment'][:24],
        })
        return pkg

    def verify_remote_anchor(self, remote_pkg: dict) -> dict:
        if self.anchor is None: raise RuntimeError("Call build_anchor() first")
        local_ms = np.array(self.ms_recon['ms_est'], dtype=np.float32)
        result   = AnchorProof.verify_remote(remote_pkg, self.key, local_ms)
        self.remote_vfy  = result
        self.handshake_ok= bool(result.get('ok', False))
        self.chain.append({
            'event':       'ANCHOR_VERIFY',
            'peer_id':     self.peer_id,
            'remote_peer': result.get('peer_id','?'),
            'ok':          self.handshake_ok,
            'l2_distance': result.get('l2_distance', 999),
            'reason':      result.get('reason','?'),
        })
        return result

    def verify(self, claimed_id: int, expected_w: int,
               tokens=None, meas=None, true_ms=None) -> dict:
        if tokens is None:
            g=claimed_id*CFG['N_WINDOWS_PER_ID']+expected_w
            ms=MS_ALL[claimed_id] if 0<=claimed_id<CFG['N_IDENTITIES'] else np.zeros(CFG['MS_DIM'])
            tokens,meas=generate_os_chain(ms,g)
        t0=time.perf_counter()
        result=verify_chain(self.p,tokens,meas,claimed_id,expected_w,true_ms)
        latency_us=(time.perf_counter()-t0)*1e6
        entry=self.chain.append({
            'event':'VERIFY','peer_id':self.peer_id,
            'claimed_id':claimed_id,'expected_w':expected_w,
            'ok':result['ok'],'p_valid':round(result['p_valid'],4),
            'latency_us':round(latency_us,2),
        })
        result.update({'proof':entry['proof'],'seq':entry['seq'],
                       'counter':entry['event']['counter'],
                       'delta_ms':entry['event']['delta_ms'],
                       'latency_us':latency_us})
        return result

    def chain_status(self) -> dict:
        s = self.chain.status()
        s.update({'peer_id':self.peer_id,'handshake_ok':self.handshake_ok,
                  'recon_done':self.ms_recon is not None,
                  'anchor_done':self.anchor is not None,
                  'recent_events':self.chain.events[-12:]})
        return s

    def encrypt_payload(self, data: bytes) -> dict:
        return aes_encrypt(data, self.key)

    def decrypt_payload(self, pkg: dict) -> bytes:
        return aes_decrypt(pkg, self.key)

# ═══════════════════════════════════════════════════════════════════
# Legacy PIMSession (backwards compat with old server/UI)
# ═══════════════════════════════════════════════════════════════════
class PIMSession:
    def __init__(self, params, session_key=None):
        self.p=params; self.key=session_key or secrets.token_bytes(32)
        self.chain=ChainProof(); self.model_hash=self._hash_model()

    def _hash_model(self):
        h=hashlib.sha3_256()
        for k in self.p.__slots__: h.update(getattr(self.p,k).tobytes())
        return h.hexdigest()

    def verify(self, claimed_id, expected_w, tokens=None, meas=None, true_ms=None):
        if tokens is None:
            g=claimed_id*CFG['N_WINDOWS_PER_ID']+expected_w
            ms=MS_ALL[claimed_id] if 0<=claimed_id<CFG['N_IDENTITIES'] else np.zeros(CFG['MS_DIM'])
            tokens,meas=generate_os_chain(ms,g)
        t0=time.perf_counter()
        r=verify_chain(self.p,tokens,meas,claimed_id,expected_w,true_ms)
        lu=(time.perf_counter()-t0)*1e6
        e=self.chain.append({'event':'VERIFY','claimed_id':claimed_id,
                              'expected_w':expected_w,'ok':r['ok'],
                              'p_valid':round(r['p_valid'],4),'latency_us':round(lu,2)})
        r.update({'proof':e['proof'],'seq':e['seq'],
                  'counter':e['event']['counter'],'delta_ms':e['event']['delta_ms'],
                  'latency_us':lu})
        return r

    def encrypt_tokens(self, tokens, meas):
        return aes_encrypt(json.dumps({'tokens':tokens.tolist(),'meas':meas.tolist()}).encode(), self.key)

    def decrypt_tokens(self, pkg):
        d=json.loads(aes_decrypt(pkg, self.key))
        return np.array(d['tokens'],dtype=np.int32), np.array(d['meas'],dtype=np.float32)

    def chain_status(self):
        s=self.chain.status(); s['recent_events']=self.chain.events[-10:]
        return s
