# tab5_live_attack_predictor.py
# ─────────────────────────────────────────────────────────────────────────────
# Standalone Tab 6 — Live Manual Attack Editor + Real-time RNN Retrain + Predict
#
# Run as a standalone app:
#   streamlit run tab5_live_attack_predictor.py
#
# Or drop the `with tab5:` block into warlok_master_v3_rnn_predictor.py
# and add  tab5  to the st.tabs(…) call.
#
# What this does:
#   1. User queues manual field-edits (msg_idx, field, new_value)
#   2. On "Run + Retrain + Predict":
#      a) simulate_manual_attack → produces a tampered trace
#      b) build_attack_dataset_with_manual → injects that trace as a new
#         labelled "manual" class alongside the 6 standard classes
#      c) train_pred_model_incremental → trains a FRESH 7-class GRU from
#         scratch on the combined dataset (fast, ~40 epochs)
#      d) predict_attack_7class → runs inference on the just-simulated trace
#         and shows live probability bars, confidence gauge and full trace
# ─────────────────────────────────────────────────────────────────────────────

import time, secrets, hashlib, hmac, queue, base64
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# ── Shared crypto / protocol helpers (self-contained copy) ───────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def _hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    out, t, c = b"", b"", 1
    while len(out) < length:
        t = hmac.new(key, t + info + bytes([c]), hashlib.sha256).digest()
        out += t; c += 1
    return out[:length]

def _mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def _b64e(b: bytes) -> bytes: return base64.urlsafe_b64encode(b)
def _b64d(b: bytes) -> bytes: return base64.urlsafe_b64decode(b)
def _bhex(b: bytes, n: int = 16) -> str: return b.hex()[:n]

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _softmax1d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def _softmax2d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=-1, keepdims=True) + 1e-12)

# ── DuplexWire ────────────────────────────────────────────────────────────────

class _DuplexWire:
    def __init__(self):
        self.a2b: queue.Queue = queue.Queue()
        self.b2a: queue.Queue = queue.Queue()
    def send_a(self, f): self.a2b.put(f)
    def send_b(self, f): self.b2a.put(f)
    def recv_a(self, timeout=0.5):
        try: return self.b2a.get(timeout=timeout)
        except queue.Empty: return None
    def recv_b(self, timeout=0.5):
        try: return self.a2b.get(timeout=timeout)
        except queue.Empty: return None

# ── P2P TLS record layer ──────────────────────────────────────────────────────

def _xor_stream(data, key, nonce):
    out, ctr, i = bytearray(), 0, 0
    while i < len(data):
        blk = _hkdf(key, nonce + ctr.to_bytes(4, "big"), 32)
        for bb in blk:
            if i >= len(data): break
            out.append(data[i] ^ bb); i += 1
        ctr += 1
    return bytes(out)

@dataclass
class _RecordKeys:
    enc_key: bytes
    mac_key: bytes

class _P2PTLS:
    def __init__(self, my_id, peer_id, psk, send_fn, recv_fn):
        self.my_id=my_id; self.peer_id=peer_id; self.psk=psk
        self.send_fn=send_fn; self.recv_fn=recv_fn
        self.keys: Optional[_RecordKeys]=None
        self.my_nonce=b""; self.peer_nonce=b""

    def hs1_send(self):
        self.my_nonce=secrets.token_bytes(16)
        self.send_fn(b"HS1|"+self.my_id.encode()+b"|"+_b64e(self.my_nonce))

    def hs2_recv_derive(self):
        msg=self.recv_fn()
        if not msg: return False
        _,pid,pn=msg.split(b"|",2)
        if pid.decode()!=self.peer_id: raise ValueError("peer id mismatch")
        self.peer_nonce=_b64d(pn)
        lo,hi=sorted([self.my_id,self.peer_id])
        nl,nh=(self.my_nonce,self.peer_nonce) if self.my_id==lo else (self.peer_nonce,self.my_nonce)
        prk=_H(self.psk+b"|"+lo.encode()+b"|"+hi.encode()+b"|"+nl+b"|"+nh)
        self.keys=_RecordKeys(_hkdf(prk,b"enc",32),_hkdf(prk,b"mac",32))
        return True

    def hs3_send_fin(self):
        assert self.keys
        fin=_mac(self.keys.mac_key,b"FIN|"+self.my_nonce+self.peer_nonce)
        self.send_fn(b"HS2|"+_b64e(fin))

    def hs4_recv_verify(self):
        assert self.keys
        msg=self.recv_fn()
        if not msg: return False
        fin2=_b64d(msg.split(b"|",1)[1])
        expect=_mac(self.keys.mac_key,b"FIN|"+self.peer_nonce+self.my_nonce)
        if not hmac.compare_digest(fin2,expect): raise ValueError("FIN mismatch")
        return True

    def send_rec(self, typ, payload):
        assert self.keys
        nonce=secrets.token_bytes(12)
        ct=_xor_stream(payload,self.keys.enc_key,nonce)
        tag=_mac(self.keys.mac_key,typ.encode()+b"|"+nonce+b"|"+ct)
        self.send_fn(b"REC|"+typ.encode()+b"|"+_b64e(nonce)+b"|"+_b64e(ct)+b"|"+_b64e(tag))

    def recv_rec(self):
        assert self.keys
        frame=self.recv_fn()
        if not frame: raise TimeoutError("no frame")
        _,typ_b,nb,ctb,tagb=frame.split(b"|",4)
        nonce=_b64d(nb); ct=_b64d(ctb); tag=_b64d(tagb)
        expect=_mac(self.keys.mac_key,typ_b+b"|"+nonce+b"|"+ct)
        if not hmac.compare_digest(tag,expect): raise ValueError("MAC failed")
        return typ_b.decode(),_xor_stream(ct,self.keys.enc_key,nonce)

# ── Protocol dataclasses ──────────────────────────────────────────────────────

WINDOW_SIZE = 48

@dataclass
class _StartGrant:
    ok: bool; reason: str=""; session_id: str=""
    window_id_start: int=0
    anchor_state_hash: bytes=b""; anchor_policy_hash: bytes=b""; signature: bytes=b""

@dataclass
class _WindowState:
    session_id: str; window_id: int
    expected_next_counter: int; expected_step_idx: int
    last_ts_ms: int; prev_mac_chain: bytes

@dataclass
class _ChainMsg:
    session_id: str; window_id: int; step_idx: int; global_counter: int
    dt_ms: int; op_code: str; payload_hash: bytes; os_token: bytes
    os_meas: float; mac_chain: bytes

    def to_bytes(self):
        return b"|".join([
            self.session_id.encode(), str(self.window_id).encode(),
            str(self.step_idx).encode(), str(self.global_counter).encode(),
            str(self.dt_ms).encode(), self.op_code.encode(),
            _b64e(self.payload_hash), _b64e(self.os_token),
            f"{self.os_meas:.6f}".encode(), _b64e(self.mac_chain),
        ])

    @staticmethod
    def from_bytes(b):
        p=b.split(b"|")
        return _ChainMsg(
            session_id=p[0].decode(), window_id=int(p[1]), step_idx=int(p[2]),
            global_counter=int(p[3]), dt_ms=int(p[4]), op_code=p[5].decode(),
            payload_hash=_b64d(p[6]), os_token=_b64d(p[7]),
            os_meas=float(p[8].decode()), mac_chain=_b64d(p[9]),
        )

# ── IAM / PAM / Chain helpers ─────────────────────────────────────────────────

def _iam(peer_id, creds, posture):
    if peer_id not in creds or len(posture)<8: return {"ok":False,"reason":"IAM deny"}
    return {"ok":True,"peer_id":peer_id,"roles":["operator"] if "op" in creds else ["viewer"],
            "session_token":secrets.token_hex(16)}

def _pam(claim,target,actions):
    if not claim.get("ok"): return False,"PAM deny"
    if "operator" not in claim["roles"] and any(a in ("WRITE","DEPLOY","CONTROL") for a in actions):
        return False,"PAM insufficient role"
    return True,"ok"

def _warlok_start(local_peer,claim,target,actions,posture):
    ok,reason=_pam(claim,target,actions)
    if not ok: return _StartGrant(ok=False,reason=reason)
    sid=secrets.token_hex(8)
    anc=_H(b"WARLOK_ANCHOR|"+local_peer.encode()+b"|"+claim["peer_id"].encode()+b"|"+
           posture.encode()+b"|"+target.encode()+b"|"+b",".join(a.encode() for a in actions)+b"|"+
           str(int(time.time())).encode()+b"|"+sid.encode())
    pol=_H(b"POLICY|"+target.encode()+b"|"+b",".join(a.encode() for a in actions))
    return _StartGrant(ok=True,session_id=sid,window_id_start=0,
                       anchor_state_hash=anc,anchor_policy_hash=pol,signature=_H(b"SIGN|"+anc+pol))

def _os_fp(op,phash,ws):
    tok=_H(b"OS|"+op.encode()+b"|"+phash+b"|"+ws.prev_mac_chain)[:16]
    meas=int.from_bytes(_H(tok)[:4],"big")/(2**32)
    return tok,float(meas)

def _chain_fields(msg):
    return b"|".join([msg.session_id.encode(),str(msg.window_id).encode(),
                      str(msg.step_idx).encode(),str(msg.global_counter).encode(),
                      str(msg.dt_ms).encode(),msg.op_code.encode(),
                      msg.payload_hash,msg.os_token,f"{msg.os_meas:.6f}".encode()])

def _build_msg(chain_key,grant,ws,op,payload,sleep_ms=0):
    phash=_H(payload); tok,meas=_os_fp(op,phash,ws)
    now=int(time.time()*1000); dt=max(0,now-ws.last_ts_ms)
    msg=_ChainMsg(session_id=grant.session_id,window_id=ws.window_id,
                  step_idx=ws.expected_step_idx,global_counter=ws.expected_next_counter,
                  dt_ms=dt,op_code=op,payload_hash=phash,os_token=tok,os_meas=meas,mac_chain=b"")
    msg.mac_chain=_mac(chain_key,ws.prev_mac_chain+b"|"+_chain_fields(msg))
    ws.prev_mac_chain=msg.mac_chain; ws.last_ts_ms=now
    ws.expected_next_counter+=1; ws.expected_step_idx+=1
    if ws.expected_step_idx==WINDOW_SIZE:
        ws.window_id+=1; ws.expected_step_idx=0
        ws.prev_mac_chain=_H(b"WINDOW_PILOT|"+grant.session_id.encode()+b"|"+str(ws.window_id).encode())
    if sleep_ms>0: time.sleep(sleep_ms/1000.0)
    return msg

@dataclass
class _NanoBundle:
    peer_id:str; anchor_state_hash:bytes; anchor_policy_hash:bytes
    dt_ms_range:Tuple[int,int]; meas_range:Tuple[float,float]; op_allowlist:set

def _train_profile(peer_id,grant,window,slack_dt=10,slack_meas=0.02):
    dt_vals=[m.dt_ms for m in window] or [0]
    ms_vals=[m.os_meas for m in window] or [0.0]
    return _NanoBundle(peer_id=peer_id,anchor_state_hash=grant.anchor_state_hash,
                       anchor_policy_hash=grant.anchor_policy_hash,
                       dt_ms_range=(max(0,int(min(dt_vals))),int(max(dt_vals)+slack_dt)),
                       meas_range=(float(min(ms_vals))-slack_meas,float(max(ms_vals))+slack_meas),
                       op_allowlist={m.op_code for m in window})

def _verify_msg(chain_key,bundle,ws,msg):
    if msg.session_id!=ws.session_id:         return False,"DROP: wrong session_id"
    if msg.window_id!=ws.window_id:           return False,"DROP: wrong window_id (drift/replay)"
    if msg.step_idx!=ws.expected_step_idx:    return False,"DROP: step mismatch (reorder/drop)"
    if msg.global_counter!=ws.expected_next_counter: return False,"DROP: counter mismatch (replay/fork)"
    dt_min,dt_max=bundle.dt_ms_range
    if not (dt_min<=msg.dt_ms<=dt_max):       return False,"DROP: dt_ms anomaly (time-warp/burst)"
    if msg.op_code not in bundle.op_allowlist: return False,"DROP: op_code not in allowlist"
    expect=_mac(chain_key,ws.prev_mac_chain+b"|"+_chain_fields(msg))
    if not hmac.compare_digest(expect,msg.mac_chain): return False,"DROP: mac_chain mismatch (splice/tamper)"
    mn,mx=bundle.meas_range
    if not (mn<=msg.os_meas<=mx): return False,"DROP: os_meas outside learned range (mimic/impersonation)"
    ws.prev_mac_chain=msg.mac_chain; ws.expected_next_counter+=1; ws.expected_step_idx+=1
    ws.last_ts_ms=int(time.time()*1000)
    if ws.expected_step_idx==WINDOW_SIZE:
        ws.window_id+=1; ws.expected_step_idx=0
        ws.prev_mac_chain=_H(b"WINDOW_PILOT|"+ws.session_id.encode()+b"|"+str(ws.window_id).encode())
    return True,"ACCEPT"

# ── Named attack mutators ─────────────────────────────────────────────────────

def _atk_reorder(w):
    w=w[:];
    if len(w)>=12: w[10],w[11]=w[11],w[10]
    return w

def _atk_drop(w):    return [m for i,m in enumerate(w) if i!=20]
def _atk_replay(w):
    out=[]
    for i,m in enumerate(w):
        out.append(m)
        if i==5: out.append(m)
    return out

def _atk_timewarp(w,dt=999999):
    out=[]
    for i,m in enumerate(w):
        if i==7: mm=_ChainMsg(**m.__dict__); mm.dt_ms=dt; out.append(mm)
        else: out.append(m)
    return out

def _atk_splice(w):
    out=[]
    for i,m in enumerate(w):
        if i==12: mm=_ChainMsg(**m.__dict__); mm.op_code="CONTROL"; out.append(mm)
        else: out.append(m)
    return out

# ── Core simulate function ────────────────────────────────────────────────────

def _simulate(steps, attack_mode="none", edits=None, sleep_ms=0,
              slack_dt=10, slack_meas=0.02, op_pattern="rw",
              forensic_continue=True):
    """
    Unified simulator. attack_mode="manual" uses edits list.
    Returns full result dict including trace.
    """
    edits = edits or []
    wire=_DuplexWire()
    pA,pB="peerA","peerB"
    psk=_H(b"mutual-trust-root|"+pA.encode()+b"|"+pB.encode())
    tlsA=_P2PTLS(pA,pB,psk,wire.send_a,lambda:wire.recv_a(0.5))
    tlsB=_P2PTLS(pB,pA,psk,wire.send_b,lambda:wire.recv_b(0.5))

    tlsA.hs1_send(); tlsB.hs1_send()
    for _ in range(10):
        if not tlsA.keys: tlsA.hs2_recv_derive()
        if not tlsB.keys: tlsB.hs2_recv_derive()
        if tlsA.keys and tlsB.keys: break
    if not(tlsA.keys and tlsB.keys): return {"ok":False,"reason":"HS1 timeout"}

    tlsA.hs3_send_fin(); tlsB.hs3_send_fin()
    okA=okB=False
    for _ in range(10):
        okA=okA or tlsA.hs4_recv_verify()
        okB=okB or tlsB.hs4_recv_verify()
        if okA and okB: break
    if not(okA and okB): return {"ok":False,"reason":"HS2 FIN timeout"}

    posture="posturehashA_12345678"
    claim=_iam(pA,"peerA_op_creds",posture)
    grant=_warlok_start(pA,claim,"pump-controller",["READ","WRITE"],posture)
    if not grant.ok: return {"ok":False,"reason":"START denied: "+grant.reason}

    chain_key=_hkdf(_H(b"chain|"+grant.anchor_state_hash+grant.anchor_policy_hash),b"chain-key",32)

    def mk_ws():
        return _WindowState(session_id=grant.session_id,window_id=0,
                            expected_next_counter=1,expected_step_idx=0,
                            last_ts_ms=int(time.time()*1000),
                            prev_mac_chain=_H(b"WINDOW_PILOT|"+grant.session_id.encode()+b"|0"))

    def get_op(i):
        if op_pattern=="rw":  return "READ" if i%3 else "WRITE"
        if op_pattern=="read": return "READ"
        return "WRITE"

    # Train profile (clean)
    wsT=mk_ws(); tw=[]
    for i in range(min(WINDOW_SIZE,steps)):
        tw.append(_build_msg(chain_key,grant,wsT,get_op(i),f"op{i}".encode(),sleep_ms))
    bundle=_train_profile(pA,grant,tw,slack_dt,slack_meas)

    # Build clean send window
    wsS=mk_ws(); sw=[]
    for i in range(steps):
        sw.append(_build_msg(chain_key,grant,wsS,get_op(i),f"op{i}".encode(),sleep_ms))

    # Apply attack
    if   attack_mode=="reorder":  attacked=_atk_reorder(sw)
    elif attack_mode=="drop":     attacked=_atk_drop(sw)
    elif attack_mode=="replay":   attacked=_atk_replay(sw)
    elif attack_mode=="timewarp": attacked=_atk_timewarp(sw)
    elif attack_mode=="splice":   attacked=_atk_splice(sw)
    elif attack_mode=="manual":
        INT_F={"dt_ms","step_idx","global_counter","window_id"}
        FLOAT_F={"os_meas"}
        em: Dict[int,list]={}
        for e in edits:
            f=str(e["field"]); v=e["value"]
            v=int(v) if f in INT_F else (float(v) if f in FLOAT_F else str(v))
            em.setdefault(int(e["msg_idx"]),[]).append((f,v))
        attacked=[]
        for i,msg in enumerate(sw):
            if i in em:
                mm=_ChainMsg(**msg.__dict__)
                for f,v in em[i]: setattr(mm,f,v)
                attacked.append(mm)
            else: attacked.append(msg)
    else: attacked=sw

    tampered_set={int(e["msg_idx"]) for e in edits} if attack_mode=="manual" else set()

    # Verify stream
    wsB=mk_ws(); trace=[]; accepted=0; first_drop=None
    rx_tok,rx_meas,rx_ops,rx_dt,rx_step,rx_ctr,rx_win=[],[],[],[],[],[],[]

    for idx,m in enumerate(attacked):
        tlsA.send_rec("CHAIN",m.to_bytes())
        _,pt=tlsB.recv_rec()
        rx=_ChainMsg.from_bytes(pt)
        ok,reason=_verify_msg(chain_key,bundle,wsB,rx)
        dec="ACCEPT" if ok else "DROP"
        if dec=="DROP" and first_drop is None: first_drop=reason
        trace.append({"i":idx,"win":rx.window_id,"step":rx.step_idx,"ctr":rx.global_counter,
                      "dt_ms":rx.dt_ms,"op":rx.op_code,"meas":round(rx.os_meas,6),
                      "decision":dec,"reason":reason,"tampered":"✏️" if idx in tampered_set else "",
                      "B_win":wsB.window_id,"B_step":wsB.expected_step_idx,"B_ctr":wsB.expected_next_counter})
        rx_tok.append(rx.os_token); rx_meas.append(float(rx.os_meas)); rx_ops.append(rx.op_code)
        rx_dt.append(int(rx.dt_ms)); rx_step.append(int(rx.step_idx))
        rx_ctr.append(int(rx.global_counter)); rx_win.append(int(rx.window_id))
        if ok: accepted+=1
        elif not forensic_continue: break

    return {"ok":True,"attack_mode":attack_mode,"session_id":grant.session_id,
            "anchor":_bhex(grant.anchor_state_hash),"chain_key":_bhex(chain_key),
            "dt_range":bundle.dt_ms_range,
            "meas_range":(round(bundle.meas_range[0],6),round(bundle.meas_range[1],6)),
            "op_allowlist":sorted(bundle.op_allowlist),
            "accepted":accepted,"sent":len(rx_meas),"dropped_reason":first_drop,
            "tampered_indices":sorted(tampered_set),
            "edits_applied":edits,"trace":trace,
            "stream":{"os_token":rx_tok,"meas":rx_meas,"op":rx_ops,"dt_ms":rx_dt,
                      "step":rx_step,"ctr":rx_ctr,"win":rx_win}}

# ══════════════════════════════════════════════════════════════════════════════
# ── RNN predictor (7-class: 6 named attacks + "manual") ──────────────────────
# ══════════════════════════════════════════════════════════════════════════════

STD_ATTACKS  = ["none","reorder","drop","replay","timewarp","splice"]
ALL_LABELS   = STD_ATTACKS + ["manual"]   # 7 classes
N_CLASSES    = len(ALL_LABELS)            # 7
RNN_HDIM     = 64
RNN_IN_DIM   = 9
RNN_SEQ      = 48
RNN_LR       = 0.008
RNN_EPOCHS   = 50
RNN_BATCH    = 24
N_PER_STD    = 30   # synthetic samples per standard attack class


def _featurise(trace: List[Dict], max_dt: float = 200.0) -> np.ndarray:
    """Convert a trace list → (RNN_SEQ, RNN_IN_DIM) float32 array."""
    op_map={"READ":0,"WRITE":1,"CONTROL":2}
    X=np.zeros((RNN_SEQ,RNN_IN_DIM),dtype=np.float32)
    for t,r in enumerate(trace[:RNN_SEQ]):
        oi=op_map.get(r["op"],1)
        X[t,0]=float(r["meas"])
        X[t,1]=float(r["dt_ms"])/max(max_dt,1.0)
        X[t,2]=float(r["step"])/(WINDOW_SIZE-1)
        X[t,3]=float(r["ctr"])/max(RNN_SEQ,1.0)
        X[t,4]=1.0 if r["decision"]=="ACCEPT" else 0.0
        X[t,5+oi]=1.0
        X[t,8]=float(t)/max(RNN_SEQ-1,1.0)
    return X


def _init_rnn(seed: int = 0, n_classes: int = N_CLASSES) -> dict:
    rng=np.random.RandomState(seed); s=0.08
    D,H,C=RNN_IN_DIM,RNN_HDIM,n_classes
    p={"_H_placeholder":np.zeros(H,dtype=np.float32)}
    for g in("z","r","h"):
        p[f"W{g}"]=(rng.randn(H,D)*s).astype(np.float32)
        p[f"U{g}"]=(rng.randn(H,H)*s).astype(np.float32)
        p[f"b{g}"]=np.zeros(H,dtype=np.float32)
    p["Wc"]=(rng.randn(C,H)*s).astype(np.float32)
    p["bc"]=np.zeros(C,dtype=np.float32)
    return p


def _rnn_forward(Xb: np.ndarray, p: dict) -> np.ndarray:
    """(B,T,D) → (B,H) mean-pooled context."""
    B,T,_=Xb.shape; H=p["Wz"].shape[0]
    h=np.zeros((B,H),dtype=np.float32); s=np.zeros((B,H),dtype=np.float32)
    for t in range(T):
        x=Xb[:,t,:]
        z=_sigmoid(x@p["Wz"].T+h@p["Uz"].T+p["bz"])
        r=_sigmoid(x@p["Wr"].T+h@p["Ur"].T+p["br"])
        g=np.tanh(x@p["Wh"].T+(r*h)@p["Uh"].T+p["bh"])
        h=(1-z)*h+z*g; s+=h
    return s/T


def _rnn_step(p, Xb, yb, opt, lr):
    """One Adam gradient step. Returns scalar loss."""
    B,T,D=Xb.shape; H=p["Wz"].shape[0]

    # ── forward with cache ────────────────────────────────────────────────────
    hs=np.zeros((B,T,H),dtype=np.float32)
    zs=np.zeros((B,T,H),dtype=np.float32)
    rs=np.zeros((B,T,H),dtype=np.float32)
    gs=np.zeros((B,T,H),dtype=np.float32)
    hp_arr=np.zeros((B,T,H),dtype=np.float32)
    h=np.zeros((B,H),dtype=np.float32)
    for t in range(T):
        x=Xb[:,t,:]
        hp_arr[:,t,:]=h
        z=_sigmoid(x@p["Wz"].T+h@p["Uz"].T+p["bz"])
        r=_sigmoid(x@p["Wr"].T+h@p["Ur"].T+p["br"])
        g=np.tanh(x@p["Wh"].T+(r*h)@p["Uh"].T+p["bh"])
        h=(1-z)*h+z*g
        hs[:,t,:]=h; zs[:,t,:]=z; rs[:,t,:]=r; gs[:,t,:]=g

    ctx=hs.mean(axis=1)                         # (B,H)
    logits=ctx@p["Wc"].T+p["bc"]               # (B,C)
    probs=_softmax2d(logits)
    loss=float(-np.mean(np.log(probs[np.arange(B),yb]+1e-12)))

    # ── backward ──────────────────────────────────────────────────────────────
    dlog=probs.copy(); dlog[np.arange(B),yb]-=1.0; dlog/=B
    grads={"Wc":dlog.T@ctx,"bc":dlog.sum(0)}
    dctx=dlog@p["Wc"]                           # (B,H)
    dh_ctx=dctx/T                               # distribute to every step

    dWz=np.zeros_like(p["Wz"]); dUz=np.zeros_like(p["Uz"]); dbz=np.zeros_like(p["bz"])
    dWr=np.zeros_like(p["Wr"]); dUr=np.zeros_like(p["Ur"]); dbr=np.zeros_like(p["br"])
    dWh=np.zeros_like(p["Wh"]); dUh=np.zeros_like(p["Uh"]); dbh=np.zeros_like(p["bh"])
    dhn=np.zeros((B,H),dtype=np.float32)

    for t in reversed(range(T)):
        x=Xb[:,t,:]; hp=hp_arr[:,t,:]
        z,r,g=zs[:,t,:],rs[:,t,:],gs[:,t,:]
        dh=dh_ctx+dhn
        dg=dh*z; dz=dh*(g-hp); dhp=dh*(1-z)
        dag=dg*(1-g*g)
        dWh+=dag.T@x; dUh+=dag.T@(r*hp); dbh+=dag.sum(0)
        dhp+=(dag@p["Uh"])*r; dr=(dag@p["Uh"])*hp
        dar=dr*r*(1-r)
        dWr+=dar.T@x; dUr+=dar.T@hp; dbr+=dar.sum(0); dhp+=dar@p["Ur"]
        daz=dz*z*(1-z)
        dWz+=daz.T@x; dUz+=daz.T@hp; dbz+=daz.sum(0); dhp+=daz@p["Uz"]
        dhn=dhp

    grads.update({"Wz":dWz,"Uz":dUz,"bz":dbz,
                  "Wr":dWr,"Ur":dUr,"br":dbr,
                  "Wh":dWh,"Uh":dUh,"bh":dbh})

    # ── Adam ──────────────────────────────────────────────────────────────────
    opt["t"]+=1; b1,b2,eps=0.9,0.999,1e-8; t_=opt["t"]
    for k,gv in grads.items():
        if k not in opt["m"]: opt["m"][k]=np.zeros_like(gv); opt["v"][k]=np.zeros_like(gv)
        opt["m"][k]=b1*opt["m"][k]+(1-b1)*gv
        opt["v"][k]=b2*opt["v"][k]+(1-b2)*(gv*gv)
        p[k]-=lr*opt["m"][k]/(1-b1**t_)/(np.sqrt(opt["v"][k]/(1-b2**t_))+eps)

    return loss


def _build_dataset(manual_traces: List[np.ndarray], seed: int = 0,
                   n_per_std: int = N_PER_STD) -> Tuple[np.ndarray,np.ndarray]:
    """
    Build (X, y) with n_per_std samples for each of the 6 standard attack
    classes, PLUS all provided manual_traces labelled as class 6 ("manual").
    """
    Xs,ys=[],[]
    for ci,atk in enumerate(STD_ATTACKS):
        for i in range(n_per_std):
            np.random.seed(seed*1000+ci*100+i)
            sl=int(np.random.randint(0,4))
            try:
                meta=_simulate(48,atk,sleep_ms=sl,forensic_continue=True)
                if not meta.get("ok") or not meta["trace"]: continue
                Xs.append(_featurise(meta["trace"])); ys.append(ci)
            except Exception: pass

    # Manual traces → class 6
    for X in manual_traces:
        Xs.append(X); ys.append(6)

    return np.stack(Xs).astype(np.float32), np.array(ys,dtype=np.int32)


def _train(manual_traces: List[np.ndarray], epochs: int = RNN_EPOCHS,
           seed: int = 0, log_cb=None) -> Tuple[dict,List[float]]:
    """Train a fresh 7-class RNN on std attacks + all collected manual traces."""
    X,y=_build_dataset(manual_traces,seed=seed)
    N=X.shape[0]; p=_init_rnn(seed,N_CLASSES)
    opt={"t":0,"m":{},"v":{}}
    losses=[]
    for ep in range(1,epochs+1):
        idx=np.random.permutation(N); ep_loss=0.0; nb=0
        for s in range(0,N,RNN_BATCH):
            b=idx[s:s+RNN_BATCH]
            if len(b)<2: continue
            ep_loss+=_rnn_step(p,X[b],y[b],opt,RNN_LR); nb+=1
        avg=ep_loss/max(nb,1); losses.append(avg)
        if log_cb and (ep==1 or ep%max(1,epochs//8)==0):
            log_cb(f"epoch {ep:3d}/{epochs}  loss={avg:.4f}")
    return p,losses


def _predict(p: dict, trace: List[Dict]) -> Dict[str,Any]:
    X=_featurise(trace)[None,:,:]   # (1,T,D)
    ctx=_rnn_forward(X,p)
    logits=(ctx@p["Wc"].T+p["bc"])[0]
    probs=_softmax1d(logits)
    pi=int(np.argmax(probs))
    return {"label":ALL_LABELS[pi],"idx":pi,"conf":float(probs[pi]),
            "probs":{lbl:float(probs[i]) for i,lbl in enumerate(ALL_LABELS)}}


# ══════════════════════════════════════════════════════════════════════════════
# ── Streamlit UI — Tab 5 only ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="WARL0K — Live Attack Predictor", layout="wide")
st.title("⚔️ WARL0K — Live Manual Attack Editor + Real-time RNN Predictor")
st.caption(
    "Edit individual protocol messages, run the simulation, watch the RNN "
    "retrain from scratch (7 classes: 6 named + your manual attack), "
    "and get an instant prediction on your crafted trace."
)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "t5_edits":          [],    # queued edit dicts
    "t5_manual_traces":  [],    # list of featurised np arrays (one per run)
    "t5_result":         None,  # last simulate result dict
    "t5_model":          None,  # trained RNN params
    "t5_losses":         [],    # training loss curve
    "t5_pred":           None,  # last prediction result
    "t5_logs":           [],    # training log lines
    "t5_run_count":      0,     # how many manual runs done so far
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

EDITABLE_FIELDS = ["dt_ms","op_code","step_idx","global_counter",
                   "window_id","os_meas","session_id"]
FIELD_DEFAULTS  = {"dt_ms":"999999","op_code":"CONTROL","step_idx":"99",
                   "global_counter":"0","window_id":"99","os_meas":"0.999",
                   "session_id":"deadbeef"}
GATE_HINT = {
    "dt_ms":          "fires → dt_ms anomaly (time-warp/burst)",
    "op_code":        "fires → op_code not in allowlist",
    "step_idx":       "fires → step mismatch (reorder/drop)",
    "global_counter": "fires → counter mismatch (replay/fork)",
    "window_id":      "fires → wrong window_id (drift/replay)",
    "os_meas":        "fires → os_meas outside learned range",
    "session_id":     "fires → wrong session_id",
}

# ── Sidebar: simulation knobs ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Simulation settings")
    t5_steps    = st.slider("Steps", 16, 96, 48, 8,   key="t5_steps")
    t5_op       = st.selectbox("Op pattern", ["rw","read","write"], key="t5_op")
    t5_forensic = st.checkbox("Forensic continue", value=True, key="t5_forensic")
    st.divider()
    st.header("🧠 Training settings")
    t5_epochs   = st.slider("RNN epochs", 20, 150, 50, 10, key="t5_epochs")
    t5_seed     = st.number_input("Seed", 0, 9999, 42, 1, key="t5_seed")
    t5_n_std    = st.slider("Std samples/class", 10, 60, 30, 5, key="t5_n_std")
    st.divider()
    st.markdown(f"**Manual traces collected:** {st.session_state.t5_run_count}")
    if st.button("🗑 Reset all collected traces", key="t5_reset"):
        st.session_state.t5_manual_traces = []
        st.session_state.t5_run_count     = 0
        st.session_state.t5_model         = None
        st.session_state.t5_losses        = []
        st.session_state.t5_pred          = None
        st.session_state.t5_result        = None

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Edit queue
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1 · Build your attack — edit individual messages")

ea, eb, ec, ed = st.columns([1, 1.4, 1.8, 0.8])
with ea:
    new_idx   = st.number_input("Msg index", 0, t5_steps-1, 0, 1, key="t5_new_idx")
with eb:
    new_field = st.selectbox("Field", EDITABLE_FIELDS, key="t5_new_field")
with ec:
    new_val   = st.text_input("New value",
                               value=FIELD_DEFAULTS.get(new_field,""),
                               help=GATE_HINT.get(new_field,""),
                               key="t5_new_val")
with ed:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➕ Add", key="t5_add"):
        st.session_state.t5_edits.append(
            {"msg_idx":int(new_idx),"field":new_field,"value":new_val}
        )

# ── Edit queue display ────────────────────────────────────────────────────────
if st.session_state.t5_edits:
    qcol, rcol = st.columns([4, 1])
    with qcol:
        st.dataframe(st.session_state.t5_edits, use_container_width=True,
                     height=min(200, 44+35*len(st.session_state.t5_edits)))
    with rcol:
        ri = st.number_input("Remove #", 0,
                             max(0,len(st.session_state.t5_edits)-1), 0, 1,
                             key="t5_ri")
        if st.button("🗑 Remove", key="t5_rem"):
            if 0<=ri<len(st.session_state.t5_edits):
                st.session_state.t5_edits.pop(ri)
        if st.button("🧹 Clear all", key="t5_clr"):
            st.session_state.t5_edits=[]
else:
    st.info("Queue is empty — add at least one edit above, then hit **Run**.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Run + Retrain + Predict
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 2 · Run simulation → retrain RNN → predict in real-time")

run_btn = st.button(
    "🚀 Run attack  ·  Retrain RNN  ·  Predict",
    type="primary", key="t5_run",
    disabled=len(st.session_state.t5_edits)==0,
)

if run_btn:
    # ── (a) simulate ──────────────────────────────────────────────────────────
    with st.spinner("Simulating manual attack…"):
        result = _simulate(
            steps=t5_steps,
            attack_mode="manual",
            edits=st.session_state.t5_edits,
            op_pattern=t5_op,
            forensic_continue=t5_forensic,
        )
    st.session_state.t5_result = result

    if not result.get("ok"):
        st.error("Simulation failed: " + result.get("reason",""))
        st.stop()

    # Featurise and store this trace
    feat_X = _featurise(result["trace"])
    st.session_state.t5_manual_traces.append(feat_X)
    st.session_state.t5_run_count += 1

    # ── (b) retrain ───────────────────────────────────────────────────────────
    logs_box  = st.empty()
    prog_bar  = st.progress(0.0, text="Training…")
    log_lines: List[str] = []

    def _log(msg):
        log_lines.append(msg)
        logs_box.code("\n".join(log_lines[-12:]))
        # update progress bar heuristically
        try:
            ep=int(msg.split()[1].split("/")[0])
            tot=int(msg.split()[1].split("/")[1])
            prog_bar.progress(ep/tot, text=f"Training epoch {ep}/{tot}")
        except Exception: pass

    t0=time.time()
    p_model, losses = _train(
        manual_traces=st.session_state.t5_manual_traces,
        epochs=int(t5_epochs),
        seed=int(t5_seed),
        log_cb=_log,
    )
    prog_bar.progress(1.0, text=f"Done in {time.time()-t0:.1f}s  —  final loss: {losses[-1]:.4f}")
    st.session_state.t5_model  = p_model
    st.session_state.t5_losses = losses
    st.session_state.t5_logs   = log_lines

    # ── (c) predict ───────────────────────────────────────────────────────────
    st.session_state.t5_pred = _predict(p_model, result["trace"])

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Live results (always shown if data exists)
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.t5_result and st.session_state.t5_result.get("ok"):
    mr  = st.session_state.t5_result
    prd = st.session_state.t5_pred

    st.divider()
    st.markdown("## 3 · Results")

    # ── 3a: Simulation summary ────────────────────────────────────────────────
    st.markdown("### Simulation summary")
    s1,s2,s3,s4 = st.columns(4)
    with s1: st.metric("Steps sent",   mr["sent"])
    with s2: st.metric("Accepted",     mr["accepted"])
    with s3: st.metric("Dropped",      mr["sent"]-mr["accepted"])
    with s4:
        if mr["dropped_reason"]:
            st.error("First DROP:\n" + mr["dropped_reason"])
        else:
            st.success("✅ All accepted")

    col_ed, col_prof = st.columns([1,1])
    with col_ed:
        st.markdown("**Edits applied**")
        st.dataframe(mr["edits_applied"], use_container_width=True,
                     height=min(180,44+35*len(mr["edits_applied"])))
    with col_prof:
        st.markdown("**Learned profile**")
        st.write("dt gate:", mr["dt_range"])
        st.write("meas gate:", mr["meas_range"])
        st.write("op allowlist:", ", ".join(mr["op_allowlist"]))

    # ── 3b: Prediction ────────────────────────────────────────────────────────
    if prd:
        st.divider()
        st.markdown("### 🤖 RNN prediction")

        # Confidence gauge (native metric)
        g1, g2, g3 = st.columns(3)
        with g1:
            st.metric("Predicted class", prd["label"].upper(),
                      delta="manual" if prd["label"]=="manual" else prd["label"])
        with g2:
            st.metric("Confidence", f"{prd['conf']*100:.1f}%")
        with g3:
            trained_on = st.session_state.t5_run_count
            st.metric("Trained on # manual traces", trained_on)

        # Full probability bar chart
        st.markdown("**Per-class probabilities**")
        bar_data = {k: round(v,4) for k,v in prd["probs"].items()}
        st.bar_chart(bar_data)

    # ── 3c: Training loss ─────────────────────────────────────────────────────
    if st.session_state.t5_losses:
        st.divider()
        st.markdown("### Training loss curve")
        st.line_chart(st.session_state.t5_losses)

    # ── 3d: Timelines ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### Signal timelines")
    trace = mr["trace"]
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        st.markdown("**os_meas**")
        st.line_chart([r["meas"] for r in trace], height=160)
    with tc2:
        st.markdown("**dt_ms**")
        st.line_chart([r["dt_ms"] for r in trace], height=160)
    with tc3:
        st.markdown("**accept=1 / drop=0**")
        st.line_chart([1 if r["decision"]=="ACCEPT" else 0 for r in trace], height=160)

    # ── 3e: Full trace table ──────────────────────────────────────────────────
    st.divider()
    st.markdown("### Full trace  (✏️ = tampered message)")
    disp_cols = ["i","tampered","win","step","ctr","dt_ms","op","meas","decision","reason"]
    st.dataframe([{k:r[k] for k in disp_cols} for r in trace],
                 use_container_width=True, height=420)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Collected {st.session_state.t5_run_count} manual trace(s) this session.  "
    "Each new run appends to the training set so the RNN sees more examples of "
    "'manual' attacks over time.  Reset via the sidebar button."
)
