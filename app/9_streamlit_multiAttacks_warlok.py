# warlok_multi_attack.py
# ─────────────────────────────────────────────────────────────────────────────
# WARL0K — Multi-Attack GRU Predictor  (upgraded from single-attack version)
#
# Key upgrades over original:
#   ✦ Multi-label architecture  — model outputs a probability for EVERY attack
#     class simultaneously (multi-hot sigmoid, not single softmax).
#   ✦ Simultaneous multi-attack injection  — user can queue edits that combine
#     multiple attack types in one trace (e.g. timewarp + splice at once).
#   ✦ Combined-attack synthetic training data  — dataset now includes blended
#     attack traces so the model has seen compound patterns.
#   ✦ Top-K detection display  — shows ALL classes above a threshold with
#     individual confidence bars rather than just the single top prediction.
#   ✦ Attack correlation heatmap  — shows which attack classes fire together.
#   ✦ Per-attack accuracy tracking  — session scorecard per attack type.
#   ✦ Slicker dark-mode UI  — custom CSS, colour-coded attack badges.
#
# Run:  streamlit run warlok_multi_attack.py
# ─────────────────────────────────────────────────────────────────────────────

import time, secrets, hashlib, hmac, queue, base64
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# Crypto / protocol helpers  (self-contained)
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
    x = x - np.max(x); e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def _softmax2d(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True); e = np.exp(x)
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
    enc_key: bytes; mac_key: bytes

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
    ok: bool; reason: str=""
    session_id: str=""
    window_id_start: int=0
    anchor_state_hash: bytes=b""
    anchor_policy_hash: bytes=b""
    signature: bytes=b""

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
    return {"ok":True,"peer_id":peer_id,
            "roles":["operator"] if "op" in creds else ["viewer"],
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
           posture.encode()+b"|"+target.encode()+b"|"+
           b",".join(a.encode() for a in actions)+b"|"+
           str(int(time.time())).encode()+b"|"+sid.encode())
    pol=_H(b"POLICY|"+target.encode()+b"|"+b",".join(a.encode() for a in actions))
    return _StartGrant(ok=True,session_id=sid,window_id_start=0,
                       anchor_state_hash=anc,anchor_policy_hash=pol,
                       signature=_H(b"SIGN|"+anc+pol))

def _os_fp(op,phash,ws):
    tok=_H(b"OS|"+op.encode()+b"|"+phash+b"|"+ws.prev_mac_chain)[:16]
    meas=int.from_bytes(_H(tok)[:4],"big")/(2**32)
    return tok,float(meas)

def _chain_fields(msg):
    return b"|".join([
        msg.session_id.encode(),str(msg.window_id).encode(),
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
    if msg.session_id!=ws.session_id:          return False,"DROP: wrong session_id"
    if msg.window_id!=ws.window_id:            return False,"DROP: wrong window_id (drift/replay)"
    if msg.step_idx!=ws.expected_step_idx:     return False,"DROP: step mismatch (reorder/drop)"
    if msg.global_counter!=ws.expected_next_counter: return False,"DROP: counter mismatch (replay/fork)"
    dt_min,dt_max=bundle.dt_ms_range
    if not (dt_min<=msg.dt_ms<=dt_max):        return False,"DROP: dt_ms anomaly (time-warp/burst)"
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
    w=w[:]
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

# ── NEW: multi-attack combiner — applies a list of named attacks sequentially ─
def _atk_multi(w, attack_modes: List[str]) -> List:
    """Apply multiple named attacks in sequence to produce a compound trace."""
    current = w
    for mode in attack_modes:
        if   mode == "reorder":  current = _atk_reorder(current)
        elif mode == "drop":     current = _atk_drop(current)
        elif mode == "replay":   current = _atk_replay(current)
        elif mode == "timewarp": current = _atk_timewarp(current)
        elif mode == "splice":   current = _atk_splice(current)
    return current

# ── Core simulate function ────────────────────────────────────────────────────
def _simulate(steps, attack_modes: List[str] = None, edits=None, sleep_ms=0,
              slack_dt=10, slack_meas=0.02, op_pattern="rw",
              forensic_continue=True):
    """
    Unified simulator supporting multiple simultaneous attacks + manual edits.
    attack_modes: list of attack class strings, e.g. ["timewarp","splice"]
    Returns full result dict including trace.
    """
    attack_modes = attack_modes or ["none"]
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
        if op_pattern=="rw":   return "READ" if i%3 else "WRITE"
        if op_pattern=="read": return "READ"
        return "WRITE"

    wsT=mk_ws(); tw=[]
    for i in range(min(WINDOW_SIZE,steps)):
        tw.append(_build_msg(chain_key,grant,wsT,get_op(i),f"op{i}".encode(),sleep_ms))
    bundle=_train_profile(pA,grant,tw,slack_dt,slack_meas)

    wsS=mk_ws(); sw=[]
    for i in range(steps):
        sw.append(_build_msg(chain_key,grant,wsS,get_op(i),f"op{i}".encode(),sleep_ms))

    # Apply named attacks (multi-attack pipeline)
    active_attacks = [m for m in attack_modes if m != "none"]
    if active_attacks:
        attacked = _atk_multi(sw, active_attacks)
    else:
        attacked = sw[:]

    # Apply manual edits on top
    tampered_set: Set[int] = set()
    if edits:
        INT_F={"dt_ms","step_idx","global_counter","window_id"}
        FLOAT_F={"os_meas"}
        em: Dict[int,list]={}
        for e in edits:
            f=str(e["field"]); v=e["value"]
            v=int(v) if f in INT_F else (float(v) if f in FLOAT_F else str(v))
            em.setdefault(int(e["msg_idx"]),[]).append((f,v))
        patched=[]
        for i,msg in enumerate(attacked):
            if i in em:
                mm=_ChainMsg(**msg.__dict__)
                for f,v in em[i]: setattr(mm,f,v)
                patched.append(mm)
                tampered_set.add(i)
            else: patched.append(msg)
        attacked=patched

    wsB=mk_ws(); trace=[]; accepted=0; first_drop=None
    rx_tok,rx_meas,rx_ops,rx_dt,rx_step,rx_ctr,rx_win=[],[],[],[],[],[],[]

    for idx_m,m in enumerate(attacked):
        tlsA.send_rec("CHAIN",m.to_bytes())
        _,pt=tlsB.recv_rec()
        rx=_ChainMsg.from_bytes(pt)
        ok,reason=_verify_msg(chain_key,bundle,wsB,rx)
        dec="ACCEPT" if ok else "DROP"
        if dec=="DROP" and first_drop is None: first_drop=reason
        trace.append({"i":idx_m,"win":rx.window_id,"step":rx.step_idx,"ctr":rx.global_counter,
                      "dt_ms":rx.dt_ms,"op":rx.op_code,"meas":round(rx.os_meas,6),
                      "decision":dec,"reason":reason,
                      "tampered":"✏️" if idx_m in tampered_set else "",
                      "B_win":wsB.window_id,"B_step":wsB.expected_step_idx,"B_ctr":wsB.expected_next_counter})
        rx_tok.append(rx.os_token); rx_meas.append(float(rx.os_meas))
        rx_ops.append(rx.op_code); rx_dt.append(int(rx.dt_ms))
        rx_step.append(int(rx.step_idx)); rx_ctr.append(int(rx.global_counter))
        rx_win.append(int(rx.window_id))
        if ok: accepted+=1
        elif not forensic_continue: break

    return {"ok":True,"attack_modes":attack_modes,"session_id":grant.session_id,
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
# Multi-label GRU  — outputs sigmoid probability per attack class independently
# ══════════════════════════════════════════════════════════════════════════════

ATK_LABELS  = ["none","reorder","drop","replay","timewarp","splice"]
N_CLASSES   = len(ATK_LABELS)
RNN_HDIM    = 96     # wider hidden dim for multi-label capacity
RNN_IN_DIM  = 9
RNN_SEQ     = 48
RNN_LR      = 0.006
RNN_EPOCHS  = 60
RNN_BATCH   = 32
N_PER_STD   = 30     # synthetic single-attack traces per class
N_COMBO     = 20     # synthetic 2-attack combo traces per pair

# Map editable field → attack label(s) it signals
FIELD_TO_ATKS: Dict[str, List[str]] = {
    "dt_ms":          ["timewarp"],
    "op_code":        ["splice"],
    "step_idx":       ["reorder"],
    "global_counter": ["replay"],
    "window_id":      ["reorder"],
    "os_meas":        ["splice"],
    "session_id":     ["replay"],
}

ATK_COLORS = {
    "none":     "#4ade80",
    "reorder":  "#fb923c",
    "drop":     "#f87171",
    "replay":   "#c084fc",
    "timewarp": "#fbbf24",
    "splice":   "#38bdf8",
}

def infer_attack_labels(edits: List[Dict], named_attacks: List[str]) -> List[int]:
    """
    Return a multi-hot label vector (list of active class indices).
    Combines named attack modes + inferences from manual edits.
    """
    active: Set[str] = set()
    for a in named_attacks:
        if a != "none":
            active.add(a)
    for e in edits:
        for lbl in FIELD_TO_ATKS.get(str(e.get("field","")), []):
            active.add(lbl)
    if not active:
        active.add("none")
    return sorted([ATK_LABELS.index(a) for a in active if a in ATK_LABELS])


def _featurise(trace: List[Dict], max_dt: float = 200.0) -> np.ndarray:
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


def _init_rnn(seed: int = 0) -> dict:
    rng=np.random.RandomState(seed); s=0.06
    D,H,C=RNN_IN_DIM,RNN_HDIM,N_CLASSES
    p={}
    for g in("z","r","h"):
        p[f"W{g}"]=(rng.randn(H,D)*s).astype(np.float32)
        p[f"U{g}"]=(rng.randn(H,H)*s).astype(np.float32)
        p[f"b{g}"]=np.zeros(H,dtype=np.float32)
    # Multi-label head: sigmoid per class (not softmax)
    p["Wc"]=(rng.randn(C,H)*s).astype(np.float32)
    p["bc"]=np.zeros(C,dtype=np.float32)
    return p


def _rnn_forward(Xb: np.ndarray, p: dict) -> np.ndarray:
    """(B,T,D) → (B,H) mean-pooled."""
    B,T,_=Xb.shape; H=p["Wz"].shape[0]
    h=np.zeros((B,H),dtype=np.float32); s=np.zeros((B,H),dtype=np.float32)
    for t in range(T):
        x=Xb[:,t,:]
        z=_sigmoid(x@p["Wz"].T+h@p["Uz"].T+p["bz"])
        r=_sigmoid(x@p["Wr"].T+h@p["Ur"].T+p["br"])
        g=np.tanh(x@p["Wh"].T+(r*h)@p["Uh"].T+p["bh"])
        h=(1-z)*h+z*g; s+=h
    return s/T


def _rnn_step_multilabel(p, Xb, yb_multihot, opt, lr):
    """
    One Adam step for multi-label binary cross-entropy.
    yb_multihot: (B, C) float32 multi-hot targets.
    Returns scalar loss.
    """
    B,T,D=Xb.shape; H=p["Wz"].shape[0]; C=p["Wc"].shape[0]

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

    ctx=hs.mean(axis=1)                        # (B,H)
    logits=ctx@p["Wc"].T+p["bc"]              # (B,C)
    probs=_sigmoid(logits)                     # multi-label sigmoid

    # Binary cross-entropy
    eps=1e-7
    loss=float(-np.mean(
        yb_multihot*np.log(probs+eps)+(1-yb_multihot)*np.log(1-probs+eps)
    ))

    # Backward
    dlog=(probs-yb_multihot)/B               # (B,C)
    grads={"Wc":dlog.T@ctx,"bc":dlog.sum(0)}
    dctx=dlog@p["Wc"]                         # (B,H)
    dh_ctx=dctx/T

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

    opt["t"]+=1; b1,b2,eps_=0.9,0.999,1e-8; t_=opt["t"]
    for k,gv in grads.items():
        if k not in opt["m"]: opt["m"][k]=np.zeros_like(gv); opt["v"][k]=np.zeros_like(gv)
        opt["m"][k]=b1*opt["m"][k]+(1-b1)*gv
        opt["v"][k]=b2*opt["v"][k]+(1-b2)*(gv*gv)
        p[k]-=lr*opt["m"][k]/(1-b1**t_)/(np.sqrt(opt["v"][k]/(1-b2**t_))+eps_)

    return loss


def _make_multihot(label_indices: List[int]) -> np.ndarray:
    v=np.zeros(N_CLASSES,dtype=np.float32)
    for i in label_indices:
        v[i]=1.0
    return v


def _build_dataset(
    labelled_traces: List[Tuple[np.ndarray, List[int]]],
    seed: int = 0,
    n_per_std: int = N_PER_STD,
    n_combo: int = N_COMBO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (X, Y_multihot) for multi-label training.
    - n_per_std single-attack synthetic traces per class.
    - n_combo compound 2-attack combo synthetic traces.
    - labelled_traces: list of (featurised_X, [label_indices]) from user runs.
    """
    Xs, Ys = [], []
    single_attacks = [a for a in ATK_LABELS if a != "none"]

    # Single-attack traces (one class active)
    for ci, atk in enumerate(ATK_LABELS):
        for i in range(n_per_std):
            np.random.seed(seed*1000+ci*100+i)
            sl=int(np.random.randint(0,4))
            try:
                meta=_simulate(48,[atk],sleep_ms=sl,forensic_continue=True)
                if not meta.get("ok") or not meta["trace"]: continue
                Xs.append(_featurise(meta["trace"]))
                Ys.append(_make_multihot([ci]))
            except Exception: pass

    # Multi-attack combo traces (two classes active simultaneously)
    import itertools
    combo_pairs = list(itertools.combinations(single_attacks, 2))
    for pair_idx, (a1, a2) in enumerate(combo_pairs):
        for i in range(n_combo):
            np.random.seed(seed*500+pair_idx*50+i)
            sl=int(np.random.randint(0,3))
            try:
                meta=_simulate(48,[a1,a2],sleep_ms=sl,forensic_continue=True)
                if not meta.get("ok") or not meta["trace"]: continue
                Xs.append(_featurise(meta["trace"]))
                label_idxs=[ATK_LABELS.index(a1),ATK_LABELS.index(a2)]
                Ys.append(_make_multihot(label_idxs))
            except Exception: pass

    # User simulation traces
    for X, label_indices in labelled_traces:
        Xs.append(X)
        Ys.append(_make_multihot(label_indices))

    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


def _train(
    labelled_traces: List[Tuple[np.ndarray, List[int]]],
    epochs: int = RNN_EPOCHS,
    seed: int = 0,
    log_cb=None,
) -> Tuple[dict, List[float]]:
    X, Y = _build_dataset(labelled_traces, seed=seed)
    N=X.shape[0]; p=_init_rnn(seed)
    opt={"t":0,"m":{},"v":{}}
    losses=[]
    for ep in range(1,epochs+1):
        idx=np.random.permutation(N); ep_loss=0.0; nb=0
        for s in range(0,N,RNN_BATCH):
            b=idx[s:s+RNN_BATCH]
            if len(b)<2: continue
            ep_loss+=_rnn_step_multilabel(p,X[b],Y[b],opt,RNN_LR); nb+=1
        avg=ep_loss/max(nb,1); losses.append(avg)
        if log_cb and (ep==1 or ep%max(1,epochs//8)==0):
            log_cb(f"epoch {ep:3d}/{epochs}  loss={avg:.4f}")
    return p, losses


def _predict(p: dict, trace: List[Dict], threshold: float = 0.35) -> Dict[str, Any]:
    X=_featurise(trace)[None,:,:]
    ctx=_rnn_forward(X,p)
    logits=(ctx@p["Wc"].T+p["bc"])[0]
    probs=_sigmoid(logits)
    detected=[ATK_LABELS[i] for i in range(N_CLASSES) if probs[i]>=threshold]
    if not detected:
        detected=[ATK_LABELS[int(np.argmax(probs))]]  # always report at least top-1
    return {
        "detected":   detected,
        "probs":      {lbl: float(probs[i]) for i,lbl in enumerate(ATK_LABELS)},
        "threshold":  threshold,
        "top_label":  ATK_LABELS[int(np.argmax(probs))],
        "top_conf":   float(probs[int(np.argmax(probs))]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# ── Streamlit UI
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="WARL0K — Multi-Attack Predictor", layout="wide")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;600;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
    background-color: #0a0d14;
    color: #e2e8f0;
  }
  h1, h2, h3 { font-family: 'Exo 2', sans-serif; font-weight: 800; letter-spacing: -0.02em; }
  code, .stCode { font-family: 'Share Tech Mono', monospace !important; }

  .stButton > button {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2040 100%);
    color: #7dd3fc; border: 1px solid #1e4d7a; border-radius: 6px;
    font-family: 'Share Tech Mono', monospace; letter-spacing: 0.05em;
    transition: all 0.2s ease;
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    color: #fff; border-color: #60a5fa; transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(96,165,250,0.3);
  }
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #4f46e5 100%);
    color: #fff; border-color: #a78bfa;
  }
  .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
    box-shadow: 0 4px 24px rgba(139,92,246,0.5);
  }

  .attack-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-family: 'Share Tech Mono', monospace; font-size: 0.82em;
    font-weight: 600; letter-spacing: 0.08em; margin: 2px;
    text-transform: uppercase;
  }
  .attack-none     { background:#14532d; color:#4ade80; border:1px solid #166534; }
  .attack-reorder  { background:#431407; color:#fb923c; border:1px solid #7c2d12; }
  .attack-drop     { background:#450a0a; color:#f87171; border:1px solid #991b1b; }
  .attack-replay   { background:#3b0764; color:#c084fc; border:1px solid #6b21a8; }
  .attack-timewarp { background:#451a03; color:#fbbf24; border:1px solid #92400e; }
  .attack-splice   { background:#082f49; color:#38bdf8; border:1px solid #0369a1; }

  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #0f172a 100%);
    border: 1px solid #1f2937; border-radius: 10px; padding: 16px 20px;
    margin: 4px 0;
  }
  .verdict-correct {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #166534; border-radius: 10px; padding: 16px;
  }
  .verdict-wrong {
    background: linear-gradient(135deg, #450a0a 0%, #3f0808 100%);
    border: 1px solid #991b1b; border-radius: 10px; padding: 16px;
  }
  .verdict-partial {
    background: linear-gradient(135deg, #451a03 0%, #3f1a03 100%);
    border: 1px solid #92400e; border-radius: 10px; padding: 16px;
  }

  .section-header {
    font-family: 'Share Tech Mono', monospace;
    color: #64748b; font-size: 0.75em; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 4px; border-bottom: 1px solid #1f2937;
    padding-bottom: 6px;
  }
  div[data-testid="stMetricValue"] { color: #e2e8f0 !important; font-weight: 700; }
  div[data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.8em; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="background: linear-gradient(90deg, #7c3aed, #2563eb, #0891b2);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
           font-size: 2.2rem; margin-bottom:0">
  ⚔️ WARL0K — Multi-Attack GRU Predictor
</h1>
<p style="color:#64748b; font-family:'Share Tech Mono',monospace; font-size:0.85em; margin-top:4px">
  Multi-label RNN · Simultaneous attack injection · Real-time retrain · Top-K detection
</p>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
defaults = {
    "t5_edits":           [],
    "t5_named_attacks":   [],      # list of active named attack modes
    "t5_labelled_traces": [],      # list of (X, [label_indices])
    "t5_result":          None,
    "t5_model":           None,
    "t5_losses":          [],
    "t5_pred":            None,
    "t5_logs":            [],
    "t5_run_count":       0,
    "t5_eval_history":    [],      # list of {"true":set,"pred":set,"correct":bool}
    "t5_threshold":       0.35,
}
for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k]=v

EDITABLE_FIELDS = ["dt_ms","op_code","step_idx","global_counter","window_id","os_meas","session_id"]
FIELD_DEFAULTS  = {"dt_ms":"999999","op_code":"CONTROL","step_idx":"99",
                   "global_counter":"0","window_id":"99","os_meas":"0.999","session_id":"deadbeef"}
GATE_HINT = {
    "dt_ms":          "→ timewarp: dt_ms anomaly",
    "op_code":        "→ splice: op_code not in allowlist",
    "step_idx":       "→ reorder: step mismatch",
    "global_counter": "→ replay: counter mismatch",
    "window_id":      "→ reorder: wrong window_id",
    "os_meas":        "→ splice: os_meas outside range",
    "session_id":     "→ replay: wrong session_id",
}

ATK_DESC = {
    "none":     "Clean traffic — no attack",
    "reorder":  "Messages arrive out of sequence",
    "drop":     "Messages silently removed (counter gap)",
    "replay":   "Messages replayed or counter forged",
    "timewarp": "Timing field (dt_ms) abnormal",
    "splice":   "Forbidden op-code injected or sensor forged",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Simulation")
    t5_steps    = st.slider("Steps", 16, 96, 48, 8,   key="t5_steps")
    t5_op       = st.selectbox("Op pattern", ["rw","read","write"], key="t5_op")
    t5_forensic = st.checkbox("Forensic continue", value=True, key="t5_forensic")

    st.markdown("---")
    st.markdown("### 🧠 Training")
    t5_epochs = st.slider("GRU epochs", 20, 200, 60, 10, key="t5_epochs")
    t5_seed   = st.number_input("Seed", 0, 9999, 42, 1, key="t5_seed")
    t5_n_std  = st.slider("Std samples/class", 10, 60, 30, 5, key="t5_n_std")
    t5_n_cmb  = st.slider("Combo samples/pair", 5, 40, 20, 5, key="t5_n_cmb")

    st.markdown("---")
    st.markdown("### 🎯 Detection threshold")
    t5_thresh = st.slider("Sigmoid threshold", 0.10, 0.75, 0.35, 0.05, key="t5_thresh",
                          help="Classes with probability ≥ threshold are flagged as detected.")
    st.session_state.t5_threshold = t5_thresh

    st.markdown("---")
    st.markdown(f"**Runs:** {st.session_state.t5_run_count}")
    history = st.session_state.t5_eval_history
    if history:
        n_correct = sum(1 for h in history if h["correct"])
        st.markdown(f"**Session accuracy:** {n_correct}/{len(history)} = {n_correct/len(history)*100:.0f}%")
    if st.button("🗑 Reset session", key="t5_reset"):
        for k in ["t5_labelled_traces","t5_run_count","t5_model","t5_losses",
                  "t5_pred","t5_result","t5_eval_history","t5_edits","t5_named_attacks"]:
            st.session_state[k] = [] if isinstance(defaults.get(k),[]) else None if defaults.get(k) is None else defaults.get(k,0)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Named attack selector
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">SECTION 1 — SELECT ATTACK TYPES</div>', unsafe_allow_html=True)
st.markdown("### 🎯 Choose which attacks to inject simultaneously")
st.markdown(
    "Select **one or more** attack types. The model will be trained to detect "
    "each one independently — including compound combinations."
)

atk_cols = st.columns(len(ATK_LABELS))
active_named: List[str] = []
for i, lbl in enumerate(ATK_LABELS):
    with atk_cols[i]:
        cls = f"attack-{lbl}"
        checked = st.checkbox(lbl.upper(), value=(lbl in st.session_state.t5_named_attacks),
                              key=f"t5_atk_{lbl}")
        if checked:
            active_named.append(lbl)
st.session_state.t5_named_attacks = active_named

if active_named:
    badges = " ".join(f'<span class="attack-badge attack-{a}">{a}</span>' for a in active_named)
    st.markdown(f"**Active attack mix:** {badges}", unsafe_allow_html=True)
    for a in active_named:
        if a != "none":
            st.caption(f"• **{a}**: {ATK_DESC[a]}")
else:
    st.info("No attacks selected — check at least one box above (or add manual edits below).")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Manual fine-grained edits
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">SECTION 2 — MANUAL FIELD EDITS (OPTIONAL)</div>', unsafe_allow_html=True)
st.markdown("### ✏️ Fine-tune: edit individual message fields")
st.markdown(
    "These are applied **on top of** the named attacks above. "
    "Each edited field will also be inferred as an additional attack label."
)

ea, eb, ec, ed = st.columns([1, 1.4, 1.8, 0.8])
with ea:
    new_idx   = st.number_input("Msg #", 0, t5_steps-1, 0, 1, key="t5_new_idx")
with eb:
    new_field = st.selectbox("Field", EDITABLE_FIELDS, key="t5_new_field")
with ec:
    new_val   = st.text_input("Value", value=FIELD_DEFAULTS.get(new_field,""),
                               help=GATE_HINT.get(new_field,""), key="t5_new_val")
with ed:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("➕ Add edit", key="t5_add"):
        st.session_state.t5_edits.append({"msg_idx":int(new_idx),"field":new_field,"value":new_val})

if st.session_state.t5_edits:
    qcol, rcol = st.columns([4, 1])
    with qcol:
        st.dataframe(st.session_state.t5_edits, use_container_width=True,
                     height=min(200, 44+35*len(st.session_state.t5_edits)))
    with rcol:
        ri = st.number_input("Remove #", 0, max(0,len(st.session_state.t5_edits)-1), 0, 1, key="t5_ri")
        if st.button("🗑 Remove", key="t5_rem"):
            if 0<=ri<len(st.session_state.t5_edits):
                st.session_state.t5_edits.pop(ri)
        if st.button("🧹 Clear all", key="t5_clr"):
            st.session_state.t5_edits=[]
else:
    st.caption("No manual edits queued — named attacks above will be used.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Run + Retrain + Predict
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">SECTION 3 — RUN SIMULATION · RETRAIN · PREDICT</div>', unsafe_allow_html=True)

has_something = bool(active_named) or bool(st.session_state.t5_edits)
run_btn = st.button(
    "🚀  Run Attack  ·  Retrain Multi-Label GRU  ·  Predict",
    type="primary", key="t5_run",
    disabled=not has_something,
)

if not has_something:
    st.caption("← Select at least one attack type (Section 1) or add a manual edit (Section 2) to enable.")

if run_btn:
    named = active_named if active_named else ["none"]

    with st.spinner("Simulating compound attack…"):
        result = _simulate(
            steps=t5_steps,
            attack_modes=named,
            edits=st.session_state.t5_edits,
            op_pattern=t5_op,
            forensic_continue=t5_forensic,
        )
    st.session_state.t5_result = result

    if not result.get("ok"):
        st.error("Simulation failed: " + result.get("reason",""))
        st.stop()

    # Infer multi-label ground truth
    true_indices = infer_attack_labels(st.session_state.t5_edits, named)
    feat_X = _featurise(result["trace"])
    st.session_state.t5_labelled_traces.append((feat_X, true_indices))
    st.session_state.t5_run_count += 1
    result["true_indices"] = true_indices
    result["true_labels"]  = [ATK_LABELS[i] for i in true_indices]
    st.session_state.t5_result = result

    # Retrain
    logs_box = st.empty(); prog_bar = st.progress(0.0, text="Training…")
    log_lines: List[str] = []

    def _log(msg):
        log_lines.append(msg)
        logs_box.code("\n".join(log_lines[-12:]))
        try:
            ep  = int(msg.split()[1].split("/")[0])
            tot = int(msg.split()[1].split("/")[1])
            prog_bar.progress(ep/tot, text=f"Epoch {ep}/{tot}")
        except Exception: pass

    t0 = time.time()
    p_model, losses = _train(
        labelled_traces=st.session_state.t5_labelled_traces,
        epochs=int(t5_epochs), seed=int(t5_seed), log_cb=_log,
    )
    prog_bar.progress(1.0, text=f"✅ Done in {time.time()-t0:.1f}s  —  final loss: {losses[-1]:.4f}")
    st.session_state.t5_model  = p_model
    st.session_state.t5_losses = losses
    st.session_state.t5_logs   = log_lines

    # Predict
    prd = _predict(p_model, result["trace"], threshold=st.session_state.t5_threshold)
    st.session_state.t5_pred = prd

    # Score
    true_set = set(result["true_labels"])
    pred_set = set(prd["detected"])
    correct  = true_set == pred_set
    partial  = bool(true_set & pred_set) and not correct
    st.session_state.t5_eval_history.append(
        {"true": true_set, "pred": pred_set, "correct": correct, "partial": partial}
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Results
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.t5_result and st.session_state.t5_result.get("ok"):
    mr    = st.session_state.t5_result
    prd   = st.session_state.t5_pred
    trace = mr["trace"]
    true_labels = mr.get("true_labels", ["none"])
    true_set    = set(true_labels)

    st.divider()
    st.markdown('<div class="section-header">SECTION 4 — RESULTS</div>', unsafe_allow_html=True)
    st.markdown("## 📊 Results")

    # ── Ground truth ─────────────────────────────────────────────────────────
    true_badges = " ".join(f'<span class="attack-badge attack-{a}">{a}</span>' for a in true_labels)
    st.markdown(f"**Ground-truth attacks injected:** {true_badges}", unsafe_allow_html=True)

    left_col, right_col = st.columns(2, gap="large")

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT — Rule-based verifier
    # ══════════════════════════════════════════════════════════════════════════
    with left_col:
        st.markdown("### 🔒 Rule-Based Verifier")
        st.caption("Deterministic cryptographic gate — fires first rule violated. No AI, no learning.")

        if mr["dropped_reason"]:
            st.error(f"**ATTACK DETECTED**\n\nFirst rule violated: `{mr['dropped_reason']}`")
        else:
            st.success("**ALL MESSAGES ACCEPTED** — no rule violations")

        v1,v2,v3 = st.columns(3)
        with v1: st.metric("Sent",     mr["sent"])
        with v2: st.metric("Accepted", mr["accepted"])
        with v3: st.metric("Dropped",  mr["sent"]-mr["accepted"])

        with st.expander("Verifier gate parameters"):
            st.code(
                f"dt_ms range  : {mr['dt_range'][0]} – {mr['dt_range'][1]} ms\n"
                f"meas range   : {mr['meas_range'][0]:.4f} – {mr['meas_range'][1]:.4f}\n"
                f"op allowlist : {', '.join(mr['op_allowlist'])}"
            )

        st.markdown("**Accept / Drop per message**")
        st.line_chart([1 if r["decision"]=="ACCEPT" else 0 for r in trace], height=140)

        tampered=[r for r in trace if r["tampered"]=="✏️"]
        if tampered:
            st.markdown("**Manually tampered messages ✏️**")
            st.dataframe(
                [{"#":r["i"],"op":r["op"],"dt_ms":r["dt_ms"],
                  "meas":r["meas"],"verdict":r["decision"],"rule":r["reason"]}
                 for r in tampered],
                use_container_width=True,
                height=min(200,44+35*len(tampered)),
            )

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT — Multi-label AI prediction
    # ══════════════════════════════════════════════════════════════════════════
    with right_col:
        st.markdown("### 🤖 Multi-Label GRU — AI Prediction")
        st.caption(
            "Simultaneous per-class sigmoid probabilities. "
            f"Classes ≥ **{st.session_state.t5_threshold:.0%}** are flagged as detected. "
            "Model trained with combined single + compound attack data."
        )

        if prd is None:
            st.warning("Run an attack first.")
        else:
            pred_set_   = set(prd["detected"])
            correct_    = true_set == pred_set_
            partial_    = bool(true_set & pred_set_) and not correct_

            # Verdict
            if correct_:
                st.markdown(
                    '<div class="verdict-correct"><b>✅ AI CORRECT</b><br>'
                    f'All {len(true_set)} attack class(es) correctly identified.</div>',
                    unsafe_allow_html=True
                )
            elif partial_:
                missed  = true_set - pred_set_
                extra   = pred_set_ - true_set
                msg_parts=[]
                if missed:  msg_parts.append(f"Missed: {', '.join(missed)}")
                if extra:   msg_parts.append(f"False positive: {', '.join(extra)}")
                st.markdown(
                    '<div class="verdict-partial"><b>⚠️ PARTIAL DETECTION</b><br>'
                    f'{" · ".join(msg_parts)}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="verdict-wrong"><b>❌ AI MISSED</b><br>'
                    f'None of the true attacks ({", ".join(true_set)}) were detected above threshold.</div>',
                    unsafe_allow_html=True
                )

            st.markdown("")

            # Metrics row
            m1,m2,m3 = st.columns(3)
            with m1:
                st.metric("True attacks", len(true_set))
            with m2:
                st.metric("AI detected",  len(pred_set_),
                          delta="✅ correct" if correct_ else ("⚠️ partial" if partial_ else "❌ missed"))
            with m3:
                history = st.session_state.t5_eval_history
                n_correct = sum(1 for h in history if h["correct"])
                st.metric("Session accuracy",
                          f"{n_correct/max(len(history),1)*100:.0f}%",
                          delta=f"{len(history)} run(s)")

            # Per-class probability bars with colour coding
            st.markdown(
                f"**Per-class sigmoid probabilities** "
                f"(threshold line at {st.session_state.t5_threshold:.0%}):"
            )
            bar_data = {
                f"{'→' if lbl in true_set else ''}{lbl}": round(prd["probs"][lbl], 4)
                for lbl in ATK_LABELS
            }
            st.bar_chart(bar_data, height=240)

            # Detected badges
            if pred_set_:
                det_badges = " ".join(
                    f'<span class="attack-badge attack-{a}">{a} {prd["probs"][a]*100:.0f}%</span>'
                    for a in ATK_LABELS if a in pred_set_
                )
                st.markdown(f"**Detected:** {det_badges}", unsafe_allow_html=True)

            # Detailed per-class table
            with st.expander("Full per-class probability table"):
                rows=[]
                for lbl in ATK_LABELS:
                    prob = prd["probs"][lbl]
                    rows.append({
                        "Class": lbl,
                        "Probability": f"{prob*100:.1f}%",
                        "Detected": "✅" if lbl in pred_set_ else "",
                        "True label": "🎯" if lbl in true_set else "",
                        "Status": (
                            "TP" if lbl in pred_set_ and lbl in true_set else
                            "FP" if lbl in pred_set_ and lbl not in true_set else
                            "FN" if lbl not in pred_set_ and lbl in true_set else
                            "TN"
                        )
                    })
                st.dataframe(rows, use_container_width=True, hide_index=True)

            with st.expander("How to improve multi-attack detection"):
                st.markdown(
                    "- **Run more compound attacks** of the same type mix — each adds a combined training sample.\n"
                    "- **Lower the threshold** (sidebar) to catch more attacks at the cost of more false positives.\n"
                    "- **Increase epochs** for harder-to-distinguish combinations.\n"
                    "- **Increase combo samples/pair** (sidebar) to generate more synthetic compound data.\n"
                    "- The model has wider hidden dim (96) and compound training pairs built-in."
                )

        # Training loss
        if st.session_state.t5_losses:
            st.markdown("**Training loss** (binary cross-entropy, multi-label)")
            final=st.session_state.t5_losses[-1]
            st.line_chart(st.session_state.t5_losses, height=130)
            quality = "✅ converged" if final<0.3 else ("⚠️ partial" if final<0.6 else "❌ still learning")
            st.caption(f"Final loss: {final:.4f} — {quality}")

    # ── Session scorecard ─────────────────────────────────────────────────────
    history = st.session_state.t5_eval_history
    if len(history) > 1:
        st.divider()
        st.markdown("### 📈 Session Scorecard")
        sc1,sc2,sc3,sc4 = st.columns(4)
        n_correct = sum(1 for h in history if h["correct"])
        n_partial = sum(1 for h in history if h["partial"])
        n_missed  = len(history) - n_correct - n_partial
        with sc1: st.metric("Total runs",       len(history))
        with sc2: st.metric("✅ Exact correct",  n_correct)
        with sc3: st.metric("⚠️ Partial detect", n_partial)
        with sc4: st.metric("❌ Missed",         n_missed)

        # Per-attack-type accuracy
        atk_true_counts  = {a:0 for a in ATK_LABELS}
        atk_hit_counts   = {a:0 for a in ATK_LABELS}
        for h in history:
            for a in h["true"]:
                atk_true_counts[a] += 1
                if a in h["pred"]:
                    atk_hit_counts[a] += 1
        score_rows=[]
        for a in ATK_LABELS:
            tc=atk_true_counts[a]; hc=atk_hit_counts[a]
            score_rows.append({
                "Attack class": a,
                "Times injected": tc,
                "Times detected": hc,
                "Recall": f"{hc/max(tc,1)*100:.0f}%" if tc>0 else "—"
            })
        st.dataframe(score_rows, use_container_width=True, hide_index=True, height=270)

    # ── Full trace ─────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📋 Full message trace")
    disp_cols=["i","tampered","win","step","ctr","dt_ms","op","meas","decision","reason"]
    st.dataframe([{k:r[k] for k in disp_cols} for r in trace],
                 use_container_width=True, height=380)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
history = st.session_state.t5_eval_history
acc_str = (f"{sum(1 for h in history if h['correct'])}/{len(history)} exact correct"
           if history else "n/a")
st.markdown(
    f'<p style="color:#475569;font-family:\'Share Tech Mono\',monospace;font-size:0.78em">'
    f"Session: {st.session_state.t5_run_count} run(s)  ·  "
    f"Accuracy: {acc_str}  ·  "
    f"Multi-label GRU (sigmoid heads) · compound attack training data"
    f"</p>",
    unsafe_allow_html=True
)
