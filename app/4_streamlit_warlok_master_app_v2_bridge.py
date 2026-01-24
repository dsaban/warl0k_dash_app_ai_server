# warlok_master_streamlit_app_FIXED.py
# Run:
#   streamlit run warlok_master_streamlit_app_FIXED.py
#
# FIXES INCLUDED (from your errors):
# 1) Streamlit slider edge-case: min_value == max_value -> no slider shown; auto-select start=0.
# 2) Bridge short-stream issue: if Tab1 stops early (e.g., attack DROP at step 13), Bridge auto-pads
#    to SEQ_LEN so UI still runs, and it visualizes a pad mask (1=real, 0=padded).
# 3) Optional forensic mode in Tab1: keep collecting after first DROP (so you always get enough
#    samples for the Bridge), while still showing the first DROP reason.

import time, secrets, hashlib, hmac, queue, base64
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import streamlit as st

# ============================================================
# Shared helpers
# ============================================================

def H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    out, t, c = b"", b"", 1
    while len(out) < length:
        t = hmac.new(key, t + info + bytes([c]), hashlib.sha256).digest()
        out += t
        c += 1
    return out[:length]

def mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def b64e(b: bytes) -> bytes:
    return base64.urlsafe_b64encode(b)

def b64d(b: bytes) -> bytes:
    return base64.urlsafe_b64decode(b)

def bhex(b: bytes, n=16) -> str:
    return b.hex()[:n]

# ============================================================
# TAB 1 — P2P Protocol / Continuity Proof Simulator
# ============================================================

def xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out = bytearray()
    counter, i = 0, 0
    while i < len(data):
        block = hkdf(key, nonce + counter.to_bytes(4, "big"), 32)
        for bb in block:
            if i >= len(data):
                break
            out.append(data[i] ^ bb)
            i += 1
        counter += 1
    return bytes(out)

class DuplexWire:
    def __init__(self):
        self.a_to_b = queue.Queue()
        self.b_to_a = queue.Queue()
    def send_a(self, frame: bytes): self.a_to_b.put(frame)
    def send_b(self, frame: bytes): self.b_to_a.put(frame)
    def recv_a(self, timeout=0.5) -> Optional[bytes]:
        try: return self.b_to_a.get(timeout=timeout)
        except queue.Empty: return None
    def recv_b(self, timeout=0.5) -> Optional[bytes]:
        try: return self.a_to_b.get(timeout=timeout)
        except queue.Empty: return None

def iam_authenticate(peer_id: str, credentials: str, device_posture_hash: str) -> Dict[str, Any]:
    if peer_id not in credentials or len(device_posture_hash) < 8:
        return {"ok": False, "reason": "IAM deny"}
    token = secrets.token_hex(16)
    roles = ["operator"] if "op" in credentials else ["viewer"]
    return {"ok": True, "peer_id": peer_id, "roles": roles, "session_token": token}

def pam_authorize_start(claim: Dict[str, Any], target: str, actions: List[str]) -> Tuple[bool, str]:
    if not claim.get("ok"):
        return False, "PAM deny: invalid claim"
    roles = claim["roles"]
    if "operator" not in roles and any(a in ("WRITE", "DEPLOY", "CONTROL") for a in actions):
        return False, "PAM deny: insufficient role"
    return True, "ok"

WINDOW_SIZE = 48

@dataclass
class StartGrant:
    ok: bool
    reason: str = ""
    session_id: str = ""
    window_id_start: int = 0
    anchor_state_hash: bytes = b""
    anchor_policy_hash: bytes = b""
    signature: bytes = b""

@dataclass
class WindowState:
    session_id: str
    window_id: int
    expected_next_counter: int
    expected_step_idx: int
    last_ts_ms: int
    prev_mac_chain: bytes

@dataclass
class ChainMsg:
    session_id: str
    window_id: int
    step_idx: int
    global_counter: int
    dt_ms: int
    op_code: str
    payload_hash: bytes
    os_token: bytes
    os_meas: float
    mac_chain: bytes

    def to_bytes(self) -> bytes:
        parts = [
            self.session_id.encode(),
            str(self.window_id).encode(),
            str(self.step_idx).encode(),
            str(self.global_counter).encode(),
            str(self.dt_ms).encode(),
            self.op_code.encode(),
            b64e(self.payload_hash),
            b64e(self.os_token),
            f"{self.os_meas:.6f}".encode(),
            b64e(self.mac_chain),
        ]
        return b"|".join(parts)

    @staticmethod
    def from_bytes(b: bytes) -> "ChainMsg":
        p = b.split(b"|")
        return ChainMsg(
            session_id=p[0].decode(),
            window_id=int(p[1]),
            step_idx=int(p[2]),
            global_counter=int(p[3]),
            dt_ms=int(p[4]),
            op_code=p[5].decode(),
            payload_hash=b64d(p[6]),
            os_token=b64d(p[7]),
            os_meas=float(p[8].decode()),
            mac_chain=b64d(p[9]),
        )

@dataclass
class NanoProfileBundle:
    peer_id: str
    anchor_state_hash: bytes
    anchor_policy_hash: bytes
    dt_ms_range: Tuple[int, int]
    meas_range: Tuple[float, float]
    op_allowlist: set

@dataclass
class RecordKeys:
    enc_key: bytes
    mac_key: bytes

class P2PTLS:
    """
    TLS-like record channel for P2P simulation (not real TLS).
    Fixes your prior "FIN mismatch" and "record MAC failed" by:
    - symmetric nonce ordering using sorted peer IDs
    - distinct FIN expectations per side
    - record tag includes (typ|nonce|ct) consistently
    """
    def __init__(self, my_id: str, peer_id: str, psk: bytes, send_fn, recv_fn):
        self.my_id = my_id
        self.peer_id = peer_id
        self.psk = psk
        self.send_fn = send_fn
        self.recv_fn = recv_fn
        self.keys: Optional[RecordKeys] = None
        self.my_nonce = b""
        self.peer_nonce = b""

    def handshake_step1_send(self):
        self.my_nonce = secrets.token_bytes(16)
        self.send_fn(b"HS1|" + self.my_id.encode() + b"|" + b64e(self.my_nonce))

    def handshake_step2_recv_and_derive(self) -> bool:
        msg = self.recv_fn()
        if not msg:
            return False
        if not msg.startswith(b"HS1|"):
            raise ValueError("expected HS1")
        _, peer_id, peer_nonce_b64 = msg.split(b"|", 2)
        if peer_id.decode() != self.peer_id:
            raise ValueError("peer id mismatch")
        self.peer_nonce = b64d(peer_nonce_b64)

        id_low, id_high = sorted([self.my_id, self.peer_id])
        if self.my_id == id_low:
            nonce_low, nonce_high = self.my_nonce, self.peer_nonce
        else:
            nonce_low, nonce_high = self.peer_nonce, self.my_nonce

        prk = H(
            self.psk + b"|" +
            id_low.encode() + b"|" + id_high.encode() + b"|" +
            nonce_low + b"|" + nonce_high
        )
        self.keys = RecordKeys(
            enc_key=hkdf(prk, b"enc", 32),
            mac_key=hkdf(prk, b"mac", 32),
        )
        return True

    def handshake_step3_send_fin(self):
        assert self.keys
        fin = mac(self.keys.mac_key, b"FIN|" + self.my_nonce + self.peer_nonce)
        self.send_fn(b"HS2|" + b64e(fin))

    def handshake_step4_recv_verify(self) -> bool:
        assert self.keys
        msg = self.recv_fn()
        if not msg:
            return False
        if not msg.startswith(b"HS2|"):
            raise ValueError("expected HS2")
        fin2 = b64d(msg.split(b"|", 1)[1])
        expect = mac(self.keys.mac_key, b"FIN|" + self.peer_nonce + self.my_nonce)
        if not hmac.compare_digest(fin2, expect):
            raise ValueError("handshake FIN mismatch")
        return True

    def send_record(self, typ: str, payload: bytes):
        assert self.keys
        nonce = secrets.token_bytes(12)
        ct = xor_stream(payload, self.keys.enc_key, nonce)
        tag = mac(self.keys.mac_key, typ.encode() + b"|" + nonce + b"|" + ct)
        frame = b"REC|" + typ.encode() + b"|" + b64e(nonce) + b"|" + b64e(ct) + b"|" + b64e(tag)
        self.send_fn(frame)

    def recv_record(self) -> Tuple[str, bytes]:
        assert self.keys
        frame = self.recv_fn()
        if not frame:
            raise TimeoutError("no frame")
        if not frame.startswith(b"REC|"):
            raise ValueError("expected REC")
        parts = frame.split(b"|", 4)
        if len(parts) != 5:
            raise ValueError("bad REC frame")
        _, typ_b, nonce_b64, ct_b64, tag_b64 = parts
        nonce = b64d(nonce_b64)
        ct = b64d(ct_b64)
        tag = b64d(tag_b64)

        expect = mac(self.keys.mac_key, typ_b + b"|" + nonce + b"|" + ct)
        if not hmac.compare_digest(tag, expect):
            raise ValueError("TLS record MAC failed")
        pt = xor_stream(ct, self.keys.enc_key, nonce)
        return typ_b.decode(), pt

def warlok_start_hook(local_peer_id: str, claim: Dict[str, Any], target: str, actions: List[str], posture_hash: str) -> StartGrant:
    ok, reason = pam_authorize_start(claim, target, actions)
    if not ok:
        return StartGrant(ok=False, reason=reason)

    session_id = secrets.token_hex(8)
    anchor_material = H(
        b"WARLOK_ANCHOR|" +
        local_peer_id.encode() + b"|" +
        claim["peer_id"].encode() + b"|" +
        posture_hash.encode() + b"|" +
        target.encode() + b"|" +
        b",".join(a.encode() for a in actions) + b"|" +
        str(int(time.time())).encode() + b"|" +
        session_id.encode()
    )
    policy_hash = H(b"POLICY|" + target.encode() + b"|" + b",".join(a.encode() for a in actions))
    signature = H(b"SIGN|" + anchor_material + policy_hash)

    return StartGrant(
        ok=True,
        session_id=session_id,
        window_id_start=0,
        anchor_state_hash=anchor_material,
        anchor_policy_hash=policy_hash,
        signature=signature
    )

def os_fingerprint_sample(op_code: str, payload_hash: bytes, ws: WindowState) -> Tuple[bytes, float]:
    tok = H(b"OS|" + op_code.encode() + b"|" + payload_hash + b"|" + ws.prev_mac_chain)[:16]
    meas = int.from_bytes(H(tok)[:4], "big") / (2**32)
    return tok, float(meas)

def chain_fields_for_mac(msg: ChainMsg) -> bytes:
    return b"|".join([
        msg.session_id.encode(),
        str(msg.window_id).encode(),
        str(msg.step_idx).encode(),
        str(msg.global_counter).encode(),
        str(msg.dt_ms).encode(),
        msg.op_code.encode(),
        msg.payload_hash,
        msg.os_token,
        f"{msg.os_meas:.6f}".encode(),
    ])

def build_chain_msg(chain_key: bytes, grant: StartGrant, ws: WindowState, op_code: str, payload: bytes, sleep_ms: int = 0) -> ChainMsg:
    payload_hash = H(payload)
    os_token, os_meas = os_fingerprint_sample(op_code, payload_hash, ws)
    now_ms = int(time.time() * 1000)
    dt_ms = max(0, now_ms - ws.last_ts_ms)

    msg = ChainMsg(
        session_id=grant.session_id,
        window_id=ws.window_id,
        step_idx=ws.expected_step_idx,
        global_counter=ws.expected_next_counter,
        dt_ms=dt_ms,
        op_code=op_code,
        payload_hash=payload_hash,
        os_token=os_token,
        os_meas=os_meas,
        mac_chain=b""
    )
    msg.mac_chain = mac(chain_key, ws.prev_mac_chain + b"|" + chain_fields_for_mac(msg))

    ws.prev_mac_chain = msg.mac_chain
    ws.last_ts_ms = now_ms
    ws.expected_next_counter += 1
    ws.expected_step_idx += 1

    if ws.expected_step_idx == WINDOW_SIZE:
        ws.window_id += 1
        ws.expected_step_idx = 0
        ws.prev_mac_chain = H(b"WINDOW_PILOT|" + grant.session_id.encode() + b"|" + str(ws.window_id).encode())

    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)

    return msg

def train_profile(peer_id: str, grant: StartGrant, window: List[ChainMsg], slack_dt=10, slack_meas=0.02) -> NanoProfileBundle:
    dt_vals = [m.dt_ms for m in window] or [0]
    meas_vals = [m.os_meas for m in window] or [0.0]
    ops = {m.op_code for m in window}
    dt_min = max(0, int(min(dt_vals)))
    dt_max = int(max(dt_vals) + slack_dt)
    mn, mx = float(min(meas_vals)), float(max(meas_vals))
    return NanoProfileBundle(
        peer_id=peer_id,
        anchor_state_hash=grant.anchor_state_hash,
        anchor_policy_hash=grant.anchor_policy_hash,
        dt_ms_range=(dt_min, dt_max),
        meas_range=(mn - slack_meas, mx + slack_meas),
        op_allowlist=ops
    )

DETERMINISTIC_GATES = {
    "DROP: wrong session_id": "deterministic",
    "DROP: wrong window_id (drift/replay)": "deterministic",
    "DROP: step mismatch (reorder/drop)": "deterministic",
    "DROP: counter mismatch (replay/fork)": "deterministic",
    "DROP: mac_chain mismatch (splice/tamper)": "deterministic",
}
NANO_GATES = {
    "DROP: dt_ms anomaly (time-warp/burst)": "nano",
    "DROP: op_code not in allowlist": "nano",
    "DROP: os_meas outside learned range (mimic/impersonation)": "nano",
}

def verify_msg(chain_key: bytes, bundle: NanoProfileBundle, ws: WindowState, msg: ChainMsg) -> Tuple[bool, str]:
    if msg.session_id != ws.session_id:
        return False, "DROP: wrong session_id"
    if msg.window_id != ws.window_id:
        return False, "DROP: wrong window_id (drift/replay)"
    if msg.step_idx != ws.expected_step_idx:
        return False, "DROP: step mismatch (reorder/drop)"
    if msg.global_counter != ws.expected_next_counter:
        return False, "DROP: counter mismatch (replay/fork)"

    dt_min, dt_max = bundle.dt_ms_range
    if not (dt_min <= msg.dt_ms <= dt_max):
        return False, "DROP: dt_ms anomaly (time-warp/burst)"

    if msg.op_code not in bundle.op_allowlist:
        return False, "DROP: op_code not in allowlist"

    expect = mac(chain_key, ws.prev_mac_chain + b"|" + chain_fields_for_mac(msg))
    if not hmac.compare_digest(expect, msg.mac_chain):
        return False, "DROP: mac_chain mismatch (splice/tamper)"

    mn, mx = bundle.meas_range
    if not (mn <= msg.os_meas <= mx):
        return False, "DROP: os_meas outside learned range (mimic/impersonation)"

    ws.prev_mac_chain = msg.mac_chain
    ws.expected_next_counter += 1
    ws.expected_step_idx += 1
    ws.last_ts_ms = int(time.time() * 1000)

    if ws.expected_step_idx == WINDOW_SIZE:
        ws.window_id += 1
        ws.expected_step_idx = 0
        ws.prev_mac_chain = H(b"WINDOW_PILOT|" + ws.session_id.encode() + b"|" + str(ws.window_id).encode())

    return True, "ACCEPT"

def attack_reorder(window: List[ChainMsg]) -> List[ChainMsg]:
    w = window[:]
    if len(w) >= 12:
        w[10], w[11] = w[11], w[10]
    return w

def attack_drop(window: List[ChainMsg]) -> List[ChainMsg]:
    return [m for i, m in enumerate(window) if i != 20]

def attack_replay(window: List[ChainMsg]) -> List[ChainMsg]:
    w = []
    for i, m in enumerate(window):
        w.append(m)
        if i == 5:
            w.append(m)
    return w

def attack_timewarp(window: List[ChainMsg], new_dt: int = 999999) -> List[ChainMsg]:
    w = []
    for i, m in enumerate(window):
        if i == 7:
            mm = ChainMsg(**m.__dict__)
            mm.dt_ms = new_dt
            w.append(mm)
        else:
            w.append(m)
    return w

def attack_splice(window: List[ChainMsg]) -> List[ChainMsg]:
    w = []
    for i, m in enumerate(window):
        if i == 12:
            mm = ChainMsg(**m.__dict__)
            mm.op_code = "CONTROL"
            w.append(mm)
        else:
            w.append(m)
    return w

ATTACK_DESC = {
    "none": ("No mutation", "No gate should fire."),
    "reorder": ("Swap step 10 and 11", "Expect step mismatch (and/or MAC mismatch)."),
    "drop": ("Remove step 20", "Expect step mismatch at the next received step."),
    "replay": ("Replay step 5 once", "Expect counter mismatch / step mismatch."),
    "timewarp": ("Set dt_ms to absurd value at step 7", "Expect dt anomaly (or MAC mismatch)."),
    "splice": ("Change op_code to CONTROL at step 12", "Expect op allowlist fail (or MAC mismatch)."),
}

def compute_scoreboard(trace: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = {}
    for r in trace:
        if r["decision"] == "DROP":
            counts[r["reason"]] = counts.get(r["reason"], 0) + 1
    cat = {"deterministic": 0, "nano": 0, "other": 0}
    for reason, n in counts.items():
        if reason in DETERMINISTIC_GATES: cat["deterministic"] += n
        elif reason in NANO_GATES: cat["nano"] += n
        else: cat["other"] += n
    return {"reasons": counts, "categories": cat}

def simulate_protocol_run(
    steps: int,
    attack_mode: str,
    sleep_ms: int,
    slack_dt: int,
    slack_meas: float,
    op_pattern: str,
    forensic_continue: bool,
    target: str = "pump-controller",
    actions: List[str] = None,
) -> Dict[str, Any]:
    if actions is None:
        actions = ["READ", "WRITE"]

    wire = DuplexWire()
    peerA_id, peerB_id = "peerA", "peerB"
    shared_psk = H(b"mutual-trust-root|" + peerA_id.encode() + b"|" + peerB_id.encode())
    tlsA = P2PTLS(peerA_id, peerB_id, shared_psk, wire.send_a, lambda: wire.recv_a(timeout=0.5))
    tlsB = P2PTLS(peerB_id, peerA_id, shared_psk, wire.send_b, lambda: wire.recv_b(timeout=0.5))

    # Symmetric handshake
    tlsA.handshake_step1_send()
    tlsB.handshake_step1_send()
    for _ in range(10):
        if not tlsA.keys: tlsA.handshake_step2_recv_and_derive()
        if not tlsB.keys: tlsB.handshake_step2_recv_and_derive()
        if tlsA.keys and tlsB.keys: break
    if not (tlsA.keys and tlsB.keys):
        return {"ok": False, "reason": "handshake timeout (HS1)"}

    tlsA.handshake_step3_send_fin()
    tlsB.handshake_step3_send_fin()
    okA = okB = False
    for _ in range(10):
        okA = okA or tlsA.handshake_step4_recv_verify()
        okB = okB or tlsB.handshake_step4_recv_verify()
        if okA and okB: break
    if not (okA and okB):
        return {"ok": False, "reason": "handshake timeout (HS2 FIN)"}

    # IAM + START
    postureA = "posturehashA_12345678"
    claimA = iam_authenticate(peerA_id, "peerA_op_creds", postureA)

    grantA = warlok_start_hook(peerA_id, claimA, target, actions, postureA)
    if not grantA.ok:
        return {"ok": False, "reason": "START denied: " + grantA.reason}

    grant = grantA
    chain_key = hkdf(H(b"chain|" + grant.anchor_state_hash + grant.anchor_policy_hash), b"chain-key", 32)

    def init_ws() -> WindowState:
        return WindowState(
            session_id=grant.session_id,
            window_id=0,
            expected_next_counter=1,
            expected_step_idx=0,
            last_ts_ms=int(time.time() * 1000),
            prev_mac_chain=H(b"WINDOW_PILOT|" + grant.session_id.encode() + b"|0")
        )

    # Train profile on first <=48 steps (clean generation)
    wsA_train = init_ws()
    train_steps = min(WINDOW_SIZE, steps)
    train_window: List[ChainMsg] = []
    for i in range(train_steps):
        if op_pattern == "rw":
            op = "READ" if i % 3 else "WRITE"
        elif op_pattern == "read":
            op = "READ"
        else:
            op = "WRITE"
        payload = f"op{i}".encode()
        train_window.append(build_chain_msg(chain_key, grant, wsA_train, op, payload, sleep_ms=sleep_ms))

    bundle = train_profile("peerA", grant, train_window, slack_dt=slack_dt, slack_meas=slack_meas)

    # Build sending window (clean) then mutate for attacks
    wsA_send = init_ws()
    send_window: List[ChainMsg] = []
    for i in range(steps):
        if op_pattern == "rw":
            op = "READ" if i % 3 else "WRITE"
        elif op_pattern == "read":
            op = "READ"
        else:
            op = "WRITE"
        payload = f"op{i}".encode()
        send_window.append(build_chain_msg(chain_key, grant, wsA_send, op, payload, sleep_ms=sleep_ms))

    if attack_mode == "none":
        attacked = send_window
    elif attack_mode == "reorder":
        attacked = attack_reorder(send_window)
    elif attack_mode == "drop":
        attacked = attack_drop(send_window)
    elif attack_mode == "replay":
        attacked = attack_replay(send_window)
    elif attack_mode == "timewarp":
        attacked = attack_timewarp(send_window)
    elif attack_mode == "splice":
        attacked = attack_splice(send_window)
    else:
        attacked = send_window

    wsB = init_ws()
    trace = []
    accepted = 0
    first_drop_reason = None

    rx_os_tokens: List[bytes] = []
    rx_meas: List[float] = []
    rx_ops: List[str] = []
    rx_dt: List[int] = []
    rx_steps: List[int] = []
    rx_ctrs: List[int] = []
    rx_wins: List[int] = []

    for idx, m in enumerate(attacked):
        tlsA.send_record("CHAIN", m.to_bytes())
        _, pt = tlsB.recv_record()
        rx = ChainMsg.from_bytes(pt)

        ok, reason = verify_msg(chain_key, bundle, wsB, rx)

        decision = "ACCEPT" if ok else "DROP"
        if decision == "DROP" and first_drop_reason is None:
            first_drop_reason = reason

        trace.append({
            "i": idx,
            "win": rx.window_id,
            "step": rx.step_idx,
            "ctr": rx.global_counter,
            "dt_ms": rx.dt_ms,
            "op": rx.op_code,
            "meas": round(rx.os_meas, 6),
            "decision": decision,
            "reason": reason,
            "B_win_now": wsB.window_id,
            "B_step_expected": wsB.expected_step_idx,
            "B_ctr_expected": wsB.expected_next_counter,
        })

        rx_os_tokens.append(rx.os_token)
        rx_meas.append(float(rx.os_meas))
        rx_ops.append(rx.op_code)
        rx_dt.append(int(rx.dt_ms))
        rx_steps.append(int(rx.step_idx))
        rx_ctrs.append(int(rx.global_counter))
        rx_wins.append(int(rx.window_id))

        if ok:
            accepted += 1
        else:
            if not forensic_continue:
                break
            # forensic_continue: keep collecting frames even after first DROP
            # (verification state wsB is already "stuck" by definition; that's the point)

    return {
        "ok": True,
        "attack_mode": attack_mode,
        "attack_desc": ATTACK_DESC.get(attack_mode, ("", "")),
        "session_id": grant.session_id,
        "anchor": bhex(grant.anchor_state_hash, 16),
        "policy": bhex(grant.anchor_policy_hash, 16),
        "chain_key": bhex(chain_key, 16),
        "dt_range": bundle.dt_ms_range,
        "meas_range": (round(bundle.meas_range[0], 6), round(bundle.meas_range[1], 6)),
        "op_allowlist": sorted(list(bundle.op_allowlist)),
        "accepted": accepted,
        "sent": len(rx_meas),
        "dropped_reason": first_drop_reason,
        "trace": trace,
        "stream": {
            "os_token": rx_os_tokens,
            "meas": rx_meas,
            "op": rx_ops,
            "dt_ms": rx_dt,
            "step": rx_steps,
            "ctr": rx_ctrs,
            "win": rx_wins,
        }
    }

# ============================================================
# TAB 2 — Nano-AI MLEI (NumPy-only GRU+Attention)
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

LR_PHASE1      = 0.006
LR_PHASE2_BASE = 0.03

CLIP_NORM     = 5.0
WEIGHT_DECAY  = 1e-4

LAMBDA_MS         = 1.0
LAMBDA_TOK        = 0.10
TOK_STOP_EPS      = 0.25
TOK_WARMUP_EPOCHS = 60

LAMBDA_ID   = 1.0
LAMBDA_W    = 1.0
LAMBDA_BCE  = 1.0
POS_WEIGHT  = 10.0

THRESH_P_VALID = 0.80
PID_MIN        = 0.70
PW_MIN         = 0.70

PILOT_AMP_DEFAULT = 0.55

def sigmoid_np(x): return 1.0 / (1.0 + np.exp(-x))

def softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

def softmax1d_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def softmax_masked_np(scores, mask):
    huge_neg = -1e9
    s = np.where(mask > 0, scores, huge_neg)
    s = s - np.max(s, axis=1, keepdims=True)
    e = np.exp(s) * mask
    return e / (np.sum(e, axis=1, keepdims=True) + 1e-12)

def clip_grads_np(grads, max_norm=CLIP_NORM):
    norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-12)
        for k in grads:
            grads[k] *= scale
    return grads

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

def window_pilot(window_global_id, seq_len=SEQ_LEN, pilot_amp=0.55):
    rng = np.random.RandomState((window_global_id * 9176 + 11) & 0xFFFFFFFF)
    chips = rng.randint(0, 2, size=seq_len).astype(np.float32)
    chips = 2.0 * chips - 1.0
    pilot = pilot_amp * chips
    pilot = pilot - pilot.mean()
    return pilot

def generate_os_chain(ms_vec, window_global_id, A_base, seq_len=SEQ_LEN, pilot_amp=0.55):
    zs = np.zeros((seq_len,), dtype=np.float32)
    for t in range(seq_len):
        a_t = A_base[t] + window_delta(window_global_id, t)
        zs[t] = float(a_t @ ms_vec)
    zs = zs + window_pilot(window_global_id, seq_len, pilot_amp=pilot_amp)

    noise_seed = (window_global_id * 1337 + int((ms_vec * 1000).sum())) & 0xFFFFFFFF
    rng = np.random.RandomState(noise_seed)
    zs = zs + rng.normal(scale=0.02, size=seq_len).astype(np.float32)

    m = (zs - zs.mean()) / (zs.std() + 1e-6)
    scaled = np.clip((m + 3.0) / 6.0, 0.0, 0.999999)
    tokens = (scaled * VOCAB_SIZE).astype(np.int32)
    return tokens, m

def build_X_backbone(tokens, m):
    T = len(tokens)
    D = VOCAB_SIZE + 2
    X = np.zeros((T, D), dtype=np.float32)
    for t in range(T):
        X[t, int(tokens[t])] = 1.0
        X[t, VOCAB_SIZE] = float(m[t])
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

@st.cache_data(show_spinner=False)
def build_dataset(seed: int, pilot_amp: float):
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

            # NEG wrong-window
            wrong_w = (w_true + 7) % N_WINDOWS_PER_ID
            g_wrong = id_true * N_WINDOWS_PER_ID + wrong_w
            toks_w, meas_w = generate_os_chain(ms_true, g_wrong, A_base, pilot_amp=pilot_amp)
            X = build_X_backbone(toks_w, meas_w)
            Xp, Mp, Tp = pad_to_T(X, toks_w)
            Xs.append(Xp); Ms.append(Mp); Tok.append(Tp)
            y_ms.append(ms_true); y_cls.append(0)
            true_id.append(id_true); true_w.append(wrong_w)
            claim_id.append(id_true); expect_w.append(w_true)

            # NEG wrong identity
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
        z = sigmoid_np(a_z); r = sigmoid_np(a_r)

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
    alphas = softmax_masked_np(scores, Mb)
    ctx = np.sum(hs * alphas[:, :, None], axis=1)
    cache = {"hs": hs, "Mb": Mb, "u": u, "alphas": alphas}
    return ctx, alphas, cache

def ms_head(ctx, p):
    h = np.tanh(ctx @ p["W_ms1"].T + p["b_ms1"])
    ms_hat = h @ p["W_ms2"].T + p["b_ms2"]
    return ms_hat, h

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

def token_ce_one(logits, target):
    p_ = softmax1d_np(logits)
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

            grads = clip_grads_np(grads)
            opt.step(p, grads, weight_decay=WEIGHT_DECAY)

        avg = total / N
        losses.append(avg)
        if log_cb and (ep == 1 or ep == 2 or ep % max(1, epochs//10) == 0):
            log_cb(f"[Phase1] Epoch {ep}/{epochs} avg_loss={avg:.6f} tok_enabled={tok_enabled}")

    return p, losses

def compute_embeddings(p, X, M):
    hs, _ = gru_forward_batch(X, M, p)
    ctx, alphas, _ = attention_forward_batch(hs, M, p)
    B, T, Hh = hs.shape
    denom = np.sum(M, axis=1, keepdims=True) + 1e-6
    h_mean = np.sum(hs * M[:,:,None], axis=1) / denom
    last_idx = np.maximum(0, np.sum(M, axis=1).astype(np.int32) - 1)
    h_last = np.zeros((B,Hh), dtype=np.float32)
    for i in range(B):
        h_last[i] = hs[i, last_idx[i], :]
    return ctx, h_last, h_mean, alphas

def ce_loss_batch_masked(logits, targets, mask_01):
    idxs = np.where(mask_01 > 0.5)[0]
    if len(idxs) == 0:
        return 0.0, np.zeros_like(logits)
    L = 0.0
    dlog = np.zeros_like(logits)
    for i in idxs:
        p_ = softmax1d_np(logits[i])
        L += -np.log(p_[targets[i]] + 1e-12)
        d = p_; d[targets[i]] -= 1.0
        dlog[i] = d
    L /= len(idxs)
    dlog /= len(idxs)
    return L, dlog

def bce_loss_batch(logits, y):
    p_ = sigmoid_np(logits)
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

            prob_id = softmax_np(logits_id, axis=1)
            prob_w  = softmax_np(logits_w, axis=1)

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

            grads = clip_grads_np(grads)
            opt.step(p, grads, weight_decay=WEIGHT_DECAY, freeze_keys=freeze)

        avg = total / N
        losses.append(avg)
        if log_cb and (ep == 1 or ep == 2 or ep % max(1, epochs//10) == 0):
            log_cb(f"[Phase2] Epoch {ep}/{epochs} avg_loss={avg:.6f} lr={lr:.5f}")

    return p, losses

def verify_chain_ai(p, tokens, meas, claimed_id, expected_w):
    X = build_X_backbone(tokens, meas)
    Xp, Mp, _ = pad_to_T(X, np.array(tokens, dtype=np.int32))

    ctx, h_last, h_mean, alphas = compute_embeddings(p, Xp[None,:,:], Mp[None,:])
    ctx1 = ctx[0]

    logits_id = (ctx @ p["W_id"].T + p["b_id"])[0]
    id_pred = int(np.argmax(logits_id))
    pid = float(softmax1d_np(logits_id)[claimed_id])

    feat_w = np.concatenate([ctx1, h_last[0], h_mean[0]], axis=0)[None,:]
    logits_w = (feat_w @ p["W_w"].T + p["b_w"])[0]
    w_pred = int(np.argmax(logits_w))
    pw = float(softmax1d_np(logits_w)[expected_w])

    cid = claimed_id / max(1, N_IDENTITIES-1)
    ew  = expected_w / max(1, N_WINDOWS_PER_ID-1)
    vb_in = np.concatenate([ctx1, np.array([cid, ew, pid, pw], dtype=np.float32)], axis=0)[None,:]
    logit_v = float(np.squeeze(vb_in @ p["W_beh"].T + p["b_beh"]))
    p_valid = float(sigmoid_np(logit_v))

    checks = {
        f"p_valid >= {THRESH_P_VALID}": (p_valid >= THRESH_P_VALID),
        "id_pred == claimed_id": (id_pred == claimed_id),
        "w_pred == expected_w": (w_pred == expected_w),
        f"pid >= {PID_MIN}": (pid >= PID_MIN),
        f"pw >= {PW_MIN}": (pw >= PW_MIN),
    }
    ok = all(checks.values())

    return {
        "ok": ok,
        "p_valid": p_valid,
        "id_pred": id_pred,
        "w_pred": w_pred,
        "pid": pid,
        "pw": pw,
        "checks": checks,
        "alphas": alphas[0],
    }

# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="WARL0K Master (FIXED)", layout="wide")
st.title("WARL0K Master Demo — P2P Continuity + Nano-AI MLEI + Bridge (FIXED)")
st.caption("Includes fixes for Bridge slider edge-case + short stream padding + optional forensic continuation.")

tab1, tab2, tab3 = st.tabs([
    "1) P2P Protocol + Continuity Proof",
    "2) Nano-AI MLEI (NumPy GRU+Attention)",
    "3) Bridge: Tab1 Trace → AI Verification"
])

# ----------------------------
# Tab 1
# ----------------------------
with tab1:
    with st.sidebar:
        st.subheader("Tab 1 — Protocol")
        steps = st.slider("Steps to send", min_value=8, max_value=144, value=48, step=1, key="p_steps")
        attack_mode = st.selectbox("Attack scenario", ["none", "reorder", "drop", "replay", "timewarp", "splice"], index=0, key="p_attack")
        sleep_ms = st.slider("Inter-step delay (ms)", 0, 10, 1, 1, key="p_sleep")

        st.divider()
        st.subheader("Tab 1 profile sensitivity")
        slack_dt = st.slider("dt slack (ms)", 0, 80, 10, 1, key="p_slack_dt")
        slack_meas = st.slider("meas slack", 0.0, 0.10, 0.02, 0.01, key="p_slack_meas")
        op_pattern = st.selectbox("Operation pattern", ["rw", "read", "write"], index=0, key="p_op")

        st.divider()
        st.subheader("Run mode")
        forensic_continue = st.checkbox("Forensic mode: keep collecting after first DROP", value=True, key="p_forensic")

        st.divider()
        st.subheader("Display")
        compact = st.checkbox("Compact trace", value=True, key="p_compact")
        show_charts = st.checkbox("Show timelines", value=True, key="p_charts")
        show_state_machine = st.checkbox("Show state machine", value=True, key="p_sm")
        show_scoreboard = st.checkbox("Show gate scoreboard", value=True, key="p_score")

        run = st.button("Run Tab 1 simulation", type="primary", key="p_run")

    if not run:
        st.info("Use the sidebar controls and click **Run Tab 1 simulation**.")
    else:
        meta = simulate_protocol_run(steps, attack_mode, sleep_ms, slack_dt, slack_meas, op_pattern, forensic_continue)
        if not meta.get("ok"):
            st.error(meta.get("reason", "Simulation failed"))
        else:
            st.session_state["tab1_last"] = meta  # for Bridge tab

            trace = meta["trace"]
            win_now = trace[-1]["B_win_now"] if trace else 0
            step_expected = trace[-1]["B_step_expected"] if trace else 0
            ctr_expected = trace[-1]["B_ctr_expected"] if trace else 1
            progress_in_window = min(WINDOW_SIZE, max(0, int(step_expected)))

            c1, c2, c3, c4, c5 = st.columns([1.4, 1.1, 1.1, 1.1, 2.3])
            with c1:
                st.markdown("### Session")
                st.write("session_id:", meta["session_id"])
                st.write("attack:", meta["attack_mode"])
                st.write("forensic:", forensic_continue)
            with c2:
                st.markdown("### Anchor")
                st.code(meta["anchor"])
            with c3:
                st.markdown("### Policy")
                st.code(meta["policy"])
            with c4:
                st.markdown("### Chain key")
                st.code(meta["chain_key"])
            with c5:
                st.markdown("### Result")
                st.metric("Accepted", meta["accepted"])
                st.metric("Collected", meta["sent"])
                if meta["dropped_reason"]:
                    st.error("First DROP: " + meta["dropped_reason"])
                else:
                    st.success("PASS: no drops")

            a_title, a_expect = meta["attack_desc"]
            st.markdown("### Attack mutation")
            st.write(f"- **What changed:** {a_title}")
            st.write(f"- **Expected gate to fire:** {a_expect}")

            if show_state_machine:
                st.divider()
                s1, s2, s3 = st.columns([1.3, 1.7, 2.0])
                with s1:
                    st.markdown("### Window progress")
                    st.progress(progress_in_window / WINDOW_SIZE)
                    st.caption(f"B expects step **{step_expected}** in window **{win_now}** (0..{WINDOW_SIZE-1}).")
                with s2:
                    st.markdown("### State machine")
                    st.markdown(
                        f"""
**START (IAM/PAM OK)**
→ **Execution Anchor** (anchor={meta['anchor']})
→ **Window {win_now}**
→ **Next expected:** step={step_expected}, counter={ctr_expected}
→ **Fast-path gates:** session/window/step/counter/MAC + dt/meas/op profile
"""
                    )
                with s3:
                    st.markdown("### Learned profile gates (Tab1)")
                    g1, g2 = st.columns(2)
                    with g1:
                        st.write("dt gate:", meta["dt_range"])
                        st.write("op allowlist:", ", ".join(meta["op_allowlist"]) if meta["op_allowlist"] else "(empty)")
                    with g2:
                        st.write("meas gate:", meta["meas_range"])

            if show_charts and trace:
                st.divider()
                st.markdown("## Timelines")
                dt_series = [r["dt_ms"] for r in trace]
                meas_series = [r["meas"] for r in trace]
                ctr_series = [r["ctr"] for r in trace]
                step_series = [r["step"] for r in trace]
                decision_series = [1 if r["decision"] == "ACCEPT" else 0 for r in trace]

                ch1, ch2, ch3 = st.columns([1.3, 1.3, 1.0])
                with ch1:
                    st.markdown("### dt_ms")
                    st.line_chart(dt_series)
                with ch2:
                    st.markdown("### os_meas")
                    st.line_chart(meas_series)
                with ch3:
                    st.markdown("### accept=1 / drop=0")
                    st.line_chart(decision_series)

                ch4, ch5 = st.columns(2)
                with ch4:
                    st.markdown("### global_counter")
                    st.line_chart(ctr_series)
                with ch5:
                    st.markdown("### step_idx")
                    st.line_chart(step_series)

            if show_scoreboard:
                st.divider()
                st.markdown("## Gate scoreboard")
                sb = compute_scoreboard(trace)
                left, right = st.columns([1.1, 1.9])
                with left:
                    st.markdown("### Categories")
                    st.write(sb["categories"])
                with right:
                    st.markdown("### DROP reasons")
                    if not sb["reasons"]:
                        st.info("No DROP reasons (PASS run).")
                    else:
                        rows = [{"reason": k, "count": v, "category": (DETERMINISTIC_GATES.get(k) or NANO_GATES.get(k) or "other")} for k, v in sb["reasons"].items()]
                        st.dataframe(rows, use_container_width=True, height=220)

            st.divider()
            st.markdown("## Step-by-step trace (B verification)")
            if compact:
                cols = ["i", "win", "step", "ctr", "dt_ms", "op", "meas", "decision", "reason"]
                st.dataframe([{k: r[k] for k in cols} for r in trace], use_container_width=True, height=460)
            else:
                st.dataframe(trace, use_container_width=True, height=460)

# ----------------------------
# Tab 2
# ----------------------------
with tab2:
    with st.sidebar:
        st.subheader("Tab 2 — Nano-AI")
        ai_seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=0, step=1, key="ai_seed")
        pilot_amp = st.slider("PN Pilot amplitude", 0.10, 0.90, float(PILOT_AMP_DEFAULT), 0.05, key="ai_pilot")

        st.divider()
        st.subheader("Training")
        epochs1 = st.slider("Phase1 epochs", 1, 200, 60, 1, key="ai_ep1")
        epochs2 = st.slider("Phase2 epochs", 1, 500, 120, 1, key="ai_ep2")
        show_attention = st.checkbox("Show attention weights", value=True, key="ai_att")
        run_train = st.button("Train Tab 2 model", type="primary", key="ai_train")

    X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W = build_dataset(ai_seed, pilot_amp)
    N = X_ALL.shape[0]

    top = st.columns([1.2, 1.2, 1.6, 1.0])
    with top[0]:
        st.metric("Samples", N)
    with top[1]:
        st.metric("Positives", int(np.sum(Y_CLS_ALL)))
    with top[2]:
        st.write("PN pilot:", f"amp={pilot_amp:.2f}, seq_len={SEQ_LEN}")
    with top[3]:
        st.write("Shapes:", f"X={X_ALL.shape}, M={M_ALL.shape}")

    if "ai_trained" not in st.session_state:
        st.session_state.ai_trained = False
        st.session_state.ai_p = None
        st.session_state.ai_loss1 = []
        st.session_state.ai_loss2 = []
        st.session_state.ai_logs = []

    def ai_log(msg: str):
        st.session_state.ai_logs.append(msg)

    if run_train:
        st.session_state.ai_trained = False
        st.session_state.ai_logs = []
        st.session_state.ai_loss1 = []
        st.session_state.ai_loss2 = []

        log_box = st.empty()
        prog = st.progress(0.0)

        p = init_model(X_ALL.shape[2], seed=ai_seed)

        t0 = time.time()
        def cb1(m):
            ai_log(m)
            log_box.code("\n".join(st.session_state.ai_logs[-12:]))
        p, loss1 = train_phase1(p, X_ALL, M_ALL, TOK_ALL, Y_MS_ALL, Y_CLS_ALL, epochs=epochs1, log_cb=cb1)
        st.session_state.ai_loss1 = loss1
        prog.progress(0.45)

        def cb2(m):
            ai_log(m)
            log_box.code("\n".join(st.session_state.ai_logs[-12:]))
        p, loss2 = train_phase2(p, X_ALL, M_ALL, Y_CLS_ALL, TRUE_ID, TRUE_W, CLAIM_ID, EXPECT_W, epochs=epochs2, log_cb=cb2)
        st.session_state.ai_loss2 = loss2
        prog.progress(1.0)

        st.session_state.ai_p = p
        st.session_state.ai_trained = True
        st.success(f"Training complete in {time.time() - t0:.2f}s")

    left, right = st.columns([1.25, 1.75])
    with left:
        st.markdown("## Process (AI)")
        st.markdown(
            """
**Phase1:** GRU+Attention → MS reconstruction (positives) + optional next-token scaffold early.
**Phase2:** identity head + window head + validity head (uses claim + pid + pw).
**Decision:** OK only if all gates pass: p_valid, id match, window match, pid/pw confidence.
"""
        )
    with right:
        st.markdown("## Training dashboard")
        if st.session_state.ai_loss1:
            st.write("Phase1 loss")
            st.line_chart(st.session_state.ai_loss1)
        if st.session_state.ai_loss2:
            st.write("Phase2 loss")
            st.line_chart(st.session_state.ai_loss2)
        if st.session_state.ai_logs:
            st.write("Logs (latest)")
            st.code("\n".join(st.session_state.ai_logs[-16:]))

    st.divider()
    st.markdown("## Quick AI check (synthetic chain)")
    if not st.session_state.ai_trained:
        st.info("Train Tab 2 model to run AI verification.")
    else:
        MS_all, A_base = init_world(ai_seed)

        v1, v2, v3, v4 = st.columns([1.0, 1.0, 1.6, 1.2])
        with v1:
            claimed_id = st.selectbox("Claimed identity", list(range(N_IDENTITIES)), index=0, key="ai_claim_id")
        with v2:
            expected_w = st.selectbox("Expected window", list(range(N_WINDOWS_PER_ID)), index=5, key="ai_expect_w")
        with v3:
            scenario = st.selectbox(
                "Scenario",
                ["LEGIT", "ATTACK: SHUFFLED", "ATTACK: TRUNCATED", "ATTACK: WRONG WINDOW", "ATTACK: WRONG IDENTITY"],
                index=0,
                key="ai_scn"
            )
        with v4:
            attack_seed = st.number_input("Attack seed (shuffle)", min_value=0, max_value=10_000, value=0, step=1, key="ai_shuf_seed")

        ms_true = MS_all[claimed_id]
        g_true = claimed_id * N_WINDOWS_PER_ID + expected_w
        toks, meas = generate_os_chain(ms_true, g_true, A_base, pilot_amp=pilot_amp)

        if scenario == "ATTACK: SHUFFLED":
            rng = np.random.RandomState(int(attack_seed))
            idxs = np.arange(len(toks)); rng.shuffle(idxs)
            toks2, meas2 = toks[idxs], meas[idxs]
        elif scenario == "ATTACK: TRUNCATED":
            Ltr = len(toks)//2
            toks2, meas2 = toks[:Ltr], meas[:Ltr]
        elif scenario == "ATTACK: WRONG WINDOW":
            wrong_w = (expected_w + 7) % N_WINDOWS_PER_ID
            g_wrong = claimed_id * N_WINDOWS_PER_ID + wrong_w
            toks2, meas2 = generate_os_chain(ms_true, g_wrong, A_base, pilot_amp=pilot_amp)
        elif scenario == "ATTACK: WRONG IDENTITY":
            other_id = (claimed_id + 1) % N_IDENTITIES
            other_w  = int((expected_w + 13) % N_WINDOWS_PER_ID)
            ms_other = MS_all[other_id]
            g_other  = other_id * N_WINDOWS_PER_ID + other_w
            toks2, meas2 = generate_os_chain(ms_other, g_other, A_base, pilot_amp=pilot_amp)
        else:
            toks2, meas2 = toks, meas

        r = verify_chain_ai(st.session_state.ai_p, toks2, meas2, claimed_id=claimed_id, expected_w=expected_w)

        cA, cB, cC, cD = st.columns([1.1, 1.1, 1.1, 1.7])
        with cA: st.metric("p_valid", f"{r['p_valid']:.4f}")
        with cB: st.metric("pid", f"{r['pid']:.4f}")
        with cC: st.metric("pw", f"{r['pw']:.4f}")
        with cD: st.metric("Decision", "OK ✅" if r["ok"] else "REJECT ❌")

        st.markdown("### Gate-by-gate validation")
        st.dataframe([{"gate": k, "pass": bool(v)} for k, v in r["checks"].items()], use_container_width=True, height=220)

        st.divider()
        st.markdown("### Tokens / meas / attention")
        a1, a2 = st.columns(2)
        with a1:
            st.line_chart([int(x) for x in toks2])
        with a2:
            st.line_chart([float(x) for x in meas2])
        if show_attention:
            st.line_chart([float(x) for x in r["alphas"]])

# ----------------------------
# Tab 3 (BRIDGE) — FIXED
# ----------------------------
with tab3:
    st.markdown("## Bridge: feed Tab 1 continuity stream into the AI verifier")
    st.caption("Fixed: handles short streams (pads) + handles slider min==max (auto start=0).")

    has_tab1 = "tab1_last" in st.session_state
    has_ai = bool(st.session_state.get("ai_trained", False)) and (st.session_state.get("ai_p") is not None)

    c0, c1 = st.columns(2)
    with c0:
        st.write("Tab 1 last run:", "✅ available" if has_tab1 else "❌ run Tab 1")
    with c1:
        st.write("AI model:", "✅ trained" if has_ai else "❌ train Tab 2")

    if not has_tab1:
        st.info("Go to Tab 1, run a simulation, then come back here.")
        st.stop()
    if not has_ai:
        st.info("Go to Tab 2, train the AI model, then come back here.")
        st.stop()

    meta = st.session_state["tab1_last"]
    stream = meta["stream"]

    max_len = len(stream["meas"])
    st.write(f"Tab 1 stream length: **{max_len}** (AI slice length SEQ_LEN={SEQ_LEN})")

    b1, b2, b3 = st.columns([1.0, 1.0, 2.0])
    with b1:
        claimed_id = st.selectbox("Claimed identity (AI)", list(range(N_IDENTITIES)), index=0, key="bridge_claim_id")
    with b2:
        expected_w = st.selectbox("Expected window (AI)", list(range(N_WINDOWS_PER_ID)), index=0, key="bridge_expected_w")
    with b3:
        st.write("Tab 1 context:", f"attack={meta['attack_mode']}, first_drop={meta['dropped_reason'] or 'None'}")

    # ---------
    # FIX #1 + FIX #2:
    # - If max_len < SEQ_LEN: pad to SEQ_LEN and show pad_mask
    # - Else: safe slider; if max_start==0 -> auto start=0 (no slider)
    # ---------
    if max_len == 0:
        st.error("Tab 1 stream is empty. Re-run Tab 1.")
        st.stop()

    if max_len < SEQ_LEN:
        st.warning(
            f"Stream shorter than SEQ_LEN ({SEQ_LEN}). Padding from {max_len} → {SEQ_LEN}. "
            "AI is expected to reject padded tails."
        )

        os_raw = stream["os_token"]
        meas_raw = np.clip(np.array(stream["meas"], dtype=np.float32), 0.0, 1.0)
        tokens_raw = np.array([(int.from_bytes(t[:2], "big") % VOCAB_SIZE) for t in os_raw], dtype=np.int32)

        pad_n = SEQ_LEN - max_len
        tokens_slice = np.pad(tokens_raw, (0, pad_n), mode="edge")
        meas_slice = np.pad(meas_raw, (0, pad_n), mode="edge")

        pad_mask = np.concatenate([
            np.ones((max_len,), dtype=np.float32),
            np.zeros((pad_n,), dtype=np.float32)
        ])
        start = 0

    else:
        max_start = max_len - SEQ_LEN
        if max_start == 0:
            st.info("Only one valid slice available (start = 0).")
            start = 0
        else:
            start = st.slider("Slice start index", min_value=0, max_value=max_start, value=0, step=1)

        os_slice = stream["os_token"][start:start+SEQ_LEN]
        meas_slice = np.clip(np.array(stream["meas"][start:start+SEQ_LEN], dtype=np.float32), 0.0, 1.0)
        tokens_slice = np.array([(int.from_bytes(t[:2], "big") % VOCAB_SIZE) for t in os_slice], dtype=np.int32)
        pad_mask = np.ones((SEQ_LEN,), dtype=np.float32)

    r = verify_chain_ai(st.session_state.ai_p, tokens_slice, meas_slice, claimed_id=claimed_id, expected_w=expected_w)

    st.divider()
    st.markdown("### AI decision on bridged slice")
    cA, cB, cC, cD = st.columns([1.1, 1.1, 1.1, 1.7])
    with cA: st.metric("p_valid", f"{r['p_valid']:.4f}")
    with cB: st.metric("pid", f"{r['pid']:.4f}")
    with cC: st.metric("pw", f"{r['pw']:.4f}")
    with cD: st.metric("Decision", "OK ✅" if r["ok"] else "REJECT ❌")

    st.markdown("#### Gate-by-gate validation")
    st.dataframe([{"gate": k, "pass": bool(v)} for k, v in r["checks"].items()], use_container_width=True, height=220)

    st.divider()
    st.markdown("### Bridged signals")
    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown("**Tokens (from Tab1 os_token)**")
        st.line_chart([int(x) for x in tokens_slice])
    with p2:
        st.markdown("**os_meas**")
        st.line_chart([float(x) for x in meas_slice])
    with p3:
        st.markdown("**Pad mask (1=real, 0=padded)**")
        st.line_chart([float(x) for x in pad_mask])

    st.divider()
    st.markdown("### Slice inspection (Tab 1 trace rows)")
    rows = []
    trace = meta["trace"]
    for i in range(start, min(start + SEQ_LEN, len(trace))):
        tr = trace[i]
        rows.append({
            "i": tr["i"],
            "win": tr["win"],
            "step": tr["step"],
            "ctr": tr["ctr"],
            "dt_ms": tr["dt_ms"],
            "op": tr["op"],
            "meas": tr["meas"],
            "decision": tr["decision"],
            "reason": tr["reason"],
        })
    st.dataframe(rows, use_container_width=True, height=380)
