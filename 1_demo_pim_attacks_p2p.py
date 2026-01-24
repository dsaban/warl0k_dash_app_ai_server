# streamlit_app_warlok_p2p_fixed2.py
# Run:
#   streamlit run streamlit_app_warlok_p2p_fixed2.py
#
# Fixes included:
# 1) Handshake FIN mismatch fixed via canonical nonce ordering (by peer IDs).
# 2) TLS record MAC failed fixed via Base64 encoding of binary fields (nonce/ct/tag),
#    so frame parsing is delimiter-safe.

import time, secrets, hashlib, hmac, queue, base64
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import streamlit as st


# =========================
# Helpers (toy crypto)
# =========================

def H(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def hkdf(key: bytes, info: bytes, length: int = 32) -> bytes:
    out, t, c = b"", b"", 1
    while len(out) < length:
        t = hmac.new(key, t + info + bytes([c]), hashlib.sha256).digest()
        out += t
        c += 1
    return out[:length]

def xor_stream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out = bytearray()
    counter, i = 0, 0
    while i < len(data):
        block = hkdf(key, nonce + counter.to_bytes(4, "big"), 32)
        for b in block:
            if i >= len(data):
                break
            out.append(data[i] ^ b)
            i += 1
        counter += 1
    return bytes(out)

def mac(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()

def bhex(b: bytes, n=8) -> str:
    return b.hex()[:n]

def b64e(b: bytes) -> bytes:
    return base64.urlsafe_b64encode(b)

def b64d(b: bytes) -> bytes:
    return base64.urlsafe_b64decode(b)


# =========================
# In-memory duplex wire
# =========================

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


# =========================
# IAM / PAM (simulation)
# =========================

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


# =========================
# WARL0K MLEI structs
# =========================

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
        # NOTE: these fields are safe because we control encoding
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
class NanoModelBundle:
    peer_id: str
    anchor_state_hash: bytes
    anchor_policy_hash: bytes
    dt_ms_range: Tuple[int, int]
    meas_range: Tuple[float, float]
    op_allowlist: set


# =========================
# TLS-like P2P record layer (toy) — FIXED (B64 framing)
# =========================

@dataclass
class RecordKeys:
    enc_key: bytes
    mac_key: bytes

class P2PTLS:
    """
    P2P "TLS-ish" (toy):
    - HS1 exchange nonces
    - derive PRK with canonical nonce ordering (by peer IDs) => stable keys
    - HS2 FIN exchange
    - Record layer: b64 encodes nonce/ct/tag to avoid delimiter parsing bugs
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

        # canonical ordering
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

        # delimiter-safe frame
        frame = b"REC|" + typ.encode() + b"|" + b64e(nonce) + b"|" + b64e(ct) + b"|" + b64e(tag)
        self.send_fn(frame)

    def recv_record(self) -> Tuple[str, bytes]:
        assert self.keys
        frame = self.recv_fn()
        if not frame:
            raise TimeoutError("no frame")
        if not frame.startswith(b"REC|"):
            raise ValueError("expected REC")

        # All binary pieces were base64, so split is safe now.
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


# =========================
# WARLOK START hook
# =========================

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
        ok=True, session_id=session_id, window_id_start=0,
        anchor_state_hash=anchor_material,
        anchor_policy_hash=policy_hash,
        signature=signature
    )


# =========================
# Chain + gates
# =========================

def os_fingerprint_sample(op_code: str, payload_hash: bytes, window_state: WindowState) -> Tuple[bytes, float]:
    tok = H(b"OS|" + op_code.encode() + b"|" + payload_hash + b"|" + window_state.prev_mac_chain)[:16]
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

def build_chain_msg(chain_key: bytes, grant: StartGrant, ws: WindowState, op_code: str, payload: bytes) -> ChainMsg:
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

    return msg

def train_nano_bundle(peer_id: str, grant: StartGrant, window: List[ChainMsg], slack_dt=10, slack_meas=0.02) -> NanoModelBundle:
    dt_vals = [m.dt_ms for m in window] or [0]
    meas_vals = [m.os_meas for m in window] or [0.0]
    ops = {m.op_code for m in window}

    dt_min = max(0, int(min(dt_vals)))
    dt_max = int(max(dt_vals) + slack_dt)
    mn = float(min(meas_vals))
    mx = float(max(meas_vals))

    return NanoModelBundle(
        peer_id=peer_id,
        anchor_state_hash=grant.anchor_state_hash,
        anchor_policy_hash=grant.anchor_policy_hash,
        dt_ms_range=(dt_min, dt_max),
        meas_range=(mn - slack_meas, mx + slack_meas),
        op_allowlist=ops
    )

def verify_msg(chain_key: bytes, bundle: NanoModelBundle, ws: WindowState, msg: ChainMsg) -> Tuple[bool, str]:
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


# =========================
# Attacks
# =========================

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
            w.append(mm)  # MAC won't match
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


# =========================
# Simulation
# =========================

def simulate_run(steps: int, attack_mode: str, sleep_ms: int, slack_dt: int, slack_meas: float, op_pattern: str) -> Dict[str, Any]:
    wire = DuplexWire()
    peerA_id, peerB_id = "peerA", "peerB"
    shared_psk = H(b"mutual-trust-root|" + peerA_id.encode() + b"|" + peerB_id.encode())

    tlsA = P2PTLS(peerA_id, peerB_id, shared_psk, wire.send_a, lambda: wire.recv_a(timeout=0.5))
    tlsB = P2PTLS(peerB_id, peerA_id, shared_psk, wire.send_b, lambda: wire.recv_b(timeout=0.5))

    # P2P handshake (no client/server)
    tlsA.handshake_step1_send()
    tlsB.handshake_step1_send()

    for _ in range(10):
        if not tlsA.keys:
            tlsA.handshake_step2_recv_and_derive()
        if not tlsB.keys:
            tlsB.handshake_step2_recv_and_derive()
        if tlsA.keys and tlsB.keys:
            break
    if not (tlsA.keys and tlsB.keys):
        return {"ok": False, "reason": "handshake timeout (HS1)"}

    tlsA.handshake_step3_send_fin()
    tlsB.handshake_step3_send_fin()

    okA = okB = False
    for _ in range(10):
        okA = okA or tlsA.handshake_step4_recv_verify()
        okB = okB or tlsB.handshake_step4_recv_verify()
        if okA and okB:
            break
    if not (okA and okB):
        return {"ok": False, "reason": "handshake timeout (HS2 FIN)"}

    # IAM + START
    postureA = "posturehashA_12345678"
    postureB = "posturehashB_12345678"
    claimA = iam_authenticate(peerA_id, "peerA_op_creds", postureA)
    claimB = iam_authenticate(peerB_id, "peerB_op_creds", postureB)

    target = "pump-controller"
    actions = ["READ", "WRITE"]
    grantA = warlok_start_hook(peerA_id, claimA, target, actions, postureA)
    grantB = warlok_start_hook(peerB_id, claimB, target, actions, postureB)
    if not (grantA.ok and grantB.ok):
        return {"ok": False, "reason": "START denied"}

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

    wsA = init_ws()
    wsB = init_ws()

    # Training window (<=48)
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
        time.sleep(max(0, sleep_ms) / 1000.0)
        train_window.append(build_chain_msg(chain_key, grant, wsA, op, payload))

    bundle = train_nano_bundle(peerA_id, grant, train_window, slack_dt=slack_dt, slack_meas=slack_meas)

    # Reset and build sending window
    wsA = init_ws()
    send_window: List[ChainMsg] = []
    for i in range(steps):
        if op_pattern == "rw":
            op = "READ" if i % 3 else "WRITE"
        elif op_pattern == "read":
            op = "READ"
        else:
            op = "WRITE"
        payload = f"op{i}".encode()
        time.sleep(max(0, sleep_ms) / 1000.0)
        send_window.append(build_chain_msg(chain_key, grant, wsA, op, payload))

    # Attack mutate
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

    trace = []
    accepted = 0
    dropped_reason = None

    for idx, m in enumerate(attacked):
        tlsA.send_record("CHAIN", m.to_bytes())
        typ, pt = tlsB.recv_record()  # now safe parse (b64)
        rx = ChainMsg.from_bytes(pt)

        ok, reason = verify_msg(chain_key, bundle, wsB, rx)
        trace.append({
            "i": idx,
            "win": rx.window_id,
            "step": rx.step_idx,
            "ctr": rx.global_counter,
            "dt_ms": rx.dt_ms,
            "op": rx.op_code,
            "meas": round(rx.os_meas, 6),
            "decision": "ACCEPT" if ok else "DROP",
            "reason": reason
        })

        if ok:
            accepted += 1
        else:
            dropped_reason = reason
            break

    return {
        "ok": True,
        "attack_mode": attack_mode,
        "session_id": grant.session_id,
        "anchor": bhex(grant.anchor_state_hash, 16),
        "policy": bhex(grant.anchor_policy_hash, 16),
        "chain_key": bhex(chain_key, 16),
        "dt_range": bundle.dt_ms_range,
        "meas_range": (round(bundle.meas_range[0], 6), round(bundle.meas_range[1], 6)),
        "op_allowlist": sorted(list(bundle.op_allowlist)),
        "accepted": accepted,
        "sent": len(attacked),
        "dropped_reason": dropped_reason,
        "trace": trace
    }


# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="WARL0K MLEI P2P Visual Simulator (Fixed2)", layout="wide")

st.title("WARL0K MLEI — P2P Visual Simulator (Handshake + Record Framing Fixed)")
st.caption("In-memory duplex wire + symmetric handshake + Base64 record framing + START hook + 48-step windows + gates.")

with st.sidebar:
    st.subheader("Run controls")
    steps = st.slider("Steps to send", min_value=8, max_value=96, value=48, step=1)
    attack_mode = st.selectbox("Attack scenario", ["none", "reorder", "drop", "replay", "timewarp", "splice"], index=0)
    sleep_ms = st.slider("Inter-step delay (ms)", 0, 10, 1, 1)

    st.divider()
    st.subheader("Gate sensitivity")
    slack_dt = st.slider("dt slack (ms)", 0, 50, 10, 1)
    slack_meas = st.slider("meas slack", 0.0, 0.10, 0.02, 0.01)
    op_pattern = st.selectbox("Operation pattern", ["rw", "read", "write"], index=0)

    run = st.button("Run simulation", type="primary")

if run:
    try:
        meta = simulate_run(steps, attack_mode, sleep_ms, slack_dt, slack_meas, op_pattern)
    except Exception as e:
        st.exception(e)
        st.stop()

    if not meta.get("ok"):
        st.error(meta.get("reason", "Simulation failed"))
        st.stop()

    c1, c2, c3, c4 = st.columns([1.6, 1.2, 1.2, 2.0])
    with c1:
        st.markdown("### Session")
        st.write("session_id:", meta["session_id"])
        st.write("attack:", meta["attack_mode"])
    with c2:
        st.markdown("### Anchor/Policy")
        st.code(f"A:{meta['anchor']}\nP:{meta['policy']}")
    with c3:
        st.markdown("### Chain key")
        st.code(meta["chain_key"])
    with c4:
        st.markdown("### Result")
        st.metric("Accepted", meta["accepted"])
        st.metric("Sent", meta["sent"])
        if meta["dropped_reason"]:
            st.error(meta["dropped_reason"])
        else:
            st.success("PASS: no drops")

    st.divider()

    g1, g2, g3 = st.columns([1.2, 1.2, 2.0])
    with g1:
        st.markdown("### dt gate")
        st.write(meta["dt_range"])
    with g2:
        st.markdown("### meas gate")
        st.write(meta["meas_range"])
    with g3:
        st.markdown("### op allowlist")
        st.write(", ".join(meta["op_allowlist"]) if meta["op_allowlist"] else "(empty)")

    st.divider()
    st.markdown("## Step trace (B verification)")
    st.dataframe(meta["trace"], use_container_width=True, height=440)

else:
    st.info("Choose an attack and click **Run simulation**.")
