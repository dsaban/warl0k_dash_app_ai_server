# warlok/chain.py — ChainParamBundle, ChainMsg, WindowState, NanoBundle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from crypto import H, hkdf, mac, b64e, b64d, rand_bytes, bhex, RunningAccumulator

WINDOW_SIZE_DEFAULT = 48

# ══════════════════════════════════════════════════════════════════════════════
# ChainParamBundle  — issued by HUB, governs both peers
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChainParamBundle:
    # Identity
    session_epoch:       int   = 0
    peer_a_id:           str   = ""
    peer_b_id:           str   = ""
    hub_signature:       bytes = b""
    bundle_hash:         bytes = b""

    # Chain params
    window_size:         int   = WINDOW_SIZE_DEFAULT
    counter_init:        int   = 1
    counter_stride:      int   = 1
    dt_ms_min:           int   = 0
    dt_ms_max:           int   = 150
    dt_ms_slack:         int   = 10
    meas_slack:          float = 0.02
    op_allowlist:        List[str] = field(default_factory=lambda: ["READ","WRITE"])
    forensic_mode:       bool  = True

    # OS / hardware seed params
    os_seed:             int   = 42
    hw_pcr_hex:          str   = ""
    kernel_hash_hex:     str   = ""
    process_snap_hex:    str   = ""
    enclave_nonce_hex:   str   = ""

    # Merkle / accumulator params
    leaf_hash_algo:      str   = "sha256"
    tree_arity:          int   = 2
    acc_init_salt_hex:   str   = ""

    # Training params
    shared_train_seed:   int   = 0
    n_per_std:           int   = 30
    n_combo:             int   = 20
    rnn_hdim:            int   = 96
    rnn_epochs:          int   = 60
    rnn_lr:              float = 0.006
    rnn_batch:           int   = 32
    detection_threshold: float = 0.35
    feature_dim:         int   = 15

    # Anchor fingerprints (set after peer attestation)
    anchor_a_fp:         str   = ""
    anchor_b_fp:         str   = ""

    def compute_hash(self, hub_key: bytes) -> "ChainParamBundle":
        """Compute bundle_hash and hub_signature."""
        payload = (
            f"{self.session_epoch}|{self.peer_a_id}|{self.peer_b_id}|"
            f"{self.window_size}|{self.counter_init}|{self.dt_ms_min}|"
            f"{self.dt_ms_max}|{self.os_seed}|{self.shared_train_seed}|"
            f"{self.rnn_hdim}|{self.rnn_epochs}|{self.acc_init_salt_hex}"
        ).encode()
        self.bundle_hash  = H(payload)
        self.hub_signature = mac(hub_key, self.bundle_hash)
        return self

    def verify(self, hub_key: bytes) -> Tuple[bool, str]:
        expected_hash = H((
            f"{self.session_epoch}|{self.peer_a_id}|{self.peer_b_id}|"
            f"{self.window_size}|{self.counter_init}|{self.dt_ms_min}|"
            f"{self.dt_ms_max}|{self.os_seed}|{self.shared_train_seed}|"
            f"{self.rnn_hdim}|{self.rnn_epochs}|{self.acc_init_salt_hex}"
        ).encode())
        import hmac as _hmac
        if not _hmac.compare_digest(H(b""), H(b"")[:0] + expected_hash,
                                    H(b"") + expected_hash):
            pass
        if self.bundle_hash != expected_hash:
            return False, "bundle_hash mismatch"
        expected_sig = mac(hub_key, self.bundle_hash)
        if not _hmac.compare_digest(self.hub_signature, expected_sig):
            return False, "hub_signature invalid"
        return True, "ok"

    def acc_init_value(self, anchor_a: bytes, anchor_b: bytes) -> bytes:
        """Deterministic accumulator seed from both anchors."""
        salt = bytes.fromhex(self.acc_init_salt_hex) if self.acc_init_salt_hex else b"ACC_SALT"
        return H(anchor_a + anchor_b + salt + b"ACC_INIT")


# ══════════════════════════════════════════════════════════════════════════════
# Protocol messages
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class StartGrant:
    ok: bool; reason: str = ""
    session_id: str = ""; window_id_start: int = 0
    anchor_state_hash: bytes = b""; anchor_policy_hash: bytes = b""
    signature: bytes = b""

@dataclass
class WindowState:
    session_id: str; window_id: int
    expected_next_counter: int; expected_step_idx: int
    last_ts_ms: int; prev_mac_chain: bytes

@dataclass
class ChainMsg:
    session_id: str; window_id: int; step_idx: int; global_counter: int
    dt_ms: int; op_code: str; payload_hash: bytes; os_token: bytes
    os_meas: float; mac_chain: bytes
    # NEW: proof-in-motion fields
    acc_value:   bytes = b""   # running accumulator snapshot
    leaf_hash:   bytes = b""   # H(this message's fields, pre-mac)

    def canonical_bytes(self) -> bytes:
        """Deterministic bytes for hashing / Merkle leaf."""
        return b"|".join([
            self.session_id.encode(), str(self.window_id).encode(),
            str(self.step_idx).encode(), str(self.global_counter).encode(),
            str(self.dt_ms).encode(), self.op_code.encode(),
            b64e(self.payload_hash), b64e(self.os_token),
            f"{self.os_meas:.6f}".encode(),
        ])

    def to_bytes(self) -> bytes:
        return b"|".join([
            self.canonical_bytes(),
            b64e(self.mac_chain),
            b64e(self.acc_value) if self.acc_value else b"AA==",
            b64e(self.leaf_hash) if self.leaf_hash else b"AA==",
        ])

    @staticmethod
    def from_bytes(b: bytes) -> "ChainMsg":
        p = b.split(b"|")
        return ChainMsg(
            session_id=p[0].decode(), window_id=int(p[1]), step_idx=int(p[2]),
            global_counter=int(p[3]), dt_ms=int(p[4]), op_code=p[5].decode(),
            payload_hash=b64d(p[6]), os_token=b64d(p[7]),
            os_meas=float(p[8].decode()), mac_chain=b64d(p[9]),
            acc_value=b64d(p[10]) if len(p) > 10 else b"",
            leaf_hash=b64d(p[11]) if len(p) > 11 else b"",
        )


@dataclass
class NanoBundle:
    peer_id: str; anchor_state_hash: bytes; anchor_policy_hash: bytes
    dt_ms_range: Tuple[int,int]; meas_range: Tuple[float,float]; op_allowlist: set


@dataclass
class WindowCertificate:
    """Exchanged between peers at end of each Merkle window."""
    session_id:      str
    window_id:       int
    merkle_root:     bytes
    prev_root:       bytes
    acc_final:       bytes
    messages_seen:   int
    attacks_blocked: int
    peer_id:         str
    peer_sig:        bytes = b""
    timestamp:       int   = 0

    def sign(self, chain_key: bytes) -> "WindowCertificate":
        self.timestamp = int(time.time())
        payload = (
            self.session_id.encode() +
            self.window_id.to_bytes(4,"big") +
            self.merkle_root + self.prev_root + self.acc_final +
            self.messages_seen.to_bytes(4,"big") +
            self.peer_id.encode() +
            self.timestamp.to_bytes(8,"big")
        )
        self.peer_sig = mac(chain_key, payload)
        return self

    def verify_sig(self, chain_key: bytes) -> bool:
        payload = (
            self.session_id.encode() +
            self.window_id.to_bytes(4,"big") +
            self.merkle_root + self.prev_root + self.acc_final +
            self.messages_seen.to_bytes(4,"big") +
            self.peer_id.encode() +
            self.timestamp.to_bytes(8,"big")
        )
        import hmac as _hmac
        return _hmac.compare_digest(self.peer_sig, mac(chain_key, payload))


@dataclass
class IncidentCertificate:
    """Self-verifying attack record."""
    session_id:    str
    window_id:     int
    message_idx:   int
    true_leaf:     bytes
    received_leaf: bytes
    merkle_path:   list
    window_root:   bytes
    anchor_hash:   bytes
    attack_classes: List[str]
    gru_probs:     Dict[str, float]
    action_taken:  str   # WARN / QUARANTINE / BLOCK
    peer_id:       str
    peer_sig:      bytes = b""
    timestamp:     int   = 0

    def sign(self, chain_key: bytes) -> "IncidentCertificate":
        self.timestamp = int(time.time())
        payload = (
            self.session_id.encode() +
            self.window_id.to_bytes(4,"big") +
            self.message_idx.to_bytes(4,"big") +
            self.received_leaf + self.window_root + self.anchor_hash +
            (",".join(self.attack_classes)).encode() +
            self.action_taken.encode() +
            self.peer_id.encode() +
            self.timestamp.to_bytes(8,"big")
        )
        self.peer_sig = mac(chain_key, payload)
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Chain building helpers
# ══════════════════════════════════════════════════════════════════════════════

def os_fp(op: str, phash: bytes, ws: WindowState) -> Tuple[bytes, float]:
    tok  = H(b"OS|" + op.encode() + b"|" + phash + b"|" + ws.prev_mac_chain)[:16]
    meas = int.from_bytes(H(tok)[:4], "big") / (2**32)
    return tok, float(meas)

def chain_fields(msg: ChainMsg) -> bytes:
    return b"|".join([
        msg.session_id.encode(), str(msg.window_id).encode(),
        str(msg.step_idx).encode(), str(msg.global_counter).encode(),
        str(msg.dt_ms).encode(), msg.op_code.encode(),
        msg.payload_hash, msg.os_token,
        f"{msg.os_meas:.6f}".encode(),
    ])

def build_msg(chain_key: bytes, grant: StartGrant, ws: WindowState,
              op: str, payload: bytes, acc: RunningAccumulator,
              sleep_ms: int = 0) -> ChainMsg:
    phash = H(payload)
    tok, meas = os_fp(op, phash, ws)
    now = int(time.time()*1000); dt = max(0, now - ws.last_ts_ms)
    msg = ChainMsg(
        session_id=grant.session_id, window_id=ws.window_id,
        step_idx=ws.expected_step_idx, global_counter=ws.expected_next_counter,
        dt_ms=dt, op_code=op, payload_hash=phash, os_token=tok,
        os_meas=meas, mac_chain=b"",
        acc_value=b"", leaf_hash=b"",
    )
    msg.mac_chain = mac(chain_key, ws.prev_mac_chain + b"|" + chain_fields(msg))
    msg.leaf_hash = H(msg.canonical_bytes())
    msg.acc_value = acc.update(msg.leaf_hash)

    ws.prev_mac_chain = msg.mac_chain
    ws.last_ts_ms = now
    ws.expected_next_counter += 1
    ws.expected_step_idx += 1
    if ws.expected_step_idx == WINDOW_SIZE_DEFAULT:
        ws.window_id += 1; ws.expected_step_idx = 0
        ws.prev_mac_chain = H(b"WINDOW_PILOT|" + grant.session_id.encode() +
                               b"|" + str(ws.window_id).encode())
    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    return msg

def train_profile(peer_id, grant, window, slack_dt=10, slack_meas=0.02) -> NanoBundle:
    dt_vals  = [m.dt_ms   for m in window] or [0]
    ms_vals  = [m.os_meas for m in window] or [0.0]
    return NanoBundle(
        peer_id=peer_id,
        anchor_state_hash=grant.anchor_state_hash,
        anchor_policy_hash=grant.anchor_policy_hash,
        dt_ms_range=(max(0,int(min(dt_vals))), int(max(dt_vals)+slack_dt)),
        meas_range=(float(min(ms_vals))-slack_meas, float(max(ms_vals))+slack_meas),
        op_allowlist={m.op_code for m in window},
    )

def verify_msg(chain_key: bytes, bundle: NanoBundle, ws: WindowState,
               msg: ChainMsg, acc: RunningAccumulator) -> Tuple[bool, str, float]:
    """
    Returns (ok, reason, acc_divergence).
    """
    import hmac as _hmac
    if msg.session_id != ws.session_id:
        return False, "DROP: wrong session_id", 0.0
    if msg.window_id != ws.window_id:
        return False, "DROP: wrong window_id (drift/replay)", 0.0
    if msg.step_idx != ws.expected_step_idx:
        return False, "DROP: step mismatch (reorder/drop)", 0.0
    if msg.global_counter != ws.expected_next_counter:
        return False, "DROP: counter mismatch (replay/fork)", 0.0
    dt_min, dt_max = bundle.dt_ms_range
    if not (dt_min <= msg.dt_ms <= dt_max):
        return False, "DROP: dt_ms anomaly (time-warp/burst)", 0.0
    if msg.op_code not in bundle.op_allowlist:
        return False, "DROP: op_code not in allowlist", 0.0
    expect = mac(chain_key, ws.prev_mac_chain + b"|" + chain_fields(msg))
    if not _hmac.compare_digest(expect, msg.mac_chain):
        return False, "DROP: mac_chain mismatch (splice/tamper)", 0.0
    mn, mx = bundle.meas_range
    if not (mn <= msg.os_meas <= mx):
        return False, "DROP: os_meas outside learned range (mimic)", 0.0

    # Accumulator check
    expected_acc = acc.update(H(msg.canonical_bytes()))
    acc_div = acc.divergence(msg.acc_value) if msg.acc_value else 0.0

    ws.prev_mac_chain = msg.mac_chain
    ws.expected_next_counter += 1
    ws.expected_step_idx += 1
    ws.last_ts_ms = int(time.time()*1000)
    if ws.expected_step_idx == WINDOW_SIZE_DEFAULT:
        ws.window_id += 1; ws.expected_step_idx = 0
        ws.prev_mac_chain = H(b"WINDOW_PILOT|" + ws.session_id.encode() +
                               b"|" + str(ws.window_id).encode())
    return True, "ACCEPT", acc_div
