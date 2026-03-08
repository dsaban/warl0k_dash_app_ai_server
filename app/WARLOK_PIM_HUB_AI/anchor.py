# warlok/anchor.py — Solid-State Anchor, IAM, PAM, P2P TLS
import time, queue
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from crypto import H, hkdf, mac, b64e, b64d, rand_bytes, rand_hex, xor_stream

# ══════════════════════════════════════════════════════════════════════════════
# Solid-State Anchor
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HardwareAttestation:
    """Simulated TPM / secure enclave attestation."""
    peer_id:         str
    cpu_serial:      str   = ""
    tpm_pcr_values:  bytes = b""
    enclave_nonce:   bytes = b""
    hw_hash:         bytes = b""

    def compute(self) -> "HardwareAttestation":
        self.hw_hash = H(
            self.peer_id.encode() +
            self.cpu_serial.encode() +
            self.tpm_pcr_values +
            self.enclave_nonce
        )
        return self

@dataclass
class OSPosture:
    """OS and kernel state snapshot."""
    peer_id:              str
    kernel_version:       str   = ""
    patch_level:          str   = ""
    process_snapshot:     str   = ""
    os_hash:              bytes = b""

    def compute(self) -> "OSPosture":
        self.os_hash = H(
            self.peer_id.encode() +
            self.kernel_version.encode() +
            self.patch_level.encode() +
            self.process_snapshot.encode()
        )
        return self

@dataclass
class IAMToken:
    """Identity & Access Management token (post-login)."""
    peer_id:       str
    username:      str
    roles:         List[str]
    issued_at:     int   = 0
    expires_at:    int   = 0
    token_sig:     bytes = b""

    def sign(self, hub_key: bytes) -> "IAMToken":
        self.issued_at  = int(time.time())
        self.expires_at = self.issued_at + 3600
        payload = (self.peer_id + self.username +
                   ",".join(self.roles) +
                   str(self.issued_at)).encode()
        self.token_sig = mac(hub_key, payload)
        return self

    def verify(self, hub_key: bytes) -> bool:
        payload = (self.peer_id + self.username +
                   ",".join(self.roles) +
                   str(self.issued_at)).encode()
        expected = mac(hub_key, payload)
        import hmac as _hmac
        return (_hmac.compare_digest(self.token_sig, expected) and
                int(time.time()) < self.expires_at)

@dataclass
class PAMGrant:
    """Privilege Access Management grant."""
    peer_id:         str
    target_resource: str
    allowed_actions: List[str]
    grant_hash:      bytes = b""

    def compute(self) -> "PAMGrant":
        self.grant_hash = H(
            self.peer_id.encode() +
            self.target_resource.encode() +
            b",".join(a.encode() for a in sorted(self.allowed_actions))
        )
        return self

@dataclass
class SolidStateAnchor:
    """
    Layered attestation hash:
      anchor = H(hw_hash ∥ os_hash ∥ iam_sig ∥ pam_hash ∥ epoch ∥ peer_id)
    """
    peer_id:       str
    hw:            HardwareAttestation
    os:            OSPosture
    iam:           IAMToken
    pam:           PAMGrant
    epoch:         int   = 0
    anchor_hash:   bytes = b""
    public_fp:     str   = ""   # first 16 hex chars — safe to share

    def compute(self) -> "SolidStateAnchor":
        self.epoch = int(time.time()) // 3600   # 1-hour epoch
        self.anchor_hash = H(
            self.hw.hw_hash +
            self.os.os_hash +
            self.iam.token_sig +
            self.pam.grant_hash +
            self.epoch.to_bytes(8, "big") +
            self.peer_id.encode()
        )
        self.public_fp = self.anchor_hash.hex()[:16]
        return self

    def is_valid(self, hub_key: bytes) -> Tuple[bool, str]:
        if not self.iam.verify(hub_key):
            return False, "IAM token invalid or expired"
        if self.hw.hw_hash == b"":
            return False, "Hardware attestation missing"
        if self.os.os_hash == b"":
            return False, "OS posture missing"
        if self.pam.grant_hash == b"":
            return False, "PAM grant missing"
        return True, "ok"


def make_anchor(peer_id: str, hub_key: bytes,
                username: str = "operator",
                roles: List[str] = None,
                target: str = "pump-controller",
                actions: List[str] = None,
                seed: int = 0) -> SolidStateAnchor:
    """
    Convenience factory — creates a fully-populated anchor for a peer.
    In production each component comes from real hw/os/iam/pam systems.
    Here we derive them deterministically from seed for reproducibility.
    """
    import hashlib
    roles   = roles   or ["operator"]
    actions = actions or ["READ", "WRITE"]

    rng_bytes = lambda n, s: hashlib.sha256(
        seed.to_bytes(4,"big") + peer_id.encode() + s.encode()
    ).digest()[:n]

    hw = HardwareAttestation(
        peer_id       = peer_id,
        cpu_serial    = rng_bytes(8, "cpu").hex(),
        tpm_pcr_values= rng_bytes(32, "pcr"),
        enclave_nonce = rng_bytes(16, "enc"),
    ).compute()

    os_ = OSPosture(
        peer_id           = peer_id,
        kernel_version    = f"5.{seed%10}.{seed%7}-warlok",
        patch_level       = f"2025.{seed%12+1:02d}",
        process_snapshot  = rng_bytes(16, "proc").hex(),
    ).compute()

    iam = IAMToken(
        peer_id  = peer_id,
        username = username,
        roles    = roles,
    ).sign(hub_key)

    pam = PAMGrant(
        peer_id         = peer_id,
        target_resource = target,
        allowed_actions = actions,
    ).compute()

    return SolidStateAnchor(
        peer_id = peer_id, hw=hw, os=os_, iam=iam, pam=pam
    ).compute()


# ══════════════════════════════════════════════════════════════════════════════
# P2P TLS  (unchanged from original, now in own module)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _RecordKeys:
    enc_key: bytes
    mac_key: bytes

class DuplexWire:
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

class P2PTLS:
    def __init__(self, my_id, peer_id, psk, send_fn, recv_fn):
        self.my_id=my_id; self.peer_id=peer_id; self.psk=psk
        self.send_fn=send_fn; self.recv_fn=recv_fn
        self.keys: Optional[_RecordKeys]=None
        self.my_nonce=b""; self.peer_nonce=b""

    def hs1_send(self):
        self.my_nonce=rand_bytes(16)
        self.send_fn(b"HS1|"+self.my_id.encode()+b"|"+b64e(self.my_nonce))

    def hs2_recv_derive(self):
        msg=self.recv_fn()
        if not msg: return False
        _,pid,pn=msg.split(b"|",2)
        if pid.decode()!=self.peer_id: raise ValueError("peer id mismatch")
        self.peer_nonce=b64d(pn)
        lo,hi=sorted([self.my_id,self.peer_id])
        nl,nh=(self.my_nonce,self.peer_nonce) if self.my_id==lo else (self.peer_nonce,self.my_nonce)
        prk=H(self.psk+b"|"+lo.encode()+b"|"+hi.encode()+b"|"+nl+b"|"+nh)
        self.keys=_RecordKeys(hkdf(prk,b"enc",32),hkdf(prk,b"mac",32))
        return True

    def hs3_send_fin(self):
        assert self.keys
        fin=mac(self.keys.mac_key,b"FIN|"+self.my_nonce+self.peer_nonce)
        self.send_fn(b"HS2|"+b64e(fin))

    def hs4_recv_verify(self):
        assert self.keys
        msg=self.recv_fn()
        if not msg: return False
        fin2=b64d(msg.split(b"|",1)[1])
        expect=mac(self.keys.mac_key,b"FIN|"+self.peer_nonce+self.my_nonce)
        import hmac as _hmac
        if not _hmac.compare_digest(fin2,expect): raise ValueError("FIN mismatch")
        return True

    def send_rec(self, typ, payload):
        assert self.keys
        nonce=rand_bytes(12)
        ct=xor_stream(payload,self.keys.enc_key,nonce)
        tag=mac(self.keys.mac_key,typ.encode()+b"|"+nonce+b"|"+ct)
        self.send_fn(b"REC|"+typ.encode()+b"|"+b64e(nonce)+b"|"+b64e(ct)+b"|"+b64e(tag))

    def recv_rec(self):
        assert self.keys
        frame=self.recv_fn()
        if not frame: raise TimeoutError("no frame")
        _,typ_b,nb,ctb,tagb=frame.split(b"|",4)
        nonce=b64d(nb); ct=b64d(ctb); tag=b64d(tagb)
        expect=mac(self.keys.mac_key,typ_b+b"|"+nonce+b"|"+ct)
        import hmac as _hmac
        if not _hmac.compare_digest(tag,expect): raise ValueError("MAC failed")
        return typ_b.decode(),xor_stream(ct,self.keys.enc_key,nonce)
