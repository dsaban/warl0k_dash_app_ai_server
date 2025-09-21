"""
Peer class for WARL0K demo.
Handles enrollment, master_seed derivation, ephemeral session key derivation,
envelope creation and verification.
"""
from __future__ import annotations

from .hub import *
from dataclasses import dataclass, field
from typing import Dict
from .crypto_utils import HMAC, H, hkdf, dh_generate_keypair, dh_shared_secret, random_bytes, bhex
import binascii, hmac

@dataclass
class Peer:
    device_id: str
    hub: object  # Hub reference
    device_nonce: bytes = field(default_factory=lambda: random_bytes(32))
    seed0: bytes = b""
    seed_sig: bytes = b""
    master_seed: bytes = b""
    last_counter_from_peer: Dict[str, int] = field(default_factory=dict)

    # Enrollment with hub
    def enroll_with_hub(self):
        self.seed0, self.seed_sig = self.hub.enroll(self.device_id)
        if not self.hub.verify_seed(self.device_id, self.seed0, self.seed_sig):
            raise RuntimeError("Invalid seed signature from hub")
        # joint-seed derivation (device contributes entropy)
        self.master_seed = HMAC(self.seed0, self.device_nonce)
        # best-effort zeroization: drop seed0 reference
        self.seed0 = b""

    # Non-revealing contribution derived from master_seed
    def contrib(self) -> bytes:
        return HMAC(self.master_seed, b"contrib:", self.device_id.encode())

    # Derive K_session (both peers must compute identical value)
    def derive_session_key(self, peer_device_id: str, my_priv: int, peer_pub: int,
                           policy_id: str, counter: int, challenge: bytes,
                           contrib_self: bytes, contrib_peer: bytes) -> bytes:
        # 1) salt from ephemeral DH
        salt = dh_shared_secret(my_priv, peer_pub)
        # 2) deterministic combined IKM - canonical ordering by device id
        left_id, right_id = sorted([self.device_id, peer_device_id])
        if self.device_id == left_id:
            combined_IKM = H(contrib_self, contrib_peer)
        else:
            combined_IKM = H(contrib_peer, contrib_self)
        # 3) info binds identities, policy, counter, challenge
        info = b"|".join([b"WARL0Kv1",
                          left_id.encode(), right_id.encode(),
                          policy_id.encode(),
                          counter.to_bytes(8, "big"),
                          challenge])
        # 4) HKDF -> session key
        return hkdf(salt=salt, ikm=combined_IKM, info=info, length=32)

    # Envelope build (demo uses HMAC as tag; production should use AEAD)
    def build_envelope(self, peer_pub: int, my_priv: int, peer_device_id: str,
                       policy_id: str, counter: int, challenge: bytes, plaintext: bytes,
                       contrib_self: bytes, contrib_peer: bytes) -> dict:
        k_sess = self.derive_session_key(peer_device_id, my_priv, peer_pub,
                                         policy_id, counter, challenge,
                                         contrib_self, contrib_peer)
        header = {
            "from": self.device_id,
            "to": peer_device_id,
            "policy_id": policy_id,
            "ctr": counter,
            "challenge": bhex(challenge),
        }
        aad = f"{header['from']}|{header['to']}|{header['policy_id']}|{header['ctr']}|{header['challenge']}".encode()
        tag = HMAC(k_sess, aad, plaintext)
        return {"header": header, "peer_pub": peer_pub, "payload": binascii.hexlify(plaintext).decode(), "tag": bhex(tag)}

    def verify_envelope(self, env: dict, my_priv: int, peer_pub: int, contrib_self: bytes, contrib_peer: bytes) -> bool:
        hdr = env["header"]
        counter = int(hdr["ctr"])
        last = self.last_counter_from_peer.get(hdr["from"], -1)
        if counter <= last:
            print(f"[{self.device_id}] REPLAY/STALE detected: counter {counter} <= last {last}")
            return False
        challenge = binascii.unhexlify(hdr["challenge"])
        k_sess = self.derive_session_key(hdr["from"], my_priv, peer_pub, hdr["policy_id"], counter, challenge, contrib_self, contrib_peer)
        aad = f"{hdr['from']}|{hdr['to']}|{hdr['policy_id']}|{hdr['ctr']}|{hdr['challenge']}".encode()
        tag = HMAC(k_sess, aad, binascii.unhexlify(env["payload"]))
        if not hmac.compare_digest(tag, binascii.unhexlify(env["tag"])):
            print(f"[{self.device_id}] TAG MISMATCH")
            return False
        self.last_counter_from_peer[hdr["from"]] = counter
        return True
