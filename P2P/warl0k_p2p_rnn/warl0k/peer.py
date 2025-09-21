from dataclasses import dataclass, field
from typing import Dict
from .crypto_utils import HMAC, H, hkdf, dh_generate_keypair, dh_shared_secret, random_bytes, bhex
import binascii, hmac

@dataclass
class Peer:
    device_id: str
    hub: object
    device_nonce: bytes = field(default_factory=lambda: random_bytes(32))
    seed0: bytes = b""
    seed_sig: bytes = b""
    master_seed: bytes = b""
    # local device secret string (a short "master identity") to be predicted by RNN
    device_identity_string: str = None
    last_counter_from_peer: Dict[str, int] = field(default_factory=dict)

    def enroll_with_hub(self, identity_string: str = None):
        self.seed0, self.seed_sig = self.hub.enroll(self.device_id)
        assert self.hub.verify_seed(self.device_id, self.seed0, self.seed_sig)
        self.master_seed = HMAC(self.seed0, self.device_nonce)
        self.seed0 = b""  # drop
        # device_identity_string: short string (letters/digits from vocab) representing device
        self.device_identity_string = identity_string or self._generate_identity()

    def _generate_identity(self, length: int = 8) -> str:
        # deterministic-ish per-run identity for demo (in prod, user chooses or securely store)
        import os, binascii
        return binascii.hexlify(os.urandom(length//2)).decode()[:length]

    def contrib(self) -> bytes:
        return HMAC(self.master_seed, b"contrib:", self.device_id.encode())

    # derive_session_key same approach as earlier
    def derive_session_key(self, peer_device_id: str, my_priv: int, peer_pub: int,
                           policy_id: str, counter: int, challenge: bytes,
                           contrib_self: bytes, contrib_peer: bytes) -> bytes:
        salt = dh_shared_secret(my_priv, peer_pub)
        left_id, right_id = sorted([self.device_id, peer_device_id])
        if self.device_id == left_id:
            combined_IKM = H(contrib_self, contrib_peer)
        else:
            combined_IKM = H(contrib_peer, contrib_self)
        info = b"|".join([b"WARL0Kv1",
                          left_id.encode(), right_id.encode(),
                          policy_id.encode(),
                          counter.to_bytes(8, "big"),
                          challenge])
        return hkdf(salt=salt, ikm=combined_IKM, info=info, length=32)

    # envelope uses HMAC as tag for demo
    def build_envelope(self, peer_pub: int, my_priv: int, peer_device_id: str,
                       policy_id: str, counter: int, challenge: bytes, plaintext: bytes,
                       contrib_self: bytes, contrib_peer: bytes) -> dict:
        k_sess = self.derive_session_key(peer_device_id, my_priv, peer_pub, policy_id, counter, challenge, contrib_self, contrib_peer)
        header = {"from": self.device_id, "to": peer_device_id, "policy_id": policy_id, "ctr": counter, "challenge": bhex(challenge)}
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
        print(f"[{self.device_id}] computed tag: {bhex(tag)}, received tag: {env['tag']}")
        if not hmac.compare_digest(tag, binascii.unhexlify(env["tag"])):
            print(f"[{self.device_id}] TAG MISMATCH")
            return False
        self.last_counter_from_peer[hdr["from"]] = counter
        return True
