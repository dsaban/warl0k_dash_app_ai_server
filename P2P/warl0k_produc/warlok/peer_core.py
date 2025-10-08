import os
from dataclasses import dataclass, field
from typing import Dict, Tuple
from .crypto import x25519_shared, hkdf_sha256, aead_encrypt, aead_decrypt, hexlify, unhex

@dataclass
class PeerCore:
    device_id: str
    last_ctr_from_peer: Dict[str, int] = field(default_factory=dict)

    def derive_k_session(self, my_priv, peer_pub_bytes: bytes, my_id: str, their_id: str,
                         policy_id: str, counter: int, challenge: bytes) -> bytes:
        shared = x25519_shared(my_priv, peer_pub_bytes)
        left, right = sorted([my_id, their_id])
        info = b"|".join([b"WARL0K", left.encode(), right.encode(),
                          policy_id.encode(), counter.to_bytes(8,"big"), challenge])
        return hkdf_sha256(salt=shared, ikm=b"p2p", info=info, length=32)

    def obf_from_k(self, k_sess: bytes, obf_len: int) -> str:
        o = hkdf_sha256(salt=b"obf", ikm=k_sess, info=b"obf", length=32)
        return hexlify(o)[:obf_len]

    def build_envelope(self, my_id: str, their_id: str, my_priv, peer_pub_bytes: bytes,
                       policy_id: str, counter: int, challenge: bytes, plaintext: bytes) -> dict:
        k = self.derive_k_session(my_priv, peer_pub_bytes, my_id, their_id, policy_id, counter, challenge)
        nonce = os.urandom(12)
        hdr = {"from": my_id, "to": their_id, "policy_id": policy_id,
               "ctr": counter, "challenge": hexlify(challenge)}
        aad = f"{hdr['from']}|{hdr['to']}|{hdr['policy_id']}|{hdr['ctr']}|{hdr['challenge']}".encode()
        ct = aead_encrypt(k, nonce, plaintext, aad)
        return {"header": hdr, "nonce": hexlify(nonce), "ciphertext": hexlify(ct)}

    def verify_envelope(self, env: dict, my_id: str, their_id: str, my_priv, peer_pub_bytes: bytes) -> Tuple[bool, bytes]:
        hdr = env["header"]
        ctr = int(hdr["ctr"])
        last = self.last_ctr_from_peer.get(hdr["from"], -1)
        if ctr <= last: return False, b""
        challenge = unhex(hdr["challenge"])
        k = self.derive_k_session(my_priv, peer_pub_bytes, my_id, their_id, hdr["policy_id"], ctr, challenge)
        nonce = unhex(env["nonce"])
        aad = f"{hdr['from']}|{hdr['to']}|{hdr['policy_id']}|{hdr['ctr']}|{hdr['challenge']}".encode()
        try:
            pt = aead_decrypt(k, nonce, unhex(env["ciphertext"]), aad)
        except Exception:
            return False, b""
        self.last_ctr_from_peer[hdr["from"]] = ctr
        return True, pt
