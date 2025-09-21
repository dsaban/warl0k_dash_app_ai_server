"""
Simple crypto utilities for WARL0K demo.
NOTE: Educational implementations only. Replace with standards libs in prod.
"""

import os, hmac, hashlib, secrets, binascii
from typing import Tuple

# ---------- HKDF (HMAC-SHA256) ----------
def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    out = b""
    t = b""
    i = 1
    while len(out) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        out += t
        i += 1
    return out[:length]

def hkdf(salt: bytes, ikm: bytes, info: bytes, length: int = 32) -> bytes:
    prk = hkdf_extract(salt, ikm)
    return hkdf_expand(prk, info, length)

# ---------- H / HMAC helpers ----------
def H(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()

def HMAC(key: bytes, *parts: bytes) -> bytes:
    return hmac.new(key, b"".join(parts), hashlib.sha256).digest()

def bhex(b: bytes) -> str:
    return binascii.hexlify(b).decode()

# ---------- Simple MODP 2048-bit DH (demo only) ----------
MODP_2048_P_HEX = (
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08"
    "8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B"
    "302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9"
    "A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6"
    "49286651ECE65381FFFFFFFFFFFFFFFF"
)
MODP_P = int(MODP_2048_P_HEX, 16)
MODP_G = 2

def dh_generate_keypair() -> Tuple[int, int]:
    # Use a modest private exponent for demo; in prod use secure curve X25519
    priv = secrets.randbits(256)
    pub = pow(MODP_G, priv, MODP_P)
    return priv, pub

def dh_shared_secret(priv: int, peer_pub: int) -> bytes:
    shared = pow(peer_pub, priv, MODP_P)
    shared_bytes = shared.to_bytes((shared.bit_length() + 7) // 8, "big")
    return hashlib.sha256(shared_bytes).digest()

# ---------- Random helpers ----------
def random_bytes(n: int) -> bytes:
    return os.urandom(n)
