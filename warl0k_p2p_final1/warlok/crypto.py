from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, hmac, serialization
import os, binascii

# For old cryptography versions that require backend=default_backend()
try:
    from cryptography.hazmat.backends import default_backend  # old API
    _BACKEND = default_backend()
except Exception:
    _BACKEND = None  # new API doesn't need it

def rand(n=32) -> bytes: return os.urandom(n)
def hexlify(b: bytes) -> str: return binascii.hexlify(b).decode()
def unhex(s: str) -> bytes: return binascii.unhexlify(s)

def gen_x25519_keypair():
    priv = x25519.X25519PrivateKey.generate()
    pub  = priv.public_key()
    return priv, pub

def pub_bytes_x25519(pub) -> bytes:
    return pub.public_bytes(encoding=serialization.Encoding.Raw,
                            format=serialization.PublicFormat.Raw)

def x25519_shared(priv, peer_pub_bytes: bytes) -> bytes:
    peer_pub = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)
    return priv.exchange(peer_pub)

def hkdf_sha256(salt: bytes, ikm: bytes, info: bytes, length=32) -> bytes:
    try:
        hk = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
    except TypeError:
        # old cryptography needs backend
        hk = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info, backend=_BACKEND)
    return hk.derive(ikm)

def hmac_sha256(key: bytes, *parts: bytes) -> bytes:
    try:
        h = hmac.HMAC(key, hashes.SHA256())
    except TypeError:
        # old cryptography needs backend
        h = hmac.HMAC(key, hashes.SHA256(), backend=_BACKEND)
    for p in parts:
        h.update(p)
    return h.finalize()

def aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    return ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)

def aead_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, aad)

# from cryptography.hazmat.primitives.asymmetric import x25519
# from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
# from cryptography.hazmat.primitives.kdf.hkdf import HKDF
# from cryptography.hazmat.primitives import hashes, hmac, serialization
# import os, binascii
#
# def rand(n=32) -> bytes: return os.urandom(n)
# def hexlify(b: bytes) -> str: return binascii.hexlify(b).decode()
# def unhex(s: str) -> bytes: return binascii.unhexlify(s)
#
# def gen_x25519_keypair():
#     priv = x25519.X25519PrivateKey.generate()
#     pub  = priv.public_key()
#     return priv, pub
#
# def pub_bytes_x25519(pub) -> bytes:
#     return pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
#
# def x25519_shared(priv, peer_pub_bytes: bytes) -> bytes:
#     peer_pub = x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes)
#     return priv.exchange(peer_pub)
#
# def hkdf_sha256(salt: bytes, ikm: bytes, info: bytes, length=32) -> bytes:
#     hk = HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info)
#     return hk.derive(ikm)
#
# def hmac_sha256(key: bytes, *parts: bytes) -> bytes:
#     h = hmac.HMAC(key, hashes.SHA256())
#     for p in parts: h.update(p)
#     return h.finalize()
#
# def aead_encrypt(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
#     return ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
#
# def aead_decrypt(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
#     return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, aad)
