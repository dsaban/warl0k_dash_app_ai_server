from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes, hmac, serialization
import os, binascii

def rand(n=32): return os.urandom(n)
def hexlify(b): return binascii.hexlify(b).decode()
def unhex(s): return binascii.unhexlify(s)

def gen_x25519_keypair():
 p=x25519.X25519PrivateKey.generate(); return p, p.public_key()

def pub_bytes_x25519(pub):
 return pub.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)

def x25519_shared(priv, peer_pub_bytes):
 return priv.exchange(x25519.X25519PublicKey.from_public_bytes(peer_pub_bytes))

def hkdf_sha256(salt, ikm, info, length=32):
 return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(ikm)

def hmac_sha256(key, *parts):
 h=hmac.HMAC(key, hashes.SHA256()); [h.update(p) for p in parts]; return h.finalize()

def aead_encrypt(key, nonce, pt, aad): return ChaCha20Poly1305(key).encrypt(nonce, pt, aad)

def aead_decrypt(key, nonce, ct, aad): return ChaCha20Poly1305(key).decrypt(nonce, ct, aad)
