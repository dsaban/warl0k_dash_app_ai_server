
import base64, os, hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

def derive_key_from_secret(secret_string: str) -> bytes:
    digest = hashlib.sha256(secret_string.encode()).digest()
    return digest[:16]

def encrypt_aead(secret_string: str, plaintext: str, aad_bytes: bytes):
    key = derive_key_from_secret(secret_string)
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext.encode(), aad_bytes)
    return {
        "nonce_b64": base64.b64encode(nonce).decode(),
        "cipher_b64": base64.b64encode(ct).decode(),
    }

def decrypt_aead(secret_string: str, nonce_b64: str, cipher_b64: str, aad_bytes: bytes) -> str:
    key = derive_key_from_secret(secret_string)
    aesgcm = AESGCM(key)
    nonce = base64.b64decode(nonce_b64)
    ct = base64.b64decode(cipher_b64)
    pt = aesgcm.decrypt(nonce, ct, aad_bytes)
    return pt.decode()
