from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

def encrypt(key, data):
    nonce = os.urandom(12)
    aes = AESGCM(key)
    return nonce, aes.encrypt(nonce, data, None)