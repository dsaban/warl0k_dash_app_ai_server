from dataclasses import dataclass
from typing import Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except Exception as e:
    AESGCM = None
    _CRYPTO_IMPORT_ERROR = e

import os

@dataclass
class CryptoBox:
    key: bytes  # 32 bytes for AES-256-GCM

    @staticmethod
    def new() -> "CryptoBox":
        return CryptoBox(key=os.urandom(32))

    def encrypt(self, plaintext: bytes, aad: bytes = b"") -> Tuple[bytes, bytes]:
        if AESGCM is None:
            raise RuntimeError(f"cryptography not available: {_CRYPTO_IMPORT_ERROR}")
        nonce = os.urandom(12)
        ct = AESGCM(self.key).encrypt(nonce, plaintext, aad)
        return nonce, ct

    def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes = b"") -> bytes:
        if AESGCM is None:
            raise RuntimeError(f"cryptography not available: {_CRYPTO_IMPORT_ERROR}")
        return AESGCM(self.key).decrypt(nonce, ciphertext, aad)
