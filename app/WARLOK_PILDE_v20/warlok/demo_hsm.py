"""
DemoHSM v2 — per-block key derivation + epoch rotation.

Key hierarchy:
  root_key
    └── epoch_key  = HKDF(root_key, "epoch:{id}")     — rotates every EPOCH_SIZE blocks
          └── block_key = HKDF(epoch_key, block_hash)  — unique per block

sign(data, block_hash)  -> HMAC-SHA256(block_key, data)
verify(...)             -> re-derive, compare
epoch_token()           -> shareable proof-of-epoch for peer cross-validation
"""
from __future__ import annotations

import hashlib, hmac as _hmac
from core.events import log_event

EPOCH_SIZE = 10  # blocks per epoch before root rotation


def _derive(parent: bytes, context: bytes) -> bytes:
    """HKDF-lite: PRK = HMAC(parent, context), OKM = HMAC(PRK, context+0x01)"""
    prk = _hmac.new(parent, context, hashlib.sha256).digest()
    return _hmac.new(prk, context + b'\x01', hashlib.sha256).digest()


class DemoHSM:
    def __init__(self, root_key: bytes):
        self.root_key    = root_key
        self._block_count = 0
        self._epoch_id   = 0
        self._epoch_key  = _derive(root_key, b"epoch:0")
        self._block_keys: dict = {}   # block_hash -> derived key (cache)

    # ── Epoch ─────────────────────────────────────────────────────────────────

    @property
    def epoch_id(self) -> int:
        return self._epoch_id

    def _maybe_rotate(self):
        new_epoch = self._block_count // EPOCH_SIZE
        if new_epoch != self._epoch_id:
            self._epoch_id  = new_epoch
            self._epoch_key = _derive(self.root_key, f"epoch:{self._epoch_id}".encode())
            log_event("HSM_EPOCH_ROTATE", {"epoch": self._epoch_id,
                                            "at_block": self._block_count})

    def epoch_token(self, epoch_id: int | None = None) -> bytes:
        """Shareable peer-validation token — HMAC of epoch key, NOT the key itself."""
        eid = epoch_id if epoch_id is not None else self._epoch_id
        ek  = _derive(self.root_key, f"epoch:{eid}".encode())
        tok = _hmac.new(ek, b"peer-token", hashlib.sha256).digest()
        log_event("HSM_EPOCH_TOKEN", {"epoch": eid})
        return tok

    # ── Per-block key ─────────────────────────────────────────────────────────

    def _bkey(self, block_hash: str) -> bytes:
        if block_hash not in self._block_keys:
            self._block_keys[block_hash] = _derive(self._epoch_key,
                                                    block_hash.encode())
        return self._block_keys[block_hash]

    # ── Sign / Verify ─────────────────────────────────────────────────────────

    def sign(self, data: bytes, block_hash: str | None = None) -> bytes:
        self._block_count += 1
        self._maybe_rotate()
        key = self._bkey(block_hash) if block_hash else self._epoch_key
        sig = _hmac.new(key, data, hashlib.sha256).digest()
        log_event("HSM_SIGN", {"len": len(data),
                                "bh": (block_hash or "")[:16],
                                "epoch": self._epoch_id,
                                "mode": "block" if block_hash else "epoch"})
        return sig

    def verify(self, data: bytes, sig: bytes,
               block_hash: str | None = None) -> bool:
        key      = self._bkey(block_hash) if block_hash else self._epoch_key
        expected = _hmac.new(key, data, hashlib.sha256).digest()
        ok       = _hmac.compare_digest(expected, sig)
        log_event("HSM_VERIFY", {"bh": (block_hash or "")[:16],
                                  "epoch": self._epoch_id, "ok": ok})
        return ok

    def verify_epoch(self, data: bytes, sig: bytes,
                     block_hash: str, epoch_id: int) -> bool:
        """Cross-epoch verify — for peers validating blocks from past epochs."""
        ek  = _derive(self.root_key, f"epoch:{epoch_id}".encode())
        bk  = _derive(ek, block_hash.encode())
        ok  = _hmac.compare_digest(_hmac.new(bk, data, hashlib.sha256).digest(), sig)
        log_event("HSM_CROSS_VERIFY", {"bh": block_hash[:16],
                                        "epoch": epoch_id, "ok": ok})
        return ok

    def status(self) -> dict:
        return {"epoch_id": self._epoch_id,
                "block_count": self._block_count,
                "epoch_size": EPOCH_SIZE,
                "next_rotation_in": EPOCH_SIZE - (self._block_count % EPOCH_SIZE),
                "cached_keys": len(self._block_keys)}
