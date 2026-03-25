"""
storage/store.py — Pluggable storage interface.

Interface contract (all backends must implement):
  put(key, value, meta)  -> StorageReceipt
  get(key)               -> StorageRecord | None
  list_keys(prefix)      -> list[str]
  delete(key)            -> bool
  stats()                -> dict

Backends shipped:
  MemoryStore   — in-process dict (demo default)
  FileStore     — one JSON file per record (swap-in for local persistence)

To add Redis/S3:
  class RedisStore(BaseStore): ...
  class S3Store(BaseStore): ...
"""

from __future__ import annotations
import hashlib, json, time, os
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from core.events import log_event


@dataclass
class StorageRecord:
    key:       str
    value:     bytes
    meta:      dict
    timestamp: float
    checksum:  str   # SHA-256 of value — integrity check on retrieval

    def verify(self) -> bool:
        return hashlib.sha256(self.value).hexdigest() == self.checksum

    def to_dict(self) -> dict:
        return {**asdict(self), "value": self.value.hex()}

    @staticmethod
    def from_dict(d: dict) -> "StorageRecord":
        return StorageRecord(key=d["key"], value=bytes.fromhex(d["value"]),
                             meta=d["meta"], timestamp=d["timestamp"],
                             checksum=d["checksum"])


@dataclass
class StorageReceipt:
    key:       str
    checksum:  str
    timestamp: float
    backend:   str
    peer_id:   str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class BaseStore(ABC):
    backend_name: str = "base"

    @abstractmethod
    def put(self, key: str, value: bytes, meta: dict | None = None) -> StorageReceipt: ...

    @abstractmethod
    def get(self, key: str) -> StorageRecord | None: ...

    @abstractmethod
    def list_keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def stats(self) -> dict: ...

    def _make_record(self, key: str, value: bytes, meta: dict) -> StorageRecord:
        return StorageRecord(key=key, value=value, meta=meta or {},
                             timestamp=time.time(),
                             checksum=hashlib.sha256(value).hexdigest())

    def _make_receipt(self, rec: StorageRecord, peer_id: str = "") -> StorageReceipt:
        return StorageReceipt(key=rec.key, checksum=rec.checksum,
                              timestamp=rec.timestamp, backend=self.backend_name,
                              peer_id=peer_id)


# ── MemoryStore ───────────────────────────────────────────────────────────────

class MemoryStore(BaseStore):
    """
    In-memory key-value store. Fast, zero-dependency, resets on restart.
    Drop-in replacement: swap with FileStore / RedisStore without changing callers.
    """
    backend_name = "memory"

    def __init__(self, peer_id: str = "local"):
        self._store: dict[str, StorageRecord] = {}
        self.peer_id = peer_id

    def put(self, key: str, value: bytes, meta: dict | None = None) -> StorageReceipt:
        rec = self._make_record(key, value, meta or {})
        self._store[key] = rec
        log_event("STORE_PUT", {"key": key, "backend": self.backend_name,
                                 "peer": self.peer_id, "bytes": len(value)})
        return self._make_receipt(rec, self.peer_id)

    def get(self, key: str) -> StorageRecord | None:
        rec = self._store.get(key)
        if rec and not rec.verify():
            log_event("STORE_CORRUPT", {"key": key, "peer": self.peer_id})
            return None
        log_event("STORE_GET", {"key": key, "found": rec is not None,
                                 "peer": self.peer_id})
        return rec

    def list_keys(self, prefix: str = "") -> list[str]:
        return [k for k in self._store if k.startswith(prefix)]

    def delete(self, key: str) -> bool:
        existed = key in self._store
        self._store.pop(key, None)
        log_event("STORE_DELETE", {"key": key, "existed": existed})
        return existed

    def stats(self) -> dict:
        total = sum(len(r.value) for r in self._store.values())
        return {"backend": self.backend_name, "peer": self.peer_id,
                "records": len(self._store), "bytes_total": total}


# ── FileStore ─────────────────────────────────────────────────────────────────

class FileStore(BaseStore):
    """
    JSON file-per-record store. Survives restarts. Swap-in for MemoryStore.
    Each record is stored as <root>/<key_hash>.json
    """
    backend_name = "file"

    def __init__(self, root: str, peer_id: str = "local"):
        self.root = root
        self.peer_id = peer_id
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        safe = hashlib.sha256(key.encode()).hexdigest()[:32]
        return os.path.join(self.root, safe + ".json")

    def put(self, key: str, value: bytes, meta: dict | None = None) -> StorageReceipt:
        rec = self._make_record(key, value, meta or {})
        with open(self._path(key), "w") as f:
            json.dump(rec.to_dict(), f)
        log_event("STORE_PUT", {"key": key, "backend": self.backend_name,
                                 "peer": self.peer_id, "bytes": len(value)})
        return self._make_receipt(rec, self.peer_id)

    def get(self, key: str) -> StorageRecord | None:
        p = self._path(key)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            rec = StorageRecord.from_dict(json.load(f))
        if not rec.verify():
            log_event("STORE_CORRUPT", {"key": key, "peer": self.peer_id})
            return None
        return rec

    def list_keys(self, prefix: str = "") -> list[str]:
        keys = []
        for fname in os.listdir(self.root):
            p = os.path.join(self.root, fname)
            try:
                with open(p) as f:
                    d = json.load(f)
                if d["key"].startswith(prefix):
                    keys.append(d["key"])
            except Exception:
                pass
        return keys

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if os.path.exists(p):
            os.remove(p)
            return True
        return False

    def stats(self) -> dict:
        count = len([f for f in os.listdir(self.root) if f.endswith(".json")])
        return {"backend": self.backend_name, "peer": self.peer_id,
                "records": count, "root": self.root}
