import time
import json
import hashlib
from typing import Any, Dict

def now_ts() -> float:
    return time.time()

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canon_json(obj: Dict[str, Any]) -> bytes:
    # Canonical JSON: stable ordering and compact separators
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
