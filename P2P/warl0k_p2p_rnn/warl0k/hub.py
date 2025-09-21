from dataclasses import dataclass, field
from typing import Dict
from .crypto_utils import random_bytes, HMAC

@dataclass
class Hub:
    hub_key: bytes = field(default_factory=lambda: random_bytes(32))
    registry: Dict[str, bytes] = field(default_factory=dict)

    def enroll(self, device_id: str):
        seed0 = random_bytes(32)
        self.registry[device_id] = seed0
        signature = HMAC(self.hub_key, device_id.encode(), seed0)
        return seed0, signature

    def verify_seed(self, device_id: str, seed0: bytes, signature: bytes) -> bool:
        expected = HMAC(self.hub_key, device_id.encode(), seed0)
        return expected == signature
