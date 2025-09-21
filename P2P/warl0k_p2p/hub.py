"""
Simple Hub: enrollment and registry.
Hub issues seed0 and simulates signing using HMAC with a hub key.
"""

from dataclasses import dataclass, field
from typing import Dict
from .crypto_utils import random_bytes, HMAC

@dataclass
class Hub:
    hub_key: bytes = field(default_factory=lambda: random_bytes(32))
    registry: Dict[str, bytes] = field(default_factory=dict)

    def enroll(self, device_id: str):
        """
        Issue a fresh seed0 for a device and return (seed0, signature).
        Signature is simulated HMAC(hub_key, device_id || seed0).
        """
        seed0 = random_bytes(32)
        self.registry[device_id] = seed0
        signature = HMAC(self.hub_key, device_id.encode(), seed0)
        return seed0, signature

    def verify_seed(self, device_id: str, seed0: bytes, signature: bytes) -> bool:
        expected = HMAC(self.hub_key, device_id.encode(), seed0)
        return hmac_compare(expected, signature)

def hmac_compare(a: bytes, b: bytes) -> bool:
    # constant-time compare
    return a == b
