"""
warlok_pim_hash
===============

A conceptual Proof-in-Motion inspired "learned hash" / HMAC-like primitive.

This library demonstrates how a tiny model can learn to reconstruct a fixed
master secret from obfuscated secrets derived from (master_secret, seed, counter),
and allows monitoring of behavior across sliding windows of counters.

⚠ IMPORTANT: This is NOT a real cryptographic primitive and MUST NOT be used
for production security. It is for experimentation and research only.
"""

from .config import PIMHashConfig
from .model import PIMHashModel
from .exceptions import VerificationError

__all__ = [
    "PIMHashConfig",
    "PIMHashModel",
    "VerificationError",
]
