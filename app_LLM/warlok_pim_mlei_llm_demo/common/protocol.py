from dataclasses import dataclass
from typing import Dict, Any

from .crypto import CryptoBox
from .util import canon_json
import json

@dataclass
class SecureChannel:
    crypto: CryptoBox

    def seal(self, env: Dict[str, Any]) -> Dict[str, Any]:
        pt = canon_json(env)
        aad = b"WARLOK-PIM"
        nonce, ct = self.crypto.encrypt(pt, aad=aad)
        return {"nonce": nonce.hex(), "ct": ct.hex()}

    def open(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        aad = b"WARLOK-PIM"
        nonce = bytes.fromhex(blob["nonce"])
        ct = bytes.fromhex(blob["ct"])
        pt = self.crypto.decrypt(nonce, ct, aad=aad)
        return json.loads(pt.decode("utf-8"))
