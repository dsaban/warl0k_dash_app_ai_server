from __future__ import annotations
import re
from typing import Dict, List

MECH_PATTERNS = {
    "CHALLENGE_RESPONSE": [r"challenge", r"response", r"challenge[- ]response"],
    "NONCE": [r"nonce", r"freshness"],
    "TIME_WINDOW": [r"time[- ](?:window|allowed|limited|stamp)", r"expires", r"expiration"],
    "COUNTER": [r"counter", r"sequence number", r"monotonic"],
    "DEVICE_FINGERPRINT": [r"fingerprint", r"device identification", r"identif(?:y|ying) value", r"mapped identification"],
    "PUF": [r"\bpuf\b", r"physical unclonable", r"sram puf", r"ring oscillator"],
    "OUT_OF_BAND": [r"li[- ]?fi", r"visible light", r"optical", r"ultrasonic", r"out[- ]of[- ]band"],
    "PAIRING": [r"pairing", r"ble", r"bluetooth", r"gesture", r"passcode"],
    "ATTESTATION": [r"attestation", r"secure boot", r"measurement"],
    "ANOMALY_SCORING": [r"anomal(y|ies)", r"risk score", r"continuous", r"behavior"],
}

CRYPTO_PATTERNS = {
    "MAC_AEAD": [r"\bmac\b", r"aead", r"aes[- ]ccm", r"aes[- ]gcm", r"poly1305", r"chacha"],
    "HASH": [r"\bhash\b", r"sha[- ]?\d+"],
    "KDF": [r"\bkdf\b", r"derive", r"hkdf"],
    "SIGNATURE": [r"signature", r"certificate", r"pki", r"public key"],
}

CONSTRAINT_PATTERNS = {
    "MCU_CONSTRAINED": [r"microcontroller", r"\bmcu\b", r"embedded", r"low power"],
    "NO_PKI": [r"without.*pki", r"no.*certificate", r"without.*certificate"],
    "OFFLINE": [r"offline", r"disconnected"],
    "BROKERED": [r"mqtt", r"broker", r"topic"],
    "OT_GATEWAY": [r"\bplc\b", r"\bot\b", r"gateway"],
}

THREAT_PATTERNS = {
    "REPLAY": [r"replay"],
    "MITM": [r"man[- ]in[- ]the[- ]middle", r"\bmitm\b"],
    "CLONING": [r"clone", r"cloning", r"counterfeit"],
    "SPOOFING": [r"spoof"],
    "HIJACK": [r"hijack", r"takeover"],
}

def _match_any(text: str, patterns: List[str]) -> bool:
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False

def tag_text(text: str) -> Dict[str, List[str]]:
    tags = {"mechanism": [], "crypto": [], "constraint": [], "threat": []}
    for k, pats in MECH_PATTERNS.items():
        if _match_any(text, pats):
            tags["mechanism"].append(k)
    for k, pats in CRYPTO_PATTERNS.items():
        if _match_any(text, pats):
            tags["crypto"].append(k)
    for k, pats in CONSTRAINT_PATTERNS.items():
        if _match_any(text, pats):
            tags["constraint"].append(k)
    for k, pats in THREAT_PATTERNS.items():
        if _match_any(text, pats):
            tags["threat"].append(k)
    return tags

def strength_score(text: str, is_independent: bool) -> float:
    verbs = len(re.findall(r"\b(?:generat|verif|transmit|receiv|deriv|authenticat|determin|comput|encrypt|decrypt)\w*\b", text, re.IGNORECASE))
    base = min(1.0, 0.25 + 0.08 * verbs)
    if is_independent:
        base = min(1.0, base + 0.10)
    return round(base, 3)
