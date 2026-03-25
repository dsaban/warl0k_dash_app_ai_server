#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PCD ARCHIVE MODULE  ·  pcd_archive.py  ·  v3.3                          ║
║                                                                           ║
║  1. EpochSecretArchive — three-tier (HOT·WARM·COLD) epoch secret store  ║
║  2. DataRecordArchive  — append-only archived Envelope store             ║
║  3. SimulatedHSM       — PKCS#11-style boundary simulation:             ║
║       · seal / unseal epoch secrets (C_Sign / C_Decrypt analogs)        ║
║       · derive_ktn()  — K(t,n) derived INSIDE HSM, never exported      ║
║       · issue_resource_token() — signed token for distributed peers     ║
║       · verify_resource_token() — token verification without secret     ║
║       · hsm_decrypt_with_token() — full gated decrypt for remote peers  ║
║       · HSM session log — every operation recorded with latency         ║
║                                                                           ║
║  HSM Model A (production): epoch_secret + K(t,n) derivation in silicon  ║
║  HSM Model C (demo):       XOR cipher, same API surface                 ║
║                                                                           ║
║  Distributed resource flow:                                              ║
║    Peer A seals data (epoch_secret in HSM)                               ║
║    Peer B wants old data → POST /api/hsm/token (gets signed token)      ║
║    Peer B sends token → POST /api/hsm/read   → HSM verifies + decrypts  ║
║    K(t,n) never leaves HSM boundary                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import os
import time
import threading
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).parent
ARCHIVE_DATA_FILE = BASE_DIR / "pcd_archive.json"
ARCHIVE_HSM_FILE  = BASE_DIR / "pcd_hsm_archive.json"
ARCHIVE_LOG_FILE  = BASE_DIR / "pcd_archive.log"
HSM_SESSION_FILE  = BASE_DIR / "pcd_hsm_session.json"

# ═══════════════════════════════════════════════════════════════════════
#  LOGGER
# ═══════════════════════════════════════════════════════════════════════
_arc_handler = logging.FileHandler(ARCHIVE_LOG_FILE, encoding="utf-8")
_arc_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"))
arc_log = logging.getLogger("pcd.archive")
arc_log.setLevel(logging.DEBUG)
arc_log.addHandler(_arc_handler)
arc_log.propagate = False

def arc_audit(level: str, actor: str, op: str, detail: str = "") -> None:
    msg = f"[{actor:14s}] {op:22s} {detail}"
    getattr(arc_log, level.lower(), arc_log.info)(msg)


# ═══════════════════════════════════════════════════════════════════════
#  CRYPTO HELPERS
# ═══════════════════════════════════════════════════════════════════════
def _sha256(data: bytes | str) -> str:
    if isinstance(data, str): data = data.encode()
    return hashlib.sha256(data).hexdigest()

def _hmac256(key: bytes | str, msg: bytes | str) -> str:
    if isinstance(key, str): key = key.encode()
    if isinstance(msg, str): msg = msg.encode()
    return _hmac.new(key, msg, hashlib.sha256).hexdigest()

def _xor_cipher(data: bytes, key_hex: str) -> bytes:
    key_bytes = bytes.fromhex(hashlib.sha256(key_hex.encode()).hexdigest() * 4)
    return bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))

def _xor_encrypt(pt: str, key_hex: str) -> str:
    return _xor_cipher(pt.encode("utf-8"), key_hex).hex()

def _xor_decrypt(ct_hex: str, key_hex: str) -> str:
    return _xor_cipher(bytes.fromhex(ct_hex), key_hex).decode("utf-8")

def _short(h: str) -> str:
    return f"{h[:8]}…{h[-6:]}" if h and len(h) > 16 else (h or "—")

def _rng(n: int = 16) -> str:
    return os.urandom(n).hex()


# ═══════════════════════════════════════════════════════════════════════
#  ADMIN POLICY
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class ArchivePolicy:
    """All limits configurable via PUT /api/archive/policy without restart."""
    max_epoch_keys:      int   = 50
    max_epoch_age_days:  float = 0.0
    max_epoch_bytes:     int   = 0
    max_record_count:    int   = 10_000
    max_record_age_days: float = 365.0
    max_record_bytes:    int   = 0
    hot_epochs:          int   = 5
    warm_epochs:         int   = 20
    enforce_on_rotate:   bool  = True
    enforce_on_read:     bool  = False
    resource_token_ttl:  int   = 300    # seconds — HSM-issued resource token lifetime

    def to_dict(self) -> dict:  return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchivePolicy":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ═══════════════════════════════════════════════════════════════════════
#  HSM OPERATION LOG ENTRY
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class HSMOperation:
    op_id:      str
    operation:  str     # SEAL_EPOCH|UNSEAL_EPOCH|DERIVE_KTN|ISSUE_TOKEN|VERIFY_TOKEN|HSM_DECRYPT|COLD_STORE|COLD_LOAD|COLD_DELETE
    epoch:      Optional[int]
    actor:      str
    node_id:    Optional[str]
    tier:       str
    latency_ms: float
    success:    bool
    detail:     str
    ts:         float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.ts))
        return d


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATED HSM  — PKCS#11-style boundary
# ═══════════════════════════════════════════════════════════════════════
class SimulatedHSM:
    """
    Simulates a FIPS 140-3 Level 4 HSM with PKCS#11 interface.

    Production replacement map:
      seal()                     → C_Sign(key_handle=EPOCH_KEY, CKM_SHA256_HMAC, data)
      unseal()                   → C_Decrypt(key_handle=EPOCH_KEY, CKM_AES_GCM, ct)
      derive_ktn()               → C_DeriveKey(CKM_SHA256_HMAC, epoch_key, params)
      issue_resource_token()     → C_Sign(key_handle=TOKEN_KEY, CKM_SHA256_HMAC, claims)
      verify_resource_token()    → C_Verify(TOKEN_KEY, CKM_SHA256_HMAC, token, sig)
      hsm_decrypt_with_token()   → all of the above chained inside the HSM firmware

    The master_key and token_key NEVER leave this class boundary.
    In production deployment these are key handles in HSM hardware — the
    actual key material never enters application memory.
    """

    def __init__(self, master_key: str):
        self._master_key = master_key
        self._token_key  = _hmac256(master_key, "TOKEN-SIGNING-KEY-SLOT-2")
        self._file       = ARCHIVE_HSM_FILE
        self._lock       = threading.Lock()
        self._session_log: List[HSMOperation] = []
        self._sf_lock    = threading.Lock()

    # ── Internal: log every HSM operation ─────────────────────────────
    def _log(self, operation: str, epoch: Optional[int], actor: str,
             node_id: Optional[str], tier: str, latency_ms: float,
             success: bool, detail: str) -> HSMOperation:
        op = HSMOperation(
            op_id=_rng(6), operation=operation, epoch=epoch,
            actor=actor, node_id=node_id, tier=tier,
            latency_ms=round(latency_ms, 3), success=success, detail=detail
        )
        with self._sf_lock:
            self._session_log.append(op)
            if len(self._session_log) > 500:
                self._session_log = self._session_log[-400:]
            try:
                existing: list = []
                if HSM_SESSION_FILE.exists():
                    existing = json.loads(HSM_SESSION_FILE.read_text(encoding="utf-8"))
                existing.append(op.to_dict())
                if len(existing) > 1000:
                    existing = existing[-800:]
                HSM_SESSION_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
            except Exception:
                pass
        arc_audit("info" if success else "error", "HSM",
                  operation, f"epoch={epoch} actor={actor} ok={success} {detail}")
        return op

    # ── PKCS#11 analog: C_Sign → seal epoch_secret ────────────────────
    def seal(self, epoch: int, secret: str) -> str:
        """
        Encrypts epoch_secret under archive_master_key.
        Production: C_Sign(EPOCH_ARCHIVE_KEY, CKM_AES_256_GCM, secret)
        epoch_secret never leaves HSM in production.
        """
        t0 = time.perf_counter()
        ct = _xor_encrypt(secret, f"{self._master_key}|epoch|{epoch}")
        ms = (time.perf_counter() - t0) * 1000
        self._log("SEAL_EPOCH", epoch, "HSM", None, "COLD", ms, True, f"ct={_short(ct)}")
        return ct

    # ── PKCS#11 analog: C_Decrypt → unseal epoch_secret ───────────────
    def unseal(self, epoch: int, ciphertext: str) -> str:
        """
        Decrypts epoch_secret from ciphertext. Used internally before derive_ktn.
        Production: C_Decrypt(EPOCH_ARCHIVE_KEY, CKM_AES_256_GCM, ciphertext)
        """
        t0 = time.perf_counter()
        pt = _xor_decrypt(ciphertext, f"{self._master_key}|epoch|{epoch}")
        ms = (time.perf_counter() - t0) * 1000
        self._log("UNSEAL_EPOCH", epoch, "HSM", None, "COLD", ms, True, "")
        return pt

    # ── PKCS#11 analog: C_DeriveKey → K(t,n) inside HSM ──────────────
    def derive_ktn(self, epoch: int, feature_fp: str, counter: int,
                   window_pos: float, actor: str = "SYSTEM") -> Optional[str]:
        """
        Derives K(t,n) = HMAC(epoch_secret, feature_fp|counter|window_pos)
        ENTIRELY INSIDE the HSM boundary.

        Production equivalent:
            C_DeriveKey(
                base_key    = epoch_key_handle,   # stays in HSM silicon
                mechanism   = CKM_SHA256_HMAC,
                derive_data = f'{feature_fp}|{counter}|{window_pos:.6f}'
            ) → derived_key_handle                # also stays in HSM
            # then C_Encrypt(derived_key_handle, ...) to use it in-place

        K(t,n) is returned here for demo purposes.
        In production only the ENCRYPTED output exits the HSM.
        """
        t0 = time.perf_counter()
        ct = self.load_cold(epoch)
        if ct is None:
            ms = (time.perf_counter() - t0) * 1000
            self._log("DERIVE_KTN", epoch, actor, None, "COLD", ms, False,
                      "epoch not in cold store")
            return None
        ep_sec = self.unseal(epoch, ct)
        ktn    = _hmac256(ep_sec, f"{feature_fp}|{counter}|{window_pos:.6f}")
        ep_sec = "WIPED"   # immediately wipe in simulation
        ms = (time.perf_counter() - t0) * 1000
        self._log("DERIVE_KTN", epoch, actor, None, "COLD", ms, True,
                  f"fp={_short(feature_fp)} ctr={counter} ktn={_short(ktn)}")
        return ktn

    # ── Issue a signed resource token for distributed peer access ──────
    def issue_resource_token(self, node_id: str, epoch: int, feature_fp: str,
                              requesting_peer: str, ttl_seconds: int = 300,
                              actor: str = "SYSTEM") -> dict:
        """
        Issues a time-bound, HSM-signed access token for a distributed peer.

        Token grants: peer 'requesting_peer' may request HSM-gated decryption
        of 'node_id' (which was sealed in 'epoch' and whose feature fingerprint
        is 'feature_fp') until 'expires_at'.

        Production:
            claims_bytes = canonical_claims.encode()
            sig = C_Sign(TOKEN_KEY_HANDLE, CKM_SHA256_HMAC, claims_bytes)
            # TOKEN_KEY_HANDLE never leaves HSM silicon

        The token is safe to transmit over any channel — without the HSM's
        token_key (which never leaves the HSM) it cannot be forged.
        """
        t0         = time.perf_counter()
        issued_at  = time.time()
        expires_at = issued_at + ttl_seconds
        nonce      = _rng(8)
        claims = {
            "node_id":         node_id,
            "epoch":           epoch,
            "feature_fp":      feature_fp,
            "requesting_peer": requesting_peer,
            "issued_at":       round(issued_at, 6),
            "expires_at":      round(expires_at, 6),
            "nonce":           nonce,
        }
        canon = "|".join(str(claims[k]) for k in sorted(claims))
        sig   = _hmac256(self._token_key, canon)
        token = {
            **claims,
            "sig":            sig,
            "issued_at_iso":  time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(issued_at)),
            "expires_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(expires_at)),
            "ttl_seconds":    ttl_seconds,
        }
        ms = (time.perf_counter() - t0) * 1000
        self._log("ISSUE_TOKEN", epoch, actor, node_id, "TOKEN", ms, True,
                  f"peer={requesting_peer} ttl={ttl_seconds}s nonce={nonce}")
        arc_audit("info", "HSM", "TOKEN_ISSUED",
                  f"node={_short(node_id)} peer={requesting_peer} ttl={ttl_seconds}s")
        return token

    # ── Verify a resource token (without revealing token_key) ──────────
    def verify_resource_token(self, token: dict,
                               actor: str = "SYSTEM") -> Tuple[bool, str]:
        """
        Verifies signature and expiry of a resource token.

        Production: C_Verify(TOKEN_KEY_HANDLE, CKM_SHA256_HMAC, canon, sig)
        token_key never leaves HSM — verification is done inside the HSM.
        """
        t0 = time.perf_counter()
        try:
            sig_rx = token.get("sig", "")
            claims = {k: token[k] for k in
                      ["node_id","epoch","feature_fp","requesting_peer",
                       "issued_at","expires_at","nonce"]}
            canon  = "|".join(str(claims[k]) for k in sorted(claims))
            sig_ok = _hmac256(self._token_key, canon) == sig_rx
            time_ok= time.time() <= float(token.get("expires_at", 0))
            valid  = sig_ok and time_ok
            reason = ("OK" if valid else
                      "TOKEN_EXPIRED" if sig_ok else "SIGNATURE_INVALID")
        except Exception as e:
            valid, reason = False, f"MALFORMED:{e}"
        ms = (time.perf_counter() - t0) * 1000
        self._log("VERIFY_TOKEN", token.get("epoch"), actor,
                  token.get("node_id"), "TOKEN", ms, valid, reason)
        return valid, reason

    # ── Full HSM-gated decrypt for distributed peer access ─────────────
    def hsm_decrypt_with_token(self, token: dict, envelope: dict,
                                actor: str = "SYSTEM") -> Tuple[Optional[str], str]:
        """
        Complete gated decrypt pipeline — all sensitive operations inside HSM:
          1. Verify resource token (sig + expiry)
          2. Bind token to this specific envelope (node_id + feature_fp match)
          3. Retrieve epoch_secret from cold store
          4. Rederive feature fingerprint from stored ai_features
          5. Derive K(t,n) inside HSM boundary
          6. Verify ZK proof (proves key possession without revealing key)
          7. Decrypt cipher_payload with K(t,n)
          8. Verify payload_hash (SHA-256 of plaintext)
          9. Wipe all intermediate secrets

        Production: steps 3–8 all execute inside HSM firmware; only the
        plaintext (or an encrypted version of it) exits the HSM boundary.

        Returns (plaintext | None, status_message)
        """
        t0      = time.perf_counter()
        node_id = envelope.get("node_id", "")

        # 1. Token verification
        valid, reason = self.verify_resource_token(token, actor)
        if not valid:
            return None, f"TOKEN_INVALID:{reason}"

        # 2. Token–envelope binding
        if token.get("node_id") != node_id:
            return None, "TOKEN_NODE_MISMATCH"
        if token.get("feature_fp") != envelope.get("feature_fp", ""):
            return None, "TOKEN_FP_MISMATCH"

        # 3. Retrieve epoch_secret
        epoch = envelope.get("epoch", 0)
        ct    = self.load_cold(epoch)
        if ct is None:
            return None, f"EPOCH_{epoch}_NOT_IN_COLD_STORE"
        ep_sec = self.unseal(epoch, ct)

        # 4. Rederive feature fingerprint (verifies stored features are intact)
        F = envelope.get("ai_features", {})
        fp_local = _sha256(
            f"{F.get('f1',0):.6f}|{F.get('f2',0):.6f}|{F.get('f3',0):.6f}"
            f"|{int(F.get('f6',0))}|{F.get('f4',0):.6f}|{F.get('f5',0):.6f}"
        )
        if fp_local != envelope.get("feature_fp", ""):
            ep_sec = "WIPED"
            return None, "FEATURE_FP_MISMATCH"

        # 5. Derive K(t,n) inside HSM
        ktn    = _hmac256(ep_sec, f"{fp_local}|{int(F.get('f6',0))}|{F.get('f4',0):.6f}")
        ep_sec = "WIPED"

        # 6. ZK proof verification
        exp_zk = _hmac256(ktn, f"zk|{envelope.get('actor','')}|{node_id}")
        if exp_zk != envelope.get("zk_proof", ""):
            ktn = "WIPED"
            return None, "ZK_PROOF_FAILED"

        # 7. Decrypt
        try:
            key_bytes = bytes.fromhex(
                hashlib.sha256(ktn.encode()).hexdigest() * 4)
            ct_bytes  = bytes.fromhex(envelope.get("cipher_payload", ""))
            plaintext = bytes(b ^ key_bytes[i % len(key_bytes)]
                              for i, b in enumerate(ct_bytes)).decode("utf-8")
        except Exception as e:
            ktn = "WIPED";  return None, f"DECRYPT_ERROR:{e}"

        # 8. Hash verify
        if _sha256(plaintext) != envelope.get("payload_hash", ""):
            ktn = "WIPED";  return None, "PAYLOAD_HASH_MISMATCH"

        ktn = "WIPED"   # 9. Wipe K(t,n)
        ms  = (time.perf_counter() - t0) * 1000
        self._log("HSM_DECRYPT", epoch, actor, node_id, "COLD", ms, True,
                  f"peer={token.get('requesting_peer','?')} lat={ms:.2f}ms")
        arc_audit("info", "HSM", "HSM_DECRYPT_OK",
                  f"node={_short(node_id)} peer={actor} lat={ms:.2f}ms")
        return plaintext, "OK"

    # ── Cold store persistence ─────────────────────────────────────────
    def store_cold(self, epoch: int, ciphertext: str) -> None:
        with self._lock:
            store = {}
            if self._file.exists():
                try: store = json.loads(self._file.read_text(encoding="utf-8"))
                except Exception: pass
            store[str(epoch)] = {
                "epoch": epoch, "ciphertext": ciphertext,
                "stored_at": time.time(),
                "integrity": _sha256(f"{epoch}|{ciphertext}"),
            }
            self._file.write_text(json.dumps(store, indent=2), encoding="utf-8")
        self._log("COLD_STORE", epoch, "HSM", None, "COLD", 0, True, "")

    def load_cold(self, epoch: int) -> Optional[str]:
        with self._lock:
            if not self._file.exists(): return None
            try: store = json.loads(self._file.read_text(encoding="utf-8"))
            except Exception: return None
        entry = store.get(str(epoch))
        if not entry: return None
        if _sha256(f"{epoch}|{entry['ciphertext']}") != entry.get("integrity",""):
            self._log("COLD_LOAD", epoch, "HSM", None, "COLD", 0, False, "INTEGRITY_FAIL")
            return None
        self._log("COLD_LOAD", epoch, "HSM", None, "COLD", 0, True, "")
        return entry["ciphertext"]

    def delete_cold(self, epoch: int) -> bool:
        with self._lock:
            if not self._file.exists(): return False
            try: store = json.loads(self._file.read_text(encoding="utf-8"))
            except Exception: return False
            if str(epoch) not in store: return False
            del store[str(epoch)]
            self._file.write_text(json.dumps(store, indent=2), encoding="utf-8")
        self._log("COLD_DELETE", epoch, "HSM", None, "COLD", 0, True, "")
        arc_audit("warning", "HSM", "COLD_DELETE", f"epoch={epoch}")
        return True

    def list_cold(self) -> List[dict]:
        if not self._file.exists(): return []
        try: store = json.loads(self._file.read_text(encoding="utf-8"))
        except Exception: return []
        return [{"epoch": int(k), "stored_at": v["stored_at"],
                 "stored_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime(v["stored_at"]))}
                for k, v in store.items()]

    @property
    def cold_stats(self) -> dict:
        entries = self.list_cold()
        size    = self._file.stat().st_size if self._file.exists() else 0
        return {"cold_entries": len(entries), "cold_bytes": size}

    def session_log(self, n: int = 50) -> List[dict]:
        return [op.to_dict() for op in self._session_log[-n:]]

    def session_stats(self) -> dict:
        ops = self._session_log
        if not ops:
            return {"total_ops": 0, "by_operation": {}, "avg_latency_ms": 0,
                    "success_rate": 1.0, "last_op": None}
        by_op: Dict[str, int] = {}
        total_lat = 0.0
        failures  = 0
        for op in ops:
            by_op[op.operation] = by_op.get(op.operation, 0) + 1
            total_lat += op.latency_ms
            if not op.success: failures += 1
        return {
            "total_ops":      len(ops),
            "by_operation":   by_op,
            "avg_latency_ms": round(total_lat / len(ops), 3),
            "success_rate":   round((len(ops) - failures) / len(ops), 4),
            "last_op":        ops[-1].to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════
#  EPOCH SECRET ARCHIVE  — three-tier
# ═══════════════════════════════════════════════════════════════════════
class EpochSecretArchive:
    def __init__(self, policy: ArchivePolicy, hsm: SimulatedHSM):
        self._policy    = policy
        self._hsm       = hsm
        self._hot:      Dict[int, str] = {}
        self._warm_lock = threading.Lock()

    def update_policy(self, p: ArchivePolicy) -> None:
        old = self._policy;  self._policy = p
        arc_audit("info","EPOCH_ARCHIVE","POLICY_UPDATE",
                  f"hot={p.hot_epochs} warm={p.warm_epochs}")
        if p.max_epoch_keys < old.max_epoch_keys or \
           p.max_epoch_age_days != old.max_epoch_age_days:
            self._enforce_limits()

    def archive_epoch(self, epoch: int, secret: str) -> str:
        current = max(self._hot.keys(), default=epoch)
        age     = current - epoch
        if age < self._policy.hot_epochs:
            self._hot[epoch] = secret;  tier = "HOT"
        elif age < self._policy.hot_epochs + self._policy.warm_epochs:
            self._store_warm(epoch, secret);  tier = "WARM"
        else:
            ct = self._hsm.seal(epoch, secret)
            self._hsm.store_cold(epoch, ct);  tier = "COLD"
        arc_audit("info","EPOCH_ARCHIVE","ARCHIVE",f"epoch={epoch} tier={tier}")
        if self._policy.enforce_on_rotate: self._enforce_limits()
        return tier

    def retrieve_epoch(self, epoch: int) -> Optional[str]:
        if self._policy.enforce_on_read: self._enforce_limits()
        if epoch in self._hot:
            arc_audit("info","EPOCH_ARCHIVE","RETRIEVE_HOT",f"epoch={epoch}")
            return self._hot[epoch]
        warm = self._load_warm()
        if str(epoch) in warm:
            ct = warm[str(epoch)]["ciphertext"]
            arc_audit("info","EPOCH_ARCHIVE","RETRIEVE_WARM",f"epoch={epoch}")
            return _xor_decrypt(ct, f"{self._hsm._master_key}|warm|{epoch}")
        ct = self._hsm.load_cold(epoch)
        if ct:
            arc_audit("info","EPOCH_ARCHIVE","RETRIEVE_COLD",f"epoch={epoch}")
            return self._hsm.unseal(epoch, ct)
        arc_audit("warning","EPOCH_ARCHIVE","RETRIEVE_MISS",f"epoch={epoch}")
        return None

    def retrieve_epochs_batch(self, epochs: List[int]) -> Dict[int, Optional[str]]:
        result: Dict[int, Optional[str]] = {}
        need_warm, need_cold = [], []
        for ep in epochs:
            if ep in self._hot:  result[ep] = self._hot[ep]
            else: need_warm.append(ep)
        if need_warm:
            warm = self._load_warm()
            for ep in need_warm:
                if str(ep) in warm:
                    ct = warm[str(ep)]["ciphertext"]
                    result[ep] = _xor_decrypt(ct, f"{self._hsm._master_key}|warm|{ep}")
                else:
                    need_cold.append(ep)
        for ep in need_cold:
            ct = self._hsm.load_cold(ep)
            result[ep] = self._hsm.unseal(ep, ct) if ct else None
        return result

    def _promote_tiers(self, current: int) -> None:
        warm = self._load_warm();  changed = False
        for ep, secret in list(self._hot.items()):
            age = current - ep
            if age >= self._policy.hot_epochs + self._policy.warm_epochs:
                ct = self._hsm.seal(ep, secret)
                self._hsm.store_cold(ep, ct);  del self._hot[ep]
            elif age >= self._policy.hot_epochs:
                self._store_warm_dict(warm, ep, secret)
                del self._hot[ep];  changed = True
        for ep_str in list(warm.keys()):
            ep  = int(ep_str);  age = current - ep
            if age >= self._policy.hot_epochs + self._policy.warm_epochs:
                sec = _xor_decrypt(warm[ep_str]["ciphertext"],
                                   f"{self._hsm._master_key}|warm|{ep}")
                ct  = self._hsm.seal(ep, sec)
                self._hsm.store_cold(ep, ct)
                del warm[ep_str];  changed = True
        if changed: self._save_warm(warm)

    def _enforce_limits(self) -> None:
        now, evicted = time.time(), []
        if self._policy.max_epoch_age_days > 0:
            max_age = self._policy.max_epoch_age_days * 86400
            warm = self._load_warm();  changed = False
            for ep_str in list(warm.keys()):
                if now - warm[ep_str].get("archived_at", now) > max_age:
                    del warm[ep_str];  evicted.append(int(ep_str));  changed = True
            if changed: self._save_warm(warm)
        if self._policy.max_epoch_keys > 0:
            cold  = [e["epoch"] for e in self._hsm.list_cold()]
            total = len(self._hot) + len(self._load_warm()) + len(cold)
            for ep in sorted(cold)[:max(0, total - self._policy.max_epoch_keys)]:
                self._hsm.delete_cold(ep);  evicted.append(ep)
        if evicted:
            arc_audit("warning","EPOCH_ARCHIVE","EVICT",f"epochs={evicted}")

    def _warm_file(self) -> Path: return BASE_DIR / "pcd_warm_epochs.json"
    def _load_warm(self) -> dict:
        with self._warm_lock:
            f = self._warm_file()
            if f.exists():
                try: return json.loads(f.read_text(encoding="utf-8"))
                except Exception: pass
            return {}
    def _save_warm(self, data: dict) -> None:
        with self._warm_lock:
            self._warm_file().write_text(json.dumps(data, indent=2), encoding="utf-8")
    def _store_warm(self, epoch: int, secret: str) -> None:
        warm = self._load_warm()
        self._store_warm_dict(warm, epoch, secret)
        self._save_warm(warm)
    def _store_warm_dict(self, warm: dict, epoch: int, secret: str) -> None:
        ct = _xor_encrypt(secret, f"{self._hsm._master_key}|warm|{epoch}")
        warm[str(epoch)] = {"epoch": epoch, "ciphertext": ct,
                            "archived_at": time.time(),
                            "integrity": _sha256(f"{epoch}|{ct}")}

    @property
    def stats(self) -> dict:
        warm  = self._load_warm()
        cold  = self._hsm.list_cold()
        warm_f = self._warm_file()
        return {
            "hot_count":   len(self._hot),
            "warm_count":  len(warm),
            "cold_count":  len(cold),
            "total_keys":  len(self._hot) + len(warm) + len(cold),
            "hot_epochs":  sorted(self._hot.keys()),
            "warm_epochs": sorted(int(k) for k in warm.keys()),
            "cold_epochs": sorted(e["epoch"] for e in cold),
            "warm_bytes":  warm_f.stat().st_size if warm_f.exists() else 0,
            "cold_bytes":  self._hsm.cold_stats["cold_bytes"],
        }

    def full_listing(self) -> List[dict]:
        rows = []
        warm = self._load_warm()
        cold = {e["epoch"]: e for e in self._hsm.list_cold()}
        for ep, secret in self._hot.items():
            rows.append({"epoch": ep, "tier": "HOT",
                         "short": _short(secret), "archived_at": None})
        for ep_str, entry in warm.items():
            rows.append({"epoch": int(ep_str), "tier": "WARM",
                         "short": _short(entry["ciphertext"]),
                         "archived_at": entry.get("archived_at")})
        for ep, meta in cold.items():
            rows.append({"epoch": ep, "tier": "COLD",
                         "short": "sealed", "archived_at": meta.get("stored_at")})
        rows.sort(key=lambda r: r["epoch"], reverse=True)
        return rows


# ═══════════════════════════════════════════════════════════════════════
#  DATA RECORD ARCHIVE
# ═══════════════════════════════════════════════════════════════════════
_arc_data_lock = threading.Lock()

def _load_arc_data() -> dict:
    if ARCHIVE_DATA_FILE.exists():
        try: return json.loads(ARCHIVE_DATA_FILE.read_text(encoding="utf-8"))
        except Exception: pass
    return {"nodes": {}, "meta": {"created": time.time(), "ops": 0, "total_archived": 0}}

def _save_arc_data(store: dict) -> None:
    store["meta"]["last_updated"] = time.time()
    store["meta"]["ops"] = store["meta"].get("ops", 0) + 1
    ARCHIVE_DATA_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


class DataRecordArchive:
    def __init__(self, policy: ArchivePolicy):
        self._policy = policy

    def update_policy(self, p: ArchivePolicy) -> None:
        self._policy = p;  self._enforce_limits()

    def archive_node(self, env_dict: dict, reason: str = "POLICY_EVICT") -> bool:
        with _arc_data_lock:
            s = _load_arc_data();  nid = env_dict["node_id"]
            if nid in s["nodes"]: return False
            s["nodes"][nid] = {**env_dict, "archived_at": time.time(),
                               "archive_reason": reason}
            s["meta"]["total_archived"] = s["meta"].get("total_archived", 0) + 1
            _save_arc_data(s)
        arc_audit("info","DATA_ARCHIVE","ARCHIVE_NODE",
                  f"node={_short(nid)} reason={reason}")
        return True

    def get_archived(self, node_id: str) -> Optional[dict]:
        with _arc_data_lock: s = _load_arc_data()
        return s["nodes"].get(node_id)

    def restore_node(self, node_id: str) -> Optional[dict]:
        with _arc_data_lock:
            s = _load_arc_data()
            if node_id not in s["nodes"]: return None
            env = s["nodes"].pop(node_id);  _save_arc_data(s)
        arc_audit("info","DATA_ARCHIVE","RESTORE_NODE",f"node={_short(node_id)}")
        return env

    def all_archived(self) -> List[dict]:
        with _arc_data_lock: s = _load_arc_data()
        return list(s["nodes"].values())

    def _enforce_limits(self) -> None:
        evicted = []
        with _arc_data_lock:
            s = _load_arc_data();  nodes = list(s["nodes"].values());  changed = False
            if self._policy.max_record_age_days > 0:
                max_age = self._policy.max_record_age_days * 86400;  now = time.time()
                for n in nodes:
                    if now - n.get("archived_at", now) > max_age:
                        del s["nodes"][n["node_id"]]
                        evicted.append(n["node_id"]);  changed = True
                nodes = list(s["nodes"].values())
            if self._policy.max_record_count > 0 and \
               len(nodes) > self._policy.max_record_count:
                for n in sorted(nodes, key=lambda n: n.get("chain_counter",0))\
                         [:len(nodes)-self._policy.max_record_count]:
                    del s["nodes"][n["node_id"]]
                    evicted.append(n["node_id"]);  changed = True
            if changed: _save_arc_data(s)
        if evicted:
            arc_audit("warning","DATA_ARCHIVE","EVICT",f"n={len(evicted)}")

    def stats(self) -> dict:
        with _arc_data_lock: s = _load_arc_data()
        nodes = list(s["nodes"].values())
        size  = ARCHIVE_DATA_FILE.stat().st_size if ARCHIVE_DATA_FILE.exists() else 0
        tiers: Dict[str,int] = {}
        for n in nodes:
            c = n.get("classification","UNKNOWN");  tiers[c] = tiers.get(c,0)+1
        return {"total_archived": len(nodes), "total_ops": s["meta"].get("ops",0),
                "cumulative_archived": s["meta"].get("total_archived",0),
                "file_bytes": size, "by_classification": tiers,
                "oldest_counter": min((n.get("chain_counter",0) for n in nodes), default=None),
                "newest_counter": max((n.get("chain_counter",0) for n in nodes), default=None)}

    def listing(self, limit: int = 100, offset: int = 0) -> List[dict]:
        with _arc_data_lock: s = _load_arc_data()
        nodes = sorted(s["nodes"].values(), key=lambda n: n.get("chain_counter",0), reverse=True)
        return [{"node_id": n["node_id"], "label": n.get("label",""),
                 "actor": n.get("actor",""), "epoch": n.get("epoch",0),
                 "classification": n.get("classification",""),
                 "ai_tier": n.get("ai_tier",""), "ai_score": n.get("ai_score",0),
                 "status": n.get("status",""), "chain_counter": n.get("chain_counter",0),
                 "archived_at": n.get("archived_at"),
                 "archive_reason": n.get("archive_reason",""),
                 "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                           time.gmtime(n.get("timestamp",0)))}
                for n in nodes[offset:offset+limit]]


# ═══════════════════════════════════════════════════════════════════════
#  ARCHIVE MANAGER — top-level facade
# ═══════════════════════════════════════════════════════════════════════
class ArchiveManager:
    def __init__(self):
        self._policy = ArchivePolicy()
        self._hsm    = SimulatedHSM(master_key=_sha256(
            f"HSM-ARCHIVE-KEY-{os.urandom(16).hex()}"))
        self._epoch_archive = EpochSecretArchive(self._policy, self._hsm)
        self._data_archive  = DataRecordArchive(self._policy)
        self._lock          = threading.Lock()

    def init_with_master(self, master: str) -> None:
        hsm_key = _hmac256(master, "HSM-ARCHIVE-MASTER-KEY")
        self._hsm = SimulatedHSM(master_key=hsm_key)
        self._epoch_archive = EpochSecretArchive(self._policy, self._hsm)
        arc_audit("info","ARCHIVE_MGR","INIT",f"hsm_key={_short(hsm_key)}")

    # ── Policy ─────────────────────────────────────────────────────────
    def get_policy(self) -> ArchivePolicy:  return self._policy
    def update_policy(self, p: ArchivePolicy) -> None:
        with self._lock:
            self._policy = p
            self._epoch_archive.update_policy(p)
            self._data_archive.update_policy(p)
        arc_audit("info","ARCHIVE_MGR","POLICY_UPDATED",
                  f"hot={p.hot_epochs} warm={p.warm_epochs} ttl={p.resource_token_ttl}s")

    # ── Epoch secrets ──────────────────────────────────────────────────
    def archive_epoch(self, epoch: int, secret: str) -> str:
        tier = self._epoch_archive.archive_epoch(epoch, secret)
        arc_audit("info","ARCHIVE_MGR","EPOCH_ARCHIVED",f"epoch={epoch} tier={tier}")
        return tier

    def get_epoch_secret(self, epoch: int) -> Optional[str]:
        s = self._epoch_archive.retrieve_epoch(epoch)
        if s is None:
            arc_audit("warning","ARCHIVE_MGR","EPOCH_MISS",f"epoch={epoch}")
        return s

    def get_epoch_secrets_batch(self, epochs: List[int]) -> Dict[int, Optional[str]]:
        return self._epoch_archive.retrieve_epochs_batch(epochs)

    def promote_tiers(self, current_epoch: int) -> None:
        self._epoch_archive._promote_tiers(current_epoch)

    # ── HSM operations ─────────────────────────────────────────────────
    def hsm_derive_ktn(self, epoch: int, feature_fp: str, counter: int,
                       window_pos: float, actor: str = "SYSTEM") -> Optional[str]:
        """Derive K(t,n) inside HSM — only works for COLD-stored epoch secrets."""
        return self._hsm.derive_ktn(epoch, feature_fp, counter, window_pos, actor)

    def hsm_issue_token(self, node_id: str, epoch: int, feature_fp: str,
                        requesting_peer: str, actor: str = "SYSTEM") -> dict:
        return self._hsm.issue_resource_token(
            node_id, epoch, feature_fp, requesting_peer,
            self._policy.resource_token_ttl, actor)

    def hsm_verify_token(self, token: dict,
                          actor: str = "SYSTEM") -> Tuple[bool, str]:
        return self._hsm.verify_resource_token(token, actor)

    def hsm_decrypt_with_token(self, token: dict, envelope: dict,
                                actor: str = "SYSTEM") -> Tuple[Optional[str], str]:
        return self._hsm.hsm_decrypt_with_token(token, envelope, actor)

    def hsm_session_log(self, n: int = 50) -> List[dict]:
        return self._hsm.session_log(n)

    def hsm_session_stats(self) -> dict:
        return self._hsm.session_stats()

    # ── Data records ───────────────────────────────────────────────────
    def archive_record(self, env_dict: dict, reason: str = "POLICY_EVICT") -> bool:
        return self._data_archive.archive_node(env_dict, reason)

    def restore_record(self, node_id: str) -> Optional[dict]:
        return self._data_archive.restore_node(node_id)

    def get_archived_record(self, node_id: str) -> Optional[dict]:
        return self._data_archive.get_archived(node_id)

    def enforce_data_limits(self) -> None:
        self._data_archive._enforce_limits()

    def all_archived_envelopes(self) -> List[dict]:
        return self._data_archive.all_archived()

    # ── Stats ──────────────────────────────────────────────────────────
    def full_stats(self) -> dict:
        warm_f = self._epoch_archive._warm_file()
        return {
            "epoch_archive": self._epoch_archive.stats,
            "data_archive":  self._data_archive.stats(),
            "hsm_session":   self._hsm.session_stats(),
            "policy":        self._policy.to_dict(),
            "files": {
                "warm_epochs":  str(warm_f),
                "warm_bytes":   warm_f.stat().st_size if warm_f.exists() else 0,
                "cold_hsm":     str(ARCHIVE_HSM_FILE),
                "cold_bytes":   ARCHIVE_HSM_FILE.stat().st_size
                                if ARCHIVE_HSM_FILE.exists() else 0,
                "data_archive": str(ARCHIVE_DATA_FILE),
                "data_bytes":   ARCHIVE_DATA_FILE.stat().st_size
                                if ARCHIVE_DATA_FILE.exists() else 0,
                "hsm_session":  str(HSM_SESSION_FILE),
            }
        }

    def epoch_listing(self) -> List[dict]:
        return self._epoch_archive.full_listing()

    def data_listing(self, limit: int = 100, offset: int = 0) -> List[dict]:
        return self._data_archive.listing(limit, offset)


# ═══════════════════════════════════════════════════════════════════════
#  SINGLETON
# ═══════════════════════════════════════════════════════════════════════
ARCHIVE = ArchiveManager()
