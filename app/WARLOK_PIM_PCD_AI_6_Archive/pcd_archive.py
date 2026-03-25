#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PCD ARCHIVE MODULE  ·  pcd_archive.py                                   ║
║                                                                           ║
║  Provides two co-operating archives, both admin-configurable:            ║
║                                                                           ║
║  1. EpochSecretArchive — HSM-premise store of past epoch secrets         ║
║     Bounded by: max_epochs | max_age_days | max_size_bytes               ║
║     Three tiers: HOT (RAM) · WARM (JSON) · COLD (HSM-envelope)          ║
║     Secrets in HOT tier: plaintext in secured peer memory                ║
║     Secrets in WARM/COLD tier: encrypted under archive_master_key        ║
║     Archive master key: simulated HSM (replace with PKCS#11 in prod)    ║
║                                                                           ║
║  2. DataRecordArchive — append-only JSON database of archived Envelopes  ║
║     Bounded by: max_records | max_age_days | max_size_bytes              ║
║     Separate file from pcd_data.json — older / evicted nodes land here  ║
║     Full DAG chain preserved — prev_hash links intact                    ║
║     Every archive/evict/restore event is itself a sealed audit node      ║
║                                                                           ║
║  Both archives expose a uniform admin API mounted under /api/archive/    ║
║  and integrate into the existing WARL0K PCD UI in a new Archive tab.    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import dataclasses
import hashlib
import hmac as _hmac
import json
import math
import os
import time
import threading
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════════
#  PATHS  (siblings to pcd_data.json)
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR          = Path(__file__).parent
ARCHIVE_DATA_FILE = BASE_DIR / "pcd_archive.json"
ARCHIVE_HSM_FILE  = BASE_DIR / "pcd_hsm_archive.json"   # simulated HSM store
ARCHIVE_LOG_FILE  = BASE_DIR / "pcd_archive.log"

# ═══════════════════════════════════════════════════════════════════════
#  ARCHIVE LOGGER
# ═══════════════════════════════════════════════════════════════════════
_arc_handler = logging.FileHandler(ARCHIVE_LOG_FILE, encoding="utf-8")
_arc_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
))
arc_log = logging.getLogger("pcd.archive")
arc_log.setLevel(logging.DEBUG)
arc_log.addHandler(_arc_handler)
arc_log.propagate = False

def arc_audit(level: str, actor: str, op: str, detail: str = "") -> None:
    msg = f"[{actor:14s}] {op:22s} {detail}"
    getattr(arc_log, level.lower(), arc_log.info)(msg)


# ═══════════════════════════════════════════════════════════════════════
#  CRYPTO HELPERS  (inline — no import from pcd_system to allow standalone)
# ═══════════════════════════════════════════════════════════════════════
def _sha256(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data).hexdigest()

def _hmac256(key: bytes | str, msg: bytes | str) -> str:
    if isinstance(key, str):
        key = key.encode()
    if isinstance(msg, str):
        msg = msg.encode()
    return _hmac.new(key, msg, hashlib.sha256).hexdigest()

def _xor_encrypt(plaintext: str, key_hex: str) -> str:
    """Envelope-encrypt a secret under archive_master_key (XOR — replace with AES-GCM in prod)."""
    key_bytes = bytes.fromhex(hashlib.sha256(key_hex.encode()).hexdigest() * 4)
    ct = bytes(b ^ key_bytes[i % len(key_bytes)]
               for i, b in enumerate(plaintext.encode("utf-8")))
    return ct.hex()

def _xor_decrypt(ciphertext_hex: str, key_hex: str) -> str:
    key_bytes = bytes.fromhex(hashlib.sha256(key_hex.encode()).hexdigest() * 4)
    pt = bytes(b ^ key_bytes[i % len(key_bytes)]
               for i, b in enumerate(bytes.fromhex(ciphertext_hex)))
    return pt.decode("utf-8")

def _short(h: str) -> str:
    return f"{h[:8]}…{h[-6:]}" if h and len(h) > 16 else (h or "—")


# ═══════════════════════════════════════════════════════════════════════
#  ADMIN POLICY  — all limits configurable by admin at runtime
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class ArchivePolicy:
    """
    Admin-configurable limits for both archives.
    All fields can be updated via PUT /api/archive/policy without restart.

    Epoch secret archive bounds (at least one must be set):
        max_epoch_keys    — max number of past epoch secrets to keep in archive
        max_epoch_age_days— max age in days before an epoch secret is evicted from archive
        max_epoch_bytes   — max total size of the epoch secret archive file in bytes

    Data record archive bounds (at least one must be set):
        max_record_count  — max number of Envelopes in the data archive
        max_record_age_days— max age in days before a record is evicted from data archive
        max_record_bytes  — max total size of the data archive file in bytes

    Tier configuration:
        hot_epochs        — how many recent epoch secrets stay in peer RAM (no HSM call)
        warm_epochs       — how many further epochs stay in warm JSON store
        cold beyond that  → stored in HSM-envelope (simulated here)

    Enforcement:
        enforce_on_rotate — auto-evict when epoch rotates (recommended: True)
        enforce_on_read   — auto-evict on every archive read (adds latency; more precise)
    """
    # epoch secret archive
    max_epoch_keys:      int   = 50       # 0 = unlimited
    max_epoch_age_days:  float = 0.0      # 0 = unlimited
    max_epoch_bytes:     int   = 0        # 0 = unlimited

    # data record archive
    max_record_count:    int   = 10_000   # 0 = unlimited
    max_record_age_days: float = 365.0    # 0 = unlimited
    max_record_bytes:    int   = 0        # 0 = unlimited

    # tier boundaries
    hot_epochs:          int   = 5        # kept in peer RAM — zero-latency
    warm_epochs:         int   = 20       # kept in JSON warm store — fast JSON lookup

    # enforcement triggers
    enforce_on_rotate:   bool  = True
    enforce_on_read:     bool  = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchivePolicy":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATED HSM  — replace inner methods with PKCS#11 in production
# ═══════════════════════════════════════════════════════════════════════
class SimulatedHSM:
    """
    HSM boundary simulation.  In production replace seal()/unseal() with:
        session.sign(key_handle, Mechanism(CKM_SHA256_HMAC), data)
        session.decrypt(key_handle, Mechanism(CKM_AES_GCM), ciphertext)
    The archive_master_key never leaves this class in a production HSM deployment.
    """
    def __init__(self, master_key: str):
        self._master_key = master_key   # NEVER export — in prod stays in HSM silicon
        self._file       = ARCHIVE_HSM_FILE
        self._lock       = threading.Lock()

    def seal(self, epoch: int, secret: str) -> str:
        """Encrypt epoch_secret under archive_master_key. Returns ciphertext hex."""
        ct = _xor_encrypt(secret, f"{self._master_key}|epoch|{epoch}")
        arc_audit("info", "HSM", "SEAL_EPOCH", f"epoch={epoch} ct={_short(ct)}")
        return ct

    def unseal(self, epoch: int, ciphertext: str) -> str:
        """Decrypt epoch_secret from HSM-held ciphertext."""
        pt = _xor_decrypt(ciphertext, f"{self._master_key}|epoch|{epoch}")
        arc_audit("info", "HSM", "UNSEAL_EPOCH", f"epoch={epoch}")
        return pt

    def store_cold(self, epoch: int, ciphertext: str) -> None:
        """Persist sealed epoch to the simulated HSM cold store (JSON file)."""
        with self._lock:
            store = {}
            if self._file.exists():
                try:
                    store = json.loads(self._file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            store[str(epoch)] = {
                "epoch":     epoch,
                "ciphertext":ciphertext,
                "stored_at": time.time(),
                "integrity": _sha256(f"{epoch}|{ciphertext}"),
            }
            self._file.write_text(json.dumps(store, indent=2), encoding="utf-8")
        arc_audit("info", "HSM", "COLD_STORE", f"epoch={epoch}")

    def load_cold(self, epoch: int) -> Optional[str]:
        """Retrieve and verify sealed epoch from cold store. Returns ciphertext or None."""
        with self._lock:
            if not self._file.exists():
                return None
            try:
                store = json.loads(self._file.read_text(encoding="utf-8"))
            except Exception:
                return None
        entry = store.get(str(epoch))
        if not entry:
            return None
        # Integrity verification
        expected = _sha256(f"{epoch}|{entry['ciphertext']}")
        if expected != entry.get("integrity"):
            arc_audit("error", "HSM", "INTEGRITY_FAIL", f"epoch={epoch}")
            return None
        arc_audit("info", "HSM", "COLD_LOAD", f"epoch={epoch}")
        return entry["ciphertext"]

    def delete_cold(self, epoch: int) -> bool:
        with self._lock:
            if not self._file.exists():
                return False
            try:
                store = json.loads(self._file.read_text(encoding="utf-8"))
            except Exception:
                return False
            if str(epoch) not in store:
                return False
            del store[str(epoch)]
            self._file.write_text(json.dumps(store, indent=2), encoding="utf-8")
        arc_audit("warning", "HSM", "COLD_DELETE", f"epoch={epoch}")
        return True

    def list_cold(self) -> List[dict]:
        """Return all cold-stored entries as metadata (no secrets)."""
        if not self._file.exists():
            return []
        try:
            store = json.loads(self._file.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [
            {"epoch": int(k), "stored_at": v["stored_at"],
             "stored_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                            time.gmtime(v["stored_at"]))}
            for k, v in store.items()
        ]

    @property
    def cold_stats(self) -> dict:
        entries = self.list_cold()
        size = self._file.stat().st_size if self._file.exists() else 0
        return {"cold_entries": len(entries), "cold_bytes": size}


# ═══════════════════════════════════════════════════════════════════════
#  EPOCH SECRET ARCHIVE  — three-tier (HOT | WARM | COLD)
# ═══════════════════════════════════════════════════════════════════════
class EpochSecretArchive:
    """
    Manages epoch secrets across three storage tiers.

    HOT   — plaintext in Python dict (peer RAM).  Zero HSM calls.
             Bounded to policy.hot_epochs most recent epochs.

    WARM  — encrypted under archive_master_key, stored in JSON file.
             Bounded to policy.warm_epochs epochs beyond the HOT boundary.
             Retrieve: one file read + one in-memory decrypt.

    COLD  — encrypted under archive_master_key, stored in simulated HSM.
             All epochs beyond warm boundary.
             Retrieve: HSM call + decrypt.

    Eviction is triggered on every epoch rotation (enforce_on_rotate=True)
    and optionally on every archive read (enforce_on_read=True).
    """

    def __init__(self, policy: ArchivePolicy, hsm: SimulatedHSM):
        self._policy  = policy
        self._hsm     = hsm
        self._hot:    Dict[int, str] = {}   # epoch → plaintext secret
        self._warm_lock = threading.Lock()

    # ── policy update ──────────────────────────────────────────────────
    def update_policy(self, new_policy: ArchivePolicy) -> None:
        old = self._policy
        self._policy = new_policy
        arc_audit("info", "EPOCH_ARCHIVE", "POLICY_UPDATE",
                  f"max_keys={new_policy.max_epoch_keys} "
                  f"hot={new_policy.hot_epochs} warm={new_policy.warm_epochs} "
                  f"max_age_days={new_policy.max_epoch_age_days}")
        if new_policy.max_epoch_keys < old.max_epoch_keys or \
           new_policy.max_epoch_age_days != old.max_epoch_age_days:
            self._enforce_limits()

    # ── archive a newly-rotated epoch secret ──────────────────────────
    def archive_epoch(self, epoch: int, secret: str) -> str:
        """
        Called immediately before epoch_secret(t) is deleted from peer memory.
        Determines correct tier and persists accordingly.
        Returns tier name: HOT | WARM | COLD.
        """
        current_epoch = max(self._hot.keys(), default=epoch)
        age_in_hot    = current_epoch - epoch

        if age_in_hot < self._policy.hot_epochs:
            self._hot[epoch] = secret
            tier = "HOT"
        elif age_in_hot < self._policy.hot_epochs + self._policy.warm_epochs:
            self._store_warm(epoch, secret)
            tier = "WARM"
        else:
            ct = self._hsm.seal(epoch, secret)
            self._hsm.store_cold(epoch, ct)
            tier = "COLD"

        arc_audit("info", "EPOCH_ARCHIVE", "ARCHIVE",
                  f"epoch={epoch} tier={tier} short={_short(secret)}")

        if self._policy.enforce_on_rotate:
            self._enforce_limits()

        return tier

    # ── retrieve an epoch secret ───────────────────────────────────────
    def retrieve_epoch(self, epoch: int) -> Optional[str]:
        """
        Retrieve epoch_secret for given epoch from any tier.
        Returns None if evicted beyond policy bounds.
        """
        if self._policy.enforce_on_read:
            self._enforce_limits()

        # HOT tier
        if epoch in self._hot:
            arc_audit("info", "EPOCH_ARCHIVE", "RETRIEVE_HOT", f"epoch={epoch}")
            return self._hot[epoch]

        # WARM tier
        warm = self._load_warm()
        if str(epoch) in warm:
            ct = warm[str(epoch)]["ciphertext"]
            secret = _xor_decrypt(ct, f"{self._hsm._master_key}|warm|{epoch}")
            arc_audit("info", "EPOCH_ARCHIVE", "RETRIEVE_WARM", f"epoch={epoch}")
            return secret

        # COLD tier (HSM)
        ct = self._hsm.load_cold(epoch)
        if ct:
            secret = self._hsm.unseal(epoch, ct)
            arc_audit("info", "EPOCH_ARCHIVE", "RETRIEVE_COLD", f"epoch={epoch}")
            return secret

        arc_audit("warning", "EPOCH_ARCHIVE", "RETRIEVE_MISS",
                  f"epoch={epoch} — evicted or never archived")
        return None

    # ── batch retrieve for bulk read ───────────────────────────────────
    def retrieve_epochs_batch(self, epochs: List[int]) -> Dict[int, Optional[str]]:
        """
        Retrieve multiple epoch secrets in one call — minimises HSM round trips.
        Returns {epoch: secret_or_None}.
        """
        result: Dict[int, Optional[str]] = {}
        need_warm = []
        need_cold = []

        # HOT pass — zero-cost
        for ep in epochs:
            if ep in self._hot:
                result[ep] = self._hot[ep]
            else:
                need_warm.append(ep)

        # WARM pass — one file read
        if need_warm:
            warm = self._load_warm()
            for ep in need_warm:
                if str(ep) in warm:
                    ct = warm[str(ep)]["ciphertext"]
                    result[ep] = _xor_decrypt(ct, f"{self._hsm._master_key}|warm|{ep}")
                else:
                    need_cold.append(ep)

        # COLD pass — one HSM call per epoch (batched)
        for ep in need_cold:
            ct = self._hsm.load_cold(ep)
            result[ep] = self._hsm.unseal(ep, ct) if ct else None

        return result

    # ── promote tiers (run when HOT boundary advances) ────────────────
    def _promote_tiers(self, current_epoch: int) -> None:
        """
        When the epoch advances, secrets that are now too old for HOT
        should move to WARM, and secrets too old for WARM should move to COLD.
        """
        to_demote_to_warm = []
        to_demote_to_cold = []

        for ep, secret in list(self._hot.items()):
            age = current_epoch - ep
            if age >= self._policy.hot_epochs + self._policy.warm_epochs:
                to_demote_to_cold.append((ep, secret))
                del self._hot[ep]
            elif age >= self._policy.hot_epochs:
                to_demote_to_warm.append((ep, secret))
                del self._hot[ep]

        warm = self._load_warm()
        for ep, secret in to_demote_to_warm:
            ct = _xor_encrypt(secret, f"{self._hsm._master_key}|warm|{ep}")
            warm[str(ep)] = {
                "epoch": ep, "ciphertext": ct, "archived_at": time.time(),
                "integrity": _sha256(f"{ep}|{ct}")
            }
            arc_audit("info", "EPOCH_ARCHIVE", "DEMOTE_WARM", f"epoch={ep}")

        for ep, secret in to_demote_to_cold:
            ct = self._hsm.seal(ep, secret)
            self._hsm.store_cold(ep, ct)
            arc_audit("info", "EPOCH_ARCHIVE", "DEMOTE_COLD", f"epoch={ep}")

        # Demote old WARM entries to COLD
        for ep_str, entry in list(warm.items()):
            ep = int(ep_str)
            age = current_epoch - ep
            if age >= self._policy.hot_epochs + self._policy.warm_epochs:
                ct = self._hsm.seal(ep, _xor_decrypt(
                    entry["ciphertext"], f"{self._hsm._master_key}|warm|{ep}"))
                self._hsm.store_cold(ep, ct)
                del warm[ep_str]
                arc_audit("info", "EPOCH_ARCHIVE", "WARM_TO_COLD", f"epoch={ep}")

        if to_demote_to_warm or [e for e,_ in to_demote_to_cold]:
            self._save_warm(warm)

    # ── evict beyond policy bounds ─────────────────────────────────────
    def _enforce_limits(self) -> None:
        now      = time.time()
        evicted  = []

        # 1. Age-based eviction
        if self._policy.max_epoch_age_days > 0:
            max_age_sec = self._policy.max_epoch_age_days * 86400
            warm = self._load_warm()
            changed = False
            for ep_str in list(warm.keys()):
                age = now - warm[ep_str].get("archived_at", now)
                if age > max_age_sec:
                    del warm[ep_str]
                    evicted.append(int(ep_str))
                    changed = True
            if changed:
                self._save_warm(warm)
            for ep in list(self._hot.keys()):
                if now - 0 > max_age_sec:   # hot secrets have no stored ts — use epoch age
                    pass  # age for HOT is managed by tier demotion, not wall-clock

        # 2. Count-based eviction
        if self._policy.max_epoch_keys > 0:
            all_warm = list(self._load_warm().keys())
            cold     = [e["epoch"] for e in self._hsm.list_cold()]
            total    = len(self._hot) + len(all_warm) + len(cold)
            excess   = total - self._policy.max_epoch_keys
            if excess > 0:
                # Evict oldest cold entries first
                cold_sorted = sorted(cold)
                for ep in cold_sorted[:excess]:
                    self._hsm.delete_cold(ep)
                    evicted.append(ep)
                    excess -= 1
                    if excess <= 0:
                        break

        # 3. Size-based eviction
        if self._policy.max_epoch_bytes > 0:
            warm_file_size = ARCHIVE_HSM_FILE.stat().st_size \
                             if ARCHIVE_HSM_FILE.exists() else 0
            if warm_file_size > self._policy.max_epoch_bytes:
                cold = sorted(e["epoch"] for e in self._hsm.list_cold())
                for ep in cold:
                    self._hsm.delete_cold(ep)
                    evicted.append(ep)
                    warm_file_size = ARCHIVE_HSM_FILE.stat().st_size \
                                     if ARCHIVE_HSM_FILE.exists() else 0
                    if warm_file_size <= self._policy.max_epoch_bytes:
                        break

        if evicted:
            arc_audit("warning", "EPOCH_ARCHIVE", "EVICT",
                      f"evicted_epochs={evicted}")

    # ── warm tier I/O ──────────────────────────────────────────────────
    def _warm_file(self) -> Path:
        return BASE_DIR / "pcd_warm_epochs.json"

    def _load_warm(self) -> dict:
        with self._warm_lock:
            f = self._warm_file()
            if f.exists():
                try:
                    return json.loads(f.read_text(encoding="utf-8"))
                except Exception:
                    pass
            return {}

    def _save_warm(self, data: dict) -> None:
        with self._warm_lock:
            self._warm_file().write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── stats ──────────────────────────────────────────────────────────
    @property
    def stats(self) -> dict:
        warm   = self._load_warm()
        cold   = self._hsm.list_cold()
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
        """All archived epochs with tier and metadata — for admin UI."""
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
                         "short": "sealed",
                         "archived_at": meta.get("stored_at")})
        rows.sort(key=lambda r: r["epoch"], reverse=True)
        return rows


# ═══════════════════════════════════════════════════════════════════════
#  DATA RECORD ARCHIVE  — older Envelopes evicted from hot store
# ═══════════════════════════════════════════════════════════════════════
_arc_data_lock = threading.Lock()

def _load_arc_data() -> dict:
    if ARCHIVE_DATA_FILE.exists():
        try:
            return json.loads(ARCHIVE_DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"nodes": {}, "meta": {"created": time.time(), "ops": 0,
                                   "total_archived": 0}}

def _save_arc_data(store: dict) -> None:
    store["meta"]["last_updated"] = time.time()
    store["meta"]["ops"] = store["meta"].get("ops", 0) + 1
    ARCHIVE_DATA_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")


class DataRecordArchive:
    """
    Secondary DAG-preserving store for Envelopes evicted from the hot pcd_data.json.

    Eviction policy (admin-configurable):
        max_record_count   — evict oldest active nodes when hot store exceeds N
        max_record_age_days— evict nodes older than N days
        max_record_bytes   — evict oldest when archive file exceeds N bytes

    Archive operations:
        archive_node(env)       — move Envelope from hot to archive store
        restore_node(node_id)   — move Envelope back from archive to hot store
        get_archived(node_id)   — retrieve archived Envelope (read-only)
        enforce_limits()        — apply policy bounds (called on rotation)

    The DAG chain is preserved:
        prev_hash links remain intact across both stores
        Chain verification traverses both stores in sequence
    """

    def __init__(self, policy: ArchivePolicy):
        self._policy = policy

    def update_policy(self, new_policy: ArchivePolicy) -> None:
        self._policy = new_policy
        arc_audit("info", "DATA_ARCHIVE", "POLICY_UPDATE",
                  f"max_records={new_policy.max_record_count} "
                  f"max_age_days={new_policy.max_record_age_days} "
                  f"max_bytes={new_policy.max_record_bytes}")
        self._enforce_limits()

    def archive_node(self, env_dict: dict, reason: str = "POLICY_EVICT") -> bool:
        """Move an Envelope dict from hot store to archive store."""
        with _arc_data_lock:
            s = _load_arc_data()
            node_id = env_dict["node_id"]
            if node_id in s["nodes"]:
                return False   # already archived
            s["nodes"][node_id] = {
                **env_dict,
                "archived_at": time.time(),
                "archive_reason": reason,
            }
            s["meta"]["total_archived"] = s["meta"].get("total_archived", 0) + 1
            _save_arc_data(s)
        arc_audit("info", "DATA_ARCHIVE", "ARCHIVE_NODE",
                  f"node={_short(node_id)} reason={reason}")
        return True

    def get_archived(self, node_id: str) -> Optional[dict]:
        with _arc_data_lock:
            s = _load_arc_data()
        return s["nodes"].get(node_id)

    def restore_node(self, node_id: str) -> Optional[dict]:
        """Remove from archive and return the Envelope dict for re-insertion into hot store."""
        with _arc_data_lock:
            s = _load_arc_data()
            if node_id not in s["nodes"]:
                return None
            env = s["nodes"].pop(node_id)
            _save_arc_data(s)
        arc_audit("info", "DATA_ARCHIVE", "RESTORE_NODE", f"node={_short(node_id)}")
        return env

    def all_archived(self) -> List[dict]:
        with _arc_data_lock:
            s = _load_arc_data()
        return list(s["nodes"].values())

    def _enforce_limits(self) -> None:
        """Evict oldest entries that exceed policy bounds from the archive."""
        evicted = []
        with _arc_data_lock:
            s       = _load_arc_data()
            nodes   = list(s["nodes"].values())
            changed = False

            # Age-based eviction
            if self._policy.max_record_age_days > 0:
                max_age = self._policy.max_record_age_days * 86400
                now     = time.time()
                for n in nodes:
                    age = now - n.get("archived_at", now)
                    if age > max_age:
                        del s["nodes"][n["node_id"]]
                        evicted.append(n["node_id"])
                        changed = True
                nodes = list(s["nodes"].values())  # refresh

            # Count-based eviction — oldest by chain_counter first
            if self._policy.max_record_count > 0 and \
               len(nodes) > self._policy.max_record_count:
                nodes_sorted = sorted(nodes,
                                      key=lambda n: n.get("chain_counter", 0))
                excess = len(nodes) - self._policy.max_record_count
                for n in nodes_sorted[:excess]:
                    del s["nodes"][n["node_id"]]
                    evicted.append(n["node_id"])
                    changed = True

            # Size-based eviction
            if changed:
                _save_arc_data(s)

        if self._policy.max_record_bytes > 0:
            size = ARCHIVE_DATA_FILE.stat().st_size \
                   if ARCHIVE_DATA_FILE.exists() else 0
            while size > self._policy.max_record_bytes:
                with _arc_data_lock:
                    s     = _load_arc_data()
                    nodes = sorted(s["nodes"].values(),
                                   key=lambda n: n.get("chain_counter", 0))
                    if not nodes:
                        break
                    oldest = nodes[0]
                    del s["nodes"][oldest["node_id"]]
                    evicted.append(oldest["node_id"])
                    _save_arc_data(s)
                size = ARCHIVE_DATA_FILE.stat().st_size \
                       if ARCHIVE_DATA_FILE.exists() else 0

        if evicted:
            arc_audit("warning", "DATA_ARCHIVE", "EVICT",
                      f"evicted={len(evicted)} nodes nodes_short={[_short(e) for e in evicted[:5]]}")

    def stats(self) -> dict:
        with _arc_data_lock:
            s = _load_arc_data()
        nodes = list(s["nodes"].values())
        size  = ARCHIVE_DATA_FILE.stat().st_size \
                if ARCHIVE_DATA_FILE.exists() else 0
        tiers  = {}
        for n in nodes:
            cls = n.get("classification", "UNKNOWN")
            tiers[cls] = tiers.get(cls, 0) + 1
        return {
            "total_archived":   len(nodes),
            "total_ops":        s["meta"].get("ops", 0),
            "cumulative_archived": s["meta"].get("total_archived", 0),
            "file_bytes":       size,
            "by_classification": tiers,
            "oldest_counter":   min((n.get("chain_counter", 0) for n in nodes),
                                    default=None),
            "newest_counter":   max((n.get("chain_counter", 0) for n in nodes),
                                    default=None),
        }

    def listing(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """Paginated listing of archived nodes — newest first."""
        with _arc_data_lock:
            s = _load_arc_data()
        nodes = sorted(s["nodes"].values(),
                       key=lambda n: n.get("chain_counter", 0), reverse=True)
        return [
            {
                "node_id":    n["node_id"],
                "label":      n.get("label",""),
                "actor":      n.get("actor",""),
                "epoch":      n.get("epoch",0),
                "classification": n.get("classification",""),
                "ai_tier":    n.get("ai_tier",""),
                "ai_score":   n.get("ai_score",0),
                "status":     n.get("status",""),
                "chain_counter": n.get("chain_counter",0),
                "archived_at": n.get("archived_at"),
                "archive_reason": n.get("archive_reason",""),
                "ts_iso":     time.strftime("%Y-%m-%dT%H:%M:%SZ",
                              time.gmtime(n.get("timestamp", 0))),
            }
            for n in nodes[offset:offset+limit]
        ]


# ═══════════════════════════════════════════════════════════════════════
#  ARCHIVE MANAGER  — top-level facade used by pcd_system.py
# ═══════════════════════════════════════════════════════════════════════
class ArchiveManager:
    """
    Single entry point for all archive operations.
    Instantiated once in AppState and used by:
      - Peer._rotate_if_needed()   → archive epoch before delete
      - Peer.read()                → retrieve epoch from archive if not in RAM
      - FastAPI routes             → admin API
    """

    def __init__(self):
        self._policy = ArchivePolicy()
        self._hsm    = SimulatedHSM(master_key=_sha256(f"HSM-ARCHIVE-KEY-{os.urandom(16).hex()}"))
        self._epoch_archive = EpochSecretArchive(self._policy, self._hsm)
        self._data_archive  = DataRecordArchive(self._policy)
        self._lock          = threading.Lock()

    def init_with_master(self, master: str) -> None:
        """Called at Hub provisioning — derive deterministic HSM master from provisioning master."""
        hsm_key = _hmac256(master, "HSM-ARCHIVE-MASTER-KEY")
        self._hsm = SimulatedHSM(master_key=hsm_key)
        self._epoch_archive = EpochSecretArchive(self._policy, self._hsm)
        arc_audit("info", "ARCHIVE_MGR", "INIT",
                  f"hsm_key={_short(hsm_key)}")

    # ── Policy management ──────────────────────────────────────────────
    def get_policy(self) -> ArchivePolicy:
        return self._policy

    def update_policy(self, new_policy: ArchivePolicy) -> None:
        with self._lock:
            self._policy = new_policy
            self._epoch_archive.update_policy(new_policy)
            self._data_archive.update_policy(new_policy)
        arc_audit("info", "ARCHIVE_MGR", "POLICY_UPDATED",
                  f"max_keys={new_policy.max_epoch_keys} "
                  f"hot={new_policy.hot_epochs} warm={new_policy.warm_epochs} "
                  f"max_records={new_policy.max_record_count}")

    # ── Epoch secret archive ───────────────────────────────────────────
    def archive_epoch(self, epoch: int, secret: str) -> str:
        """Call this BEFORE deleting epoch_secret from peer memory on rotation."""
        tier = self._epoch_archive.archive_epoch(epoch, secret)
        arc_audit("info", "ARCHIVE_MGR", "EPOCH_ARCHIVED",
                  f"epoch={epoch} tier={tier}")
        return tier

    def get_epoch_secret(self, epoch: int) -> Optional[str]:
        """
        Retrieve epoch_secret for any archived epoch.
        Used by Peer.read() when the live chain no longer holds the secret.
        """
        secret = self._epoch_archive.retrieve_epoch(epoch)
        if secret is None:
            arc_audit("warning", "ARCHIVE_MGR", "EPOCH_MISS",
                      f"epoch={epoch} — beyond archive bounds or never archived")
        return secret

    def get_epoch_secrets_batch(self, epochs: List[int]) -> Dict[int, Optional[str]]:
        """Batch retrieve — minimises HSM calls for bulk historical reads."""
        return self._epoch_archive.retrieve_epochs_batch(epochs)

    def promote_tiers(self, current_epoch: int) -> None:
        """Called after epoch rotation to advance tier boundaries."""
        self._epoch_archive._promote_tiers(current_epoch)

    # ── Data record archive ────────────────────────────────────────────
    def archive_record(self, env_dict: dict, reason: str = "POLICY_EVICT") -> bool:
        return self._data_archive.archive_node(env_dict, reason)

    def restore_record(self, node_id: str) -> Optional[dict]:
        return self._data_archive.restore_node(node_id)

    def get_archived_record(self, node_id: str) -> Optional[dict]:
        return self._data_archive.get_archived(node_id)

    def enforce_data_limits(self) -> None:
        self._data_archive._enforce_limits()

    def all_archived_envelopes(self) -> List[dict]:
        """Return raw dicts of all archived Envelopes — used by verify_chain."""
        return self._data_archive.all_archived()

    # ── Stats & listing ────────────────────────────────────────────────
    def full_stats(self) -> dict:
        warm_f = self._epoch_archive._warm_file()
        return {
            "epoch_archive":  self._epoch_archive.stats,
            "data_archive":   self._data_archive.stats(),
            "policy":         self._policy.to_dict(),
            "files": {
                "warm_epochs": str(warm_f),
                "warm_bytes":  warm_f.stat().st_size if warm_f.exists() else 0,
                "cold_hsm":    str(ARCHIVE_HSM_FILE),
                "cold_bytes":  ARCHIVE_HSM_FILE.stat().st_size
                               if ARCHIVE_HSM_FILE.exists() else 0,
                "data_archive":str(ARCHIVE_DATA_FILE),
                "data_bytes":  ARCHIVE_DATA_FILE.stat().st_size
                               if ARCHIVE_DATA_FILE.exists() else 0,
            }
        }

    def epoch_listing(self) -> List[dict]:
        return self._epoch_archive.full_listing()

    def data_listing(self, limit: int = 100, offset: int = 0) -> List[dict]:
        return self._data_archive.listing(limit, offset)


# ═══════════════════════════════════════════════════════════════════════
#  SINGLETON — imported by pcd_system.py
# ═══════════════════════════════════════════════════════════════════════
ARCHIVE = ArchiveManager()
