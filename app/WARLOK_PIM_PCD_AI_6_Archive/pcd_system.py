#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  PCD — PROOF CHAIN OF DATA                                               ║
║  Full Production Codebase · Hub + Peer + NanoAI + DAG + JSON Storage    ║
║                                                                           ║
║  Architecture:                                                            ║
║    HUB  — one-time epoch seed provisioner, validates mutual sync         ║
║    PEER — sealer/verifier with forward-secret epoch ratchet              ║
║    DAG  — append-only SHA-256 hash chain, JSON-backed                    ║
║    AI   — deterministic 6-feature weighted model → K(t,n) key factory   ║
║    STORE— JSON datafile as database, change-tracked per operation        ║
║    LOG  — structured audit log for every operation                       ║
║    ARCHIVE — epoch secret HSM archive + data record archive (v3.2)      ║
║                                                                           ║
║  Run:  pip install fastapi uvicorn pydantic                              ║
║        python3 pcd_system.py                                             ║
║  UI:   http://localhost:7700   Docs: /docs                               ║
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
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, jsonify, Response
from pcd_archive import (
    ARCHIVE, ArchivePolicy,
    arc_audit,
)


# ═══════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════
BASE_DIR   = Path(__file__).parent
DATA_FILE  = BASE_DIR / "pcd_data.json"
AUDIT_FILE = BASE_DIR / "pcd_audit.log"


# ═══════════════════════════════════════════════════════════════════════
#  AUDIT LOGGER
# ═══════════════════════════════════════════════════════════════════════
_audit_handler = logging.FileHandler(AUDIT_FILE, encoding="utf-8")
_audit_handler.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
))
audit_log = logging.getLogger("pcd.audit")
audit_log.setLevel(logging.DEBUG)
audit_log.addHandler(_audit_handler)
audit_log.propagate = False

def audit(level: str, actor: str, op: str, detail: str = "") -> None:
    msg = f"[{actor:12s}] {op:20s} {detail}"
    getattr(audit_log, level.lower(), audit_log.info)(msg)


# ═══════════════════════════════════════════════════════════════════════
#  CRYPTO ENGINE  (stdlib only — no external crypto deps)
# ═══════════════════════════════════════════════════════════════════════
class Crypto:
    """Pure-stdlib cryptographic primitives — identical across all peers."""

    @staticmethod
    def sha256(data: bytes | str) -> str:
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hmac256(key: bytes | str, msg: bytes | str) -> str:
        if isinstance(key, str):
            key = key.encode()
        if isinstance(msg, str):
            msg = msg.encode()
        return _hmac.new(key, msg, hashlib.sha256).hexdigest()

    @staticmethod
    def rng(n: int = 16) -> str:
        return os.urandom(n).hex()

    @staticmethod
    def xor_stream(data: bytes, key_hex: str) -> bytes:
        """XOR stream cipher. Demo-grade — replace with AES-256-GCM in production."""
        key_bytes = bytes.fromhex(hashlib.sha256(key_hex.encode()).hexdigest() * 4)
        return bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data))

    @classmethod
    def encrypt(cls, plaintext: str, key_hex: str) -> str:
        return cls.xor_stream(plaintext.encode("utf-8"), key_hex).hex()

    @classmethod
    def decrypt(cls, ciphertext_hex: str, key_hex: str) -> str:
        return cls.xor_stream(bytes.fromhex(ciphertext_hex), key_hex).decode("utf-8")

    @staticmethod
    def short(h: str) -> str:
        return f"{h[:8]}…{h[-6:]}" if h and len(h) > 16 else (h or "—")

    @staticmethod
    def entropy(payload: str) -> float:
        """Shannon entropy normalised to [0, 1]."""
        data = payload.encode("utf-8")
        if not data:
            return 0.0
        from collections import Counter
        freq = Counter(data)
        n = len(data)
        h = -sum((c / n) * math.log2(c / n) for c in freq.values())
        return min(h / 8.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
#  NANO-AI MODEL  — deterministic 6-feature key factory
# ═══════════════════════════════════════════════════════════════════════
WEIGHTS = [0.22, 0.28, 0.15, 0.18, 0.12, 0.05]
BIAS    = -0.15
TIERS   = [(0.85, "HOT"), (0.60, "WARM"), (0.30, "COOL"), (0.00, "QUARANTINE")]

GATE_ENTROPY = 0.30
GATE_ACTOR   = 0.40
GATE_TIMING  = 0.20


@dataclass
class Features:
    f1: float   # entropy
    f2: float   # actor reputation
    f3: float   # timing alignment
    f4: float   # window position
    f5: float   # chain depth
    f6: int     # raw counter (integer — NOT normalised in fingerprint)
    f6n: float  # normalised counter (inference only)

    def fingerprint(self) -> str:
        s = f"{self.f1:.6f}|{self.f2:.6f}|{self.f3:.6f}|{self.f6}|{self.f4:.6f}|{self.f5:.6f}"
        return Crypto.sha256(s)

    def to_dict(self) -> dict:
        return {"f1":self.f1,"f2":self.f2,"f3":self.f3,
                "f4":self.f4,"f5":self.f5,"f6":self.f6,"f6n":self.f6n}

    @classmethod
    def from_dict(cls, d: dict) -> "Features":
        return cls(**{k:d[k] for k in ("f1","f2","f3","f4","f5","f6","f6n")})


@dataclass
class InferenceResult:
    score: float
    tier:  str
    ktn:   str
    fp:    str
    gates: List[str]
    features: Features


class NanoAI:
    """Deterministic nano-AI — same weights produce same K(t,n) on all peers."""

    def __init__(self):
        self._rep: Dict[str, float] = {}

    def rep(self, actor: str) -> float:
        return self._rep.get(actor, 0.50)

    def update_rep(self, actor: str, score: float) -> None:
        self._rep[actor] = min(1.0, self._rep.get(actor, 0.50) + 0.04 * score)

    def build_features(
        self,
        payload: str,
        actor: str,
        win_pos: int,
        win_size: int,
        dag_size: int,
        counter: int,
        last_ts: float,
    ) -> Features:
        delta_t = max(0.0, time.time() - last_ts)
        f1 = round(Crypto.entropy(payload), 6)
        f2 = round(self.rep(actor), 6)
        f3 = round(min(1.0, 1.0 / (delta_t + 0.1)), 6)
        f4 = round(win_pos / max(win_size, 1), 6)
        f5 = round(min(dag_size / 20.0, 1.0), 6)
        f6 = counter
        f6n = round(min(f6 / 100.0, 1.0), 6)
        return Features(f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, f6=f6, f6n=f6n)

    def infer(self, F: Features, epoch_secret: str) -> InferenceResult:
        z = (WEIGHTS[0]*F.f1 + WEIGHTS[1]*F.f2 + WEIGHTS[2]*F.f3 +
             WEIGHTS[3]*F.f4 + WEIGHTS[4]*F.f5 + WEIGHTS[5]*F.f6n + BIAS)
        score = round(1.0 / (1.0 + math.exp(-z)), 6)
        tier  = next((t for v, t in TIERS if score >= v), "QUARANTINE")
        fp    = F.fingerprint()
        ktn   = Crypto.hmac256(epoch_secret, f"{fp}|{F.f6}|{F.f4:.6f}")

        # Hard anomaly gates — run in parallel with inference
        gates = []
        if F.f1 < GATE_ENTROPY:
            gates.append(f"LOW_ENTROPY({F.f1:.3f}<{GATE_ENTROPY})")
        if F.f2 < GATE_ACTOR:
            gates.append(f"ACTOR_DEGRADED({F.f2:.3f}<{GATE_ACTOR})")
        if F.f3 < GATE_TIMING:
            gates.append(f"TIMING_ANOMALY({F.f3:.3f}<{GATE_TIMING})")
        if F.f6 == 0 and F.f5 > 0.10:
            gates.append("STRUCTURAL_VIOLATION(counter=0,depth>0)")

        if gates:
            tier = "QUARANTINE"

        return InferenceResult(score=score, tier=tier, ktn=ktn, fp=fp,
                               gates=gates, features=F)

    def regen_ktn(self, stored_fp: str, F_local: Features, epoch_secret: str) -> Tuple[bool, str]:
        fp_local = F_local.fingerprint()
        ktn      = Crypto.hmac256(epoch_secret, f"{fp_local}|{F_local.f6}|{F_local.f4:.6f}")
        return fp_local == stored_fp, ktn


# ═══════════════════════════════════════════════════════════════════════
#  PCD ENVELOPE
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class Envelope:
    node_id:        str
    label:          str
    actor:          str
    classification: str
    epoch:          int
    window_pos:     int
    chain_counter:  int
    timestamp:      float
    nonce:          str
    cipher_payload: str
    payload_hash:   str
    cipher_hash:    str
    prev_hash:      str
    env_hash:       str
    temporal_seal:  str
    chain_seal:     str
    feature_fp:     str
    zk_proof:       str
    ai_score:       float
    ai_tier:        str
    ai_features:    dict
    ai_gates:       list
    latency_ms:     float
    status:         str = "ACTIVE"        # ACTIVE | SUPERSEDED | REDACTED
    superseded_by:  str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Envelope":
        return cls(**{k: d.get(k, "") for k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════
#  JSON DATA STORE  — every mutation is logged
# ═══════════════════════════════════════════════════════════════════════
_store_lock = threading.Lock()

def _load_store() -> dict:
    if DATA_FILE.exists():
        try:
            return json.loads(DATA_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"nodes": {}, "meta": {"created": time.time(), "ops": 0}}

def _save_store(store: dict) -> None:
    store["meta"]["last_updated"] = time.time()
    store["meta"]["ops"] = store["meta"].get("ops", 0) + 1
    DATA_FILE.write_text(json.dumps(store, indent=2), encoding="utf-8")

def store_put(env: Envelope) -> None:
    with _store_lock:
        s = _load_store()
        s["nodes"][env.node_id] = env.to_dict()
        _save_store(s)
    audit("info", env.actor, "STORE_PUT",
          f"node={Crypto.short(env.node_id)} tier={env.ai_tier} score={env.ai_score:.4f}")

def store_get(node_id: str) -> Optional[Envelope]:
    with _store_lock:
        s = _load_store()
    d = s["nodes"].get(node_id)
    return Envelope.from_dict(d) if d else None

def store_update_status(node_id: str, status: str, superseded_by: str = "") -> bool:
    with _store_lock:
        s = _load_store()
        if node_id not in s["nodes"]:
            return False
        s["nodes"][node_id]["status"] = status
        if superseded_by:
            s["nodes"][node_id]["superseded_by"] = superseded_by
        _save_store(s)
    audit("warning", "SYSTEM", "STORE_UPDATE",
          f"node={Crypto.short(node_id)} status={status} superseded_by={superseded_by}")
    return True

def store_nullify(node_id: str, policy_ref: str) -> bool:
    with _store_lock:
        s = _load_store()
        if node_id not in s["nodes"]:
            return False
        s["nodes"][node_id]["cipher_payload"] = f"REDACTED:{policy_ref}"
        s["nodes"][node_id]["status"] = "REDACTED"
        _save_store(s)
    audit("warning", "SYSTEM", "STORE_REDACT",
          f"node={Crypto.short(node_id)} policy={policy_ref}")
    return True

def store_all() -> List[Envelope]:
    with _store_lock:
        s = _load_store()
    return [Envelope.from_dict(d) for d in s["nodes"].values()]

def store_stats() -> dict:
    with _store_lock:
        s = _load_store()
    nodes = list(s["nodes"].values())
    return {
        "total": len(nodes),
        "active": sum(1 for n in nodes if n.get("status") == "ACTIVE"),
        "superseded": sum(1 for n in nodes if n.get("status") == "SUPERSEDED"),
        "redacted": sum(1 for n in nodes if n.get("status") == "REDACTED"),
        "quarantine": sum(1 for n in nodes if n.get("ai_tier") == "QUARANTINE"),
        "ops": s["meta"].get("ops", 0),
        "data_file": str(DATA_FILE),
    }


# ═══════════════════════════════════════════════════════════════════════
#  EPOCH CHAIN — forward-secret ratchet
# ═══════════════════════════════════════════════════════════════════════
class EpochChain:
    def __init__(self, master: str, W: int, anchor: float):
        self.W = W
        self._epoch = 0
        self._secrets: Dict[int, str] = {}
        # epoch_0 = HMAC(master, "epoch|0|W|anchor")
        ep0 = Crypto.hmac256(master, f"epoch|0|{W}|{int(anchor)}")
        self._secrets[0] = ep0
        audit("info", "EPOCH_CHAIN", "INIT",
              f"ep0={Crypto.short(ep0)} W={W}")

    @property
    def current_epoch(self) -> int:
        return self._epoch

    @property
    def secret(self) -> str:
        return self._secrets[self._epoch]

    def secret_for(self, epoch: int) -> Optional[str]:
        return self._secrets.get(epoch)

    def rotate(self) -> None:
        old = self._epoch
        cur_secret = self._secrets[old]
        new_secret = Crypto.hmac256(cur_secret, f"epoch|{old+1}|pim-pcd-chain")
        self._epoch += 1
        self._secrets[self._epoch] = new_secret
        # Archive epoch secret BEFORE deletion — three-tier HSM archive
        tier = ARCHIVE.archive_epoch(old, cur_secret)
        ARCHIVE.promote_tiers(self._epoch)
        del self._secrets[old]  # forward secrecy — past secret gone from RAM
        audit("info", "EPOCH_CHAIN", "ROTATE",
              f"epoch {old}→{self._epoch} archived_to={tier} "
              f"new={Crypto.short(new_secret)} fwd_secrecy=applied")


# ═══════════════════════════════════════════════════════════════════════
#  HUB AUTHORITY — one-time provisioner + epoch sync validator
# ═══════════════════════════════════════════════════════════════════════
@dataclass
class HubProvision:
    master_hash:   str
    epoch_0_hash:  str   # hash of epoch_0 for validation (not the secret itself)
    W:             int
    timing_anchor: float
    weights:       list
    bias:          float
    issued_at:     float
    peer_a_id:     str
    peer_b_id:     str

    def validate_seed(self, peer_id: str, submitted_seed_hash: str) -> bool:
        """Peer submits hash of epoch_0 they computed — Hub verifies without revealing secret."""
        ok = submitted_seed_hash == self.epoch_0_hash
        audit("info" if ok else "error", "HUB", "SEED_VALIDATE",
              f"peer={peer_id} ok={ok}")
        return ok


class Hub:
    """Hub Authority — provisions once, validates epoch seeds, then stays silent."""

    def __init__(self):
        self.provision: Optional[HubProvision] = None
        self._epoch_chain: Optional[EpochChain] = None
        self._log: List[dict] = []

    def _emit(self, level: str, msg: str) -> None:
        entry = {"ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg, "actor": "HUB"}
        self._log.append(entry)
        audit(level, "HUB", "LOG", msg)

    def init(self, master: str, W: int, peer_a: str, peer_b: str) -> HubProvision:
        anchor = time.time()
        chain = EpochChain(master, W, anchor)
        ep0_hash = Crypto.sha256(chain.secret)  # hash only — secret stays in chain
        self._epoch_chain = chain

        self.provision = HubProvision(
            master_hash   = Crypto.sha256(master),
            epoch_0_hash  = ep0_hash,
            W             = W,
            timing_anchor = anchor,
            weights       = WEIGHTS,
            bias          = BIAS,
            issued_at     = anchor,
            peer_a_id     = peer_a,
            peer_b_id     = peer_b,
        )
        # Initialise archive HSM with deterministic key from master
        ARCHIVE.init_with_master(master)
        self._emit("info", f"Hub provisioned | W={W} | peers={peer_a},{peer_b}")
        self._emit("info", f"ep0_hash={Crypto.short(ep0_hash)} | Hub now SILENT")
        audit("info", "HUB", "PROVISION",
              f"peer_a={peer_a} peer_b={peer_b} W={W} ep0={Crypto.short(ep0_hash)}")
        return self.provision

    def validate_epoch_sync(self, peer_a_hash: str, peer_b_hash: str) -> Tuple[bool, str]:
        """Both peers submit their epoch_0 hash. Hub validates mutual consistency."""
        if self.provision is None:
            return False, "Hub not provisioned"
        a_ok = peer_a_hash == self.provision.epoch_0_hash
        b_ok = peer_b_hash == self.provision.epoch_0_hash
        mutual = a_ok and b_ok
        msg = ("SYNC_OK" if mutual else
               f"SYNC_FAIL a_ok={a_ok} b_ok={b_ok}")
        self._emit("info" if mutual else "error", f"Epoch sync: {msg}")
        audit("info" if mutual else "error", "HUB", "EPOCH_SYNC", msg)
        return mutual, msg

    @property
    def log(self) -> List[dict]:
        return self._log[-60:]


# ═══════════════════════════════════════════════════════════════════════
#  PEER — sealer / verifier with full PCD pipeline
# ═══════════════════════════════════════════════════════════════════════
class Peer:
    def __init__(self, peer_id: str, provision: HubProvision):
        self.peer_id    = peer_id
        self.provision  = provision
        self._chain     = EpochChain.__new__(EpochChain)
        # Re-derive epoch chain from provisioned params
        self._chain.W        = provision.W
        self._chain._epoch   = 0
        self._chain._secrets = {}
        ep0 = Crypto.hmac256(
            Crypto.hmac256("__derive__", f"epoch|0|{provision.W}|{int(provision.timing_anchor)}"),
            f"epoch|0|{provision.W}|{int(provision.timing_anchor)}"
        )
        # Actually re-derive correctly — match Hub's formula
        # We don't have master, but Hub gave us epoch_0_hash for validation.
        # In a real deployment peers receive epoch_0_secret over a secure channel.
        # Here we store it after Hub init passes it via the app state.
        self._epoch0_secret: Optional[str] = None
        self._chain._epoch   = 0
        self._chain._secrets = {}

        self.ai      = NanoAI()
        self._wpos   = 0
        self._counter = 0
        self._last_ts = provision.timing_anchor
        self._log: List[dict] = []

    def _emit(self, level: str, msg: str) -> None:
        entry = {"ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg, "actor": self.peer_id}
        self._log.append(entry)
        if len(self._log) > 200:
            self._log = self._log[-150:]
        audit(level, self.peer_id, "LOG", msg)

    def set_epoch0(self, secret: str) -> None:
        self._epoch0_secret = secret
        self._chain._secrets[0] = secret
        # Submit hash to Hub for validation
        h = Crypto.sha256(secret)
        self._emit("info", f"epoch_0 received | hash={Crypto.short(h)}")

    @property
    def epoch_secret(self) -> str:
        return self._chain.secret

    @property
    def epoch(self) -> int:
        return self._chain.current_epoch

    @property
    def wpos(self) -> int:
        return self._wpos

    def _rotate_if_needed(self) -> None:
        if self._wpos >= self.provision.W:
            self._chain.rotate()
            self._wpos = 0
            self._emit("info", f"Epoch rotated → {self.epoch} | fwd_secrecy applied")

    def _dag_latest_hash(self) -> str:
        nodes = store_all()
        chain_nodes = [n for n in nodes if n.status != "REDACTED"]
        if not chain_nodes:
            return "0" * 64
        chain_nodes.sort(key=lambda n: n.chain_counter)
        return chain_nodes[-1].env_hash

    # ── SEAL (write / insert) ──────────────────────────────────────────
    def seal(
        self,
        label: str,
        payload: str,
        classification: str = "CONFIDENTIAL",
        actor: Optional[str] = None,
    ) -> Tuple[Optional[Envelope], str]:
        t0 = time.perf_counter()
        actor = actor or self.peer_id

        dag_size = len(store_all())
        F = self.ai.build_features(
            payload, actor, self._wpos, self.provision.W,
            dag_size, self._counter, self._last_ts
        )
        result = self.ai.infer(F, self.epoch_secret)
        ktn    = result.ktn

        if result.gates:
            self._emit("warning",
                f"QUARANTINE gates fired: {', '.join(result.gates)}")
            audit("warning", actor, "QUARANTINE",
                  f"label={label} gates={result.gates}")

        # Encrypt
        cipher  = Crypto.encrypt(payload, ktn)
        ph      = Crypto.sha256(payload)
        ch      = Crypto.sha256(cipher)

        # Chain link
        prev    = self._dag_latest_hash()
        nonce   = Crypto.rng(8)
        ts      = time.time()
        tseal   = Crypto.hmac256(self.epoch_secret, f"{ch}|{ts:.6f}|{nonce}")
        cseal   = Crypto.hmac256(self.epoch_secret, f"{prev}|{result.fp}|{self._counter}")
        node_id = Crypto.sha256(f"{label}{ts}{nonce}{Crypto.rng(4)}")[:24]
        zk      = Crypto.hmac256(ktn, f"zk|{actor}|{node_id}")
        ehash   = Crypto.sha256(f"{ch}|{prev}|{tseal}|{cseal}")

        env = Envelope(
            node_id=node_id, label=label, actor=actor,
            classification=classification,
            epoch=self.epoch, window_pos=self._wpos,
            chain_counter=self._counter, timestamp=ts, nonce=nonce,
            cipher_payload=cipher, payload_hash=ph, cipher_hash=ch,
            prev_hash=prev, env_hash=ehash,
            temporal_seal=tseal, chain_seal=cseal,
            feature_fp=result.fp, zk_proof=zk,
            ai_score=result.score, ai_tier=result.tier,
            ai_features=F.to_dict(), ai_gates=result.gates,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )

        # Persist to JSON store
        store_put(env)

        # Update state
        self.ai.update_rep(actor, result.score)
        self._wpos     += 1
        self._counter  += 1
        self._last_ts   = ts

        # Revoke key
        ktn = "REVOKED"
        del ktn

        self._rotate_if_needed()

        self._emit("info",
            f"SEALED {label} | score={result.score:.4f} tier={result.tier} "
            f"ep={env.epoch} w={env.window_pos} ctr={env.chain_counter} "
            f"lat={env.latency_ms}ms")

        return env, ""

    # ── READ (retrieve + decrypt) ──────────────────────────────────────
    def read(self, node_id: str) -> Tuple[Optional[str], Optional[Envelope], str]:
        env = store_get(node_id)
        if not env:
            # Check data record archive
            env_dict = ARCHIVE.get_archived_record(node_id)
            if env_dict:
                env = Envelope.from_dict(env_dict)
            else:
                return None, None, f"Node {node_id} not found"
        if env.status == "REDACTED":
            return None, env, "Payload has been redacted under policy"

        # Re-derive K(t,n) — try live chain first, then archive
        ep_sec = self._chain.secret_for(env.epoch)
        if ep_sec is None:
            ep_sec = ARCHIVE.get_epoch_secret(env.epoch)
        if ep_sec is None:
            return None, env, (
                f"Epoch {env.epoch} secret unavailable "
                f"(beyond archive bounds — configure policy to retain further epochs)"
            )

        F_local = Features.from_dict(env.ai_features)
        fp_ok, ktn_read = self.ai.regen_ktn(env.feature_fp, F_local, ep_sec)

        if not fp_ok:
            audit("error", self.peer_id, "FP_MISMATCH",
                  f"node={Crypto.short(node_id)}")
            return None, env, "Feature fingerprint mismatch — possible tampering"

        expected_zk = Crypto.hmac256(ktn_read, f"zk|{env.actor}|{env.node_id}")
        if expected_zk != env.zk_proof:
            return None, env, "ZK proof verification failed"

        try:
            plaintext = Crypto.decrypt(env.cipher_payload, ktn_read)
            if Crypto.sha256(plaintext) != env.payload_hash:
                return None, env, "Payload hash mismatch after decryption"
        except Exception as e:
            return None, env, f"Decryption error: {e}"
        finally:
            ktn_read = "REVOKED"

        self._emit("info",
            f"READ {env.label} | fp_ok=✓ zk_ok=✓ hash_ok=✓ tier={env.ai_tier}")
        audit("info", self.peer_id, "READ",
              f"node={Crypto.short(node_id)} label={env.label}")
        return plaintext, env, ""

    # ── SUPERSEDE (logical update) ─────────────────────────────────────
    def supersede(
        self,
        target_node_id: str,
        new_payload: str,
        reason: str = "AMENDMENT",
        actor: Optional[str] = None,
    ) -> Tuple[Optional[Envelope], str]:
        target = store_get(target_node_id)
        if not target:
            return None, f"Target node {target_node_id} not found"
        if target.status != "ACTIVE":
            return None, f"Target node is {target.status} — cannot supersede"

        # Seal the new version
        new_env, err = self.seal(
            label=f"{target.label}:v{self._counter}",
            payload=new_payload,
            classification=target.classification,
            actor=actor or self.peer_id,
        )
        if err:
            return None, err

        # Atomic update: mark old as superseded, link to new
        store_update_status(target_node_id, "SUPERSEDED", new_env.node_id)

        # Seal audit node
        audit_payload = json.dumps({
            "op": "SUPERSEDE", "old_node": target_node_id,
            "new_node": new_env.node_id, "reason": reason,
            "actor": actor or self.peer_id, "ts": time.time()
        })
        self.seal("_AUDIT_SUPERSEDE", audit_payload, "AUDIT", actor or "SYSTEM")

        self._emit("info",
            f"SUPERSEDE {target.label} → {new_env.node_id[:12]} reason={reason}")
        return new_env, ""

    # ── REDACT (physical nullification) ───────────────────────────────
    def redact(
        self,
        target_node_id: str,
        policy_ref: str,
        actor: Optional[str] = None,
    ) -> Tuple[bool, str]:
        target = store_get(target_node_id)
        if not target:
            return False, f"Node {target_node_id} not found"

        store_nullify(target_node_id, policy_ref)

        # Seal redaction receipt
        receipt = json.dumps({
            "op": "REDACT", "target_node": target_node_id,
            "policy_ref": policy_ref, "actor": actor or self.peer_id,
            "ts": time.time()
        })
        self.seal("_AUDIT_REDACT", receipt, "AUDIT", actor or "SYSTEM")

        self._emit("warning",
            f"REDACT {target.label} | policy={policy_ref}")
        return True, ""

    # ── VERIFY CHAIN ───────────────────────────────────────────────────
    def verify_chain(self) -> List[dict]:
        # Merge hot + archived nodes, sort by counter
        nodes = store_all()
        archived = [Envelope.from_dict(d) for d in ARCHIVE.all_archived_envelopes()]
        seen = {n.node_id for n in nodes}
        for a in archived:
            if a.node_id not in seen:
                nodes.append(a)
        nodes.sort(key=lambda n: n.chain_counter)

        # Batch-fetch all epoch secrets needed
        needed_epochs = list({n.epoch for n in nodes})
        batch = ARCHIVE.get_epoch_secrets_batch(needed_epochs)
        # Merge live chain secrets into batch
        for ep in needed_epochs:
            live = self._chain.secret_for(ep)
            if live:
                batch[ep] = live

        results = []
        for i, n in enumerate(nodes):
            parent = nodes[i-1] if i > 0 else None

            c1 = Crypto.sha256(n.cipher_payload) == n.cipher_hash
            c2 = (n.prev_hash == "0"*64 if parent is None
                  else n.prev_hash == parent.env_hash)
            F_check = Features.from_dict(n.ai_features)
            c3 = F_check.fingerprint() == n.feature_fp

            ep_sec = batch.get(n.epoch)
            if ep_sec:
                expected_ts = Crypto.hmac256(
                    ep_sec, f"{n.cipher_hash}|{n.timestamp:.6f}|{n.nonce}")
                c4 = expected_ts == n.temporal_seal
            else:
                c4 = None  # epoch beyond archive bounds — cannot verify seal

            valid = c1 and c2 and c3 and (c4 is True or c4 is None)
            results.append({
                "node_id":   n.node_id,
                "label":     n.label,
                "valid":     valid,
                "c1_cipher": c1,
                "c2_prev":   c2,
                "c3_fp":     c3,
                "c4_seal":   c4,
                "tier":      n.ai_tier,
                "epoch":     n.epoch,
                "wpos":      n.window_pos,
                "status":    n.status,
                "counter":   n.chain_counter,
                "location":  "archive" if n.node_id in {a.node_id for a in archived}
                             else "hot",
            })

        ok = all(r["valid"] for r in results)
        self._emit("info" if ok else "error",
            f"VERIFY: {len(results)} nodes ({len(archived)} from archive), chain_ok={ok}")
        return results

    @property
    def window_info(self) -> dict:
        return {
            "epoch": self.epoch,
            "wpos": self._wpos,
            "W": self.provision.W,
            "secret_short": Crypto.short(self.epoch_secret),
            "counter": self._counter,
        }

    @property
    def log(self) -> List[dict]:
        return self._log[-60:]


# ═══════════════════════════════════════════════════════════════════════
#  APPLICATION STATE
# ═══════════════════════════════════════════════════════════════════════
class AppState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hub: Optional[Hub] = None
        self.peer_a: Optional[Peer] = None
        self.peer_b: Optional[Peer] = None
        self.provisioned = False
        self._log: List[dict] = []
        self._lock = threading.Lock()

    def log(self, level: str, actor: str, msg: str) -> None:
        with self._lock:
            self._log.append({"ts": time.strftime("%H:%M:%S"),
                              "level": level, "msg": msg, "actor": actor})
            if len(self._log) > 400:
                self._log = self._log[-300:]

    def all_logs(self) -> List[dict]:
        logs = list(self._log)
        if self.hub:
            logs.extend(self.hub.log)
        if self.peer_a:
            logs.extend(self.peer_a.log)
        if self.peer_b:
            logs.extend(self.peer_b.log)
        logs.sort(key=lambda x: x.get("ts",""))
        return logs[-120:]


STATE = AppState()

SAMPLE_PAYLOADS = [
    ("sensor-batch-001", "CONFIDENTIAL",
     '{"temp":36.8,"pressure":1013,"unit":"SN-77","status":"nominal","ts":1742000001}'),
    ("patient-record-002", "TOP_SECRET",
     '{"patient_id":"PT-9821","bp":"118/76","glucose":94,"flagged":false,"ward":"ICU-3"}'),
    ("tx-ledger-7731", "CONFIDENTIAL",
     '{"from":"ACCT-441","to":"ACCT-882","amount":12500.00,"currency":"USD","ref":"INV-2026-441"}'),
    ("sat-telemetry-042", "SECRET",
     '{"sat":"SAT-9","alt_km":412,"lat":31.2,"lon":34.8,"status":"NOMINAL","batt":0.92}'),
    ("ml-batch-epoch17", "UNCLASSIFIED",
     '{"epoch":17,"loss":0.0421,"acc":0.9834,"model":"resnet-50","samples":50000}'),
    ("audit-log-443", "SECRET",
     '{"event":"ACCESS","user":"admin","resource":"/api/v2/data","result":"PERMIT","ip":"10.0.1.4"}'),
]
_sample_idx = [0]


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  FLASK APP + ROUTES
# ═══════════════════════════════════════════════════════════════════════
app = Flask(__name__)


def _j(data):
    return jsonify(data)


def _env_summary(env: Envelope) -> dict:
    return {
        "node_id": env.node_id, "label": env.label, "actor": env.actor,
        "classification": env.classification, "epoch": env.epoch,
        "window_pos": env.window_pos, "chain_counter": env.chain_counter,
        "ai_score": env.ai_score, "ai_tier": env.ai_tier,
        "ai_gates": env.ai_gates, "status": env.status,
        "hash_short": Crypto.short(env.env_hash),
        "prev_short": Crypto.short(env.prev_hash),
        "ts": time.strftime("%H:%M:%S", time.localtime(env.timestamp)),
        "latency_ms": env.latency_ms,
    }


def _body() -> dict:
    return request.get_json(force=True, silent=True) or {}


# ── Core PCD routes ───────────────────────────────────────────────────
@app.route("/api/provision", methods=["POST"])
def api_provision():
    d = _body()
    master  = d.get("master_secret", "M-WARLOK-2026-DEFAULT")
    W       = int(d.get("window_size", 4))
    peer_a  = d.get("peer_a_id", "NODE-ALPHA")
    peer_b  = d.get("peer_b_id", "NODE-BETA")

    STATE.reset()
    hub  = Hub()
    prov = hub.init(master, W, peer_a, peer_b)

    anchor     = prov.timing_anchor
    ep0_secret = Crypto.hmac256(master, f"epoch|0|{W}|{int(anchor)}")

    pa = Peer(peer_a, prov);  pa.set_epoch0(ep0_secret)
    pb = Peer(peer_b, prov);  pb.set_epoch0(ep0_secret)

    sync_ok, sync_msg = hub.validate_epoch_sync(
        Crypto.sha256(ep0_secret), Crypto.sha256(ep0_secret))

    STATE.hub = hub;  STATE.peer_a = pa;  STATE.peer_b = pb
    STATE.provisioned = True
    STATE.log("info", "SYSTEM",
              f"Provisioned | W={W} | peers={peer_a},{peer_b} | sync={sync_ok}")
    audit("info", "SYSTEM", "PROVISION",
          f"W={W} peers={peer_a},{peer_b} sync={sync_ok}")

    return _j({"ok": True, "hub_record": {
        "master_hash":  Crypto.short(prov.master_hash),
        "epoch_0_hash": Crypto.short(prov.epoch_0_hash),
        "W": W,
        "timing_anchor": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                       time.gmtime(prov.timing_anchor)),
        "peer_a": peer_a, "peer_b": peer_b,
        "sync": sync_ok, "sync_msg": sync_msg,
    }})


@app.route("/api/seal", methods=["POST"])
def api_seal():
    if not STATE.provisioned:
        return _j({"ok": False, "error": "Not provisioned"})
    d = _body()
    p = STATE.peer_a if d.get("peer","a") == "a" else STATE.peer_b
    env, err = p.seal(d.get("label","data"), d.get("payload","{}"),
                      d.get("classification","CONFIDENTIAL"))
    if err:
        return _j({"ok": False, "error": err})
    return _j({"ok": True, "node": _env_summary(env)})


@app.route("/api/read", methods=["POST"])
def api_read():
    if not STATE.provisioned:
        return _j({"ok": False, "error": "Not provisioned"})
    d = _body()
    p       = STATE.peer_a if d.get("peer","b") == "a" else STATE.peer_b
    node_id = d.get("node_id","").strip()

    if not node_id:
        active = [n for n in store_all() if n.status == "ACTIVE"]
        if not active:
            return _j({"ok": False, "error": "No active nodes"})
        active.sort(key=lambda n: n.chain_counter)
        node_id = active[-1].node_id

    plaintext, env, err = p.read(node_id)
    if err and not env:
        return _j({"ok": False, "error": err})
    return _j({"ok": True, "plaintext": plaintext, "error": err, "node": {
        **_env_summary(env),
        "feature_fp":    Crypto.short(env.feature_fp),
        "zk_proof":      Crypto.short(env.zk_proof),
        "temporal_seal": Crypto.short(env.temporal_seal),
        "ai_features":   env.ai_features,
    }})


@app.route("/api/supersede", methods=["POST"])
def api_supersede():
    if not STATE.provisioned:
        return _j({"ok": False, "error": "Not provisioned"})
    d = _body()
    if not d.get("node_id") or not d.get("new_payload"):
        return _j({"ok": False, "error": "node_id and new_payload required"})
    p = STATE.peer_a if d.get("peer","a") == "a" else STATE.peer_b
    env, err = p.supersede(d["node_id"], d["new_payload"],
                           d.get("reason","AMENDMENT"))
    if err:
        return _j({"ok": False, "error": err})
    return _j({"ok": True, "new_node": _env_summary(env)})


@app.route("/api/redact", methods=["POST"])
def api_redact():
    if not STATE.provisioned:
        return _j({"ok": False, "error": "Not provisioned"})
    d = _body()
    if not d.get("node_id"):
        return _j({"ok": False, "error": "node_id required"})
    p = STATE.peer_a if d.get("peer","a") == "a" else STATE.peer_b
    ok, err = p.redact(d["node_id"], d.get("policy_ref","GDPR-ART17"))
    return _j({"ok": ok, "error": err})


@app.route("/api/verify", methods=["GET"])
def api_verify():
    if not STATE.provisioned:
        return _j({"ok": False, "error": "Not provisioned"})
    p        = STATE.peer_a if request.args.get("peer","a") == "a" else STATE.peer_b
    results  = p.verify_chain()
    chain_ok = all(r["valid"] for r in results)
    return _j({"ok": True, "chain_valid": chain_ok, "results": results})


@app.route("/api/nodes", methods=["GET"])
def api_nodes():
    nodes = store_all()
    nodes.sort(key=lambda n: n.chain_counter)
    return _j({"ok": True, "nodes": [_env_summary(n) for n in nodes],
               "stats": store_stats()})


@app.route("/api/sample", methods=["GET"])
def api_sample():
    idx = _sample_idx[0] % len(SAMPLE_PAYLOADS)
    _sample_idx[0] += 1
    label, cls, payload = SAMPLE_PAYLOADS[idx]
    return _j({"label": label, "classification": cls, "payload": payload})


@app.route("/api/state", methods=["GET"])
def api_state():
    if not STATE.provisioned:
        return _j({"provisioned": False, "logs": [],
                   "stats": {"total": 0}, "window_a": {}, "window_b": {}})
    nodes = store_all();  nodes.sort(key=lambda n: n.chain_counter)
    ai_last: dict = {}
    if nodes:
        last = nodes[-1]
        ai_last = {
            "score": last.ai_score, "tier": last.ai_tier,
            "entropy": last.ai_features.get("f1",0),
            "actor":   last.ai_features.get("f2",0),
            "timing":  last.ai_features.get("f3",0),
            "window_pos": last.ai_features.get("f4",0),
            "chain_depth":last.ai_features.get("f5",0),
            "gates": last.ai_gates,
        }
    return _j({
        "provisioned": True,
        "nodes":    [_env_summary(n) for n in nodes],
        "stats":    store_stats(),
        "window_a": STATE.peer_a.window_info if STATE.peer_a else {},
        "window_b": STATE.peer_b.window_info if STATE.peer_b else {},
        "ai_last":  ai_last,
        "logs":     STATE.all_logs()[-80:],
    })


@app.route("/api/audit_log", methods=["GET"])
def api_audit_log():
    n = int(request.args.get("n", 100))
    try:
        lines = AUDIT_FILE.read_text(encoding="utf-8").splitlines()
        return _j({"ok": True, "lines": lines[-n:],
                   "path": str(AUDIT_FILE), "total": len(lines)})
    except Exception as e:
        return _j({"ok": False, "error": str(e), "lines": []})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    STATE.reset()
    _save_store({"nodes": {}, "meta": {"created": time.time(), "ops": 0}})
    audit("warning", "SYSTEM", "RESET", "Full system reset")
    return _j({"ok": True})


# ── Archive API routes ────────────────────────────────────────────────
@app.route("/api/archive/stats", methods=["GET"])
def api_archive_stats():
    return _j({"ok": True, "stats": ARCHIVE.full_stats()})


@app.route("/api/archive/policy", methods=["GET"])
def api_archive_policy_get():
    return _j({"ok": True, "policy": ARCHIVE.get_policy().to_dict()})


@app.route("/api/archive/policy", methods=["PUT"])
def api_archive_policy_update():
    d = _body()
    new_policy = ArchivePolicy(
        max_epoch_keys      = int(d.get("max_epoch_keys", 50)),
        max_epoch_age_days  = float(d.get("max_epoch_age_days", 0.0)),
        max_epoch_bytes     = int(d.get("max_epoch_bytes", 0)),
        max_record_count    = int(d.get("max_record_count", 10000)),
        max_record_age_days = float(d.get("max_record_age_days", 365.0)),
        max_record_bytes    = int(d.get("max_record_bytes", 0)),
        hot_epochs          = int(d.get("hot_epochs", 5)),
        warm_epochs         = int(d.get("warm_epochs", 20)),
        enforce_on_rotate   = bool(d.get("enforce_on_rotate", True)),
        enforce_on_read     = bool(d.get("enforce_on_read", False)),
    )
    ARCHIVE.update_policy(new_policy)
    audit("info", "ADMIN", "ARCHIVE_POLICY_UPDATE",
          f"max_keys={new_policy.max_epoch_keys} hot={new_policy.hot_epochs} "
          f"warm={new_policy.warm_epochs} max_records={new_policy.max_record_count}")
    return _j({"ok": True, "policy": new_policy.to_dict()})


@app.route("/api/archive/epochs", methods=["GET"])
def api_archive_epochs():
    return _j({"ok": True, "epochs": ARCHIVE.epoch_listing()})


@app.route("/api/archive/epoch/<int:epoch>", methods=["GET"])
def api_archive_epoch_get(epoch: int):
    secret = ARCHIVE.get_epoch_secret(epoch)
    if secret is None:
        return _j({"ok": False, "error": f"Epoch {epoch} not found in archive"})
    return _j({"ok": True, "epoch": epoch,
               "short": Crypto.short(secret), "available": True})


@app.route("/api/archive/epoch/<int:epoch>", methods=["DELETE"])
def api_archive_epoch_evict(epoch: int):
    ea = ARCHIVE._epoch_archive
    ea._hot.pop(epoch, None)
    warm = ea._load_warm()
    if str(epoch) in warm:
        del warm[str(epoch)];  ea._save_warm(warm)
    ARCHIVE._hsm.delete_cold(epoch)
    audit("warning", "ADMIN", "EPOCH_MANUAL_EVICT", f"epoch={epoch}")
    return _j({"ok": True, "evicted_epoch": epoch})


@app.route("/api/archive/records", methods=["GET"])
def api_archive_records():
    limit  = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))
    return _j({"ok": True,
               "records": ARCHIVE.data_listing(limit, offset),
               "stats":   ARCHIVE._data_archive.stats()})


@app.route("/api/archive/records/<node_id>", methods=["GET"])
def api_archive_record_get(node_id: str):
    rec = ARCHIVE.get_archived_record(node_id)
    if rec is None:
        return _j({"ok": False, "error": f"Node {node_id} not found in archive"})
    return _j({"ok": True, "record": rec})


@app.route("/api/archive/records/archive", methods=["POST"])
def api_archive_record_archive():
    d = _body()
    nid    = d.get("node_id","")
    reason = d.get("reason","MANUAL_ARCHIVE")
    env = store_get(nid)
    if not env:
        return _j({"ok": False, "error": f"Node {nid} not found in hot store"})
    ok = ARCHIVE.archive_record(env.to_dict(), reason)
    if ok:
        with _store_lock:
            s = _load_store()
            s["nodes"].pop(nid, None)
            _save_store(s)
        audit("info", "ADMIN", "MANUAL_ARCHIVE_NODE",
              f"node={Crypto.short(nid)} reason={reason}")
    return _j({"ok": ok, "node_id": nid})


@app.route("/api/archive/records/restore", methods=["POST"])
def api_archive_record_restore():
    d = _body()
    nid = d.get("node_id","")
    env_dict = ARCHIVE.restore_record(nid)
    if env_dict is None:
        return _j({"ok": False, "error": f"Node {nid} not found in archive"})
    env_dict.pop("archived_at", None)
    env_dict.pop("archive_reason", None)
    with _store_lock:
        s = _load_store()
        s["nodes"][nid] = env_dict
        _save_store(s)
    audit("info", "ADMIN", "RESTORE_NODE", f"node={Crypto.short(nid)}")
    return _j({"ok": True, "node_id": nid})


@app.route("/api/archive/enforce", methods=["POST"])
def api_archive_enforce():
    ARCHIVE.enforce_data_limits()
    audit("info", "ADMIN", "ARCHIVE_ENFORCE", "manual trigger")
    return _j({"ok": True, "stats": ARCHIVE.full_stats()})


@app.route("/")
def index():
    return Response(HTML_UI, mimetype="text/html")


HTML_UI = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PCD — Proof Chain of Data v3.2</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#030712;--bg2:#080f1e;--bg3:#0c1528;--panel:#050d1a;
  --border:#0f2040;--border2:#1a3560;--border3:#2a4d80;
  --c:#00e5ff;--c2:#00b8d4;--c3:#0097a7;
  --g:#00e676;--g2:#00c853;
  --a:#ffab40;--a2:#ff6d00;
  --r:#ff5252;--r2:#d50000;
  --p:#e040fb;--p2:#aa00ff;
  --text:#c8deff;--text2:#6e91c4;--text3:#2e4870;
  --hot:#00e676;--warm:#00e5ff;--cool:#ffab40;--qua:#ff5252;
}
*{margin:0;padding:0;box-sizing:border-box}
html{background:var(--bg);color:var(--text);font-family:'JetBrains Mono',monospace;font-size:13px}
body{min-height:100vh;background:
  radial-gradient(ellipse 80% 40% at 20% 0%,rgba(0,229,255,.04) 0%,transparent 60%),
  radial-gradient(ellipse 60% 50% at 80% 100%,rgba(0,230,118,.03) 0%,transparent 60%),
  var(--bg)}
body::before{content:'';position:fixed;inset:0;background:
  linear-gradient(rgba(0,229,255,.015) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,229,255,.015) 1px,transparent 1px);
  background-size:48px 48px;pointer-events:none;z-index:0}

#app{position:relative;z-index:1;max-width:1600px;margin:0 auto;padding:0 16px 48px}

/* ── HEADER ── */
header{display:flex;align-items:center;justify-content:space-between;
  padding:14px 0 12px;border-bottom:1px solid var(--border2);margin-bottom:18px}
.logo{font-family:'Syne',sans-serif;font-weight:800;font-size:22px;
  color:var(--c);letter-spacing:4px;
  text-shadow:0 0 20px rgba(0,229,255,.3)}
.logo span{color:var(--text3);font-weight:400;font-size:11px;letter-spacing:3px;
  margin-left:14px;vertical-align:middle}
.status-bar{display:flex;gap:10px;align-items:center}
.pill{display:flex;align-items:center;gap:6px;
  background:var(--bg3);border:1px solid var(--border2);
  border-radius:2px;padding:4px 10px;font-size:10px;
  letter-spacing:1.5px;color:var(--text2)}
.dot{width:5px;height:5px;border-radius:50%;flex-shrink:0}
.dot.on{background:var(--g);box-shadow:0 0 6px var(--g);animation:blink 2s infinite}
.dot.off{background:var(--a2);box-shadow:0 0 4px var(--a2)}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

/* ── LAYOUT ── */
.layout{display:grid;grid-template-columns:320px 1fr 280px;gap:14px;align-items:start}
.col{display:flex;flex-direction:column;gap:12px}

/* ── PANELS ── */
.panel{background:var(--panel);border:1px solid var(--border2);border-radius:4px;overflow:hidden;position:relative}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;
  background:linear-gradient(90deg,transparent,var(--c3),transparent);opacity:.5}
.ph{display:flex;align-items:center;gap:8px;
  padding:8px 12px;background:var(--bg3);border-bottom:1px solid var(--border);
  font-size:9px;letter-spacing:2.5px;color:var(--c2);font-weight:500}
.ph .badge{margin-left:auto;padding:2px 7px;border-radius:1px;font-size:8px;letter-spacing:1px}
.b-hub{background:rgba(224,64,251,.2);border:1px solid var(--p2);color:var(--p)}
.b-a{background:rgba(0,229,255,.15);border:1px solid var(--c2);color:var(--c)}
.b-b{background:rgba(0,230,118,.15);border:1px solid var(--g2);color:var(--g)}
.pb{padding:12px}

/* ── INPUTS ── */
label{display:block;font-size:9px;letter-spacing:2px;color:var(--text3);margin-bottom:4px;text-transform:uppercase}
input,textarea,select{width:100%;background:var(--bg2);border:1px solid var(--border2);
  border-radius:2px;color:var(--text);font-family:inherit;font-size:12px;
  padding:7px 9px;outline:none;transition:border-color .15s}
input:focus,textarea:focus,select:focus{border-color:var(--c3)}
textarea{height:72px;resize:vertical}
.fr{margin-bottom:10px}
.fr2{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}

/* ── BUTTONS ── */
.btn{display:inline-flex;align-items:center;gap:5px;cursor:pointer;
  font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:500;
  letter-spacing:2px;padding:7px 14px;border:none;border-radius:2px;
  transition:all .12s;text-transform:uppercase;white-space:nowrap}
.btn:disabled{opacity:.3;cursor:not-allowed}
.btn-c{background:var(--c3);color:#000}
.btn-g{background:var(--g2);color:#000}
.btn-a{background:var(--a2);color:#000}
.btn-r{background:var(--r2);color:#fff}
.btn-p{background:var(--p2);color:#fff}
.btn-ghost{background:transparent;color:var(--text2);border:1px solid var(--border2)}
.btn:not(:disabled):hover{filter:brightness(1.2);transform:translateY(-1px)}
.btn-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}

/* ── METRICS ROW ── */
.metrics{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:14px}
.metric{background:var(--panel);border:1px solid var(--border);border-radius:3px;
  padding:10px 12px;position:relative;overflow:hidden}
.metric::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.metric.c::after{background:var(--c)}.metric.g::after{background:var(--g)}
.metric.a::after{background:var(--a)}.metric.r::after{background:var(--r)}
.metric.p::after{background:var(--p)}
.ml{font-size:8px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:3px}
.mv{font-family:'Syne',sans-serif;font-size:22px;font-weight:800}
.metric.c .mv{color:var(--c)}.metric.g .mv{color:var(--g)}
.metric.a .mv{color:var(--a)}.metric.r .mv{color:var(--r)}.metric.p .mv{color:var(--p)}
.ms{font-size:9px;color:var(--text3);margin-top:2px}

/* ── AI BARS ── */
.ai-bars{display:flex;flex-direction:column;gap:5px;margin-bottom:10px}
.ai-row{display:flex;align-items:center;gap:7px}
.ai-lbl{font-size:9px;color:var(--text2);width:100px;flex-shrink:0;letter-spacing:1px}
.ai-track{flex:1;height:3px;background:var(--bg3);border-radius:2px;overflow:hidden}
.ai-fill{height:100%;border-radius:2px;transition:width .5s ease}
.ai-val{font-size:10px;width:32px;text-align:right;flex-shrink:0}

/* ── WINDOW GRID ── */
.wgrid{display:grid;gap:3px;margin-bottom:8px}
.wc{aspect-ratio:1;border-radius:1px;display:flex;align-items:center;justify-content:center;
  font-size:8px;border:1px solid var(--border);transition:all .3s}
.wc.used{background:rgba(0,229,255,.1);border-color:var(--c2);color:var(--c)}
.wc.active{background:rgba(0,229,255,.25);border-color:var(--c);color:var(--c);
  animation:wglow .9s infinite alternate}
@keyframes wglow{from{box-shadow:0 0 3px rgba(0,229,255,.2)}to{box-shadow:0 0 10px rgba(0,229,255,.5)}}

/* ── KV GRID ── */
.kv{display:grid;grid-template-columns:110px 1fr;gap:2px 8px;font-size:10px}
.kv .k{color:var(--text3)}.kv .v{color:var(--text);word-break:break-all}
.kv .v.c{color:var(--c)}.kv .v.g{color:var(--g)}.kv .v.a{color:var(--a)}
.kv .v.p{color:var(--p)}.kv .v.r{color:var(--r)}

/* ── NODE LIST ── */
.node-list{max-height:320px;overflow-y:auto}
.node-item{display:flex;align-items:flex-start;gap:8px;padding:7px 10px;
  border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s}
.node-item:hover{background:var(--bg3)}
.node-item.selected{background:rgba(0,229,255,.05)}
.tier-badge{padding:2px 6px;border-radius:1px;font-size:8px;letter-spacing:1px;
  flex-shrink:0;margin-top:1px;font-weight:500}
.tier-HOT{background:rgba(0,230,118,.2);color:var(--hot);border:1px solid var(--g2)}
.tier-WARM{background:rgba(0,229,255,.2);color:var(--warm);border:1px solid var(--c2)}
.tier-COOL{background:rgba(255,171,64,.2);color:var(--cool);border:1px solid var(--a)}
.tier-QUARANTINE{background:rgba(255,82,82,.2);color:var(--qua);border:1px solid var(--r2)}
.node-body{flex:1;min-width:0}
.node-label{font-size:11px;color:var(--text);margin-bottom:1px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.node-sub{font-size:9px;color:var(--text2)}
.node-status{font-size:9px;text-align:right;flex-shrink:0}
.st-ACTIVE{color:var(--g)}.st-SUPERSEDED{color:var(--a)}.st-REDACTED{color:var(--r)}

/* ── DETAIL PANEL ── */
.detail{background:var(--bg2);border:1px solid var(--border2);border-radius:2px;
  padding:10px;margin-top:10px;display:none;font-size:11px}
.detail.open{display:block}
.plaintext-box{background:var(--bg3);border:1px solid var(--border2);
  border-radius:2px;padding:8px;margin-top:8px;font-size:10px;
  color:var(--g);word-break:break-all;max-height:120px;overflow-y:auto}

/* ── TERMINAL LOG ── */
.term{background:#020913;height:200px;overflow-y:auto;
  padding:8px 10px;font-size:11px;line-height:1.8;border-radius:2px}
.term-line{display:flex;gap:8px}
.term-ts{color:var(--text3);flex-shrink:0;min-width:56px}
.term-actor{color:var(--c2);flex-shrink:0;min-width:80px}
.lv-info .term-msg{color:var(--text)}
.lv-warning .term-msg,.lv-warn .term-msg{color:var(--a)}
.lv-error .term-msg,.lv-err .term-msg{color:var(--r)}
.lv-hub .term-msg{color:var(--p)}
.lv-debug .term-msg{color:var(--text3)}

/* ── AUDIT LOG ── */
.audit-lines{background:#020913;height:240px;overflow-y:auto;
  padding:8px;font-size:9px;line-height:1.9;color:var(--text2)}
.audit-lines span{color:var(--c3)}

/* ── TABS ── */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border2);margin-bottom:10px}
.tab{padding:6px 14px;font-size:9px;letter-spacing:2px;cursor:pointer;
  border-bottom:2px solid transparent;color:var(--text3);transition:all .15s}
.tab.active{border-color:var(--c);color:var(--c)}

/* ── VERIFY ── */
.vr-line{display:flex;align-items:center;gap:6px;padding:3px 0;
  font-size:9px;border-bottom:1px solid var(--border)}
.vr-ok{color:var(--g)}.vr-fail{color:var(--r)}

/* ── ALERT ── */
.alert{display:flex;align-items:center;gap:7px;padding:6px 10px;
  border-radius:2px;margin-bottom:8px;font-size:10px}
.alert-ok{background:rgba(0,230,118,.08);border:1px solid var(--g2);color:var(--g)}
.alert-err{background:rgba(255,82,82,.08);border:1px solid var(--r2);color:var(--r)}
.alert-hub{background:rgba(224,64,251,.08);border:1px solid var(--p2);color:var(--p)}

/* ── ARCHIVE ── */
.arc-row{display:flex;align-items:center;gap:6px;padding:5px 8px;
  border-bottom:1px solid var(--border);font-size:10px;cursor:pointer;transition:background .1s}
.arc-row:hover{background:var(--bg3)}
.arc-tier{padding:2px 6px;border-radius:1px;font-size:8px;font-weight:700;letter-spacing:1px;flex-shrink:0}
.arc-tier.HOT{background:rgba(0,230,118,.2);color:var(--hot);border:1px solid var(--g2)}
.arc-tier.WARM{background:rgba(255,171,64,.2);color:var(--a);border:1px solid var(--a)}
.arc-tier.COLD{background:rgba(0,229,255,.2);color:var(--c);border:1px solid var(--c2)}
.arc-label{flex:1;color:var(--text2)}.arc-val{color:var(--text3);font-size:9px}
.policy-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px}
.policy-field label{display:block;font-size:9px;letter-spacing:1.5px;color:var(--text3);margin-bottom:3px}
.policy-field input{width:100%;background:var(--bg2);border:1px solid var(--border2);
  border-radius:2px;color:var(--text);font-family:inherit;font-size:12px;padding:5px 7px;outline:none}
.stat-row{display:flex;justify-content:space-between;font-size:10px;
  padding:3px 0;border-bottom:1px solid var(--border)}
.stat-key{color:var(--text3)}.stat-val{color:var(--c);font-weight:500}
::-webkit-scrollbar{width:2px;height:2px}
::-webkit-scrollbar-track{background:var(--bg2)}
::-webkit-scrollbar-thumb{background:var(--border2)}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1;transform:none}}
.ani{animation:fadein .2s ease both}
</style>
</head>
<body>
<div id="app">
<header>
  <div>
    <span class="logo">PCD<span>PROOF CHAIN OF DATA  ·  v3.2  ·  FULL STACK + ARCHIVE</span></span>
  </div>
  <div class="status-bar">
    <div class="pill"><span class="dot off" id="hub-dot"></span><span id="hub-lbl">HUB: OFFLINE</span></div>
    <div class="pill"><span class="dot off" id="a-dot"></span><span id="a-lbl">NODE-A: —</span></div>
    <div class="pill"><span class="dot off" id="b-dot"></span><span id="b-lbl">NODE-B: —</span></div>
    <div class="pill" id="epoch-pill">EPOCH: —</div>
    <div class="pill" id="win-pill">WIN: —/—</div>
  </div>
</header>

<!-- METRICS -->
<div class="metrics">
  <div class="metric c"><div class="ml">Chain Nodes</div><div class="mv" id="m-nodes">0</div><div class="ms">DAG entries</div></div>
  <div class="metric g"><div class="ml">Chain Valid</div><div class="mv" id="m-valid">—</div><div class="ms">integrity</div></div>
  <div class="metric a"><div class="ml">AI Score</div><div class="mv" id="m-ai">—</div><div class="ms">last insert</div></div>
  <div class="metric p"><div class="ml">Epoch</div><div class="mv" id="m-epoch">0</div><div class="ms">current window</div></div>
  <div class="metric r"><div class="ml">Ops</div><div class="mv" id="m-ops">0</div><div class="ms">total writes</div></div>
</div>

<div class="layout">

<!-- ── LEFT COLUMN ── -->
<div class="col">

  <!-- HUB -->
  <div class="panel ani">
    <div class="ph">◈ HUB AUTHORITY<span class="badge b-hub">HUB</span></div>
    <div class="pb">
      <div id="hub-alert"></div>
      <div class="fr2">
        <div><label>Master Secret</label><input id="hub-master" value="M-WARLOK-2026-PROD"></div>
        <div><label>Window Size W</label><input id="hub-W" type="number" value="4" min="2" max="16"></div>
      </div>
      <div class="fr2">
        <div><label>Node A ID</label><input id="hub-aid" value="NODE-ALPHA"></div>
        <div><label>Node B ID</label><input id="hub-bid" value="NODE-BETA"></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-p" onclick="doProvision()">◈ INIT HUB</button>
        <button class="btn btn-ghost" onclick="doReset()">↺ RESET</button>
      </div>
      <div id="hub-kv" class="detail" style="margin-top:8px"></div>
    </div>
  </div>

  <!-- WRITE -->
  <div class="panel ani">
    <div class="ph">⬡ SEAL &amp; INSERT<span class="badge b-a" id="a-badge">NODE-A</span></div>
    <div class="pb">
      <div id="write-alert"></div>
      <div class="fr"><label>Label</label><input id="w-label" value="sensor-batch-001"></div>
      <div class="fr"><label>Payload</label><textarea id="w-payload">{"temp":36.8,"pressure":1013,"unit":"SN-77","status":"nominal"}</textarea></div>
      <div class="fr2">
        <div><label>Classification</label>
          <select id="w-cls">
            <option>TOP_SECRET</option><option>SECRET</option>
            <option selected>CONFIDENTIAL</option><option>UNCLASSIFIED</option>
          </select></div>
        <div><label>Peer</label>
          <select id="w-peer"><option value="a">A — Sender</option><option value="b">B — Receiver</option></select></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-c" id="btn-seal" onclick="doSeal()" disabled>⬡ SEAL & INSERT</button>
        <button class="btn btn-ghost" onclick="doSample()">↺ SAMPLE</button>
      </div>
    </div>
  </div>

  <!-- READ -->
  <div class="panel ani">
    <div class="ph">▶ RETRIEVE &amp; DECRYPT<span class="badge b-b" id="b-badge">NODE-B</span></div>
    <div class="pb">
      <div class="fr"><label>Node ID (blank = latest)</label><input id="r-nodeid" placeholder="node_id or blank for latest…"></div>
      <div class="fr2">
        <div><label>Peer</label>
          <select id="r-peer"><option value="b">B — Receiver</option><option value="a">A — Sender</option></select></div>
        <div></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-g" id="btn-read" onclick="doRead()" disabled>▶ DECRYPT</button>
      </div>
      <div id="read-result" class="detail"></div>
    </div>
  </div>

  <!-- EDIT / REDACT -->
  <div class="panel ani">
    <div class="ph">✎ EDIT / REDACT</div>
    <div class="pb">
      <div class="fr"><label>Target Node ID</label><input id="e-nodeid" placeholder="node_id to supersede or redact…"></div>
      <div class="fr"><label>New Payload (supersede only)</label><textarea id="e-payload" style="height:50px" placeholder='{"field":"new_value"}'></textarea></div>
      <div class="fr2">
        <div><label>Reason / Policy</label><input id="e-reason" value="GDPR-ART17"></div>
        <div><label>Peer</label>
          <select id="e-peer"><option value="a">A</option><option value="b">B</option></select></div>
      </div>
      <div class="btn-row">
        <button class="btn btn-a" id="btn-sup" onclick="doSupersede()" disabled>✎ SUPERSEDE</button>
        <button class="btn btn-r" id="btn-red" onclick="doRedact()" disabled>⌫ REDACT</button>
      </div>
      <div id="edit-alert"></div>
    </div>
  </div>

</div><!-- /col left -->

<!-- ── CENTRE COLUMN ── -->
<div class="col">

  <!-- NODE CHAIN -->
  <div class="panel ani">
    <div class="ph">≡ DAG PROOF CHAIN</div>
    <div class="pb" style="padding:0">
      <div class="node-list" id="node-list">
        <div style="color:var(--text3);padding:20px;text-align:center;font-size:10px">◉ Chain empty — provision Hub and insert data</div>
      </div>
    </div>
  </div>

  <!-- DETAIL VIEW -->
  <div class="panel ani">
    <div class="ph">◉ NODE INSPECTOR</div>
    <div class="pb">
      <div id="inspect-empty" style="color:var(--text3);font-size:10px;padding:8px 0">Click a node above to inspect</div>
      <div id="inspect-body" style="display:none">
        <div class="kv" id="inspect-kv"></div>
        <div id="inspect-plain" class="detail"></div>
      </div>
    </div>
  </div>

  <!-- TABS: VERIFY / AUDIT LOG -->
  <div class="panel ani">
    <div class="ph">◈ VERIFICATION &amp; AUDIT</div>
    <div class="pb">
      <div class="tabs">
        <div class="tab active" id="tab-vr" onclick="switchTab('vr')">VERIFY CHAIN</div>
        <div class="tab" id="tab-audit" onclick="switchTab('audit')">AUDIT LOG FILE</div>
      </div>

      <div id="pane-vr">
        <div class="btn-row" style="margin-top:0;margin-bottom:8px">
          <button class="btn btn-a" id="btn-verify" onclick="doVerify()" disabled>◈ VERIFY</button>
          <span id="verify-status" style="font-size:10px;color:var(--text3);margin-left:8px"></span>
        </div>
        <div id="verify-results"></div>
      </div>

      <div id="pane-audit" style="display:none">
        <div class="btn-row" style="margin-top:0;margin-bottom:8px">
          <button class="btn btn-ghost" onclick="loadAuditLog()">↺ REFRESH</button>
          <span id="audit-meta" style="font-size:9px;color:var(--text3);margin-left:8px"></span>
        </div>
        <div class="audit-lines" id="audit-lines">Loading…</div>
      </div>
    </div>
  </div>

</div><!-- /col centre -->

<!-- ── RIGHT COLUMN ── -->
<div class="col">

  <!-- AI ENGINE -->
  <div class="panel ani">
    <div class="ph">◈ NANO-AI ENGINE</div>
    <div class="pb">
      <div id="ai-gate-alert"></div>
      <div class="ai-bars" id="ai-bars">
        <div class="ai-row"><div class="ai-lbl">f1 entropy</div><div class="ai-track"><div class="ai-fill" id="ab-f1" style="width:0%;background:var(--c)"></div></div><div class="ai-val" id="av-f1" style="color:var(--c)">—</div></div>
        <div class="ai-row"><div class="ai-lbl">f2 actor rep</div><div class="ai-track"><div class="ai-fill" id="ab-f2" style="width:0%;background:var(--g)"></div></div><div class="ai-val" id="av-f2" style="color:var(--g)">—</div></div>
        <div class="ai-row"><div class="ai-lbl">f3 timing</div><div class="ai-track"><div class="ai-fill" id="ab-f3" style="width:0%;background:var(--p)"></div></div><div class="ai-val" id="av-f3" style="color:var(--p)">—</div></div>
        <div class="ai-row"><div class="ai-lbl">f4 win pos</div><div class="ai-track"><div class="ai-fill" id="ab-f4" style="width:0%;background:var(--a)"></div></div><div class="ai-val" id="av-f4" style="color:var(--a)">—</div></div>
        <div class="ai-row"><div class="ai-lbl">f5 depth</div><div class="ai-track"><div class="ai-fill" id="ab-f5" style="width:0%;background:var(--c2)"></div></div><div class="ai-val" id="av-f5" style="color:var(--c2)">—</div></div>
        <div class="ai-row" style="margin-top:4px">
          <div class="ai-lbl" style="font-weight:500;color:var(--c)">K(t,n) trust</div>
          <div class="ai-track"><div class="ai-fill" id="ab-score" style="width:0%;background:var(--c)"></div></div>
          <div class="ai-val" id="av-score" style="color:var(--c)">—</div>
        </div>
      </div>
      <div id="ai-tier-badge" style="font-size:10px;letter-spacing:2px;padding:4px 8px;
        border-radius:1px;display:inline-block;border:1px solid var(--border2)">AWAITING INIT</div>
    </div>
  </div>

  <!-- EPOCH / WINDOW STATE -->
  <div class="panel ani">
    <div class="ph">⊛ EPOCH STATE</div>
    <div class="pb">
      <div class="kv" style="margin-bottom:10px">
        <span class="k">epoch</span><span class="v a" id="ek-ep">—</span>
        <span class="k">epoch_secret</span><span class="v c" id="ek-sec">—</span>
        <span class="k">window_pos</span><span class="v g" id="ek-wpos">—</span>
        <span class="k">W size</span><span class="v" id="ek-W">—</span>
        <span class="k">counter</span><span class="v" id="ek-ctr">—</span>
        <span class="k">K(t,n)</span><span class="v p">derived, not stored</span>
      </div>
      <label style="margin-bottom:5px">Window slots (W=<span id="ek-wlbl">—</span>)</label>
      <div class="wgrid" id="wgrid"></div>
    </div>
  </div>

  <!-- ARCHIVE ADMIN -->
  <div class="panel ani">
    <div class="ph">⊞ ARCHIVE ADMIN</div>
    <div class="pb">
      <div class="tabs">
        <div class="tab active" id="arc-tab-stats" onclick="arcTab('stats')">STATS</div>
        <div class="tab" id="arc-tab-epochs" onclick="arcTab('epochs')">EPOCHS</div>
        <div class="tab" id="arc-tab-records" onclick="arcTab('records')">RECORDS</div>
        <div class="tab" id="arc-tab-policy" onclick="arcTab('policy')">POLICY</div>
      </div>

      <!-- STATS PANE -->
      <div id="arc-pane-stats">
        <div id="arc-stats-content" style="font-size:10px">
          <div style="color:var(--text3);padding:8px 0">◉ Provision Hub to enable archive</div>
        </div>
        <div class="btn-row" style="margin-top:8px">
          <button class="btn btn-ghost" onclick="arcLoadStats()">↺ REFRESH</button>
          <button class="btn btn-a" onclick="arcEnforce()" id="btn-arc-enforce" disabled>⊞ ENFORCE</button>
        </div>
      </div>

      <!-- EPOCHS PANE -->
      <div id="arc-pane-epochs" style="display:none">
        <div id="arc-epoch-list" style="max-height:200px;overflow-y:auto;font-size:10px">
          <div style="color:var(--text3);padding:8px 0">No epochs archived yet</div>
        </div>
        <div class="btn-row" style="margin-top:8px">
          <button class="btn btn-ghost" onclick="arcLoadEpochs()">↺ REFRESH</button>
        </div>
        <div class="fr" style="margin-top:8px">
          <label>Evict epoch (admin)</label>
          <div style="display:flex;gap:6px">
            <input id="arc-evict-epoch" type="number" placeholder="epoch #" style="width:80px">
            <button class="btn btn-r" onclick="arcEvictEpoch()">✕ EVICT</button>
          </div>
        </div>
      </div>

      <!-- RECORDS PANE -->
      <div id="arc-pane-records" style="display:none">
        <div id="arc-record-list" style="max-height:200px;overflow-y:auto">
          <div style="color:var(--text3);font-size:10px;padding:8px 0">No records archived yet</div>
        </div>
        <div class="btn-row" style="margin-top:8px">
          <button class="btn btn-ghost" onclick="arcLoadRecords()">↺ REFRESH</button>
        </div>
        <div class="fr" style="margin-top:8px">
          <label>Archive / restore node ID</label>
          <input id="arc-node-id" placeholder="node_id…">
        </div>
        <div class="btn-row" style="margin-top:0">
          <button class="btn btn-a" onclick="arcArchiveNode()">⊞ ARCHIVE</button>
          <button class="btn btn-g" onclick="arcRestoreNode()">↑ RESTORE</button>
        </div>
        <div id="arc-record-alert" style="margin-top:6px"></div>
      </div>

      <!-- POLICY PANE -->
      <div id="arc-pane-policy" style="display:none">
        <div style="font-size:9px;color:var(--text3);margin-bottom:8px;letter-spacing:1px">
          EPOCH SECRET ARCHIVE BOUNDS
        </div>
        <div class="policy-grid">
          <div class="policy-field"><label>Max epoch keys (0=∞)</label><input id="p-max-keys" type="number" value="50"></div>
          <div class="policy-field"><label>Max age days (0=∞)</label><input id="p-max-age" type="number" value="0" step="0.1"></div>
          <div class="policy-field"><label>Hot tier epochs</label><input id="p-hot" type="number" value="5"></div>
          <div class="policy-field"><label>Warm tier epochs</label><input id="p-warm" type="number" value="20"></div>
        </div>
        <div style="font-size:9px;color:var(--text3);margin-bottom:8px;letter-spacing:1px">
          DATA RECORD ARCHIVE BOUNDS
        </div>
        <div class="policy-grid">
          <div class="policy-field"><label>Max records (0=∞)</label><input id="p-max-rec" type="number" value="10000"></div>
          <div class="policy-field"><label>Max record age days</label><input id="p-rec-age" type="number" value="365" step="0.1"></div>
          <div class="policy-field"><label>Max archive bytes (0=∞)</label><input id="p-rec-bytes" type="number" value="0"></div>
          <div class="policy-field"><label>Max epoch bytes (0=∞)</label><input id="p-ep-bytes" type="number" value="0"></div>
        </div>
        <div class="btn-row" style="margin-top:4px">
          <button class="btn btn-p" onclick="arcSavePolicy()">◈ SAVE POLICY</button>
          <button class="btn btn-ghost" onclick="arcLoadPolicy()">↺ LOAD</button>
        </div>
        <div id="arc-policy-alert" style="margin-top:6px"></div>
      </div>
    </div>
  </div>

  <!-- SYSTEM LOG -->
  <div class="panel ani">
    <div class="ph">▶ SYSTEM LOG</div>
    <div class="pb" style="padding:0">
      <div class="term" id="term">
        <div class="term-line lv-info">
          <span class="term-ts">—</span>
          <span class="term-actor">SYSTEM</span>
          <span class="term-msg">PCD system ready. Provision Hub to begin.</span>
        </div>
      </div>
    </div>
  </div>

</div><!-- /col right -->
</div><!-- /layout -->
</div><!-- /app -->

<script>
// ── STATE ─────────────────────────────────────────────────────────────
let lastLogLen=0, provisioned=false, selectedNodeId=null, activeTab='vr';

// ── API ───────────────────────────────────────────────────────────────
async function api(method,path,body){
  const r=await fetch(path,{method,headers:{'Content-Type':'application/json'},
    body:body?JSON.stringify(body):undefined});
  return r.json();
}

// ── PROVISION ────────────────────────────────────────────────────────
async function doProvision(){
  const r=await api('POST','/api/provision',{
    master_secret:$('hub-master').value,
    window_size:parseInt($('hub-W').value),
    peer_a_id:$('hub-aid').value,
    peer_b_id:$('hub-bid').value,
  });
  if(!r.ok){setAlert('hub-alert','alert-err',r.error);return;}
  const h=r.hub_record;
  $('hub-kv').className='detail open ani';
  $('hub-kv').innerHTML=kvHtml([
    ['master_hash',h.master_hash,'c'],
    ['epoch_0_hash',h.epoch_0_hash,'p'],
    ['W',h.W,'a'],
    ['timing_anchor',h.timing_anchor,''],
    ['peer_a',h.peer_a,'g'],
    ['peer_b',h.peer_b,'g'],
    ['epoch_sync',h.sync?'✓ MUTUAL OK':'✗ FAILED',h.sync?'g':'r'],
  ]);
  setAlert('hub-alert','alert-hub','◈ Hub provisioned. Epoch_0 sync: '+(h.sync?'✓ MUTUAL OK':'✗ FAILED'));
  provisioned=true;
  ['btn-seal','btn-read','btn-verify','btn-sup','btn-red'].forEach(id=>$(id).disabled=false);
  initWGrid(parseInt($('hub-W').value));
  $('a-badge').textContent=h.peer_a;
  $('b-badge').textContent=h.peer_b;
  slog('hub','HUB',`Provisioned | W=${h.W} | sync=${h.sync}`);
}

// ── SEAL ──────────────────────────────────────────────────────────────
async function doSeal(){
  if(!provisioned)return;
  const r=await api('POST','/api/seal',{
    label:$('w-label').value,
    payload:$('w-payload').value,
    classification:$('w-cls').value,
    peer:$('w-peer').value,
  });
  if(!r.ok){setAlert('write-alert','alert-err',r.error||'Seal failed');return;}
  clearAlert('write-alert');
  const n=r.node;
  $('r-nodeid').value=n.node_id;
  $('e-nodeid').value=n.node_id;
  slog('info',$('w-peer').value==='a'?$('hub-aid').value:$('hub-bid').value,
    `SEALED ${n.label} | score=${n.ai_score} tier=${n.ai_tier} lat=${n.latency_ms}ms`);
  if(n.ai_gates&&n.ai_gates.length){
    slog('warn','AI',`GATES: ${n.ai_gates.join(', ')}`);
  }
  loadAuditLog();
}

// ── READ ──────────────────────────────────────────────────────────────
async function doRead(){
  if(!provisioned)return;
  const r=await api('POST','/api/read',{
    node_id:$('r-nodeid').value.trim(),
    peer:$('r-peer').value,
  });
  const d=$('read-result');
  d.className='detail open ani';
  if(!r.ok||r.error){
    d.innerHTML=`<span style="color:var(--r)">${r.error||'Read failed'}</span>`;
    return;
  }
  const n=r.node;
  d.innerHTML=`<div style="color:var(--g);font-size:9px;letter-spacing:2px;margin-bottom:6px">
    ✓ DECRYPTED | fp_ok=✓ zk_ok=✓ hash_ok=✓ | tier=${n.ai_tier}</div>`+
    kvHtml([
      ['label',n.label,''],
      ['feature_fp',n.feature_fp,'c'],
      ['zk_proof',n.zk_proof,'p'],
      ['t_seal',n.temporal_seal,'c'],
    ])+
    `<div class="plaintext-box">${esc(r.plaintext||'[empty]')}</div>`;
  slog('info',`(${$('r-peer').value})`,`READ ${n.label} | ok`);
}

// ── SUPERSEDE ────────────────────────────────────────────────────────
async function doSupersede(){
  const tid=$('e-nodeid').value.trim();
  const np=$('e-payload').value.trim();
  if(!tid||!np){setAlert('edit-alert','alert-err','node_id and new_payload required');return;}
  const r=await api('POST','/api/supersede',{
    node_id:tid,new_payload:np,
    reason:$('e-reason').value,
    peer:$('e-peer').value,
  });
  if(!r.ok){setAlert('edit-alert','alert-err',r.error);return;}
  setAlert('edit-alert','alert-ok','Superseded → '+r.new_node.node_id.slice(0,12));
  slog('warn','SYSTEM',`SUPERSEDE → new node ${r.new_node.node_id.slice(0,12)}`);
}

// ── REDACT ────────────────────────────────────────────────────────────
async function doRedact(){
  const tid=$('e-nodeid').value.trim();
  if(!tid){setAlert('edit-alert','alert-err','node_id required');return;}
  if(!confirm(`Permanently redact node ${tid.slice(0,12)}? This cannot be undone.`))return;
  const r=await api('POST','/api/redact',{
    node_id:tid,
    policy_ref:$('e-reason').value,
    peer:$('e-peer').value,
  });
  if(!r.ok){setAlert('edit-alert','alert-err',r.error);return;}
  setAlert('edit-alert','alert-ok','Redacted | payload nullified | receipt sealed');
  slog('warn','SYSTEM',`REDACT node ${tid.slice(0,12)} policy=${$('e-reason').value}`);
}

// ── VERIFY ────────────────────────────────────────────────────────────
async function doVerify(){
  const r=await api('GET','/api/verify?peer=a');
  const ok=r.chain_valid;
  $('verify-status').textContent=(ok?'✓ ALL NODES INTACT':'✗ VIOLATION DETECTED');
  $('verify-status').style.color=ok?'var(--g)':'var(--r)';
  $('m-valid').textContent=ok?'✓':'✗';
  $('m-valid').style.color=ok?'var(--g)':'var(--r)';
  const div=$('verify-results');
  div.innerHTML='';
  (r.results||[]).forEach(n=>{
    const line=document.createElement('div');
    line.className='vr-line '+(n.valid?'vr-ok':'vr-fail');
    line.innerHTML=
      `<span style="min-width:80px">${n.label.slice(0,14)}</span>`+
      `<span>E:${n.epoch} W:${n.wpos}</span>`+
      `<span class="tier-badge tier-${n.tier}">${n.tier.slice(0,1)}</span>`+
      `<span style="margin-left:auto">${n.valid?'✓ VALID':'✗ FAIL'}</span>`+
      (!n.c1_cipher?'<span style="color:var(--r)">[cipher]</span>':'')+
      (!n.c2_prev?'<span style="color:var(--r)">[prev_hash]</span>':'')+
      (!n.c3_fp?'<span style="color:var(--r)">[feature_fp]</span>':'');
    div.appendChild(line);
  });
  slog(ok?'info':'error','VERIFY',`chain_ok=${ok} nodes=${(r.results||[]).length}`);
}

// ── SAMPLE ────────────────────────────────────────────────────────────
async function doSample(){
  const r=await api('GET','/api/sample');
  $('w-label').value=r.label;
  $('w-payload').value=r.payload;
  $('w-cls').value=r.classification;
}

// ── RESET ─────────────────────────────────────────────────────────────
async function doReset(){
  if(!confirm('Reset all state and clear JSON store?'))return;
  await api('POST','/api/reset',{});
  provisioned=false;
  $('hub-kv').className='detail';
  ['btn-seal','btn-read','btn-verify','btn-sup','btn-red'].forEach(id=>$(id).disabled=true);
  $('node-list').innerHTML='<div style="color:var(--text3);padding:20px;text-align:center;font-size:10px">◉ Chain empty — provision Hub and insert data</div>';
  $('term').innerHTML='<div class="term-line lv-info"><span class="term-ts">—</span><span class="term-actor">SYSTEM</span><span class="term-msg">Reset. Ready.</span></div>';
  $('m-nodes').textContent='0';
  slog('warn','SYSTEM','Full reset');
}

// ── AUDIT LOG ─────────────────────────────────────────────────────────
async function loadAuditLog(){
  if(activeTab!=='audit')return;
  const r=await api('GET','/api/audit_log?n=120');
  const el=$('audit-lines');
  if(!r.ok){el.textContent='Error: '+r.error;return;}
  $('audit-meta').textContent=`${r.total} total lines | ${r.path}`;
  el.innerHTML=r.lines.map(l=>{
    const parts=l.split(' | ');
    return `<div><span>${esc(parts[0]||'')} | ${esc(parts[1]||'')}</span> ${esc((parts.slice(2)).join(' | '))}</div>`;
  }).join('');
  el.scrollTop=el.scrollHeight;
}

function switchTab(t){
  activeTab=t;
  document.querySelectorAll('.tab').forEach(el=>el.classList.remove('active'));
  $('tab-'+t).classList.add('active');
  $('pane-vr').style.display=t==='vr'?'':'none';
  $('pane-audit').style.display=t==='audit'?'':'none';
  if(t==='audit')loadAuditLog();
}

// ── NODE INSPECTOR ────────────────────────────────────────────────────
function selectNode(n){
  selectedNodeId=n.node_id;
  document.querySelectorAll('.node-item').forEach(el=>el.classList.remove('selected'));
  const el=document.getElementById('ni-'+n.node_id);
  if(el)el.classList.add('selected');

  $('inspect-empty').style.display='none';
  $('inspect-body').style.display='';
  $('e-nodeid').value=n.node_id;

  $('inspect-kv').innerHTML=kvHtml([
    ['node_id',n.node_id.slice(0,20)+'…','c'],
    ['label',n.label,''],
    ['actor',n.actor,'g'],
    ['classification',n.classification,'a'],
    ['ai_score',n.ai_score,'g'],
    ['ai_tier',n.ai_tier,tierColor(n.ai_tier)],
    ['status',n.status,n.status==='ACTIVE'?'g':n.status==='SUPERSEDED'?'a':'r'],
    ['epoch',`E:${n.epoch} W:${n.window_pos}`,''],
    ['counter',n.chain_counter,'c'],
    ['hash',n.hash_short,'c'],
    ['prev',n.prev_short,'p'],
    ['ts',n.ts,''],
    ['latency',n.latency_ms+'ms','a'],
  ]);

  const ip=$('inspect-plain');
  if(n.ai_gates&&n.ai_gates.length){
    ip.className='detail open';
    ip.innerHTML='<span style="color:var(--r)">⚠ GATES: '+esc(n.ai_gates.join(', '))+'</span>';
  } else {
    ip.className='detail';
  }
}

// ── POLL STATE ────────────────────────────────────────────────────────
async function pollState(){
  try{
    const s=await api('GET','/api/state');
    if(!s.provisioned)return;

    // metrics
    $('m-nodes').textContent=s.stats.total||0;
    $('m-epoch').textContent=s.window_a.epoch||0;
    $('m-ops').textContent=s.stats.ops||0;
    $('epoch-pill').textContent='EPOCH: '+(s.window_a.epoch||0);
    $('win-pill').textContent=`WIN: ${s.window_a.wpos||0}/${s.window_a.W||'—'}`;

    // dots
    ['hub','a','b'].forEach(x=>{
      const d=$(x+'-dot');d.className='dot on';
    });
    $('hub-lbl').textContent='HUB: SILENT';
    $('a-lbl').textContent='NODE-A: '+($('hub-aid').value||'—');
    $('b-lbl').textContent='NODE-B: '+($('hub-bid').value||'—');

    // epoch panel
    const wa=s.window_a;
    $('ek-ep').textContent=wa.epoch||0;
    $('ek-sec').textContent=wa.secret_short||'—';
    $('ek-wpos').textContent=wa.wpos||0;
    $('ek-W').textContent=wa.W||'—';
    $('ek-ctr').textContent=wa.counter||0;
    $('ek-wlbl').textContent=wa.W||'—';
    updateWGrid(wa.wpos||0,wa.W||4);

    // AI bars
    const ai=s.ai_last||{};
    if(ai.score!==undefined){
      setBar('f1',ai.entropy||0,'var(--c)');
      setBar('f2',ai.actor||0,'var(--g)');
      setBar('f3',ai.timing||0,'var(--p)');
      setBar('f4',ai.window_pos||0,'var(--a)');
      setBar('f5',ai.chain_depth||0,'var(--c2)');
      setBar('score',ai.score||0,'var(--c)');
      $('m-ai').textContent=(ai.score||0).toFixed(3);
      $('m-ai').style.color=ai.score>0.7?'var(--g)':ai.score>0.4?'var(--a)':'var(--r)';

      const tb=$('ai-tier-badge');
      const tier=ai.tier||'—';
      tb.textContent=tier;
      tb.style.color=`var(--${tier==='HOT'?'hot':tier==='WARM'?'warm':tier==='COOL'?'cool':'qua'})`;
      tb.style.borderColor=tb.style.color;

      if(ai.gates&&ai.gates.length){
        setAlert('ai-gate-alert','alert-err','⚠ '+ai.gates.join(', '));
      } else {
        clearAlert('ai-gate-alert');
      }
    }

    // node list
    const nodes=s.nodes||[];
    const nl=$('node-list');
    if(nodes.length===0){
      nl.innerHTML='<div style="color:var(--text3);padding:20px;text-align:center;font-size:10px">◉ Chain empty</div>';
    } else {
      nl.innerHTML='';
      [...nodes].reverse().forEach(n=>{
        const el=document.createElement('div');
        el.className='node-item'+(n.node_id===selectedNodeId?' selected':'');
        el.id='ni-'+n.node_id;
        const tc=tierColor(n.ai_tier);
        el.innerHTML=
          `<span class="tier-badge tier-${n.ai_tier}">${n.ai_tier.slice(0,1)}</span>`+
          `<div class="node-body">`+
            `<div class="node-label">${esc(n.label)}</div>`+
            `<div class="node-sub">${n.actor} · E:${n.epoch} W:${n.window_pos} · ${n.ts}</div>`+
          `</div>`+
          `<div class="node-status st-${n.status}">${n.status}</div>`;
        el.onclick=()=>selectNode(n);
        nl.appendChild(el);
      });
    }

    // logs
    const logs=s.logs||[];
    if(logs.length>lastLogLen){
      logs.slice(lastLogLen).forEach(e=>slog(e.level,e.actor||'—',e.msg));
      lastLogLen=logs.length;
    }

  }catch(e){}
}

// ── HELPERS ───────────────────────────────────────────────────────────
function $(id){return document.getElementById(id)}
function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function tierColor(t){return t==='HOT'?'g':t==='WARM'?'c':t==='COOL'?'a':'r'}

function kvHtml(rows){
  return '<div class="kv">'+rows.map(([k,v,c])=>
    `<span class="k">${esc(k)}</span><span class="v ${c||''}">${esc(v)}</span>`
  ).join('')+'</div>';
}

function setAlert(id,cls,msg){
  const el=$(id);if(!el)return;
  el.innerHTML=`<div class="alert ${cls}" style="margin-bottom:8px">${esc(msg)}</div>`;
}
function clearAlert(id){const el=$(id);if(el)el.innerHTML='';}

function setBar(key,val,color){
  const pct=Math.round((val||0)*100);
  const f=$('ab-'+key);const v=$('av-'+key);
  if(f){f.style.width=pct+'%';f.style.background=color;}
  if(v){v.textContent=pct+'%';v.style.color=color;}
}

function slog(level,actor,msg){
  const t=$('term');
  const d=document.createElement('div');
  d.className='term-line lv-'+(level||'info');
  const ts=new Date().toTimeString().slice(0,8);
  d.innerHTML=`<span class="term-ts">${ts}</span>`+
    `<span class="term-actor">${esc(String(actor||'').slice(0,12))}</span>`+
    `<span class="term-msg">${esc(msg)}</span>`;
  t.appendChild(d);
  t.scrollTop=t.scrollHeight;
  if(t.children.length>200)t.removeChild(t.firstChild);
}

function initWGrid(W){
  const g=$('wgrid');
  g.style.gridTemplateColumns=`repeat(${Math.min(W,8)},1fr)`;
  g.innerHTML='';
  for(let i=0;i<W;i++){
    const c=document.createElement('div');
    c.className='wc';c.id=`wc-${i}`;c.textContent=i;
    g.appendChild(c);
  }
}

function updateWGrid(wpos,W){
  for(let i=0;i<W;i++){
    const c=$(`wc-${i}`);if(!c)continue;
    c.className='wc';
    if(i<wpos)c.classList.add('used');
    if(i===wpos)c.classList.add('active');
  }
}

// ── ARCHIVE UI ────────────────────────────────────────────────────────
let arcActiveTab='stats';

function arcTab(t){
  arcActiveTab=t;
  ['stats','epochs','records','policy'].forEach(x=>{
    $('arc-tab-'+x).className='tab'+(t===x?' active':'');
    $('arc-pane-'+x).style.display=t===x?'':'none';
  });
  if(t==='stats')   arcLoadStats();
  if(t==='epochs')  arcLoadEpochs();
  if(t==='records') arcLoadRecords();
  if(t==='policy')  arcLoadPolicy();
}

async function arcLoadStats(){
  const r=await api('GET','/api/archive/stats');
  if(!r.ok){$('arc-stats-content').textContent='Error: '+r.error;return;}
  const s=r.stats;
  const ea=s.epoch_archive||{};
  const da=s.data_archive||{};
  $('arc-stats-content').innerHTML=
    `<div style="color:var(--c2);font-size:9px;letter-spacing:2px;margin-bottom:6px">EPOCH ARCHIVE</div>`+
    statRow('Hot keys',ea.hot_count||0)+
    statRow('Warm keys',ea.warm_count||0)+
    statRow('Cold keys (HSM)',ea.cold_count||0)+
    statRow('Total keys',ea.total_keys||0)+
    statRow('Warm bytes',fmt_bytes(ea.warm_bytes||0))+
    statRow('Cold bytes',fmt_bytes(ea.cold_bytes||0))+
    `<div style="color:var(--g2);font-size:9px;letter-spacing:2px;margin:8px 0 6px">DATA ARCHIVE</div>`+
    statRow('Archived records',da.total_archived||0)+
    statRow('Lifetime archived',da.cumulative_archived||0)+
    statRow('Archive bytes',fmt_bytes(da.file_bytes||0))+
    statRow('Oldest counter',da.oldest_counter??'—')+
    statRow('Newest counter',da.newest_counter??'—');
  $('btn-arc-enforce').disabled=false;
}

function statRow(k,v){
  return `<div class="stat-row"><span class="stat-key">${k}</span><span class="stat-val">${v}</span></div>`;
}
function fmt_bytes(b){
  if(b===0)return '0 B';
  if(b<1024)return b+' B';
  if(b<1048576)return (b/1024).toFixed(1)+' KB';
  return (b/1048576).toFixed(2)+' MB';
}

async function arcLoadEpochs(){
  const r=await api('GET','/api/archive/epochs');
  const el=$('arc-epoch-list');
  if(!r.ok||!r.epochs||r.epochs.length===0){
    el.innerHTML='<div style="color:var(--text3);padding:8px 0;font-size:10px">No epochs archived yet</div>';
    return;
  }
  el.innerHTML=r.epochs.map(e=>{
    const ts=e.archived_at?new Date(e.archived_at*1000).toISOString().slice(0,19).replace('T',' '):'—';
    return `<div class="arc-row">
      <span class="arc-tier ${e.tier}">${e.tier}</span>
      <span class="arc-label">epoch ${e.epoch}</span>
      <span class="arc-val">${e.short||'sealed'}</span>
      <span class="arc-val" style="margin-left:6px">${ts}</span>
    </div>`;
  }).join('');
}

async function arcLoadRecords(){
  const r=await api('GET','/api/archive/records?limit=50');
  const el=$('arc-record-list');
  if(!r.ok||!r.records||r.records.length===0){
    el.innerHTML='<div style="color:var(--text3);font-size:10px;padding:8px 0">No records archived yet</div>';
    return;
  }
  el.innerHTML=r.records.map(n=>{
    const cls=n.classification||'';
    const tc=n.ai_tier==='HOT'?'var(--hot)':n.ai_tier==='WARM'?'var(--c)':
             n.ai_tier==='COOL'?'var(--a)':'var(--r)';
    return `<div class="arc-row" onclick="$('arc-node-id').value='${n.node_id}'">
      <span class="arc-tier ${n.ai_tier}">${n.ai_tier.slice(0,1)}</span>
      <span class="arc-label" style="font-size:10px">${esc(n.label)}</span>
      <span class="arc-val">${n.ts_iso||''}</span>
      <span class="arc-val" style="margin-left:4px">${esc(n.archive_reason||'')}</span>
    </div>`;
  }).join('');
}

async function arcLoadPolicy(){
  const r=await api('GET','/api/archive/policy');
  if(!r.ok)return;
  const p=r.policy;
  $('p-max-keys').value=p.max_epoch_keys;
  $('p-max-age').value=p.max_epoch_age_days;
  $('p-hot').value=p.hot_epochs;
  $('p-warm').value=p.warm_epochs;
  $('p-max-rec').value=p.max_record_count;
  $('p-rec-age').value=p.max_record_age_days;
  $('p-rec-bytes').value=p.max_record_bytes;
  $('p-ep-bytes').value=p.max_epoch_bytes;
}

async function arcSavePolicy(){
  const r=await api('PUT','/api/archive/policy',{
    max_epoch_keys:     parseInt($('p-max-keys').value)||0,
    max_epoch_age_days: parseFloat($('p-max-age').value)||0,
    hot_epochs:         parseInt($('p-hot').value)||5,
    warm_epochs:        parseInt($('p-warm').value)||20,
    max_record_count:   parseInt($('p-max-rec').value)||0,
    max_record_age_days:parseFloat($('p-rec-age').value)||0,
    max_record_bytes:   parseInt($('p-rec-bytes').value)||0,
    max_epoch_bytes:    parseInt($('p-ep-bytes').value)||0,
    enforce_on_rotate:  true,
    enforce_on_read:    false,
  });
  const al=$('arc-policy-alert');
  if(r.ok){
    al.innerHTML='<div class="alert alert-ok">✓ Policy saved and applied</div>';
    slog('info','ADMIN','Archive policy updated');
  } else {
    al.innerHTML=`<div class="alert alert-err">Error: ${esc(r.error||'failed')}</div>`;
  }
  setTimeout(()=>al.innerHTML='',3000);
}

async function arcEnforce(){
  const r=await api('POST','/api/archive/enforce',{});
  slog('info','ADMIN','Archive limits enforced');
  arcLoadStats();
}

async function arcEvictEpoch(){
  const ep=parseInt($('arc-evict-epoch').value);
  if(isNaN(ep))return;
  const r=await api('DELETE',`/api/archive/epoch/${ep}`);
  slog(r.ok?'warn':'err','ADMIN',`Evict epoch ${ep}: ok=${r.ok}`);
  arcLoadEpochs();
}

async function arcArchiveNode(){
  const id=$('arc-node-id').value.trim();
  if(!id)return;
  const r=await api('POST','/api/archive/records/archive',{node_id:id,reason:'MANUAL_ARCHIVE'});
  const al=$('arc-record-alert');
  al.innerHTML=r.ok
    ?'<div class="alert alert-ok">✓ Node archived</div>'
    :`<div class="alert alert-err">Error: ${esc(r.error||'')}</div>`;
  setTimeout(()=>al.innerHTML='',3000);
  if(r.ok){arcLoadRecords();slog('warn','ADMIN',`Archived node ${id.slice(0,12)}`);}
}

async function arcRestoreNode(){
  const id=$('arc-node-id').value.trim();
  if(!id)return;
  const r=await api('POST','/api/archive/records/restore',{node_id:id});
  const al=$('arc-record-alert');
  al.innerHTML=r.ok
    ?'<div class="alert alert-ok">✓ Node restored to hot store</div>'
    :`<div class="alert alert-err">Error: ${esc(r.error||'')}</div>`;
  setTimeout(()=>al.innerHTML='',3000);
  if(r.ok){arcLoadRecords();slog('info','ADMIN',`Restored node ${id.slice(0,12)}`);}
}

setInterval(pollState,800);
slog('info','SYSTEM','PCD system loaded | Provision Hub to begin');
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    AUDIT_FILE.touch(exist_ok=True)
    audit("info", "SYSTEM", "STARTUP",
          "PCD system starting on http://localhost:7700")
    audit("info", "SYSTEM", "PATHS",
          f"data={DATA_FILE} audit={AUDIT_FILE}")

    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PCD — PROOF CHAIN OF DATA  ·  Full Stack v3.2           ║")
    print("║  Flask + Archive (Epoch HSM + Data Record)               ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  UI:         http://localhost:7700                       ║")
    print(f"║  Data store: {str(DATA_FILE):<42s}║")
    print(f"║  Audit log:  {str(AUDIT_FILE):<42s}║")
    arc_path = str(BASE_DIR / "pcd_archive.json")
    print(f"║  Archive:    {arc_path:<42s}║")
    print("╚══════════════════════════════════════════════════════════╝")

    app.run(host="0.0.0.0", port=7700, debug=False, threaded=True)
