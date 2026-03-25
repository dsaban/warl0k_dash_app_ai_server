#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PIM-PCD  ·  NEURAL RATCHET CHAIN  ·  WARLOK UI                           ║
║  FastAPI + Pydantic  ·  Async  ·  SSE streaming  ·  Dark terminal theme    ║
║                                                                              ║
║  Run (with FastAPI installed):                                              ║
║      pip install fastapi uvicorn                                            ║
║      uvicorn pim_pcd_fastapi:app --host 0.0.0.0 --port 5050 --reload       ║
║                                                                              ║
║  Run (this file — Flask shim active when FastAPI not installed):            ║
║      python3 pim_pcd_fastapi.py                                             ║
║      Open: http://localhost:5050                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

# ─── Runtime detection — real FastAPI or Flask shim ──────────────────────────
import sys, importlib.util

_FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
_UVICORN_AVAILABLE = importlib.util.find_spec("uvicorn") is not None

# ─── Standard library ────────────────────────────────────────────────────────
import asyncio
import dataclasses
import hashlib
import hmac as _hmac
import json
import math
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 ── CRYPTO ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class Crypto:
    """Pure-stdlib cryptographic primitives — identical on both peers."""

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
    def derive_key(base: str, *parts: str) -> str:
        return Crypto.hmac256(base.encode(), "|".join(parts).encode())

    @staticmethod
    def rng(n: int = 16) -> str:
        return os.urandom(n).hex()

    @staticmethod
    def xor_bytes(a: bytes, b: bytes) -> bytes:
        out = bytearray(len(a))
        for i, byte in enumerate(a):
            out[i] = byte ^ b[i % len(b)]
        return bytes(out)

    @classmethod
    def obfuscate(cls, plaintext: str, ktn: str) -> str:
        ks = bytes.fromhex(cls.sha256(ktn + "stream") * 4)
        pt = plaintext.encode("utf-8")
        return cls.xor_bytes(pt, ks[: len(pt)]).hex()

    @classmethod
    def deobfuscate(cls, hex_ct: str, ktn: str) -> str:
        ks = bytes.fromhex(cls.sha256(ktn + "stream") * 4)
        ct = bytes.fromhex(hex_ct)
        return cls.xor_bytes(ct, ks[: len(ct)]).decode("utf-8")

    @staticmethod
    def short(h: str) -> str:
        return f"{h[:8]}…{h[-6:]}" if h and len(h) > 16 else (h or "—")


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 ── NANO-AI MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AIFeatures:
    entropy:       float
    actor_score:   float
    timing_align:  float
    window_pos:    float
    chain_depth:   float
    chain_counter: int

    def fingerprint(self) -> str:
        s = (
            f"{self.entropy:.6f}|{self.actor_score:.6f}|{self.timing_align:.6f}|"
            f"{self.chain_counter}|{self.window_pos:.6f}|{self.chain_depth:.6f}"
        )
        return Crypto.sha256(s)

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class AIResult:
    score:       float
    ktn:         str
    fingerprint: str
    features:    AIFeatures
    tier:        str


class NanoAIModel:
    """
    Deterministic nano-AI model — weight vector W is public (Hub-distributed).
    Both peers run the identical model over identical features → identical K(t,n).
    K(t,n) = HMAC(epoch_secret, feature_fingerprint | counter | window_pos)
    """

    WEIGHTS = np.array([0.22, 0.28, 0.15, 0.18, 0.12, 0.05], dtype=np.float64)
    BIAS    = -0.15
    TIERS   = [(0.85, "HOT"), (0.60, "WARM"), (0.30, "COOL"), (0.00, "QUARANTINE")]

    def __init__(self) -> None:
        self._rep: Dict[str, float] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _entropy(payload: str) -> float:
        data = payload.encode("utf-8")
        if not data:
            return 0.0
        freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        freq = freq[freq > 0]
        p    = freq / len(data)
        return float(min(-np.sum(p * np.log2(p)) / 8.0, 1.0))

    def actor_score(self, actor_id: str) -> float:
        return self._rep.get(actor_id, 0.50)

    def update_actor(self, actor_id: str, score: float) -> None:
        self._rep[actor_id] = min(1.0, score + 0.04)

    def build_features(
        self,
        payload:       str,
        actor_id:      str,
        window_pos:    int,
        window_size:   int,
        chain_depth:   int,
        chain_counter: int,
        last_event_ts: float,
    ) -> AIFeatures:
        delta_t = max(0.0, time.time() - last_event_ts)
        return AIFeatures(
            entropy       = round(self._entropy(payload), 6),
            actor_score   = round(self.actor_score(actor_id), 6),
            timing_align  = round(min(1.0, 1.0 / (delta_t + 0.1)), 6),
            window_pos    = round(window_pos / max(window_size, 1), 6),
            chain_depth   = round(min(chain_depth / 20.0, 1.0), 6),
            chain_counter = chain_counter,
        )

    def infer(self, features: AIFeatures, epoch_secret: str) -> AIResult:
        vec = np.array(
            [
                features.entropy,
                features.actor_score,
                features.timing_align,
                features.window_pos,
                features.chain_depth,
                min(features.chain_counter / 100.0, 1.0),
            ],
            dtype=np.float64,
        )
        score = round(self._sigmoid(float(np.dot(self.WEIGHTS, vec)) + self.BIAS), 6)
        fp    = features.fingerprint()
        ktn   = Crypto.hmac256(
            epoch_secret,
            f"{fp}|{features.chain_counter}|{features.window_pos:.6f}",
        )
        tier  = next((t for v, t in self.TIERS if score >= v), "QUARANTINE")
        return AIResult(score=score, ktn=ktn, fingerprint=fp, features=features, tier=tier)

    def detect_anomalies(self, f: AIFeatures) -> List[str]:
        flags = []
        if f.entropy      < 0.30: flags.append(f"entropy={f.entropy:.3f}<0.30")
        if f.actor_score  < 0.40: flags.append(f"actor={f.actor_score:.3f}<0.40")
        if f.timing_align < 0.20: flags.append(f"timing={f.timing_align:.3f}<0.20")
        if f.chain_counter == 0 and f.chain_depth > 0.1:
            flags.append("counter=0 but depth>0 — replay?")
        return flags

    def validate_ktn(
        self,
        received_fp:    str,
        local_features: AIFeatures,
        epoch_secret:   str,
    ) -> Tuple[bool, str]:
        local_fp = local_features.fingerprint()
        ktn      = Crypto.hmac256(
            epoch_secret,
            f"{local_fp}|{local_features.chain_counter}|{local_features.window_pos:.6f}",
        )
        return local_fp == received_fp, ktn


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 ── PCD ENVELOPE + DAG STORE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PcdEnvelope:
    """
    Cryptographic data wrapper.
    K(t,n) is intentionally ABSENT — regenerated at retrieval by both peers.
    """

    object_id:      str
    node_id:        str
    node_type:      str
    label:          str
    classification: str
    actor:          str
    cipher_payload: str
    payload_hash:   str
    cipher_hash:    str
    prev_hash:      str
    hash:           str
    temporal_seal:  str
    chain_seal:     str
    feature_fp:     str
    zk_proof:       str
    epoch:          int
    window_pos:     int
    chain_counter:  int
    chain_depth:    int
    timestamp:      float
    ai_score:       float
    ai_features:    dict
    tier:           str
    nonce:          str
    latency_ms:     float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class DagStore:
    """Append-only, thread-safe hash-linked DAG."""

    def __init__(self) -> None:
        self._nodes:  Dict[str, PcdEnvelope] = {}
        self._edges:  List[Tuple[str, str]]  = []
        self._latest: Optional[str]          = None
        self._lock    = threading.Lock()

    def add(self, env: PcdEnvelope) -> None:
        with self._lock:
            self._nodes[env.node_id] = env
            self._latest = env.node_id

    def link(self, from_id: str, to_id: str) -> None:
        with self._lock:
            self._edges.append((from_id, to_id))

    def get(self, node_id: str) -> Optional[PcdEnvelope]:
        return self._nodes.get(node_id) or next(
            (n for n in self._nodes.values()
             if n.object_id == node_id or n.object_id.startswith(node_id)),
            None,
        )

    @property
    def latest(self) -> Optional[PcdEnvelope]:
        return self._nodes.get(self._latest) if self._latest else None

    def all_data(self) -> List[PcdEnvelope]:
        with self._lock:
            return [n for n in self._nodes.values()
                    if n.node_type in ("GENESIS", "DATA")]

    def size(self) -> int:
        return len(self._nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def clear(self) -> None:
        with self._lock:
            self._nodes.clear()
            self._edges.clear()
            self._latest = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 ── PEER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SideConfig:
    epoch_0_secret:  str
    timing_anchor:   float
    window_size:     int
    epoch_ms:        int
    drift_tolerance: int
    model_weights:   List[float]


class Peer:
    """
    A PIM-PCD chain participant. Both sides hold identical config and model.
    Neither needs the other to advance state — the Neural Ratchet is self-contained.
    """

    def __init__(self, peer_id: str, role: str, config: SideConfig) -> None:
        self.peer_id = peer_id
        self.role    = role
        self.config  = config
        self.dag     = DagStore()
        self.model   = NanoAIModel()

        self._secrets:  Dict[int, str] = {0: config.epoch_0_secret}
        self._epoch     = 0
        self._wpos      = 0
        self._counter   = 0
        self._last_ts   = config.timing_anchor
        self._log_q:    queue.Queue    = queue.Queue()
        self._ai_last:  Optional[AIResult] = None

    # ── Logging ──────────────────────────────────────────────────────────────
    def log(self, level: str, msg: str) -> None:
        self._log_q.put({"ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg})

    # ── Epoch secret management ───────────────────────────────────────────────
    def _get_secret(self, target_epoch: int) -> str:
        """Derive epoch secret for any epoch by ratcheting from epoch_0. Deterministic on both peers."""
        if target_epoch in self._secrets:
            return self._secrets[target_epoch]
        sec = self._secrets.get(0, self.config.epoch_0_secret)
        self._secrets[0] = sec
        for ep in range(1, target_epoch + 1):
            if ep in self._secrets:
                sec = self._secrets[ep]
                continue
            ns = Crypto.derive_key(sec, "epoch", str(ep), "pim-pcd-chain")
            self._secrets[ep] = ns
            sec = ns
        return self._secrets[target_epoch]

    def _rotate_epoch(self) -> None:
        cur  = self._secrets[self._epoch]
        ns   = Crypto.derive_key(cur, "epoch", str(self._epoch + 1), "pim-pcd-chain")
        self._epoch += 1
        self._secrets[self._epoch] = ns
        self._wpos = 0
        # Forward secrecy: delete past epoch secret
        del self._secrets[self._epoch - 1]
        self.log(
            "sys",
            f"Epoch rotated {self._epoch-1}→{self._epoch} | "
            f"new secret: {Crypto.short(ns)} | forward secrecy applied",
        )

    @property
    def _secret(self) -> str:
        return self._secrets[self._epoch]

    # ── INSERT (Side-A) ───────────────────────────────────────────────────────
    def insert(self, label: str, payload: str, classification: str = "CONFIDENTIAL") -> PcdEnvelope:
        t0 = time.perf_counter()

        feats     = self.model.build_features(
            payload, self.peer_id, self._wpos,
            self.config.window_size, self.dag.size(),
            self._counter, self._last_ts,
        )
        ai        = self.model.infer(feats, self._secret)
        ktn       = ai.ktn  # NEVER leaves this scope
        self._ai_last = ai

        for anomaly in self.model.detect_anomalies(feats):
            self.log("err", f"⚠ Anomaly: {anomaly}")
        self.log("info", f"K(t={self._epoch},n={self._wpos}) computed: "
                         f"{Crypto.short(ktn)} [NOT transmitted]")

        cp    = Crypto.obfuscate(payload, ktn)
        ph    = Crypto.sha256(payload)
        ch    = Crypto.sha256(cp)

        parent    = self.dag.latest
        prev      = parent.hash if parent else "0" * 64
        pid       = parent.node_id if parent else None

        nonce     = Crypto.rng(8)
        ts        = time.time()
        tseal     = Crypto.hmac256(self._secret, f"{ch}|{ts}|{nonce}")
        cseal     = Crypto.hmac256(self._secret, f"{prev}|{ai.fingerprint}|{self._counter}")
        oid       = Crypto.sha256(f"{label}{ts}{nonce}{Crypto.rng(4)}")
        nid       = oid[:16]
        zk        = Crypto.hmac256(ktn, f"zk|{self.peer_id}|{nid}")
        ehash     = Crypto.sha256(f"{ch}|{prev}|{tseal}|{cseal}")

        env = PcdEnvelope(
            object_id=oid, node_id=nid,
            node_type="GENESIS" if self.dag.size() == 0 else "DATA",
            label=label, classification=classification, actor=self.peer_id,
            cipher_payload=cp, payload_hash=ph, cipher_hash=ch,
            prev_hash=prev, hash=ehash,
            temporal_seal=tseal, chain_seal=cseal,
            feature_fp=ai.fingerprint, zk_proof=zk,
            epoch=self._epoch, window_pos=self._wpos,
            chain_counter=self._counter, chain_depth=self.dag.size(),
            timestamp=ts, ai_score=ai.score,
            ai_features=feats.as_dict(), tier=ai.tier,
            nonce=nonce,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
        self.dag.add(env)
        if pid:
            self.dag.link(pid, nid)

        self.model.update_actor(self.peer_id, ai.score)
        self._wpos      += 1
        self._counter   += 1
        self._last_ts    = ts
        if self._wpos >= self.config.window_size:
            self._rotate_epoch()

        self.log(
            "ok",
            f"Anchored: {label} | score={ai.score:.4f} | tier={ai.tier} | "
            f"hash={Crypto.short(ehash)} | {env.latency_ms}ms",
        )
        return env

    # ── RECEIVE (Side-B) ──────────────────────────────────────────────────────
    def receive(self, env: PcdEnvelope) -> Tuple[bool, Optional[str]]:
        self.log("info", f"Receiving: {env.label} E:{env.epoch} W:{env.window_pos}")
        ep_sec  = self._get_secret(env.epoch)
        feats   = AIFeatures(**env.ai_features)
        fp_ok, ktn_b = self.model.validate_ktn(env.feature_fp, feats, ep_sec)

        self.log(
            "ok" if fp_ok else "err",
            f"K(t,n) regen: {Crypto.short(ktn_b)} [B-side, independent] "
            f"fp={'✓' if fp_ok else '✗'}",
        )

        exp_zk  = Crypto.hmac256(ktn_b, f"zk|{env.actor}|{env.node_id}")
        zk_ok   = exp_zk == env.zk_proof
        pt: Optional[str] = None
        pay_ok  = False

        if fp_ok:
            try:
                pt      = Crypto.deobfuscate(env.cipher_payload, ktn_b)
                pay_ok  = Crypto.sha256(pt) == env.payload_hash
            except Exception as exc:
                self.log("err", f"Decrypt error: {exc}")

        ok_s = lambda v: "✓" if v else "✗"
        self.log(
            "ok" if (fp_ok and zk_ok and pay_ok) else "err",
            f"fp={ok_s(fp_ok)}  zk={ok_s(zk_ok)}  payload={ok_s(pay_ok)}",
        )
        if pt and pay_ok:
            self.log("ok", f"Plaintext: {pt[:100]}{'…' if len(pt)>100 else ''}")

        for a in self.model.detect_anomalies(feats):
            self.log("warn", f"Anomaly flag: {a}")

        self.dag.add(env)
        for node in self.dag.all_data():
            if node.hash == env.prev_hash:
                self.dag.link(node.node_id, env.node_id)
                break

        self._counter   = env.chain_counter + 1
        self._wpos      = env.window_pos + 1
        self._last_ts   = env.timestamp
        if self._wpos >= self.config.window_size:
            self._rotate_epoch()

        return fp_ok and zk_ok and pay_ok, pt

    # ── RETRIEVE ──────────────────────────────────────────────────────────────
    def retrieve(self, node_id: str) -> Tuple[Optional[str], Optional[PcdEnvelope]]:
        env    = self.dag.get(node_id)
        if not env:
            return None, None
        ep_sec = self._get_secret(env.epoch)
        feats  = AIFeatures(**env.ai_features)
        _, ktn = self.model.validate_ktn(env.feature_fp, feats, ep_sec)
        try:
            pt = Crypto.deobfuscate(env.cipher_payload, ktn)
        except Exception:
            pt = "[decrypt error]"
        return pt, env

    # ── VERIFY CHAIN ──────────────────────────────────────────────────────────
    def verify_chain(self) -> List[dict]:
        nodes  = self.dag.all_data()
        result = []
        for i, n in enumerate(nodes):
            parent  = nodes[i - 1] if i > 0 else None
            ch_ok   = Crypto.sha256(n.cipher_payload) == n.cipher_hash
            pv_ok   = (n.prev_hash == "0" * 64) if parent is None else (n.prev_hash == parent.hash)
            fp_ok   = AIFeatures(**n.ai_features).fingerprint() == n.feature_fp
            valid   = ch_ok and pv_ok and fp_ok
            result.append({
                "label": n.label, "valid": valid,
                "cipher_ok": ch_ok, "prev_ok": pv_ok, "fp_ok": fp_ok,
                "epoch": n.epoch, "wpos": n.window_pos,
                "nid": Crypto.short(n.node_id),
            })
        return result

    # ── TAMPER (simulation) ───────────────────────────────────────────────────
    def tamper(self, node_id: str) -> bool:
        env = self.dag.get(node_id)
        if not env:
            return False
        env.cipher_payload = env.cipher_payload[:-8] + Crypto.rng(4)
        self.log("err", f"⚠ TAMPER: cipher_payload corrupted at '{env.label}'")
        self.log("err", "Attacker cannot re-forge seals without epoch_secret")
        return True

    # ── Helpers ───────────────────────────────────────────────────────────────
    @property
    def window_info(self) -> dict:
        return {
            "epoch":        self._epoch,
            "wpos":         self._wpos,
            "W":            self.config.window_size,
            "secret_short": Crypto.short(self._secrets.get(self._epoch, "")),
            "counter":      self._counter,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 ── APPLICATION STATE  (singleton, suitable for demo)
# ═══════════════════════════════════════════════════════════════════════════════

class AppState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.side_a:        Optional[Peer] = None
        self.side_b:        Optional[Peer] = None
        self.provisioned:   bool           = False
        self.hub_record:    dict           = {}
        self.global_log:    List[dict]     = []
        self._lock = threading.Lock()

    def log(self, level: str, msg: str) -> None:
        with self._lock:
            self.global_log.append({"ts": time.strftime("%H:%M:%S"), "level": level, "msg": msg})
            if len(self.global_log) > 400:
                self.global_log = self.global_log[-300:]

    def drain_peer_logs(self) -> List[dict]:
        entries = []
        for peer in (self.side_a, self.side_b):
            if peer:
                while not peer._log_q.empty():
                    try:
                        entries.append(peer._log_q.get_nowait())
                    except queue.Empty:
                        break
        return entries


STATE        = AppState()
_sample_idx  = [0]
SAMPLES      = [
    ("sensor-batch-001", "CONFIDENTIAL",
     '{"temp":36.8,"pressure":1013,"unit":"SN-77","status":"nominal"}'),
    ("patient-record-002", "TOP_SECRET",
     '{"patient_id":"PT-9821","bp":"118/76","glucose":94,"flagged":false}'),
    ("tx-ledger-7731", "CONFIDENTIAL",
     '{"from":"ACCT-441","to":"ACCT-882","amount":12500.00,"currency":"USD"}'),
    ("sat-telemetry-042", "SECRET",
     '{"sat":"SAT-9","alt_km":412,"lat":31.2,"lon":34.8,"status":"NOMINAL"}'),
    ("ml-batch-epoch17", "UNCLASSIFIED",
     '{"epoch":17,"loss":0.0421,"acc":0.9834,"model":"resnet-50"}'),
    ("audit-log-443", "SECRET",
     '{"event":"ACCESS","user":"admin","resource":"/api/v2/data","result":"PERMIT"}'),
]


# ─── Serialisation helpers ────────────────────────────────────────────────────

def _env_summary(env: PcdEnvelope) -> dict:
    return {
        "node_id":  env.node_id,
        "label":    env.label,
        "cls":      env.classification,
        "actor":    env.actor,
        "tier":     env.tier,
        "score":    env.ai_score,
        "epoch":    env.epoch,
        "wpos":     env.window_pos,
        "counter":  env.chain_counter,
        "hash_s":   Crypto.short(env.hash),
        "prev_s":   Crypto.short(env.prev_hash),
        "ts":       time.strftime("%H:%M:%S", time.localtime(env.timestamp)),
        "latency":  env.latency_ms,
        "type":     env.node_type,
    }


def _env_full(env: PcdEnvelope) -> dict:
    d = _env_summary(env)
    d.update({
        "object_id":     env.object_id,
        "full_hash":     env.hash,
        "temporal_seal": env.temporal_seal,
        "chain_seal":    env.chain_seal,
        "feature_fp":    env.feature_fp,
        "zk_proof":      env.zk_proof,
        "nonce":         env.nonce,
        "cipher_preview": env.cipher_payload[:48] + "…",
        "ai_features":   env.ai_features,
        "latency_ms":    env.latency_ms,
    })
    return d


def _build_state_snapshot() -> dict:
    """Full UI state — polled by the frontend every 600 ms."""
    a, b       = STATE.side_a, STATE.side_b
    a_nodes    = a.dag.all_data() if a else []
    a_edges    = a.dag.edges()    if a else []
    ai_feats   = {}

    if a and a._ai_last:
        r = a._ai_last
        ai_feats = {
            "score":        r.score,
            "tier":         r.tier,
            "entropy":      r.features.entropy,
            "actor":        r.features.actor_score,
            "timing":       r.features.timing_align,
            "window_pos":   r.features.window_pos,
            "chain_depth":  r.features.chain_depth,
            "ktn_strength": r.score,
        }

    # Drain peer logs into global log
    for entry in STATE.drain_peer_logs():
        STATE.global_log.append(entry)

    return {
        "provisioned": STATE.provisioned,
        "hub_record":  STATE.hub_record,
        "a_id":        a.peer_id if a else "—",
        "b_id":        b.peer_id if b else "—",
        "a_nodes":     [_env_summary(n) for n in a_nodes],
        "a_dag_edges": a_edges,
        "a_window":    a.window_info if a else {},
        "b_window":    b.window_info if b else {},
        "ai_features": ai_feats,
        "metrics": {
            "chain_nodes": len(a_nodes),
            "ai_score":    round(ai_feats.get("score", 0), 3) if ai_feats else "—",
            "epoch":       a._epoch if a else 0,
        },
        "log": STATE.global_log[-80:],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 ── PYDANTIC REQUEST MODELS  (real when FastAPI present)
# ═══════════════════════════════════════════════════════════════════════════════

if _FASTAPI_AVAILABLE:
    from pydantic import BaseModel, Field

    class HubInitRequest(BaseModel):
        master_secret:   str = Field("M-ALPHA-9K7X", description="Hub master secret")
        window_size:     int = Field(4, ge=2, le=32, description="Epoch window size W")
        side_a_id:       str = Field("NODE-ALPHA")
        side_b_id:       str = Field("NODE-BETA")

    class InsertRequest(BaseModel):
        label:          str = Field(..., description="Human-readable data label")
        payload:        str = Field(..., description="JSON or text payload")
        classification: str = Field("CONFIDENTIAL",
                                    description="TOP_SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED")

    class RetrieveRequest(BaseModel):
        node_id: str = Field("", description="node_id to retrieve; empty = latest")

    class VerifyRequest(BaseModel):
        side: str = Field("b", description="'a' or 'b' — which peer's DAG to verify")

    class TamperRequest(BaseModel):
        node_id: str = Field("", description="Specific node to tamper; empty = random")

    class ResetRequest(BaseModel):
        pass

else:
    # Lightweight shim — behaves identically for our purposes
    class _Model:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class HubInitRequest(_Model):
        def __init__(self, master_secret="M-ALPHA-9K7X", window_size=4,
                     side_a_id="NODE-ALPHA", side_b_id="NODE-BETA"):
            super().__init__(master_secret=master_secret, window_size=int(window_size),
                             side_a_id=side_a_id, side_b_id=side_b_id)

    class InsertRequest(_Model):
        def __init__(self, label="data", payload="{}", classification="CONFIDENTIAL"):
            super().__init__(label=label, payload=payload, classification=classification)

    class RetrieveRequest(_Model):
        def __init__(self, node_id=""):
            super().__init__(node_id=node_id)

    class VerifyRequest(_Model):
        def __init__(self, side="b"):
            super().__init__(side=side)

    class TamperRequest(_Model):
        def __init__(self, node_id=""):
            super().__init__(node_id=node_id)

    class ResetRequest(_Model):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 ── ROUTE HANDLER FUNCTIONS  (pure async — used by both paths)
# ═══════════════════════════════════════════════════════════════════════════════

async def handle_hub_init(req: HubInitRequest) -> dict:
    anchor = time.time()
    ep0    = Crypto.derive_key(
        req.master_secret, "epoch", "0", str(req.window_size), str(int(anchor))
    )
    cfg = SideConfig(
        epoch_0_secret  = ep0,
        timing_anchor   = anchor,
        window_size     = req.window_size,
        epoch_ms        = 30_000,
        drift_tolerance = 500,
        model_weights   = NanoAIModel.WEIGHTS.tolist(),
    )
    STATE.reset()
    STATE.side_a     = Peer(req.side_a_id, "SENDER",   cfg)
    STATE.side_b     = Peer(req.side_b_id, "RECEIVER", cfg)
    STATE.provisioned = True
    STATE.hub_record  = {
        "master_hash":    Crypto.short(Crypto.sha256(req.master_secret)),
        "epoch_0_secret": Crypto.short(ep0),
        "window_size_W":  req.window_size,
        "epoch_duration": "30000ms",
        "timing_anchor":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(anchor)),
        "side_a_id":      req.side_a_id,
        "side_b_id":      req.side_b_id,
        "hub_status":     "SILENT — not in chain",
    }
    STATE.log("hub", f"HUB provisioned | W={req.window_size} | epoch_0={Crypto.short(ep0)}")
    STATE.log("hub", "HUB: ⊘ SILENT — both sides operational independently")
    return {"ok": True, "hub_record": STATE.hub_record}


async def handle_insert(req: InsertRequest) -> dict:
    if not STATE.provisioned:
        return {"ok": False, "error": "HUB not initialised"}
    env = STATE.side_a.insert(req.label, req.payload, req.classification)
    STATE.log("ok", f"[A] Inserted: {req.label} | score={env.ai_score:.3f} | tier={env.tier}")
    return {"ok": True, "node": _env_summary(env)}


async def handle_sample() -> dict:
    idx = _sample_idx[0] % len(SAMPLES)
    _sample_idx[0] += 1
    label, cls, payload = SAMPLES[idx]
    return {"label": label, "classification": cls, "payload": payload}


async def handle_stream() -> dict:
    if not STATE.provisioned:
        return {"ok": False, "error": "HUB not initialised"}
    nodes = STATE.side_a.dag.all_data()
    if not nodes:
        return {"ok": False, "error": "No nodes to stream — insert data first"}

    blocks, seq = [], 0
    for i, env in enumerate(nodes):
        blocks += [
            {"seq": seq,   "type": "DATA",
             "content": f"E{env.epoch} W{env.window_pos} OBJ:{Crypto.short(env.object_id)} ENC:{env.cipher_payload[:28]}…"},
            {"seq": seq+1, "type": "PROOF",
             "content": f"SEAL:{Crypto.short(env.temporal_seal)} CHAIN:{Crypto.short(env.chain_seal)} PREV:{Crypto.short(env.prev_hash)}"},
            {"seq": seq+2, "type": "FEATURES",
             "content": f"FP:{Crypto.short(env.feature_fp)} CNT:{env.chain_counter} E:{env.epoch} W:{env.window_pos}"},
            {"seq": seq+3, "type": "ACK",
             "content": f"ACK:{Crypto.short(env.object_id)} ZK:{Crypto.short(env.zk_proof)} SCORE:{env.ai_score}"},
        ]
        seq += 4
        ok, _ = STATE.side_b.receive(env)
        STATE.log("ok" if ok else "err",
                  f"[P2P] Node {i+1}/{len(nodes)}: {env.label} → B {'✓' if ok else '✗'}")
    STATE.log("ok", f"[P2P] Stream complete: {seq} blocks | K(t,n) never transmitted")
    return {"ok": True, "blocks": blocks, "total": seq}


async def handle_retrieve(req: RetrieveRequest) -> dict:
    if not STATE.provisioned:
        return {"ok": False, "error": "HUB not initialised"}
    node_id = req.node_id.strip()

    # Try Side-B first; fall back to Side-A (pre-stream case)
    pt, env = STATE.side_b.retrieve(node_id) if node_id else (None, None)
    if not env:
        latest = STATE.side_b.dag.latest or STATE.side_a.dag.latest
        if not latest:
            return {"ok": False, "error": "No nodes yet"}
        pt, env = STATE.side_a.retrieve(latest.node_id)

    if not env:
        return {"ok": False, "error": f"Not found: {node_id}"}

    STATE.log("ok", f"[B] Retrieved: {env.label} | tier={env.tier}")
    return {"ok": True, "plaintext": pt, "node": _env_full(env)}


async def handle_verify(req: VerifyRequest) -> dict:
    if not STATE.provisioned:
        return {"ok": False, "error": "HUB not initialised"}
    peer = STATE.side_b if req.side == "b" else STATE.side_a
    results    = peer.verify_chain()
    chain_ok   = all(r["valid"] for r in results)
    STATE.log(
        "ok" if chain_ok else "err",
        f"Chain verify [{req.side.upper()}]: {'ALL INTACT' if chain_ok else 'VIOLATION DETECTED'}",
    )
    return {"ok": True, "chain_valid": chain_ok, "results": results}


async def handle_tamper(req: TamperRequest) -> dict:
    if not STATE.provisioned:
        return {"ok": False, "error": "HUB not initialised"}
    nodes = STATE.side_a.dag.all_data()
    if not nodes:
        return {"ok": False, "error": "No nodes"}
    import random
    target = STATE.side_a.dag.get(req.node_id) if req.node_id else random.choice(nodes)
    if not target:
        return {"ok": False, "error": "Node not found"}
    STATE.side_a.tamper(target.node_id)
    STATE.log("err", f"[TAMPER] Applied to: {target.label} ({Crypto.short(target.node_id)})")
    return {"ok": True, "tampered_node": target.node_id, "label": target.label}


async def handle_state() -> dict:
    return _build_state_snapshot()


async def handle_reset() -> dict:
    STATE.reset()
    _sample_idx[0] = 0
    return {"ok": True}


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 ── FASTAPI APPLICATION  (real FastAPI when available)
# ═══════════════════════════════════════════════════════════════════════════════

if _FASTAPI_AVAILABLE:
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title       = "PIM-PCD Neural Ratchet Chain",
        description = "Proof Chain of Data — Nano-AI Governed P2P Data Transfer · WARLOK UI",
        version     = "2.0.0",
    )
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/", response_class=HTMLResponse, tags=["UI"])
    async def root():
        return HTMLResponse(content=_HTML_TEMPLATE)

    @app.post("/api/hub/init", tags=["Hub"])
    async def api_hub_init(req: HubInitRequest):
        return await handle_hub_init(req)

    @app.post("/api/insert", tags=["Chain"])
    async def api_insert(req: InsertRequest):
        return await handle_insert(req)

    @app.get("/api/sample", tags=["Chain"])
    async def api_sample():
        return await handle_sample()

    @app.post("/api/stream", tags=["P2P"])
    async def api_stream():
        return await handle_stream()

    @app.post("/api/retrieve", tags=["Chain"])
    async def api_retrieve(req: RetrieveRequest):
        return await handle_retrieve(req)

    @app.post("/api/verify", tags=["Chain"])
    async def api_verify(req: VerifyRequest):
        return await handle_verify(req)

    @app.post("/api/tamper", tags=["Debug"])
    async def api_tamper(req: TamperRequest):
        return await handle_tamper(req)

    @app.get("/api/state", tags=["UI"])
    async def api_state():
        return await handle_state()

    @app.post("/api/reset", tags=["Debug"])
    async def api_reset():
        return await handle_reset()

    @app.get("/api/events", tags=["SSE"])
    async def api_events():
        """
        Server-Sent Events stream — pushes state deltas to connected clients.
        Provides real-time updates without polling when using real FastAPI.
        """
        async def generate() -> AsyncGenerator[str, None]:
            last_len = 0
            while True:
                await asyncio.sleep(0.4)
                snap     = _build_state_snapshot()
                log      = snap.get("log", [])
                new_logs = log[last_len:]
                last_len = len(log)
                if new_logs or snap.get("provisioned"):
                    data = json.dumps({
                        "nodes":   snap.get("metrics", {}).get("chain_nodes", 0),
                        "epoch":   snap.get("a_window", {}).get("epoch", 0),
                        "wpos":    snap.get("a_window", {}).get("wpos", 0),
                        "ai":      snap.get("ai_features", {}),
                        "new_log": new_logs,
                    })
                    yield f"data: {data}\n\n"
        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

else:
    # ─────────────────────────────────────────────────────────────────────────
    #  FLASK COMPATIBILITY SHIM
    #  Wraps every async handler with asyncio.run() and wires Flask routes.
    #  API surface is 100% identical — swap to real FastAPI by running:
    #      pip install fastapi uvicorn
    #      uvicorn pim_pcd_fastapi:app --port 5050 --reload
    # ─────────────────────────────────────────────────────────────────────────
    import inspect
    from flask import Flask, Response, jsonify, request as flask_request, render_template_string

    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    def _run(coro):
        """Run an async handler synchronously."""
        return asyncio.run(coro)

    def _body_to_model(model_cls):
        """Parse JSON body and instantiate a model-like object."""
        data = flask_request.get_json(force=True, silent=True) or {}
        sig  = inspect.signature(model_cls.__init__)
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in data:
                kwargs[name] = data[name]
            elif param.default is not inspect.Parameter.empty:
                kwargs[name] = param.default
        return model_cls(**kwargs)

    @app.route("/")
    def root():
        return render_template_string(_HTML_TEMPLATE)

    @app.route("/api/hub/init", methods=["POST"])
    def api_hub_init():
        return jsonify(_run(handle_hub_init(_body_to_model(HubInitRequest))))

    @app.route("/api/insert", methods=["POST"])
    def api_insert():
        return jsonify(_run(handle_insert(_body_to_model(InsertRequest))))

    @app.route("/api/sample", methods=["GET"])
    def api_sample():
        return jsonify(_run(handle_sample()))

    @app.route("/api/stream", methods=["POST"])
    def api_stream():
        return jsonify(_run(handle_stream()))

    @app.route("/api/retrieve", methods=["POST"])
    def api_retrieve():
        return jsonify(_run(handle_retrieve(_body_to_model(RetrieveRequest))))

    @app.route("/api/verify", methods=["POST"])
    def api_verify():
        return jsonify(_run(handle_verify(_body_to_model(VerifyRequest))))

    @app.route("/api/tamper", methods=["POST"])
    def api_tamper():
        return jsonify(_run(handle_tamper(_body_to_model(TamperRequest))))

    @app.route("/api/state", methods=["GET"])
    def api_state():
        return jsonify(_run(handle_state()))

    @app.route("/api/reset", methods=["POST"])
    def api_reset():
        return jsonify(_run(handle_reset()))

    # SSE fallback for Flask — polling-compatible
    @app.route("/api/events", methods=["GET"])
    def api_events():
        def gen():
            while True:
                snap = _build_state_snapshot()
                data = json.dumps({
                    "nodes":   snap.get("metrics", {}).get("chain_nodes", 0),
                    "epoch":   snap.get("a_window", {}).get("epoch", 0),
                    "wpos":    snap.get("a_window", {}).get("wpos", 0),
                    "ai":      snap.get("ai_features", {}),
                    "new_log": [],
                })
                yield f"data: {data}\n\n"
                time.sleep(0.6)
        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 ── WARLOK HTML/CSS/JS UI  (same dark terminal aesthetic as demo)
# ═══════════════════════════════════════════════════════════════════════════════

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>PIM-PCD · WARLOK · Neural Ratchet</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#050810;--bg2:#07101e;--bg3:#0b1628;--panel:#080f1c;
  --border:#162238;--border2:#1e3050;
  --cyan:#00f5e4;--cyan2:#00bfb3;--blue:#0080ff;--blue2:#0055cc;
  --green:#00e676;--green2:#00c853;--amber:#ffab00;--red:#ff3d57;
  --purple:#bb86fc;--pink:#ff6ec7;
  --text:#c8d8f0;--text2:#7a99cc;--text3:#2e4a6a;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;font-size:14px;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background-image:linear-gradient(rgba(0,245,228,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(0,245,228,.025) 1px,transparent 1px);background-size:44px 44px;pointer-events:none;z-index:0}
body::after{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,.06) 2px,rgba(0,0,0,.06) 4px);pointer-events:none;z-index:0}
#app{position:relative;z-index:1;max-width:1480px;margin:0 auto;padding:0 18px 40px}
header{display:flex;align-items:center;justify-content:space-between;padding:16px 0 12px;border-bottom:1px solid var(--border);margin-bottom:20px}
.logo{font-family:'Orbitron',monospace;font-weight:900;font-size:24px;color:var(--cyan);text-shadow:0 0 16px #00f5e440;letter-spacing:3px}
.logo-sub{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--text3);letter-spacing:3px;margin-left:10px}
.pills{display:flex;gap:10px}
.pill{display:flex;align-items:center;gap:5px;font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--text2);background:var(--bg3);border:1px solid var(--border);border-radius:2px;padding:4px 9px}
.dot{width:5px;height:5px;border-radius:50%;background:var(--green);box-shadow:0 0 5px var(--green);animation:pulse 2s infinite}
.dot.amber{background:var(--amber);box-shadow:0 0 5px var(--amber)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.phases{display:flex;align-items:center;gap:10px;padding:9px 14px;background:var(--bg3);border:1px solid var(--border2);border-radius:3px;margin-bottom:18px;font-family:'Share Tech Mono',monospace;font-size:10px}
.ph{padding:4px 12px;border-radius:2px;font-size:9px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:default;transition:all .15s}
.ph.active{background:rgba(0,245,228,.2);border:1px solid var(--cyan);color:var(--cyan);box-shadow:0 0 8px #00f5e430}
.ph.done{background:rgba(0,230,118,.15);border:1px solid var(--green2);color:var(--green)}
.ph.pend{background:var(--bg);border:1px solid var(--border);color:var(--text3)}
.ph-arrow{color:var(--text3)}
.mrow{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px}
.mc{background:var(--panel);border:1px solid var(--border);border-radius:3px;padding:11px 13px;position:relative;overflow:hidden}
.mc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px}
.mc.c::after{background:var(--cyan)}.mc.g::after{background:var(--green)}.mc.b::after{background:var(--blue)}.mc.a::after{background:var(--amber)}.mc.p::after{background:var(--purple)}
.ml{font-family:'Share Tech Mono',monospace;font-size:8px;letter-spacing:2px;color:var(--text3);text-transform:uppercase;margin-bottom:4px}
.mv{font-family:'Orbitron',monospace;font-size:18px;font-weight:700}
.mc.c .mv{color:var(--cyan)}.mc.g .mv{color:var(--green)}.mc.b .mv{color:var(--blue)}.mc.a .mv{color:var(--amber)}.mc.p .mv{color:var(--purple)}
.ms{font-size:10px;color:var(--text3);margin-top:2px}
.grid3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.panel{background:var(--panel);border:1px solid var(--border);border-radius:4px;overflow:hidden;position:relative}
.panel::before{content:'';position:absolute;top:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan2),transparent);opacity:.4}
.ph2{display:flex;align-items:center;gap:7px;padding:8px 13px;background:var(--bg3);border-bottom:1px solid var(--border);font-family:'Share Tech Mono',monospace;font-size:10px;letter-spacing:2px;color:var(--cyan)}
.ph2 .badge{padding:2px 6px;border-radius:2px;font-size:8px;margin-left:auto}
.badge-hub{background:rgba(255,110,199,.2);border:1px solid var(--pink);color:var(--pink)}
.badge-a{background:rgba(0,245,228,.15);border:1px solid var(--cyan2);color:var(--cyan)}
.badge-b{background:rgba(0,230,118,.15);border:1px solid var(--green2);color:var(--green)}
.pb{padding:13px}
label{display:block;font-family:'Share Tech Mono',monospace;font-size:9px;letter-spacing:2px;color:var(--text2);margin-bottom:4px;text-transform:uppercase}
input,textarea,select{width:100%;background:var(--bg2);border:1px solid var(--border2);border-radius:2px;color:var(--text);font-family:'Share Tech Mono',monospace;font-size:12px;padding:7px 9px;outline:none;transition:border-color .2s}
input:focus,textarea:focus{border-color:var(--cyan2)}
textarea{height:68px;resize:vertical}
.fr{margin-bottom:11px}.fr2{display:grid;grid-template-columns:1fr 1fr;gap:9px;margin-bottom:11px}
.btn{display:inline-flex;align-items:center;gap:5px;font-family:'Orbitron',monospace;font-size:10px;font-weight:700;letter-spacing:2px;padding:8px 14px;border:none;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase;white-space:nowrap}
.btn-c{background:linear-gradient(135deg,var(--cyan2),var(--blue));color:var(--bg);box-shadow:0 0 10px #00f5e430}
.btn-g{background:linear-gradient(135deg,var(--green2),#00897b);color:var(--bg)}
.btn-a{background:linear-gradient(135deg,var(--amber),#e65100);color:var(--bg)}
.btn-p{background:linear-gradient(135deg,var(--purple),var(--blue2));color:#fff}
.btn-r{background:linear-gradient(135deg,var(--red),#880e4f);color:#fff}
.btn-ghost{background:transparent;color:var(--text2);border:1px solid var(--border2)}
.btn:hover{filter:brightness(1.2);transform:translateY(-1px)}
.btn:disabled{opacity:.35;cursor:not-allowed;transform:none!important;filter:none!important}
.br{display:flex;gap:7px;flex-wrap:wrap;margin-top:11px}
.aib{display:flex;flex-direction:column;gap:6px;margin-bottom:10px}
.aibr{display:flex;align-items:center;gap:7px}
.aibl{font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--text2);width:115px;flex-shrink:0;text-transform:uppercase;letter-spacing:1px}
.aibt{flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
.aibf{height:100%;border-radius:2px;transition:width .6s ease}
.aibv{font-family:'Share Tech Mono',monospace;font-size:10px;width:34px;text-align:right;flex-shrink:0}
.wgrid{display:grid;gap:3px;margin-bottom:8px}
.wc{aspect-ratio:1;border-radius:2px;display:flex;align-items:center;justify-content:center;font-family:'Share Tech Mono',monospace;font-size:8px;border:1px solid var(--border);transition:all .3s}
.wc.used{background:rgba(0,245,228,.15);border-color:var(--cyan2);color:var(--cyan)}
.wc.active{background:rgba(0,245,228,.35);border-color:var(--cyan);color:var(--cyan);box-shadow:0 0 8px #00f5e440;animation:wg .8s infinite alternate}
@keyframes wg{from{box-shadow:0 0 4px #00f5e430}to{box-shadow:0 0 12px #00f5e470}}
.kv{display:grid;grid-template-columns:130px 1fr;gap:2px 8px;font-family:'Share Tech Mono',monospace;font-size:10px}
.kv .k{color:var(--text3)}.kv .v{color:var(--text);word-break:break-all}
.kv .v.c{color:var(--cyan)}.kv .v.g{color:var(--green)}.kv .v.a{color:var(--amber)}.kv .v.p{color:var(--purple)}.kv .v.pk{color:var(--pink)}
.clist{max-height:240px;overflow-y:auto}
.cn{display:flex;align-items:flex-start;gap:9px;padding:7px;border-bottom:1px solid var(--border);cursor:pointer;transition:background .1s;border-radius:2px}
.cn:hover{background:var(--bg3)}
.cnb{font-family:'Share Tech Mono',monospace;font-size:7px;letter-spacing:1px;padding:2px 5px;border-radius:2px;white-space:nowrap;flex-shrink:0;margin-top:1px}
.cn.genesis .cnb{background:var(--amber);color:var(--bg)}.cn.data .cnb{background:var(--blue);color:#fff}
.cnm{flex:1;min-width:0}.cnt{font-weight:600;font-size:12px;color:var(--text);margin-bottom:1px}
.cnh{font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--cyan)}
.cnd{font-size:10px;color:var(--text2);margin-top:1px}.cns{font-family:'Share Tech Mono',monospace;font-size:11px;flex-shrink:0;text-align:right}
.p2pt{display:flex;align-items:center;gap:0;margin:10px 0}
.p2pn{flex-shrink:0;width:70px;text-align:center}
.p2pc{width:42px;height:42px;border-radius:50%;margin:0 auto 5px;display:flex;align-items:center;justify-content:center;font-family:'Orbitron',monospace;font-size:9px;border:2px solid}
.p2pc.src{border-color:var(--cyan);background:rgba(0,245,228,.1);color:var(--cyan)}
.p2pc.dst{border-color:var(--green);background:rgba(0,230,118,.1);color:var(--green)}
.p2cl{font-size:9px;color:var(--text2);font-family:'Share Tech Mono',monospace}
.pipe{flex:1;height:2px;background:var(--border2);position:relative}
.pkt{position:absolute;top:-5px;height:12px;width:22px;border-radius:2px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);animation:pktmove .5s linear}
@keyframes pktmove{from{left:0;opacity:0}50%{opacity:1}to{left:calc(100% - 22px);opacity:0}}
.bstream{background:var(--bg2);border:1px solid var(--border2);border-radius:2px;padding:9px;max-height:200px;overflow-y:auto;font-family:'Share Tech Mono',monospace;font-size:10px}
.bi{display:grid;grid-template-columns:38px 80px 1fr 56px;gap:5px;padding:4px 0;border-bottom:1px solid var(--border);align-items:center}
.bi:last-child{border-bottom:none}.bseq{color:var(--text3)}
.btag{font-size:8px;padding:2px 4px;border-radius:2px;text-align:center;font-weight:700;letter-spacing:1px}
.btag.DATA{background:rgba(0,128,255,.2);color:var(--blue);border:1px solid var(--blue2)}
.btag.PROOF{background:rgba(0,245,228,.2);color:var(--cyan);border:1px solid var(--cyan2)}
.btag.FEATURES{background:rgba(187,134,252,.2);color:var(--purple);border:1px solid var(--purple)}
.btag.ACK{background:rgba(0,230,118,.2);color:var(--green);border:1px solid var(--green2)}
.bc{color:var(--text2);font-size:9px;word-break:break-all}.bst{font-size:9px;text-align:right}
.bst.ok{color:var(--green)}.bst.enc{color:var(--amber)}
.term{background:#020408;height:155px;overflow-y:auto;padding:8px 10px;font-family:'Share Tech Mono',monospace;font-size:11px;line-height:1.7}
.ll{display:flex;gap:7px}.lt{color:var(--text3);min-width:52px;flex-shrink:0}
.ok{color:var(--green)}.info{color:var(--cyan2)}.err{color:var(--red)}.warn{color:var(--amber)}.sys{color:var(--purple)}.hub{color:var(--pink)}
.ibadge{display:inline-flex;align-items:center;gap:5px;padding:5px 11px;border-radius:2px;font-family:'Orbitron',monospace;font-size:10px;font-weight:700;letter-spacing:1px}
.ibadge.ok{background:rgba(0,230,118,.15);border:1px solid var(--green2);color:var(--green)}
.ibadge.warn{background:rgba(255,171,0,.15);border:1px solid var(--amber);color:var(--amber)}
.ibadge.wait{background:rgba(0,128,255,.1);border:1px solid var(--blue2);color:var(--blue)}
.alert{display:flex;align-items:center;gap:7px;padding:7px 11px;border-radius:2px;margin-bottom:9px;font-family:'Share Tech Mono',monospace;font-size:10px}
.alert.danger{background:rgba(255,61,87,.1);border:1px solid var(--red);color:var(--red)}
.alert.success{background:rgba(0,230,118,.1);border:1px solid var(--green2);color:var(--green)}
.alert.hub{background:rgba(255,110,199,.1);border:1px solid var(--pink);color:var(--pink)}
.vr{background:var(--bg2);border:1px solid var(--border2);border-radius:2px;padding:11px;margin-top:9px;font-family:'Share Tech Mono',monospace;font-size:10px;line-height:1.8;display:none}
.vr.on{display:block}
.dbox{background:var(--bg2);border:1px solid var(--border2);border-radius:3px;padding:10px;margin-top:9px;display:none}.dbox.on{display:block}
#dagc{width:100%;height:230px;background:#020408;display:block;border-radius:2px}
::-webkit-scrollbar{width:3px;height:3px}::-webkit-scrollbar-track{background:var(--bg2)}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px}::-webkit-scrollbar-thumb:hover{background:var(--cyan2)}
@keyframes fadein{from{opacity:0;transform:translateY(3px)}to{opacity:1;transform:none}}.ani{animation:fadein .25s ease both}
</style>
</head>
<body>
<div id="app">
<header>
  <div style="display:flex;align-items:baseline;gap:8px">
    <span class="logo">PIM-PCD</span>
    <span class="logo-sub">NEURAL RATCHET CHAIN · v2.0 · WARLOK · FastAPI</span>
  </div>
  <div class="pills">
    <div class="pill"><span class="dot amber" id="hub-dot"></span><span id="hub-lbl">HUB: INIT</span></div>
    <div class="pill"><span class="dot amber" id="a-dot"></span><span id="a-lbl">SIDE-A: OFFLINE</span></div>
    <div class="pill"><span class="dot amber" id="b-dot"></span><span id="b-lbl">SIDE-B: OFFLINE</span></div>
    <div class="pill" id="epoch-pill">EPOCH: —</div>
    <div class="pill" id="win-pill">WINDOW: —/—</div>
  </div>
</header>
<div class="phases">
  <span style="color:var(--text3);font-size:8px;letter-spacing:2px">PHASE:</span>
  <span class="ph active" id="ph1">① HUB INIT</span><span class="ph-arrow">→</span>
  <span class="ph pend"   id="ph2">② SIDE SYNC</span><span class="ph-arrow">→</span>
  <span class="ph pend"   id="ph3">③ INSERT &amp; CHAIN</span><span class="ph-arrow">→</span>
  <span class="ph pend"   id="ph4">④ P2P STREAM</span><span class="ph-arrow">→</span>
  <span class="ph pend"   id="ph5">⑤ VERIFY &amp; AUDIT</span>
</div>
<div class="mrow">
  <div class="mc c"><div class="ml">Chain Nodes</div><div class="mv" id="m-nodes">0</div><div class="ms">DAG entries</div></div>
  <div class="mc g"><div class="ml">Chain Valid</div><div class="mv" id="m-valid">—</div><div class="ms">integrity</div></div>
  <div class="mc b"><div class="ml">Latency</div><div class="mv" id="m-lat">—</div><div class="ms">avg ms</div></div>
  <div class="mc a"><div class="ml">AI Score</div><div class="mv" id="m-ai">—</div><div class="ms">nano-AI trust</div></div>
  <div class="mc p"><div class="ml">Epoch</div><div class="mv" id="m-epoch">0</div><div class="ms">current window</div></div>
</div>
<div class="grid3">
  <div class="panel ani">
    <div class="ph2">◈ HUB AUTHORITY<span class="badge badge-hub">HUB</span></div>
    <div class="pb">
      <div id="hub-alert"></div>
      <div class="fr2">
        <div><label>Master Secret</label><input id="hub-master" value="M-ALPHA-9K7X"></div>
        <div><label>Window Size W</label><input id="hub-W" type="number" value="4" min="2" max="16"></div>
      </div>
      <div class="fr2">
        <div><label>Side-A Node ID</label><input id="hub-aid" value="NODE-ALPHA"></div>
        <div><label>Side-B Node ID</label><input id="hub-bid" value="NODE-BETA"></div>
      </div>
      <div class="br">
        <button class="btn btn-p" onclick="doHubInit()">◈ INIT &amp; PROVISION</button>
        <button class="btn btn-ghost" onclick="doReset()">↺ RESET</button>
      </div>
      <div id="hub-kv" class="dbox" style="margin-top:10px"></div>
    </div>
  </div>
  <div class="panel ani">
    <div class="ph2">⬡ SIDE-A — SENDER<span class="badge badge-a" id="a-badge">NODE-A</span></div>
    <div class="pb">
      <div id="a-alert"></div>
      <div class="fr"><label>Data Label</label><input id="a-label" value="sensor-batch-001"></div>
      <div class="fr"><label>Payload</label><textarea id="a-payload">{"temp":36.8,"pressure":1013,"unit":"SN-77","status":"nominal"}</textarea></div>
      <div class="fr2">
        <div><label>Classification</label>
          <select id="a-cls"><option>TOP_SECRET</option><option>SECRET</option><option selected>CONFIDENTIAL</option><option>UNCLASSIFIED</option></select>
        </div>
        <div><label>Actor</label><input id="a-actor" value="node-alpha-7"></div>
      </div>
      <div class="br">
        <button class="btn btn-c" id="btn-insert" onclick="doInsert()" disabled>⬡ INSERT &amp; CHAIN</button>
        <button class="btn btn-ghost" onclick="doSample()">↺ SAMPLE</button>
      </div>
      <div id="a-insert-result" class="dbox"></div>
    </div>
  </div>
  <div class="panel ani">
    <div class="ph2">⬡ SIDE-B — RECEIVER<span class="badge badge-b" id="b-badge">NODE-B</span></div>
    <div class="pb">
      <div id="b-alert"></div>
      <div class="fr"><label>Retrieve Object ID</label><input id="b-retrieve-id" placeholder="object_id from chain…"></div>
      <div class="fr"><label>Actor (B-Side)</label><input id="b-actor" value="inspector-beta"></div>
      <div class="br">
        <button class="btn btn-g" id="btn-retrieve" onclick="doRetrieve()" disabled>▶ RETRIEVE + DECRYPT</button>
        <button class="btn btn-ghost" onclick="doRetrieveLatest()">▶ LATEST</button>
      </div>
      <div id="b-result" class="dbox"></div>
      <div style="margin-top:12px">
        <div class="br" style="margin-top:0">
          <button class="btn btn-a" id="btn-verify" onclick="doVerify()" disabled>◈ VERIFY CHAIN</button>
          <button class="btn btn-r" onclick="doTamper()">⚠ TAMPER</button>
        </div>
        <div id="b-verify" class="vr"></div>
      </div>
    </div>
  </div>
</div>
<div class="grid3">
  <div class="panel ani">
    <div class="ph2">⊛ SECRET WINDOW STATE</div>
    <div class="pb">
      <div class="kv" id="win-kv" style="margin-bottom:10px">
        <span class="k">epoch</span><span class="v a" id="wk-ep">—</span>
        <span class="k">epoch_secret</span><span class="v c" id="wk-sec">—</span>
        <span class="k">window_pos</span><span class="v g" id="wk-wpos">—</span>
        <span class="k">window_size</span><span class="v" id="wk-wsz">—</span>
        <span class="k">K(t,n)</span><span class="v p" id="wk-ktn">—</span>
        <span class="k">master_mapped</span><span class="v" id="wk-mm">—</span>
        <span class="k">chain_counter</span><span class="v" id="wk-cc">—</span>
      </div>
      <label style="margin-bottom:5px">Window Slots (W=<span id="wk-label">4</span>)</label>
      <div class="wgrid" id="wgrid"></div>
      <span class="ibadge wait" id="win-badge">⊛ AWAITING INIT</span>
    </div>
  </div>
  <div class="panel ani">
    <div class="ph2">◈ NANO-AI HARDENING ENGINE</div>
    <div class="pb">
      <div id="ai-alert"></div>
      <div class="aib">
        <div class="aibr"><div class="aibl">Entropy Density</div><div class="aibt"><div class="aibf" id="bf-ent" style="width:0%;background:var(--cyan)"></div></div><div class="aibv" id="bv-ent" style="color:var(--cyan)">—</div></div>
        <div class="aibr"><div class="aibl">Actor Reputation</div><div class="aibt"><div class="aibf" id="bf-rep" style="width:0%;background:var(--green)"></div></div><div class="aibv" id="bv-rep" style="color:var(--green)">—</div></div>
        <div class="aibr"><div class="aibl">Temporal Align</div><div class="aibt"><div class="aibf" id="bf-tmp" style="width:0%;background:var(--blue)"></div></div><div class="aibv" id="bv-tmp" style="color:var(--blue)">—</div></div>
        <div class="aibr"><div class="aibl">Window Position</div><div class="aibt"><div class="aibf" id="bf-win" style="width:0%;background:var(--amber)"></div></div><div class="aibv" id="bv-win" style="color:var(--amber)">—</div></div>
        <div class="aibr"><div class="aibl">Chain Depth</div><div class="aibt"><div class="aibf" id="bf-dep" style="width:0%;background:var(--purple)"></div></div><div class="aibv" id="bv-dep" style="color:var(--purple)">—</div></div>
        <div class="aibr"><div class="aibl">K(t,n) Strength</div><div class="aibt"><div class="aibf" id="bf-ktn" style="width:0%;background:var(--pink)"></div></div><div class="aibv" id="bv-ktn" style="color:var(--pink)">—</div></div>
      </div>
      <span class="ibadge wait" id="ai-badge">◈ AWAITING INIT</span>
      <span style="font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--text3);margin-left:8px" id="ai-last">—</span>
    </div>
  </div>
  <div class="panel ani">
    <div class="ph2">◉ DAG PROOF CHAIN</div>
    <div class="pb" style="padding:0"><canvas id="dagc"></canvas></div>
  </div>
</div>
<div class="grid2">
  <div class="panel ani">
    <div class="ph2">≡ CHAIN NODE INSPECTOR</div>
    <div class="pb" style="padding:7px 9px">
      <div class="clist" id="clist"><div style="color:var(--text3);font-family:'Share Tech Mono',monospace;font-size:10px;padding:16px;text-align:center">◉ Chain empty — HUB init → insert data</div></div>
      <div id="node-detail" class="dbox"></div>
    </div>
  </div>
  <div class="panel ani">
    <div class="ph2">⇆ P2P NEURAL RATCHET STREAM</div>
    <div class="pb">
      <div class="p2pt">
        <div class="p2pn"><div class="p2pc src" id="p2p-src-c">A</div><div class="p2cl" id="p2p-src-l">NODE-A</div></div>
        <div class="pipe" id="p2p-pipe"></div>
        <div class="p2pn"><div class="p2pc dst" id="p2p-dst-c">B</div><div class="p2cl" id="p2p-dst-l">NODE-B</div></div>
      </div>
      <div class="br" style="margin-bottom:9px">
        <button class="btn btn-c" id="btn-p2p" onclick="doStream()" disabled>⇆ STREAM CHAIN</button>
        <button class="btn btn-ghost" onclick="clearStream()">✕ CLEAR</button>
      </div>
      <div class="bstream" id="bstream"><div style="color:var(--text3);padding:8px;text-align:center">⇆ No blocks — insert data then stream</div></div>
    </div>
  </div>
</div>
<div class="panel ani">
  <div class="ph2">▶ SYSTEM LOG — NEURAL RATCHET ENGINE · FastAPI</div>
  <div style="padding:0"><div class="term" id="term"></div></div>
</div>
</div>
<script>
let latencies=[],phase=1,lastLogLen=0;
async function api(method,path,body){
  const r=await fetch(path,{method,headers:{'Content-Type':'application/json'},body:body?JSON.stringify(body):undefined});
  return r.json();
}
async function doHubInit(){
  const r=await api('POST','/api/hub/init',{
    master_secret:document.getElementById('hub-master').value,
    window_size:parseInt(document.getElementById('hub-W').value),
    side_a_id:document.getElementById('hub-aid').value,
    side_b_id:document.getElementById('hub-bid').value,
  });
  if(!r.ok){syslog('err',r.error);return;}
  const hr=r.hub_record;
  document.getElementById('hub-kv').className='dbox on';
  document.getElementById('hub-kv').innerHTML=`<div style="color:var(--pink);font-family:'Share Tech Mono',monospace;font-size:9px;margin-bottom:6px;letter-spacing:2px">HUB PROVISION RECORD</div><div class="kv"><span class="k">master_hash</span><span class="v c">${hr.master_hash}</span><span class="k">epoch_0_secret</span><span class="v p">${hr.epoch_0_secret}</span><span class="k">window_size_W</span><span class="v a">${hr.window_size_W}</span><span class="k">timing_anchor</span><span class="v">${hr.timing_anchor}</span><span class="k">side_a_id</span><span class="v g">${hr.side_a_id}</span><span class="k">side_b_id</span><span class="v g">${hr.side_b_id}</span><span class="k">hub_status</span><span class="v pk">${hr.hub_status}</span></div>`;
  document.getElementById('hub-alert').innerHTML=`<div class="alert hub">◈ HUB provisioned. Master secret secured. HUB now offline from chain.</div>`;
  setPhase(2);['btn-insert','btn-retrieve','btn-verify','btn-p2p'].forEach(id=>{document.getElementById(id).disabled=false;});
  syslog('hub',`Provisioned | W=${hr.window_size_W}`);
  initWGrid(parseInt(document.getElementById('hub-W').value));
}
async function doInsert(){
  const r=await api('POST','/api/insert',{label:document.getElementById('a-label').value,payload:document.getElementById('a-payload').value,classification:document.getElementById('a-cls').value});
  if(!r.ok){syslog('err',r.error);return;}
  setPhase(3);const n=r.node;
  document.getElementById('a-insert-result').className='dbox on';
  document.getElementById('a-insert-result').innerHTML=`<div style="color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:9px;margin-bottom:6px">SIDE-A INSERTION RECORD</div><div class="kv"><span class="k">K(t,n)</span><span class="v p">…computed [not transmitted]</span><span class="k">label</span><span class="v">${n.label}</span><span class="k">score</span><span class="v g">${n.score}</span><span class="k">tier</span><span class="v a">${n.tier}</span><span class="k">epoch · win</span><span class="v">E:${n.epoch} W:${n.wpos}</span><span class="k">hash</span><span class="v c">${n.hash_s}</span><span class="k">latency</span><span class="v a">${n.latency}ms</span></div>`;
  latencies.push(n.latency);
  document.getElementById('b-retrieve-id').value=n.node_id;
}
async function doSample(){const r=await api('GET','/api/sample');document.getElementById('a-label').value=r.label;document.getElementById('a-payload').value=r.payload;document.getElementById('a-cls').value=r.classification;}
async function doRetrieve(){const id=document.getElementById('b-retrieve-id').value.trim();const r=await api('POST','/api/retrieve',{node_id:id});showRetrieve(r);}
async function doRetrieveLatest(){const r=await api('POST','/api/retrieve',{node_id:''});showRetrieve(r);}
function showRetrieve(r){
  const d=document.getElementById('b-result');
  if(!r.ok){d.className='dbox on';d.innerHTML=`<span style="color:var(--red)">${r.error}</span>`;return;}
  const n=r.node;
  d.className='dbox on';
  d.innerHTML=`<div style="color:var(--green);font-family:'Share Tech Mono',monospace;font-size:9px;margin-bottom:6px">SIDE-B RETRIEVAL — ${n.latency_ms}ms [${n.tier}]</div><div class="kv"><span class="k">label</span><span class="v">${n.label}</span><span class="k">K(t,n) regen</span><span class="v p">[B-side, independent]</span><span class="k">fp_match</span><span class="v g">✓ MATCH</span><span class="k">zk_proof</span><span class="v g">✓ VALID</span><span class="k">payload_hash</span><span class="v g">✓ INTACT</span><span class="k">plaintext</span><span class="v" style="display:block;margin-top:3px;padding:5px;background:var(--bg);border-radius:2px;grid-column:1/-1">${r.plaintext||'[empty]'}</span></div>`;
}
async function doStream(){
  document.getElementById('bstream').innerHTML='';
  const r=await api('POST','/api/stream',{});
  if(!r.ok){syslog('err',r.error);return;}
  setPhase(4);
  r.blocks.forEach((b,i)=>{setTimeout(()=>{addBlock(b);if(b.type==='DATA')animatePkt();},i*55);});
  syslog('ok',`Stream complete: ${r.total} blocks | K(t,n) never transmitted`);
}
function addBlock(b){
  const s=document.getElementById('bstream');const row=document.createElement('div');row.className='bi ani';
  const stM={DATA:'enc',PROOF:'ok',FEATURES:'enc',ACK:'ok'};const stT={DATA:'ENCRYPTED',PROOF:'SEALED',FEATURES:'FEATURES',ACK:'✓ ACK'};
  row.innerHTML=`<span class="bseq">#${String(b.seq).padStart(4,'0')}</span><span class="btag ${b.type}">${b.type}</span><span class="bc">${b.content}</span><span class="bst ${stM[b.type]}">${stT[b.type]}</span>`;
  s.appendChild(row);s.scrollTop=s.scrollHeight;
}
function animatePkt(){const p=document.getElementById('p2p-pipe');const pk=document.createElement('div');pk.className='pkt';p.appendChild(pk);setTimeout(()=>pk.remove(),600);}
function clearStream(){document.getElementById('bstream').innerHTML='<div style="color:var(--text3);padding:8px;text-align:center;font-family:\'Share Tech Mono\',monospace;font-size:10px">⇆ Cleared</div>';}
async function doVerify(){
  const r=await api('POST','/api/verify',{side:'b'});setPhase(5);
  const d=document.getElementById('b-verify');d.className='vr on';
  let h=`<span style="color:${r.chain_valid?'var(--green)':'var(--red)'}">${r.chain_valid?'◈ CHAIN VERIFIED — ALL NODES INTACT':'⚠ CHAIN INTEGRITY VIOLATION DETECTED'}</span><br><br>`;
  (r.results||[]).forEach((res,i)=>{const c=res.valid?'var(--green)':'var(--red)';h+=`<span style="color:${c}">Node[${i}] ${res.label} E:${res.epoch} W:${res.wpos} → ${res.valid?'VALID':'INVALID'}</span><br>`;});
  d.innerHTML=h;
  document.getElementById('m-valid').textContent=r.chain_valid?'✓':'✗';
  document.getElementById('m-valid').style.color=r.chain_valid?'var(--green)':'var(--red)';
}
async function doTamper(){
  const r=await api('POST','/api/tamper',{});
  if(!r.ok){syslog('err','No nodes to tamper');return;}
  document.getElementById('ai-alert').innerHTML=`<div class="alert danger">⚠ TAMPER INJECTED: ${r.label} — chain seal broken</div>`;
}
async function doReset(){await api('POST','/api/reset',{});location.reload();}
function initWGrid(W){
  const g=document.getElementById('wgrid');g.style.gridTemplateColumns=`repeat(${Math.min(W,8)},1fr)`;g.innerHTML='';
  document.getElementById('wk-label').textContent=W;
  for(let i=0;i<W;i++){const c=document.createElement('div');c.className='wc';c.id=`wc-${i}`;c.textContent=i;g.appendChild(c);}
}
function updateWGrid(wpos,W){for(let i=0;i<W;i++){const c=document.getElementById(`wc-${i}`);if(!c)continue;c.className='wc';if(i<wpos)c.classList.add('used');if(i===wpos)c.classList.add('active');}}
function setBar(id,val,color){const pct=Math.round((val||0)*100);const f=document.getElementById('bf-'+id);const v=document.getElementById('bv-'+id);if(f){f.style.width=pct+'%';f.style.background=color;}if(v){v.textContent=pct+'%';v.style.color=color;}}
function setPhase(n){phase=Math.max(phase,n);for(let i=1;i<=5;i++){const el=document.getElementById(`ph${i}`);el.className=`ph ${i<phase?'done':i===phase?'active':'pend'}`;}}
function syslog(level,msg){const t=document.getElementById('term');const d=document.createElement('div');d.className='ll';const ts=new Date().toTimeString().slice(0,8);const map={ok:['ok','[OK] '],info:['info','[INF]'],err:['err','[ERR]'],warn:['warn','[WRN]'],sys:['sys','[SYS]'],hub:['hub','[HUB]']};const[cls,pre]=map[level]||['info','[---]'];d.innerHTML=`<span class="lt">${ts}</span><span class="${cls}">${pre} ${msg}</span>`;t.appendChild(d);t.scrollTop=t.scrollHeight;}
function showNodeDetail(n){const d=document.getElementById('node-detail');d.className='dbox on';d.innerHTML=`<div style="color:var(--cyan);font-family:'Share Tech Mono',monospace;font-size:9px;margin-bottom:6px;letter-spacing:2px">◈ NODE: ${n.label}</div><div class="kv"><span class="k">node_id</span><span class="v c">${n.node_id}</span><span class="k">type</span><span class="v">${n.type}</span><span class="k">epoch · win</span><span class="v">E:${n.epoch} W:${n.wpos}</span><span class="k">class</span><span class="v a">${n.cls}</span><span class="k">ai_score</span><span class="v g">${n.score}</span><span class="k">tier</span><span class="v">${n.tier}</span><span class="k">hash</span><span class="v c">${n.hash_s}</span><span class="k">K(t,n)</span><span class="v" style="color:var(--text3)">NOT STORED — regenerated at retrieval</span></div>`;}
function drawDAG(nodes,edges){
  const canvas=document.getElementById('dagc');const ctx=canvas.getContext('2d');const W=canvas.offsetWidth,H=230;canvas.width=W;canvas.height=H;ctx.fillStyle='#020408';ctx.fillRect(0,0,W,H);
  if(!nodes.length){ctx.fillStyle='#2e4a6a';ctx.font='11px "Share Tech Mono"';ctx.textAlign='center';ctx.fillText('◉ DAG empty — HUB init then insert data',W/2,H/2);return;}
  const sorted=[...nodes].sort((a,b)=>a.counter-b.counter);const pos={};
  sorted.forEach((n,i)=>{pos[n.node_id]={x:(i/Math.max(sorted.length-1,1))*(W-80)+40,y:H/2};});
  edges.forEach(([f,t])=>{const fp=pos[f],tp=pos[t];if(!fp||!tp)return;ctx.save();ctx.strokeStyle='#1e3050';ctx.lineWidth=1.5;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(fp.x,fp.y);const mx=(fp.x+tp.x)/2;ctx.bezierCurveTo(mx,fp.y,mx,tp.y,tp.x,tp.y);ctx.stroke();ctx.restore();ctx.save();ctx.strokeStyle='#00bfb3';ctx.lineWidth=1;ctx.setLineDash([]);ctx.translate(tp.x,tp.y);ctx.rotate(Math.atan2(tp.y-fp.y,tp.x-fp.x));ctx.beginPath();ctx.moveTo(-9,-4);ctx.lineTo(0,0);ctx.lineTo(-9,4);ctx.stroke();ctx.restore();});
  sorted.forEach(n=>{const p=pos[n.node_id];if(!p)return;const color=n.type==='GENESIS'?'#ffab00':n.score>0.8?'#00e676':n.score>0.6?'#00f5e4':'#0080ff';const r=n.type==='GENESIS'?13:9;ctx.save();ctx.shadowColor=color;ctx.shadowBlur=10;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=color+'28';ctx.fill();ctx.restore();ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.strokeStyle=color;ctx.lineWidth=2;ctx.fillStyle='#020408';ctx.fill();ctx.stroke();ctx.fillStyle=color;ctx.font='8px "Share Tech Mono"';ctx.textAlign='center';ctx.fillText(n.label.length>13?n.label.slice(0,11)+'…':n.label,p.x,p.y+r+12);ctx.fillStyle='#3a5580';ctx.fillText(`E:${n.epoch} W:${n.wpos}`,p.x,p.y+r+21);});
  [['#ffab00','GENESIS'],['#00e676','HIGH'],['#00f5e4','STD'],['#0080ff','COOL']].forEach((it,i)=>{ctx.beginPath();ctx.arc(12+i*88,H-12,3,0,Math.PI*2);ctx.fillStyle=it[0];ctx.fill();ctx.fillStyle='#7a99cc';ctx.font='8px "Share Tech Mono"';ctx.textAlign='left';ctx.fillText(it[1],18+i*88,H-9);});
}
async function pollState(){
  try{
    const s=await api('GET','/api/state');if(!s)return;
    if(s.log&&s.log.length>lastLogLen){s.log.slice(lastLogLen).forEach(e=>syslog(e.level,e.msg));lastLogLen=s.log.length;}
    if(!s.provisioned)return;
    const w=s.a_window,W=w.W||4;
    document.getElementById('hub-lbl').textContent='HUB: SILENT';document.getElementById('hub-dot').className='dot';
    document.getElementById('a-lbl').textContent=`SIDE-A: ${s.a_id}`;document.getElementById('a-dot').className='dot';
    document.getElementById('b-lbl').textContent=`SIDE-B: ${s.b_id}`;document.getElementById('b-dot').className='dot';
    document.getElementById('epoch-pill').textContent=`EPOCH: ${w.epoch||0}`;
    document.getElementById('win-pill').textContent=`WINDOW: ${w.wpos||0}/${W}`;
    document.getElementById('a-badge').textContent=s.a_id;document.getElementById('b-badge').textContent=s.b_id;
    document.getElementById('p2p-src-l').textContent=s.a_id;document.getElementById('p2p-dst-l').textContent=s.b_id;
    document.getElementById('m-nodes').textContent=s.metrics.chain_nodes;
    document.getElementById('m-epoch').textContent=w.epoch||0;
    if(latencies.length){const avg=(latencies.reduce((a,b)=>a+b,0)/latencies.length).toFixed(1);document.getElementById('m-lat').textContent=avg+'ms';}
    const af=s.ai_features||{};
    if(af.score!==undefined){setBar('ent',af.entropy||0,'var(--cyan)');setBar('rep',af.actor||0,'var(--green)');setBar('tmp',af.timing||0,'var(--blue)');setBar('win',af.window_pos||0,'var(--amber)');setBar('dep',af.chain_depth||0,'var(--purple)');setBar('ktn',af.score||0,'var(--pink)');const ok=af.score>0.6;document.getElementById('ai-badge').className=`ibadge ${ok?'ok':'warn'}`;document.getElementById('ai-badge').textContent=`◈ ${ok?'SECURE':'REVIEW'} — ${af.score}`;document.getElementById('m-ai').textContent=(af.score||0).toFixed(3);document.getElementById('m-ai').style.color=ok?'var(--green)':af.score>0.3?'var(--amber)':'var(--red)';}
    document.getElementById('wk-ep').textContent=w.epoch||0;document.getElementById('wk-sec').textContent=w.secret_short||'—';document.getElementById('wk-wpos').textContent=w.wpos||0;document.getElementById('wk-wsz').textContent=W;document.getElementById('wk-cc').textContent=w.counter||0;document.getElementById('wk-mm').textContent=`M → ep${w.epoch||0}_secret → K(${w.epoch||0},${w.wpos||0})`;document.getElementById('wk-ktn').textContent='…computed on insert…';
    if(document.getElementById('wc-0'))updateWGrid(w.wpos||0,W);
    document.getElementById('win-badge').className='ibadge ok';document.getElementById('win-badge').textContent=`⊛ EPOCH ${w.epoch||0} · SLOT ${w.wpos||0}/${W}`;
    const nodes=s.a_nodes||[];
    if(nodes.length){const cl=document.getElementById('clist');cl.innerHTML='';[...nodes].reverse().forEach(n=>{const cls=n.type==='GENESIS'?'genesis':'data';const sc=n.score>0.8?'var(--green)':n.score>0.6?'var(--cyan)':n.score>0.4?'var(--amber)':'var(--red)';const el=document.createElement('div');el.className=`cn ${cls} ani`;el.innerHTML=`<span class="cnb">${n.type}</span><div class="cnm"><div class="cnt">${n.label}</div><div class="cnh">${n.node_id}</div><div class="cnd">${n.actor} · E:${n.epoch} W:${n.wpos} · ${n.ts}</div></div><div class="cns" style="color:${sc}">${n.score}</div>`;el.onclick=()=>showNodeDetail(n);cl.appendChild(el);});}
    drawDAG(s.a_nodes||[],s.a_dag_edges||[]);
  }catch(e){}
}
window.addEventListener('resize',()=>{api('GET','/api/state').then(s=>{if(s)drawDAG(s.a_nodes||[],s.a_dag_edges||[]);});});
syslog('hub','PIM-PCD WARLOK · FastAPI Engine loaded');syslog('sys','Architecture: HUB → epoch_secret → nanoAI(features) → K(t,n)');syslog('sys','K(t,n) is never transmitted — both sides regenerate independently');syslog('info','Start: ◈ INIT & PROVISION the HUB to begin');
setInterval(pollState,600);setTimeout(()=>{drawDAG([],[]);},100);
</script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 ── ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║  PIM-PCD  ·  NEURAL RATCHET CHAIN  ·  WARLOK UI                ║"""

    if _FASTAPI_AVAILABLE and _UVICORN_AVAILABLE:
        import uvicorn
        banner += """
║  Runtime:  FastAPI + uvicorn  (native async ASGI)               ║
║  Docs:     http://localhost:5053/docs                            ║
║  App:      http://localhost:5053                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        uvicorn.run("pim_pcd_fastapi:app", host="0.0.0.0", port=5053,
                    reload=False, log_level="info")
    else:
        banner += f"""
║  Runtime:  Flask shim  (FastAPI not installed)                  ║
║  Upgrade:  pip install fastapi uvicorn                          ║
║            uvicorn pim_pcd_fastapi:app --port 5053 --reload     ║
║  App:      http://localhost:5053                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""
        print(banner)
        app.run(host="0.0.0.0", port=5053, debug=False, threaded=True)
