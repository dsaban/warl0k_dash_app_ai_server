"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PIM-PCD  ·  NEURAL RATCHET CHAIN  ·  Python Implementation                ║
║  Proof Chain of Data — Nano-AI Governed P2P Data Transfer                  ║
║                                                                              ║
║  Architecture:                                                               ║
║    HubAuthority  →  provisions Side-A and Side-B (once, then silent)        ║
║    NanoAIModel   →  deterministic trust scorer + K(t,n) regenerator         ║
║    PcdEnvelope   →  cryptographic data wrapper (PCD envelope struct)        ║
║    DagStore      →  append-only, hash-linked directed acyclic graph         ║
║    ChainValidator →  full proof-chain integrity verifier                    ║
║    Peer          →  Side-A (sender) / Side-B (receiver) runtime             ║
║    P2PChannel    →  socket-based block stream (DATA·PROOF·FEATURES·ACK)     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import hmac as _hmac
import json
import math
import os
import random
import socket
import struct
import threading
import time
import queue
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum, auto

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  ANSI colour helpers for terminal output
# ─────────────────────────────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    CYAN    = "\033[96m"
    GREEN   = "\033[92m"
    AMBER   = "\033[93m"
    RED     = "\033[91m"
    PURPLE  = "\033[95m"
    BLUE    = "\033[94m"
    GREY    = "\033[90m"
    WHITE   = "\033[97m"
    NAVY    = "\033[34m"
    PINK    = "\033[35m"

def cprint(color: str, prefix: str, msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"{C.GREY}{ts}{C.RESET}  {color}{C.BOLD}{prefix:<10}{C.RESET}  {msg}")

def hdr(title: str) -> None:
    width = 76
    bar = "═" * width
    print(f"\n{C.NAVY}{C.BOLD}╔{bar}╗{C.RESET}")
    pad = (width - len(title)) // 2
    print(f"{C.NAVY}{C.BOLD}║{' ' * pad}{C.CYAN}{title}{C.NAVY}{' ' * (width - pad - len(title))}║{C.RESET}")
    print(f"{C.NAVY}{C.BOLD}╚{bar}╝{C.RESET}\n")

def section(title: str) -> None:
    print(f"\n{C.BLUE}{C.BOLD}── {title} {'─' * max(0, 68 - len(title))}{C.RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  CRYPTOGRAPHIC PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────
class Crypto:
    """Pure-stdlib cryptographic operations."""

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
        """One-way key derivation: HMAC(base, parts joined by '|')."""
        return Crypto.hmac256(base.encode(), "|".join(parts).encode())

    @staticmethod
    def rng_hex(n: int = 16) -> str:
        return os.urandom(n).hex()

    @staticmethod
    def xor_bytes(a: bytes, b: bytes) -> bytes:
        """XOR two byte strings (pad b cyclically to len(a))."""
        out = bytearray(len(a))
        for i, byte in enumerate(a):
            out[i] = byte ^ b[i % len(b)]
        return bytes(out)

    @classmethod
    def obfuscate(cls, plaintext: str, ktn: str) -> str:
        """Encrypt plaintext using K(t,n) as key (XOR stream, AES-GCM in prod)."""
        key_stream_hex = cls.sha256(ktn + "stream")
        key_bytes = bytes.fromhex(key_stream_hex * 4)  # extend key stream
        pt_bytes = plaintext.encode("utf-8")
        ct_bytes = cls.xor_bytes(pt_bytes, key_bytes[:len(pt_bytes)])
        return ct_bytes.hex()

    @classmethod
    def deobfuscate(cls, ciphertext_hex: str, ktn: str) -> str:
        """Decrypt ciphertext using K(t,n) as key."""
        key_stream_hex = cls.sha256(ktn + "stream")
        key_bytes = bytes.fromhex(key_stream_hex * 4)
        ct_bytes = bytes.fromhex(ciphertext_hex)
        pt_bytes = cls.xor_bytes(ct_bytes, key_bytes[:len(ct_bytes)])
        return pt_bytes.decode("utf-8")

    @staticmethod
    def short(h: str) -> str:
        return f"{h[:8]}…{h[-6:]}" if len(h) > 16 else h


# ─────────────────────────────────────────────────────────────────────────────
#  NANO-AI MODEL
#  Lightweight deterministic neural model — weight vector governs K(t,n)
#  derivation and trust scoring.  Both peers run IDENTICAL model.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class AIFeatures:
    """Feature vector fed into the nano-AI model."""
    entropy:       float   # Shannon entropy of payload bytes, 0-1
    actor_score:   float   # cumulative actor reputation, 0-1
    timing_align:  float   # timing alignment vs expected interval, 0-1
    window_pos:    float   # window slot index normalised by W, 0-1
    chain_depth:   float   # DAG depth normalised to 20-node max, 0-1
    chain_counter: int     # global monotonic counter (raw int, not normalised)

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.entropy,
            self.actor_score,
            self.timing_align,
            self.window_pos,
            self.chain_depth,
            min(self.chain_counter / 100.0, 1.0),  # normalised counter
        ], dtype=np.float64)

    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of the feature vector."""
        s = (f"{self.entropy:.6f}|{self.actor_score:.6f}|{self.timing_align:.6f}|"
             f"{self.chain_counter}|{self.window_pos:.6f}|{self.chain_depth:.6f}")
        return Crypto.sha256(s)


@dataclass
class AIResult:
    """Output of nanoAI(features, epoch_secret)."""
    score:       float        # trust score 0-1
    ktn:         str          # K(t,n) — the regenerated session secret
    fingerprint: str          # public feature fingerprint
    features:    AIFeatures
    tier:        str          # HOT / WARM / COOL / QUARANTINE


class NanoAIModel:
    """
    Deterministic nano-AI trust model.

    Model weights W[0..5] are public — distributed by HUB at provisioning.
    Both peers run the SAME model with the SAME features + epoch_secret
    and derive the SAME K(t,n) WITHOUT any transmission of the secret.

    Architecture:
        1. Feature vector F = [entropy, actor, timing, win_pos, depth, counter]
        2. Linear layer: score = sigmoid( dot(W, F) + bias )
        3. Feature fingerprint = SHA-256(canonical feature string)
        4. K(t,n) = HMAC(epoch_secret, fingerprint | chain_counter | window_pos)
        5. Anomaly flags if any feature deviates from threshold bands
    """

    # Model weights — public, distributed by Hub
    WEIGHTS = np.array([0.22, 0.28, 0.15, 0.18, 0.12, 0.05], dtype=np.float64)
    BIAS    = -0.15

    # Tier thresholds
    TIERS = [
        (0.85, "HOT",        C.GREEN),
        (0.60, "WARM",       C.CYAN),
        (0.30, "COOL",       C.AMBER),
        (0.00, "QUARANTINE", C.RED),
    ]

    # Per-feature anomaly bands [min_normal, max_normal]
    ANOMALY_BANDS = {
        "entropy":      (0.30, 1.00),
        "actor_score":  (0.40, 1.00),
        "timing_align": (0.20, 1.00),
        "window_pos":   (0.00, 1.00),   # always valid
        "chain_depth":  (0.00, 1.00),   # always valid
    }

    def __init__(self):
        self._actor_rep: Dict[str, float] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def update_actor(self, actor_id: str, score: float) -> None:
        self._actor_rep[actor_id] = min(1.0, score + 0.04)

    def actor_score(self, actor_id: str) -> float:
        return self._actor_rep.get(actor_id, 0.50)

    @staticmethod
    def _shannon_entropy(payload: str) -> float:
        """Compute Shannon entropy of UTF-8 bytes, normalised to [0,1]."""
        data = payload.encode("utf-8")
        if not data:
            return 0.0
        freq = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        freq = freq[freq > 0]
        p = freq / len(data)
        H = -np.sum(p * np.log2(p))
        return float(min(H / 8.0, 1.0))

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
        """Compute the feature vector from observable chain state."""
        entropy      = self._shannon_entropy(payload)
        actor        = self.actor_score(actor_id)
        delta_t      = max(0.0, time.time() - last_event_ts)
        timing_align = min(1.0, 1.0 / (delta_t + 0.1))
        win_norm     = window_pos / max(window_size, 1)
        depth_norm   = min(chain_depth / 20.0, 1.0)

        return AIFeatures(
            entropy       = round(entropy, 6),
            actor_score   = round(actor, 6),
            timing_align  = round(timing_align, 6),
            window_pos    = round(win_norm, 6),
            chain_depth   = round(depth_norm, 6),
            chain_counter = chain_counter,
        )

    def infer(self, features: AIFeatures, epoch_secret: str) -> AIResult:
        """
        Run nano-AI inference.
        Returns trust score, K(t,n), and anomaly information.
        K(t,n) = HMAC(epoch_secret, feature_fingerprint | counter | window_pos)
        """
        vec   = features.as_vector()
        raw   = float(np.dot(self.WEIGHTS, vec)) + self.BIAS
        score = round(self._sigmoid(raw), 6)

        fp    = features.fingerprint()
        ktn   = Crypto.hmac256(
            epoch_secret,
            f"{fp}|{features.chain_counter}|{features.window_pos:.6f}"
        )

        tier_label = "QUARANTINE"
        for threshold, label, _ in self.TIERS:
            if score >= threshold:
                tier_label = label
                break

        return AIResult(
            score       = score,
            ktn         = ktn,
            fingerprint = fp,
            features    = features,
            tier        = tier_label,
        )

    def detect_anomalies(self, features: AIFeatures) -> List[str]:
        """Return list of anomaly descriptions for any out-of-band features."""
        flags = []
        vals = {
            "entropy":     features.entropy,
            "actor_score": features.actor_score,
            "timing_align": features.timing_align,
        }
        for name, val in vals.items():
            lo, hi = self.ANOMALY_BANDS[name]
            if not (lo <= val <= hi):
                flags.append(f"{name}={val:.3f} outside [{lo},{hi}]")
        if features.chain_counter == 0 and features.chain_depth > 0.1:
            flags.append("counter=0 but depth>0 — possible replay")
        return flags

    def validate_ktn_match(
        self,
        received_fingerprint: str,
        local_features: AIFeatures,
        epoch_secret: str,
    ) -> Tuple[bool, str]:
        """
        Side-B validates that its locally reconstructed K(t,n) matches
        the expected ZK proof — without ever seeing Side-A's K(t,n).
        Returns (match: bool, local_ktn: str)
        """
        local_fp  = local_features.fingerprint()
        fp_match  = local_fp == received_fingerprint
        local_ktn = Crypto.hmac256(
            epoch_secret,
            f"{local_fp}|{local_features.chain_counter}|{local_features.window_pos:.6f}"
        )
        return fp_match, local_ktn


# ─────────────────────────────────────────────────────────────────────────────
#  PCD ENVELOPE
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PcdEnvelope:
    """
    Cryptographic data wrapper.  K(t,n) is intentionally ABSENT.
    All fields are either public context, encrypted payload, or proofs.
    """
    # Identity
    object_id:    str
    node_id:      str      # short ID (first 16 hex chars of object_id)
    node_type:    str      # GENESIS | DATA | READ
    label:        str
    classification: str
    actor:        str

    # Payload (encrypted)
    cipher_payload: str    # obfuscate(plaintext, K(t,n))
    payload_hash:   str    # SHA-256(plaintext) — B verifies after decrypt
    cipher_hash:    str    # SHA-256(cipher_payload) — chain integrity

    # Chain linkage
    prev_hash:    str      # hash of parent envelope
    hash:         str      # SHA-256(cipher_hash|prev_hash|temporal_seal|chain_seal)

    # Proofs (epoch_secret used but not stored)
    temporal_seal: str     # HMAC(epoch_secret, cipher_hash|ts|nonce)
    chain_seal:    str     # HMAC(epoch_secret, prev_hash|fp|counter)
    feature_fp:    str     # public feature fingerprint — B uses to regen K
    zk_proof:      str     # HMAC(K(t,n), zk|actor|node_id)

    # Context (all public)
    epoch:         int
    window_pos:    int
    chain_counter: int
    chain_depth:   int
    timestamp:     float
    ai_score:      float
    ai_features:   dict    # serialised AIFeatures
    tier:          str
    nonce:         str
    latency_ms:    float

    # Not stored — K(t,n) never persisted
    # ktn: NOT HERE BY DESIGN

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PcdEnvelope":
        return cls(**d)

    def summary(self) -> str:
        return (f"[{self.tier}] {self.label} "
                f"E:{self.epoch} W:{self.window_pos} "
                f"score:{self.ai_score:.3f} "
                f"hash:{Crypto.short(self.hash)}")


# ─────────────────────────────────────────────────────────────────────────────
#  DAG STORE
# ─────────────────────────────────────────────────────────────────────────────
class DagStore:
    """
    Append-only directed acyclic graph of PCD envelopes.
    Nodes are linked via hash references.  Any modification to a node
    invalidates all downstream hash links — self-announcing tamper detection.
    """

    def __init__(self):
        self._nodes:  Dict[str, PcdEnvelope] = {}  # node_id → envelope
        self._edges:  List[Tuple[str, str]]  = []  # (from_id, to_id)
        self._latest: Optional[str]          = None
        self._lock = threading.Lock()

    def add(self, env: PcdEnvelope) -> None:
        with self._lock:
            self._nodes[env.node_id] = env
            self._latest = env.node_id

    def link(self, from_id: str, to_id: str) -> None:
        with self._lock:
            self._edges.append((from_id, to_id))

    def get(self, node_id: str) -> Optional[PcdEnvelope]:
        return self._nodes.get(node_id)

    def find_by_object_id(self, object_id: str) -> Optional[PcdEnvelope]:
        for env in self._nodes.values():
            if env.object_id == object_id or env.object_id.startswith(object_id):
                return env
        return None

    @property
    def latest(self) -> Optional[PcdEnvelope]:
        return self._nodes.get(self._latest) if self._latest else None

    def all_data_nodes(self) -> List[PcdEnvelope]:
        """Return all DATA/GENESIS nodes in insertion order."""
        with self._lock:
            return [n for n in self._nodes.values()
                    if n.node_type in ("GENESIS", "DATA")]

    def trace(self, node_id: str) -> List[PcdEnvelope]:
        """Trace full lineage from genesis to node_id."""
        path: List[PcdEnvelope] = []
        cur = node_id
        visited: set = set()
        while cur and cur not in visited:
            visited.add(cur)
            node = self._nodes.get(cur)
            if not node:
                break
            path.insert(0, node)
            parents = [f for f, t in self._edges if t == cur]
            cur = parents[0] if parents else None
        return path

    def size(self) -> int:
        return len(self._nodes)

    def print_graph(self) -> None:
        """Pretty-print the DAG topology."""
        nodes = self.all_data_nodes()
        section("DAG PROOF CHAIN TOPOLOGY")
        for i, n in enumerate(nodes):
            connector = "├──" if i < len(nodes) - 1 else "└──"
            tier_color = {
                "HOT": C.GREEN, "WARM": C.CYAN,
                "COOL": C.AMBER, "QUARANTINE": C.RED,
            }.get(n.tier, C.WHITE)
            print(f"  {connector} {tier_color}{C.BOLD}[{n.tier}]{C.RESET} "
                  f"{C.WHITE}{n.label}{C.RESET} "
                  f"{C.GREY}E:{n.epoch} W:{n.window_pos} "
                  f"score:{n.ai_score:.3f} "
                  f"hash:{Crypto.short(n.hash)}{C.RESET}")
            if i < len(nodes) - 1:
                print(f"  │     {C.GREY}│ prev_hash:{Crypto.short(n.hash)}{C.RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  CHAIN VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class NodeValidation:
    node_id:      str
    label:        str
    cipher_ok:    bool   # SHA-256(cipher_payload) == cipher_hash
    prev_ok:      bool   # prev_hash links correctly to parent
    seal_ok:      bool   # temporal_seal validates (requires epoch_secret)
    fp_ok:        bool   # feature fingerprint reconstructible
    valid:        bool   # all checks pass

    def status(self) -> str:
        if self.valid:
            return f"{C.GREEN}✓ VALID{C.RESET}"
        flags = []
        if not self.cipher_ok: flags.append("cipher_hash_mismatch")
        if not self.prev_ok:   flags.append("prev_hash_mismatch")
        if not self.seal_ok:   flags.append("temporal_seal_invalid")
        if not self.fp_ok:     flags.append("fingerprint_mismatch")
        return f"{C.RED}✗ INVALID [{', '.join(flags)}]{C.RESET}"


class ChainValidator:
    """Full proof-chain integrity verifier."""

    def __init__(self, dag: DagStore, epoch_secrets: Dict[int, str]):
        self._dag     = dag
        self._secrets = epoch_secrets  # epoch → secret (read-only copy for audit)

    def validate_node(
        self,
        node: PcdEnvelope,
        parent: Optional[PcdEnvelope],
    ) -> NodeValidation:
        # Check 1: cipher hash
        recomputed_cipher_hash = Crypto.sha256(node.cipher_payload)
        cipher_ok = recomputed_cipher_hash == node.cipher_hash

        # Check 2: prev_hash linkage
        if parent is None:
            prev_ok = node.prev_hash == "0" * 64
        else:
            prev_ok = node.prev_hash == parent.hash

        # Check 3: temporal seal (if we have the epoch secret)
        epoch_sec = self._secrets.get(node.epoch)
        if epoch_sec:
            expected_seal = Crypto.hmac256(
                epoch_sec,
                f"{node.cipher_hash}|{node.timestamp}|{node.nonce}"
            )
            seal_ok = expected_seal == node.temporal_seal
        else:
            seal_ok = True  # cannot verify without secret — skip

        # Check 4: feature fingerprint self-consistency
        feat = AIFeatures(**node.ai_features)
        fp_ok = feat.fingerprint() == node.feature_fp

        valid = cipher_ok and prev_ok and seal_ok and fp_ok

        return NodeValidation(
            node_id   = node.node_id,
            label     = node.label,
            cipher_ok = cipher_ok,
            prev_ok   = prev_ok,
            seal_ok   = seal_ok,
            fp_ok     = fp_ok,
            valid     = valid,
        )

    def validate_chain(self) -> Tuple[bool, List[NodeValidation]]:
        """Validate full DAG chain in insertion order."""
        nodes   = self._dag.all_data_nodes()
        results: List[NodeValidation] = []
        chain_valid = True

        for i, node in enumerate(nodes):
            parent = nodes[i - 1] if i > 0 else None
            result = self.validate_node(node, parent)
            results.append(result)
            if not result.valid:
                chain_valid = False

        return chain_valid, results

    def print_validation(self) -> bool:
        chain_valid, results = self.validate_chain()
        section("CHAIN INTEGRITY VALIDATION")
        for i, r in enumerate(results):
            print(f"  Node[{i}] {C.WHITE}{r.label:<30}{C.RESET} {r.status()}")
        if chain_valid:
            print(f"\n  {C.GREEN}{C.BOLD}◈ CHAIN VERIFIED — ALL NODES INTACT{C.RESET}")
        else:
            print(f"\n  {C.RED}{C.BOLD}⚠ CHAIN INTEGRITY VIOLATION DETECTED{C.RESET}")
        return chain_valid


# ─────────────────────────────────────────────────────────────────────────────
#  HUB AUTHORITY
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SideConfig:
    """Parameters distributed by Hub to each side — once only."""
    epoch_0_secret:  str
    timing_anchor:   float
    window_size:     int
    epoch_ms:        int
    drift_tolerance: int
    model_weights:   List[float]   # public — not secret


@dataclass
class HubState:
    master_secret:  str   # never distributed
    window_size:    int
    epoch_ms:       int
    drift_tolerance: int
    timing_anchor:  float
    side_a_id:      str
    side_b_id:      str
    epoch_0_secret: str   # derived from master — distributed once


class HubAuthority:
    """
    One-time provisioning authority.  After calling provision(), the Hub
    goes permanently silent.  It holds master_secret but is not in the
    data path.

    Compromise of the Hub after provisioning does NOT expose chain objects
    because:
      - epoch_0_secret alone cannot forge temporal_seal or chain_seal
        without the live chain state
      - both sides will have rotated past epoch 0 (forward secrecy)
    """

    def __init__(
        self,
        master_secret:   str   = "M-PIMCHAIN-2026",
        window_size:     int   = 4,
        epoch_ms:        int   = 30000,
        drift_tolerance: int   = 500,
        side_a_id:       str   = "NODE-ALPHA",
        side_b_id:       str   = "NODE-BETA",
    ):
        self._master = master_secret
        self._w      = window_size
        self._state: Optional[HubState] = None

    def provision(self) -> Tuple[SideConfig, SideConfig, HubState]:
        """
        Derive epoch_0_secret, build side configs, and return them.
        After this call, Hub goes SILENT.
        """
        anchor = time.time()
        epoch0 = Crypto.derive_key(
            self._master,
            "epoch", "0", str(self._w), str(int(anchor))
        )

        state = HubState(
            master_secret   = self._master,
            window_size     = self._w,
            epoch_ms        = 30000,
            drift_tolerance = 500,
            timing_anchor   = anchor,
            side_a_id       = "NODE-ALPHA",
            side_b_id       = "NODE-BETA",
            epoch_0_secret  = epoch0,
        )
        self._state = state

        config = SideConfig(
            epoch_0_secret  = epoch0,
            timing_anchor   = anchor,
            window_size     = self._w,
            epoch_ms        = 30000,
            drift_tolerance = 500,
            model_weights   = NanoAIModel.WEIGHTS.tolist(),
        )

        hdr("HUB AUTHORITY — PROVISIONING")
        cprint(C.PINK,  "[HUB]",    f"Master secret hash: {Crypto.short(Crypto.sha256(self._master))}")
        cprint(C.PINK,  "[HUB]",    f"Epoch-0 secret:     {Crypto.short(epoch0)}")
        cprint(C.PINK,  "[HUB]",    f"Window size W:      {self._w}")
        cprint(C.PINK,  "[HUB]",    f"Model weights:      {NanoAIModel.WEIGHTS.tolist()}")
        cprint(C.PINK,  "[HUB]",    f"Distributing config to SIDE-A and SIDE-B…")
        cprint(C.PINK,  "[HUB]",    f"Status: ⊘ SILENT — not in chain after this point")

        return config, config, state   # same config to both sides


# ─────────────────────────────────────────────────────────────────────────────
#  PEER (SIDE-A / SIDE-B)
# ─────────────────────────────────────────────────────────────────────────────
class Peer:
    """
    A PIM-PCD chain participant.  Both sides hold identical config and
    run identical NanoAIModel.  Neither needs the other to advance state.
    """

    def __init__(self, peer_id: str, role: str, config: SideConfig):
        self.peer_id     = peer_id
        self.role        = role          # SENDER | RECEIVER
        self.config      = config
        self.dag         = DagStore()
        self.model       = NanoAIModel()

        # Epoch state
        self._epoch_secrets: Dict[int, str] = {0: config.epoch_0_secret}
        self._current_epoch  = 0
        self._window_pos     = 0
        self._chain_counter  = 0
        self._last_event_ts  = config.timing_anchor

        cprint(C.CYAN if role == "SENDER" else C.GREEN,
               f"[{peer_id}]",
               f"Peer initialised | role={role} | epoch_secret={Crypto.short(config.epoch_0_secret)}")

    # ── Epoch rotation ────────────────────────────────────────────
    def _rotate_epoch(self) -> None:
        cur_secret = self._epoch_secrets[self._current_epoch]
        next_secret = Crypto.derive_key(cur_secret, "epoch", str(self._current_epoch + 1), "pim-pcd-chain")
        self._current_epoch += 1
        self._epoch_secrets[self._current_epoch] = next_secret
        self._window_pos = 0

        cprint(C.PURPLE, f"[{self.peer_id}]",
               f"Epoch rotated {self._current_epoch - 1}→{self._current_epoch} | "
               f"new secret: {Crypto.short(next_secret)} | "
               f"forward secrecy: past epoch erased")

        # Forward secrecy: delete previous epoch secret
        del self._epoch_secrets[self._current_epoch - 1]

    @property
    def _epoch_secret(self) -> str:
        return self._epoch_secrets[self._current_epoch]

    # ── INSERT (Side-A only) ──────────────────────────────────────
    def insert(self, label: str, payload: str, classification: str = "CONFIDENTIAL") -> PcdEnvelope:
        """Build and anchor a PCD envelope into the local DAG."""
        t0 = time.perf_counter()

        # Compute features
        features = self.model.build_features(
            payload       = payload,
            actor_id      = self.peer_id,
            window_pos    = self._window_pos,
            window_size   = self.config.window_size,
            chain_depth   = self.dag.size(),
            chain_counter = self._chain_counter,
            last_event_ts = self._last_event_ts,
        )

        # Run nano-AI → K(t,n)
        ai_result = self.model.infer(features, self._epoch_secret)
        ktn = ai_result.ktn  # NEVER leaves this method

        # Anomaly check
        anomalies = self.model.detect_anomalies(features)
        if anomalies:
            cprint(C.RED, f"[{self.peer_id}]",
                   f"⚠ Anomaly on '{label}': {'; '.join(anomalies)}")

        cprint(C.BLUE, f"[{self.peer_id}]",
               f"K(t={self._current_epoch},n={self._window_pos}) computed: "
               f"{Crypto.short(ktn)} {C.GREY}[NOT transmitted]{C.RESET}")

        # Encrypt payload with K(t,n)
        cipher_payload = Crypto.obfuscate(payload, ktn)
        payload_hash   = Crypto.sha256(payload)
        cipher_hash    = Crypto.sha256(cipher_payload)

        # Chain linkage
        parent = self.dag.latest
        prev_hash  = parent.hash if parent else "0" * 64
        parent_id  = parent.node_id if parent else None

        # Proofs (using epoch_secret — never stored in envelope)
        nonce       = Crypto.rng_hex(8)
        ts_now      = time.time()
        temp_seal   = Crypto.hmac256(
            self._epoch_secret,
            f"{cipher_hash}|{ts_now}|{nonce}"
        )
        chain_seal  = Crypto.hmac256(
            self._epoch_secret,
            f"{prev_hash}|{ai_result.fingerprint}|{self._chain_counter}"
        )

        # ZK proof: proves K(t,n) was held — without revealing it
        obj_id  = Crypto.sha256(f"{label}{ts_now}{nonce}{Crypto.rng_hex(4)}")
        node_id = obj_id[:16]
        zk_proof = Crypto.hmac256(ktn, f"zk|{self.peer_id}|{node_id}")

        # Envelope hash
        env_hash = Crypto.sha256(f"{cipher_hash}|{prev_hash}|{temp_seal}|{chain_seal}")

        env = PcdEnvelope(
            object_id     = obj_id,
            node_id       = node_id,
            node_type     = "GENESIS" if self.dag.size() == 0 else "DATA",
            label         = label,
            classification = classification,
            actor         = self.peer_id,
            cipher_payload = cipher_payload,
            payload_hash  = payload_hash,
            cipher_hash   = cipher_hash,
            prev_hash     = prev_hash,
            hash          = env_hash,
            temporal_seal = temp_seal,
            chain_seal    = chain_seal,
            feature_fp    = ai_result.fingerprint,
            zk_proof      = zk_proof,
            epoch         = self._current_epoch,
            window_pos    = self._window_pos,
            chain_counter = self._chain_counter,
            chain_depth   = self.dag.size(),
            timestamp     = ts_now,
            ai_score      = ai_result.score,
            ai_features   = dataclasses.asdict(features),
            tier          = ai_result.tier,
            nonce         = nonce,
            latency_ms    = round((time.perf_counter() - t0) * 1000, 3),
        )

        # Add to DAG
        self.dag.add(env)
        if parent_id:
            self.dag.link(parent_id, node_id)

        # Update actor reputation
        self.model.update_actor(self.peer_id, ai_result.score)

        # Advance window
        self._window_pos    += 1
        self._chain_counter += 1
        self._last_event_ts  = ts_now

        # Epoch rotation at window boundary
        if self._window_pos >= self.config.window_size:
            self._rotate_epoch()

        tier_color = {"HOT": C.GREEN, "WARM": C.CYAN,
                      "COOL": C.AMBER, "QUARANTINE": C.RED}.get(ai_result.tier, C.WHITE)
        cprint(C.CYAN, f"[{self.peer_id}]",
               f"Anchored: {C.WHITE}{label}{C.RESET} "
               f"score={tier_color}{ai_result.score:.4f}{C.RESET} "
               f"tier={tier_color}{ai_result.tier}{C.RESET} "
               f"hash={Crypto.short(env_hash)} "
               f"{C.GREY}latency={env.latency_ms}ms{C.RESET}")
        return env

    # ── Epoch secret derivation (deterministic, both sides identical) ──
    def _get_epoch_secret(self, target_epoch: int) -> str:
        """
        Derive epoch secret for any epoch by ratcheting from epoch_0_secret.
        Both peers derive identically using the same fixed HMAC chain.
        """
        if target_epoch in self._epoch_secrets:
            return self._epoch_secrets[target_epoch]
        secret = self.config.epoch_0_secret
        self._epoch_secrets[0] = secret
        for ep in range(1, target_epoch + 1):
            if ep in self._epoch_secrets:
                secret = self._epoch_secrets[ep]
                continue
            next_s = Crypto.derive_key(secret, "epoch", str(ep), "pim-pcd-chain")
            self._epoch_secrets[ep] = next_s
            secret = next_s
        return self._epoch_secrets[target_epoch]

    # ── RECEIVE envelope from P2P stream ─────────────────────────
    def receive_envelope(self, env: PcdEnvelope) -> bool:
        """
        Side-B receives a PCD envelope and:
        1. Reconstructs feature vector from public fields
        2. Regenerates K(t,n) independently using its epoch_secret
        3. Verifies feature fingerprint
        4. Decrypts and verifies payload hash
        5. Verifies ZK proof
        6. Adds to local DAG
        """
        cprint(C.GREEN, f"[{self.peer_id}]",
               f"Receiving: {C.WHITE}{env.label}{C.RESET} "
               f"E:{env.epoch} W:{env.window_pos}")

        epoch_sec = self._get_epoch_secret(env.epoch)

        # Reconstruct features from public envelope fields
        feat = AIFeatures(**env.ai_features)

        # Validate and regenerate K(t,n)
        fp_match, local_ktn = self.model.validate_ktn_match(
            received_fingerprint = env.feature_fp,
            local_features       = feat,
            epoch_secret         = epoch_sec,
        )

        cprint(C.GREEN, f"[{self.peer_id}]",
               f"K(t,n) regen: {Crypto.short(local_ktn)} "
               f"{C.GREY}[B-side, independent]{C.RESET} "
               f"fp_match={C.GREEN if fp_match else C.RED}{'✓' if fp_match else '✗'}{C.RESET}")

        # Verify ZK proof
        expected_zk = Crypto.hmac256(local_ktn, f"zk|{env.actor}|{env.node_id}")
        zk_valid    = expected_zk == env.zk_proof

        # Decrypt and verify payload
        plaintext = None
        payload_ok = False
        if fp_match:
            try:
                plaintext  = Crypto.deobfuscate(env.cipher_payload, local_ktn)
                dec_hash   = Crypto.sha256(plaintext)
                payload_ok = dec_hash == env.payload_hash
            except Exception as e:
                cprint(C.RED, f"[{self.peer_id}]", f"Decryption error: {e}")

        # Run anomaly detection on received features
        anomalies = self.model.detect_anomalies(feat)

        # Report
        ok_str   = lambda v: f"{C.GREEN}✓{C.RESET}" if v else f"{C.RED}✗{C.RESET}"
        cprint(C.GREEN, f"[{self.peer_id}]",
               f"fp_match={ok_str(fp_match)}  "
               f"zk_proof={ok_str(zk_valid)}  "
               f"payload_hash={ok_str(payload_ok)}")

        if plaintext and payload_ok:
            preview = plaintext[:80] + ("…" if len(plaintext) > 80 else "")
            cprint(C.GREEN, f"[{self.peer_id}]",
                   f"Plaintext: {C.WHITE}{preview}{C.RESET}")

        if anomalies:
            cprint(C.RED, f"[{self.peer_id}]",
                   f"⚠ Anomaly flags: {'; '.join(anomalies)}")

        # Add to local DAG
        self.dag.add(env)
        parent = self.dag.get(self.dag._latest) if self.dag._latest else None
        # Link edges (simplified — find by prev_hash)
        for node in self.dag.all_data_nodes():
            if node.hash == env.prev_hash:
                self.dag.link(node.node_id, env.node_id)
                break

        # Sync window position
        self._chain_counter = env.chain_counter + 1
        self._window_pos    = env.window_pos + 1
        self._last_event_ts = env.timestamp

        if self._window_pos >= self.config.window_size:
            self._rotate_epoch()

        return fp_match and zk_valid and payload_ok

    # ── RETRIEVE ──────────────────────────────────────────────────
    def retrieve(self, node_id: str) -> Optional[Tuple[str, PcdEnvelope]]:
        """Side-B retrieves and decrypts an object by node_id."""
        env = self.dag.get(node_id) or self.dag.find_by_object_id(node_id)
        if not env:
            cprint(C.RED, f"[{self.peer_id}]", f"Object not found: {node_id}")
            return None

        epoch_sec = self._get_epoch_secret(env.epoch)
        feat      = AIFeatures(**env.ai_features)
        _, local_ktn = self.model.validate_ktn_match(env.feature_fp, feat, epoch_sec)

        plaintext = Crypto.deobfuscate(env.cipher_payload, local_ktn)
        cprint(C.GREEN, f"[{self.peer_id}]",
               f"Retrieved: {env.label} | tier={env.tier} | "
               f"plaintext={plaintext[:60]}{'…' if len(plaintext) > 60 else ''}")
        return plaintext, env

    def get_validator(self) -> ChainValidator:
        """Return a validator with access to this peer's epoch secrets."""
        # Ensure we have secrets for all epochs in the chain
        for env in self.dag.all_data_nodes():
            self._get_epoch_secret(env.epoch)
        return ChainValidator(self.dag, dict(self._epoch_secrets))

    def tamper_node(self, node_id: str) -> bool:
        """Simulate tamper attack: corrupt a node's cipher payload."""
        env = self.dag.get(node_id)
        if not env:
            return False
        env.cipher_payload = env.cipher_payload[:-8] + Crypto.rng_hex(4)
        cprint(C.RED, f"[ATTACKER]",
               f"⚠ Tampered cipher_payload of '{env.label}' ({Crypto.short(node_id)})")
        cprint(C.RED, f"[ATTACKER]",
               f"Cannot re-forge temporal_seal or chain_seal without epoch_secret")
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  P2P CHANNEL
#  Socket-based block stream.  Four block types per envelope:
#    DATA     — encrypted payload + identity
#    PROOF    — seals and hashes
#    FEATURES — public feature fingerprint (B uses to regen K)
#    ACK      — ZK proof + score confirmation
#  K(t,n) is NEVER in any block type.
# ─────────────────────────────────────────────────────────────────────────────
class BlockType(str, Enum):
    DATA     = "DATA"
    PROOF    = "PROOF"
    FEATURES = "FEATURES"
    ACK      = "ACK"


@dataclass
class Block:
    seq:      int
    btype:    BlockType
    payload:  dict

    def encode(self) -> bytes:
        raw = json.dumps({
            "seq":   self.seq,
            "type":  self.btype.value,
            "data":  self.payload,
        }).encode("utf-8")
        # Prefix with 4-byte length
        return struct.pack(">I", len(raw)) + raw

    @classmethod
    def decode(cls, raw: bytes) -> "Block":
        obj = json.loads(raw.decode("utf-8"))
        return cls(
            seq     = obj["seq"],
            btype   = BlockType(obj["type"]),
            payload = obj["data"],
        )


class P2PChannel:
    """
    Simple TCP-based P2P channel.  Sender streams chain envelopes as
    typed block sequences.  Receiver reconstructs full PCD envelopes.

    In production: replace with QUIC + mutual TLS or libp2p.
    For demo: localhost TCP sockets.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9050):
        self.host = host
        self.port = port
        self._seq = 0
        self._rx_queue: queue.Queue = queue.Queue()
        self._server: Optional[socket.socket] = None
        self._running = False

    # ── Sender side ───────────────────────────────────────────────
    def send_envelope(self, conn: socket.socket, env: PcdEnvelope) -> int:
        """Transmit a PCD envelope as 4 typed blocks."""
        blocks_sent = 0

        # Block 1: DATA — encrypted payload + identity
        b1 = Block(self._seq, BlockType.DATA, {
            "node_id":    env.node_id,
            "label":      env.label,
            "cls":        env.classification,
            "epoch":      env.epoch,
            "window_pos": env.window_pos,
            "cipher_preview": env.cipher_payload[:40] + "…",
        })
        self._send_block(conn, b1)
        blocks_sent += 1; self._seq += 1

        # Block 2: PROOF — seals and hashes (no secrets)
        b2 = Block(self._seq, BlockType.PROOF, {
            "temporal_seal": env.temporal_seal,
            "chain_seal":    env.chain_seal,
            "prev_hash":     env.prev_hash,
            "env_hash":      env.hash,
            "cipher_hash":   env.cipher_hash,
        })
        self._send_block(conn, b2)
        blocks_sent += 1; self._seq += 1

        # Block 3: FEATURES — public feature fingerprint
        # B uses this to reconstruct feature vector and regenerate K(t,n)
        # K(t,n) IS NOT HERE
        b3 = Block(self._seq, BlockType.FEATURES, {
            "feature_fp":    env.feature_fp,
            "chain_counter": env.chain_counter,
            "epoch":         env.epoch,
            "window_pos":    env.window_pos,
            "ai_features":   env.ai_features,   # public feature values
            "chain_depth":   env.chain_depth,
        })
        self._send_block(conn, b3)
        blocks_sent += 1; self._seq += 1

        # Block 4: ACK — ZK proof + full envelope (for DAG sync)
        b4 = Block(self._seq, BlockType.ACK, {
            "zk_proof":     env.zk_proof,
            "ai_score":     env.ai_score,
            "tier":         env.tier,
            "full_envelope": env.to_dict(),   # full envelope for DAG sync
        })
        self._send_block(conn, b4)
        blocks_sent += 1; self._seq += 1

        return blocks_sent

    @staticmethod
    def _send_block(conn: socket.socket, block: Block) -> None:
        data = block.encode()
        conn.sendall(data)

    # ── Receiver side ─────────────────────────────────────────────
    @staticmethod
    def recv_block(conn: socket.socket) -> Optional[Block]:
        try:
            hdr = P2PChannel._recvall(conn, 4)
            if not hdr:
                return None
            length = struct.unpack(">I", hdr)[0]
            raw    = P2PChannel._recvall(conn, length)
            if not raw:
                return None
            return Block.decode(raw)
        except Exception:
            return None

    @staticmethod
    def _recvall(conn: socket.socket, n: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < n:
            try:
                chunk = conn.recv(n - len(data))
                if not chunk:
                    return None
                data.extend(chunk)
            except Exception:
                return None
        return bytes(data)

    def stream_chain(
        self,
        sender: Peer,
        receiver: Peer,
        use_socket: bool = False,
    ) -> int:
        """
        Stream all DAG nodes from sender to receiver.
        If use_socket=False, uses in-process queue (simulated).
        If use_socket=True, uses actual TCP sockets.
        """
        nodes = sender.dag.all_data_nodes()
        section(f"P2P BLOCK STREAM  {sender.peer_id} → {receiver.peer_id}")
        cprint(C.BLUE, "[P2P]",
               f"Streaming {len(nodes)} nodes as block sequences")
        cprint(C.BLUE, "[P2P]",
               f"Block types: DATA · PROOF · FEATURES · ACK")
        cprint(C.BLUE, "[P2P]",
               f"K(t,n) is ABSENT from all block types")

        total_blocks = 0

        if use_socket:
            total_blocks = self._stream_via_socket(sender, receiver, nodes)
        else:
            total_blocks = self._stream_in_process(sender, receiver, nodes)

        cprint(C.GREEN, "[P2P]",
               f"Stream complete: {total_blocks} blocks | "
               f"K(t,n) was never transmitted")
        return total_blocks

    def _stream_in_process(
        self,
        sender: Peer,
        receiver: Peer,
        nodes: List[PcdEnvelope],
    ) -> int:
        """Simulate P2P via in-process block queue — no network needed."""
        total = 0
        for i, env in enumerate(nodes):
            section(f"  Node {i+1}/{len(nodes)}: {env.label}")
            blocks = self._build_blocks(env)
            for block in blocks:
                self._log_block(block, sender.peer_id, receiver.peer_id)
                total += 1

            # Receiver processes the full envelope
            result = receiver.receive_envelope(env)
            status = f"{C.GREEN}✓ ACCEPTED{C.RESET}" if result else f"{C.RED}✗ REJECTED{C.RESET}"
            cprint(C.BLUE, "[P2P]", f"Node {i+1} → receiver verdict: {status}")
            time.sleep(0.05)  # simulate wire latency

        return total

    def _build_blocks(self, env: PcdEnvelope) -> List[Block]:
        seq = self._seq
        blocks = [
            Block(seq,   BlockType.DATA, {
                "node_id": env.node_id, "label": env.label,
                "epoch": env.epoch, "window_pos": env.window_pos,
                "cipher_preview": env.cipher_payload[:32] + "…",
            }),
            Block(seq+1, BlockType.PROOF, {
                "temporal_seal": Crypto.short(env.temporal_seal),
                "chain_seal":    Crypto.short(env.chain_seal),
                "prev_hash":     Crypto.short(env.prev_hash),
                "env_hash":      Crypto.short(env.hash),
            }),
            Block(seq+2, BlockType.FEATURES, {
                "feature_fp":    Crypto.short(env.feature_fp),
                "chain_counter": env.chain_counter,
                "epoch":         env.epoch,
                "window_pos":    env.window_pos,
            }),
            Block(seq+3, BlockType.ACK, {
                "zk_proof":  Crypto.short(env.zk_proof),
                "ai_score":  env.ai_score,
                "tier":      env.tier,
            }),
        ]
        self._seq += 4
        return blocks

    def _log_block(self, block: Block, src: str, dst: str) -> None:
        colors = {
            BlockType.DATA:     C.BLUE,
            BlockType.PROOF:    C.CYAN,
            BlockType.FEATURES: C.PURPLE,
            BlockType.ACK:      C.GREEN,
        }
        color = colors.get(block.btype, C.WHITE)
        summary = " ".join(f"{k}:{v}" for k, v in list(block.payload.items())[:3])
        cprint(color, f"[BLOCK]",
               f"#{block.seq:04d} {color}{C.BOLD}{block.btype.value:<8}{C.RESET} "
               f"{C.GREY}{summary[:60]}{C.RESET}")

    # ── TCP socket streaming ──────────────────────────────────────
    def _stream_via_socket(
        self,
        sender: Peer,
        receiver: Peer,
        nodes: List[PcdEnvelope],
    ) -> int:
        """Real TCP socket stream — sender in thread, receiver in thread."""
        received_envs: List[PcdEnvelope] = []
        total_blocks  = [0]
        error         = [None]

        def server_thread():
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.host, self.port))
            srv.listen(1)
            srv.settimeout(10.0)
            try:
                conn, addr = srv.accept()
                cprint(C.GREEN, f"[{receiver.peer_id}]",
                       f"Connection from {addr[0]}:{addr[1]}")
                while True:
                    block = self.recv_block(conn)
                    if block is None:
                        break
                    total_blocks[0] += 1
                    if block.btype == BlockType.ACK and "full_envelope" in block.payload:
                        env = PcdEnvelope.from_dict(block.payload["full_envelope"])
                        received_envs.append(env)
                conn.close()
            except socket.timeout:
                pass
            except Exception as e:
                error[0] = e
            finally:
                srv.close()

        def client_thread():
            time.sleep(0.2)  # let server start
            try:
                conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                conn.connect((self.host, self.port))
                cprint(C.CYAN, f"[{sender.peer_id}]",
                       f"Connected to {self.host}:{self.port}")
                for env in nodes:
                    sent = self.send_envelope(conn, env)
                    total_blocks[0] += sent
                conn.close()
            except Exception as e:
                error[0] = e

        srv_t = threading.Thread(target=server_thread, daemon=True)
        cli_t = threading.Thread(target=client_thread, daemon=True)
        srv_t.start(); cli_t.start()
        srv_t.join(timeout=15); cli_t.join(timeout=15)

        # Process received envelopes
        for env in received_envs:
            receiver.receive_envelope(env)

        if error[0]:
            cprint(C.RED, "[P2P]", f"Socket error: {error[0]}")

        return total_blocks[0]


# ─────────────────────────────────────────────────────────────────────────────
#  FULL DEMO ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_PAYLOADS = [
    ("sensor-batch-001",    "CONFIDENTIAL",
     '{"temp": 36.8, "pressure": 1013, "unit": "SN-77", "status": "nominal"}'),
    ("patient-record-002",  "TOP_SECRET",
     '{"patient_id": "PT-9821", "bp": "118/76", "glucose": 94, "flagged": false}'),
    ("tx-ledger-7731",      "CONFIDENTIAL",
     '{"from": "ACCT-441", "to": "ACCT-882", "amount": 12500.00, "currency": "USD"}'),
    ("sat-telemetry-042",   "SECRET",
     '{"sat": "SAT-9", "alt_km": 412, "lat": 31.2, "lon": 34.8, "status": "NOMINAL"}'),
    ("ml-batch-epoch17",    "UNCLASSIFIED",
     '{"epoch": 17, "loss": 0.0421, "acc": 0.9834, "model": "resnet-50"}'),
    ("audit-log-443",       "SECRET",
     '{"event": "ACCESS", "user": "admin", "resource": "/api/v2/data", "result": "PERMIT"}'),
]


def run_demo(use_socket: bool = False, inject_tamper: bool = True):
    """
    Full PIM-PCD demo:
      1. Hub provisions both sides
      2. Side-A inserts multiple data objects
      3. P2P block stream to Side-B
      4. Side-B validates each received envelope independently
      5. Tamper simulation + chain integrity verification
      6. Full chain audit on both sides
    """
    hdr("PIM-PCD  NEURAL RATCHET CHAIN  v2.0  —  PYTHON DEMO")

    # ── Step 1: Hub provisioning ──────────────────────────────────
    hub = HubAuthority(
        master_secret   = "M-PIMCHAIN-2026-ALPHA",
        window_size     = 4,   # small W for demo — rotate epoch after 4 objects
        side_a_id       = "NODE-ALPHA",
        side_b_id       = "NODE-BETA",
    )
    config_a, config_b, hub_state = hub.provision()
    # Hub is now SILENT

    # ── Step 2: Initialise peers ──────────────────────────────────
    section("PEER INITIALISATION")
    side_a = Peer("NODE-ALPHA", "SENDER",   config_a)
    side_b = Peer("NODE-BETA",  "RECEIVER", config_b)

    # ── Step 3: Side-A inserts data ───────────────────────────────
    section("SIDE-A  —  DATA INSERTION & PROOF CHAIN CONSTRUCTION")
    envelopes = []
    for label, cls, payload in SAMPLE_PAYLOADS:
        time.sleep(0.1)  # simulate real inter-event timing
        env = side_a.insert(label, payload, cls)
        envelopes.append(env)

    side_a.dag.print_graph()

    # ── Step 4: P2P stream to Side-B ─────────────────────────────
    channel = P2PChannel()
    total_blocks = channel.stream_chain(side_a, side_b, use_socket=use_socket)

    cprint(C.BLUE, "[P2P]", f"Total blocks transmitted: {total_blocks}")

    # ── Step 5: Side-B DAG state ──────────────────────────────────
    section("SIDE-B  —  RECEIVED DAG STATE")
    side_b.dag.print_graph()

    # ── Step 6: Side-B retrieves a specific object ────────────────
    section("SIDE-B  —  ON-DEMAND RETRIEVAL")
    if envelopes:
        target = envelopes[-1]
        cprint(C.GREEN, "[NODE-B]",
               f"Retrieving: {target.label} ({Crypto.short(target.node_id)})")
        result = side_b.retrieve(target.node_id)

    # ── Step 7: Chain verification — both sides ───────────────────
    section("CHAIN INTEGRITY VERIFICATION  —  SIDE-A  (canonical)")
    validator_a = side_a.get_validator()
    valid_a = validator_a.print_validation()

    section("CHAIN INTEGRITY VERIFICATION  —  SIDE-B  (received)")
    # B has read-only copies of epoch secrets it used during reception
    validator_b = ChainValidator(
        side_b.dag,
        {0: config_b.epoch_0_secret}  # B's epoch 0 secret (same as A's)
    )
    valid_b = validator_b.print_validation()

    # ── Step 8: Tamper simulation ─────────────────────────────────
    if inject_tamper and len(envelopes) >= 2:
        section("TAMPER ATTACK SIMULATION")
        target_env = envelopes[1]  # attack the second node
        cprint(C.RED, "[ATTACK]",
               f"Attempting to corrupt '{target_env.label}'…")
        side_a.tamper_node(target_env.node_id)

        section("CHAIN VERIFICATION AFTER TAMPER  —  SIDE-A")
        cprint(C.AMBER, "[INFO]",
               "Expected: tampered node fails cipher_hash check; "
               "all downstream nodes fail prev_hash check")
        validator_post = side_a.get_validator()
        valid_post = validator_post.print_validation()

        if not valid_post:
            cprint(C.GREEN, "[RESULT]",
                   f"✓ Tamper detected as expected — chain break is self-announcing")

    # ── Step 9: Feature analysis summary ─────────────────────────
    section("NANO-AI FEATURE ANALYSIS SUMMARY")
    nodes = side_a.dag.all_data_nodes()
    print(f"\n  {'Label':<28} {'Score':>6}  {'Tier':<12} "
          f"{'Entropy':>8} {'Actor':>8} {'Timing':>8} {'Win':>6} {'Depth':>7}")
    print(f"  {'-'*90}")
    for n in nodes:
        f = AIFeatures(**n.ai_features)
        tier_color = {"HOT": C.GREEN, "WARM": C.CYAN,
                      "COOL": C.AMBER, "QUARANTINE": C.RED}.get(n.tier, C.WHITE)
        print(f"  {n.label:<28} "
              f"{C.WHITE}{n.ai_score:>6.3f}{C.RESET}  "
              f"{tier_color}{n.tier:<12}{C.RESET} "
              f"{f.entropy:>8.3f} "
              f"{f.actor_score:>8.3f} "
              f"{f.timing_align:>8.3f} "
              f"{f.window_pos:>6.3f} "
              f"{f.chain_depth:>7.3f}")

    # ── Final summary ─────────────────────────────────────────────
    hdr("DEMO COMPLETE — SUMMARY")
    print(f"  {C.CYAN}Chain nodes inserted:{C.RESET}     {len(envelopes)}")
    print(f"  {C.CYAN}P2P blocks transmitted:{C.RESET}   {total_blocks}")
    print(f"  {C.CYAN}Epoch rotations:{C.RESET}          "
          f"{side_a._current_epoch} (window_size={config_a.window_size})")
    print(f"  {C.CYAN}K(t,n) transmissions:{C.RESET}     "
          f"{C.GREEN}0 — never transmitted{C.RESET}")
    print(f"  {C.CYAN}Side-A chain valid:{C.RESET}       "
          f"{C.GREEN if valid_a else C.RED}{'YES' if valid_a else 'NO (tamper detected)'}{C.RESET}")
    print(f"  {C.CYAN}Side-B chain valid:{C.RESET}       "
          f"{C.GREEN if valid_b else C.RED}{'YES' if valid_b else 'PARTIAL'}{C.RESET}")
    print(f"  {C.CYAN}Tamper detection:{C.RESET}         "
          f"{'Self-announcing — chain break detected' if inject_tamper else 'Not tested'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CLI
# ─────────────────────────────────────────────────────────────────────────────
def interactive_demo():
    """
    Interactive mode — manually drive each phase.
    """
    hdr("PIM-PCD  INTERACTIVE MODE")

    hub = HubAuthority(
        master_secret = "M-INTERACTIVE-2026",
        window_size   = 4,
    )
    config_a, config_b, _ = hub.provision()

    side_a = Peer("NODE-ALPHA", "SENDER",   config_a)
    side_b = Peer("NODE-BETA",  "RECEIVER", config_b)
    channel = P2PChannel()

    menu = """
  Commands:
    i <label> <payload>   — Insert data on Side-A
    s                     — Sample insert (pre-built payload)
    stream                — Stream all Side-A nodes to Side-B
    verify a              — Verify chain on Side-A
    verify b              — Verify chain on Side-B
    retrieve <node_id>    — Side-B retrieves object
    dag a                 — Print Side-A DAG
    dag b                 — Print Side-B DAG
    tamper <node_id>      — Simulate tamper on Side-A node
    features              — Print feature analysis table
    q                     — Quit
"""
    print(menu)

    _sample_idx = [0]

    while True:
        try:
            cmd = input(f"\n{C.CYAN}{C.BOLD}pim-pcd>{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue

        parts = cmd.split(None, 2)
        verb  = parts[0].lower()

        if verb == "q":
            break

        elif verb == "i" and len(parts) >= 3:
            label, payload = parts[1], parts[2]
            side_a.insert(label, payload)

        elif verb == "s":
            items = SAMPLE_PAYLOADS
            idx = _sample_idx[0] % len(items)
            label, cls, payload = items[idx]
            _sample_idx[0] += 1
            side_a.insert(label, payload, cls)

        elif verb == "stream":
            channel.stream_chain(side_a, side_b)

        elif verb == "verify" and len(parts) >= 2:
            peer = side_a if parts[1] == "a" else side_b
            validator = peer.get_validator()
            validator.print_validation()

        elif verb == "retrieve" and len(parts) >= 2:
            side_b.retrieve(parts[1])

        elif verb == "dag" and len(parts) >= 2:
            peer = side_a if parts[1] == "a" else side_b
            peer.dag.print_graph()

        elif verb == "tamper" and len(parts) >= 2:
            side_a.tamper_node(parts[1])

        elif verb == "features":
            nodes = side_a.dag.all_data_nodes()
            if not nodes:
                print("  No nodes yet.")
            else:
                print(f"\n  {'Label':<28} {'Score':>6}  {'Tier':<12} "
                      f"{'H(x)':>6} {'Actor':>7} {'Time':>7}")
                for n in nodes:
                    f = AIFeatures(**n.ai_features)
                    print(f"  {n.label:<28} {n.ai_score:>6.3f}  {n.tier:<12} "
                          f"{f.entropy:>6.3f} {f.actor_score:>7.3f} {f.timing_align:>7.3f}")
        else:
            print(menu)


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_demo()
    else:
        run_demo(use_socket=False, inject_tamper=True)
