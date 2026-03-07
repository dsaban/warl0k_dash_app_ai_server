# warlok_gw/gateway.py
# ─────────────────────────────────────────────────────────────────────────────
# WARL0K sidecar gateway — TLS message interceptor
#
# ARCHITECTURE
# ────────────
#
#   [agentic AI agent — far cloud]
#        │  each outbound message is a ChainMsg inside a TLS record
#        ▼
#   ┌─────────────────────────────────────────────────────────────┐
#   │  WarlokGateway.intercept(session_id, raw_tls_payload)       │
#   │                                                             │
#   │  1. parse_chain_msg(raw)  → structured fields               │
#   │  2. rule_verify(msg)      → ACCEPT/DROP + reason            │
#   │  3. engine.step(buf, msg) → Verdict (AI classifier)         │
#   │                                                             │
#   │  Decision logic:                                            │
#   │    rule says DROP  → always BLOCK (hard rule)               │
#   │    AI  says BLOCK  → BLOCK + emit alert (soft rule)         │
#   │    both say OK     → ALLOW → forward to asset               │
#   └─────────────────────────────────────────────────────────────┘
#        │  inbound near-sidecar
#        ▼
#   [org asset — internal command executor]
#
# USAGE
# ─────
#   gw = WarlokGateway("model.npz", threshold=0.82)
#
#   # In your TLS termination / sidecar proxy:
#   outcome = gw.intercept(session_id="sess-abc123", raw=tls_record_bytes)
#   if outcome.action == "BLOCK":
#       send_alert(outcome)
#       close_session(session_id)
#   else:
#       forward_to_asset(raw)
#
# THREAD SAFETY
# ─────────────
#   WarlokGateway is thread-safe.
#   GRUInferEngine weights are read-only after __init__.
#   SessionBuffers are per-session (keyed by session_id) and accessed
#   under a per-session lock so concurrent messages on the same session
#   are serialised correctly.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import base64, hashlib, hmac, json, logging, threading, time
from dataclasses import dataclass, field
from typing      import Callable, Dict, Optional

log = logging.getLogger("warlok.gateway")

# Import compiled Cython extension (falls back to pure Python if not built yet)
try:
    from warlok_gw.gru_infer import GRUInferEngine, SessionBuffer, Verdict
    log.info("Loaded Cython GRU engine")
except ImportError:
    log.warning("Cython extension not built — falling back to pure Python engine")
    from warlok_gw._gru_infer_py import GRUInferEngine, SessionBuffer, Verdict


# ── Outcome returned to the caller for every intercepted message ──────────────

@dataclass
class Outcome:
    action:        str            # "ALLOW" | "BLOCK"
    session_id:    str
    step:          int            # message index within session
    rule_verdict:  str            # "ACCEPT" | "DROP: <reason>"
    ai_class:      str            # predicted attack class
    ai_confidence: float          # model confidence 0–1
    ai_scores:     Dict[str,float]
    latency_us:    float          # end-to-end checkpoint latency in microseconds
    blocked_by:    str            # "rule" | "ai" | "none"


# ── Minimal ChainMsg parser (matches the protocol used in the simulator) ──────

def _parse_chain_msg(raw: bytes) -> dict:
    """
    Parse a raw TLS record payload (pipe-delimited ChainMsg.to_bytes() format)
    into a dict with the fields expected by GRUInferEngine.step().

    In production, replace this with your actual TLS record unwrap + parser.
    """
    try:
        parts = raw.split(b"|")
        # Format: session_id|window_id|step_idx|global_counter|dt_ms|
        #         op_code|payload_hash_b64|os_token_b64|os_meas|mac_chain_b64
        return {
            "session_id": parts[0].decode(),
            "window_id":  int(parts[1]),
            "step_idx":   int(parts[2]),
            "ctr":        int(parts[3]),
            "dt_ms":      int(parts[4]),
            "op":         parts[5].decode(),
            "meas":       float(parts[8].decode()),
            "decision":   "ACCEPT",   # filled in after rule check
        }
    except Exception as exc:
        log.error("ChainMsg parse failed: %s  raw=%r", exc, raw[:80])
        return {}


# ── Minimal rule verifier (stateless field checks, no crypto re-check) ────────
# In production this is your full verify_msg() from the protocol module.

_OP_ALLOWLIST = {"READ", "WRITE"}

def _rule_check(msg: dict, session_meta: dict) -> tuple[bool, str]:
    """
    Lightweight stateless rule checks that don't need the full crypto chain.
    Returns (ok, reason).
    """
    if not msg:
        return False, "DROP: parse error"
    if msg.get("op") not in _OP_ALLOWLIST:
        return False, f"DROP: op_code not in allowlist ({msg.get('op')})"
    dt = msg.get("dt_ms", 0)
    if dt > 500_000:
        return False, f"DROP: dt_ms anomaly ({dt} ms)"
    return True, "ACCEPT"


# ── Main gateway class ─────────────────────────────────────────────────────────

class WarlokGateway:
    """
    Sidecar checkpoint.  One instance per process; shared across threads.

    Parameters
    ──────────
    model_path  : path to .npz weights file (saved by model_io.save_weights)
    threshold   : AI confidence threshold to trigger a BLOCK (default 0.82)
    min_steps   : minimum messages before AI starts scoring (default 4)
    alert_cb    : optional callback(outcome: Outcome) called on every BLOCK
    max_sessions: max concurrent sessions tracked before LRU eviction (default 10_000)
    """

    def __init__(self,
                 model_path:   str,
                 threshold:    float             = 0.82,
                 min_steps:    int               = 4,
                 alert_cb:     Optional[Callable] = None,
                 max_sessions: int               = 10_000):

        self._engine      = GRUInferEngine(model_path,
                                           threshold=threshold,
                                           min_steps=min_steps)
        self._alert_cb    = alert_cb
        self._max_sessions= max_sessions

        # session_id → (SessionBuffer, threading.Lock, metadata_dict)
        self._sessions: Dict[str, tuple] = {}
        self._global_lock = threading.Lock()

        log.info("WarlokGateway ready  model=%s  threshold=%.2f",
                 model_path, threshold)

    # ── Per-session buffer management ─────────────────────────────────────────

    def _get_or_create(self, session_id: str):
        with self._global_lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    # Evict oldest session (simple FIFO)
                    oldest = next(iter(self._sessions))
                    del self._sessions[oldest]
                    log.debug("evicted session %s (LRU)", oldest)
                buf  = SessionBuffer(session_id)
                lock = threading.Lock()
                meta = {"open_at": time.monotonic(), "steps": 0}
                self._sessions[session_id] = (buf, lock, meta)
            return self._sessions[session_id]

    def close_session(self, session_id: str):
        """Call when the TLS session is torn down to free the buffer."""
        with self._global_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    # ── Main intercept entry point ────────────────────────────────────────────

    def intercept(self,
                  session_id: str,
                  raw:        bytes,
                  extra_meta: dict | None = None) -> Outcome:
        """
        Intercept one TLS record from an agentic session.

        Parameters
        ──────────
        session_id : str   — unique identifier for the TLS session
        raw        : bytes — raw TLS record payload (unwrapped but not decrypted here;
                             pass the decrypted plaintext from your TLS terminator)
        extra_meta : dict  — optional extra fields merged into the msg dict
                             (e.g. {"decision": "DROP", "reason": "..."} if you
                              already ran your own rule verifier upstream)

        Returns
        ───────
        Outcome  — caller decides whether to ALLOW or BLOCK based on .action
        """
        t0 = time.perf_counter()

        buf, lock, meta = self._get_or_create(session_id)

        with lock:
            # ── 1. Parse raw bytes → structured message ───────────────────────
            msg = _parse_chain_msg(raw)
            if extra_meta:
                msg.update(extra_meta)

            # ── 2. Lightweight rule check ─────────────────────────────────────
            rule_ok, rule_reason = _rule_check(msg, meta)
            msg["decision"] = "ACCEPT" if rule_ok else "DROP"

            # ── 3. AI inference via compiled GRU ──────────────────────────────
            verdict: Verdict = self._engine.step(buf, msg)
            meta["steps"] += 1

            # ── 4. Decision: rule gate first, AI gate second ───────────────────
            if not rule_ok:
                action     = "BLOCK"
                blocked_by = "rule"
            elif verdict.block:
                action     = "BLOCK"
                blocked_by = "ai"
            else:
                action     = "ALLOW"
                blocked_by = "none"

            latency_us = (time.perf_counter() - t0) * 1_000_000

            outcome = Outcome(
                action        = action,
                session_id    = session_id,
                step          = meta["steps"],
                rule_verdict  = rule_reason,
                ai_class      = verdict.attack_class,
                ai_confidence = verdict.confidence,
                ai_scores     = verdict.scores,
                latency_us    = latency_us,
                blocked_by    = blocked_by,
            )

            if action == "BLOCK":
                log.warning(
                    "BLOCK  session=%s  step=%d  rule=%s  ai=%s(%.2f)  by=%s  lat=%.1fµs",
                    session_id, meta["steps"], rule_reason,
                    verdict.attack_class, verdict.confidence,
                    blocked_by, latency_us,
                )
                if self._alert_cb:
                    try:
                        self._alert_cb(outcome)
                    except Exception:
                        log.exception("alert_cb raised")
            else:
                log.debug(
                    "ALLOW  session=%s  step=%d  ai=%s(%.2f)  lat=%.1fµs",
                    session_id, meta["steps"],
                    verdict.attack_class, verdict.confidence,
                    latency_us,
                )

            return outcome

    # ── Convenience: score a full pre-collected trace (offline eval) ──────────

    def score_trace(self, session_id: str, trace: list) -> Outcome:
        """
        Score a complete trace list (e.g. from the Streamlit simulator).
        Each element should be a dict with keys: meas, dt_ms, step_idx,
        ctr, decision, op.  Returns the final Outcome.
        """
        # Temporarily register a fresh session, replay, then clean up
        self.close_session(session_id)
        last = None
        for msg in trace:
            raw = _pack_msg_dict(msg)
            last = self.intercept(session_id, raw, extra_meta=msg)
        self.close_session(session_id)
        return last

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)


def _pack_msg_dict(msg: dict) -> bytes:
    """
    Re-pack a msg dict back into the pipe-delimited wire format so it can
    be passed to intercept() for offline evaluation.  Inverse of _parse_chain_msg.
    """
    b64 = lambda x: base64.urlsafe_b64encode(x).decode()
    dummy = b64(b"\x00" * 32)
    parts = [
        msg.get("session_id", "eval"),
        str(msg.get("window_id",  0)),
        str(msg.get("step_idx",   msg.get("step", 0))),
        str(msg.get("ctr",        0)),
        str(msg.get("dt_ms",      0)),
        msg.get("op",             "READ"),
        dummy,   # payload_hash
        dummy,   # os_token
        f"{msg.get('meas', 0.0):.6f}",
        dummy,   # mac_chain
    ]
    return "|".join(parts).encode()
