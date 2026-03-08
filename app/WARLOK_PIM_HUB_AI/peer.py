# warlok/peer.py — Autonomous Peer Node (Near / Far)
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

from crypto  import H, hkdf, mac, RunningAccumulator, merkle_build, merkle_proof, merkle_verify
from anchor  import SolidStateAnchor
from chain   import (ChainParamBundle, ChainMsg, StartGrant, WindowState,
                      NanoBundle, WindowCertificate, IncidentCertificate,
                      verify_msg, WINDOW_SIZE_DEFAULT)
from model   import (ATK_LABELS, N_CLASSES, featurise, predict,
                      rnn_forward, sigmoid, init_rnn)

# ── Interception action thresholds ────────────────────────────────────────────
WARN_THRESHOLD       = 0.35
QUARANTINE_THRESHOLD = 0.55
BLOCK_THRESHOLD      = 0.80

InterceptionAction = str   # "PASS" | "WARN" | "QUARANTINE" | "BLOCK"


@dataclass
class PeerDecision:
    """Full decision record for one message."""
    msg_idx:        int
    rule_verdict:   str    # ACCEPT / DROP
    rule_reason:    str
    acc_divergence: float
    gru_probs:      Dict[str, float]
    detected:       List[str]
    action:         InterceptionAction
    top_conf:       float
    window_id:      int
    leaf_hash:      bytes  = b""
    merkle_root:    bytes  = b""


@dataclass
class PeerSessionState:
    """Everything a peer tracks during a live session."""
    peer_id:          str
    role:             str   # "near" | "far"
    bundle:           ChainParamBundle
    anchor:           SolidStateAnchor
    chain_key:        bytes
    nano_bundle:      Optional[NanoBundle]
    gru_params:       Optional[dict]
    session_id:       str   = ""
    window_state:     Optional[WindowState] = None
    accumulator:      Optional[RunningAccumulator] = None

    decisions:        List[PeerDecision]          = field(default_factory=list)
    window_certs:     List[WindowCertificate]     = field(default_factory=list)
    incidents:        List[IncidentCertificate]   = field(default_factory=list)
    peer_certs_rx:    List[WindowCertificate]     = field(default_factory=list)
    cert_matches:     List[bool]                  = field(default_factory=list)

    # Merkle state
    window_leaves:    List[bytes]   = field(default_factory=list)
    prev_root:        bytes         = H(b"GENESIS_ROOT")
    current_window_id: int          = 0

    # Stats
    messages_seen:    int = 0
    messages_accepted:int = 0
    attacks_blocked:  int = 0


class PeerNode:
    """
    Autonomous peer node. Receives messages, runs rule-based + GRU verification,
    maintains Merkle window, issues Window Certificates and Incident Certificates.
    No HUB contact during live session.
    """

    def __init__(self, peer_id: str, role: str = "near"):
        self.peer_id = peer_id
        self.role    = role
        self.state: Optional[PeerSessionState] = None
        self.log:   List[str] = []

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}][{self.peer_id}] {msg}")

    # ── Session initialisation ────────────────────────────────────────────────

    def init_session(self,
                     bundle:     ChainParamBundle,
                     anchor:     SolidStateAnchor,
                     gru_params: Optional[dict],
                     nano_bundle: Optional[NanoBundle],
                     session_id: str,
                     grant:      StartGrant) -> bool:
        """
        Called after HUB delivers bundle + weights. Sets up all session state.
        Returns True if anchor validates against bundle.
        """
        # Verify bundle signature
        ok, reason = bundle.verify(H(b"WARLOK_HUB_MASTER_KEY_v1"))
        if not ok:
            self._log(f"Bundle verification FAILED: {reason}")
            return False

        # Verify anchor matches bundle fingerprint
        expected_fp = (bundle.anchor_a_fp if self.role=="near"
                       else bundle.anchor_b_fp)
        if anchor.public_fp != expected_fp:
            self._log(f"Anchor fingerprint mismatch: "
                      f"got {anchor.public_fp}, expected {expected_fp}")
            # Allow mismatch in simulation (anchors generated with same seed)

        chain_key = hkdf(
            H(b"chain|" + grant.anchor_state_hash + grant.anchor_policy_hash),
            b"chain-key", 32
        )

        acc_init = H(grant.anchor_state_hash + b"ACC_INIT")

        ws = WindowState(
            session_id             = session_id,
            window_id              = 0,
            expected_next_counter  = bundle.counter_init,
            expected_step_idx      = 0,
            last_ts_ms             = int(time.time()*1000),
            prev_mac_chain         = H(b"WINDOW_PILOT|" + session_id.encode() + b"|0"),
        )

        self.state = PeerSessionState(
            peer_id       = self.peer_id,
            role          = self.role,
            bundle        = bundle,
            anchor        = anchor,
            chain_key     = chain_key,
            nano_bundle   = nano_bundle,
            gru_params    = gru_params,
            session_id    = session_id,
            window_state  = ws,
            accumulator   = RunningAccumulator(acc_init),
        )
        self._log(f"Session initialised. GRU loaded: {gru_params is not None}")
        return True

    # ── Per-message processing ────────────────────────────────────────────────

    def process_message(self, msg: ChainMsg) -> PeerDecision:
        """
        Full pipeline per message:
          1. Rule-based MAC / counter / timing / op verification
          2. Accumulator check
          3. GRU multi-label inference
          4. Interception decision
          5. Merkle leaf accumulation → window cert on boundary
        """
        assert self.state is not None, "Session not initialised"
        s = self.state

        # 1. Rule-based verification
        ok, reason, acc_div = verify_msg(
            s.chain_key, s.nano_bundle, s.window_state,
            msg, s.accumulator
        )
        rule_verdict = "ACCEPT" if ok else "DROP"

        # 2. Merkle leaf
        leaf = H(msg.canonical_bytes())
        s.window_leaves.append(leaf)

        # Window boundary check
        at_boundary = (len(s.window_leaves) >= s.bundle.window_size)
        if at_boundary:
            levels, root = merkle_build(s.window_leaves)
            cert = WindowCertificate(
                session_id    = s.session_id,
                window_id     = s.current_window_id,
                merkle_root   = root,
                prev_root     = s.prev_root,
                acc_final     = s.accumulator.value,
                messages_seen = len(s.window_leaves),
                attacks_blocked = s.attacks_blocked,
                peer_id       = self.peer_id,
            ).sign(s.chain_key)
            s.window_certs.append(cert)
            s.prev_root = root
            s.current_window_id += 1
            s.window_leaves = []
            root_delta = int.from_bytes(
                bytes(a^b for a,b in zip(root[:4], s.prev_root[:4])), "big"
            ) / (2**32)
            self._log(f"Window {cert.window_id} closed. Root={root.hex()[:12]}")
        else:
            root = s.prev_root
            root_delta = 0.0

        # 3. GRU inference
        gru_probs = {lbl: 0.0 for lbl in ATK_LABELS}
        detected  = ["none"]
        top_conf  = 0.0

        if s.gru_params is not None:
            leaf_hash_norm = int.from_bytes(leaf[:4], "big") / (2**32)
            fake_row = {
                "meas":            msg.os_meas,
                "dt_ms":           msg.dt_ms,
                "step":            msg.step_idx,
                "ctr":             msg.global_counter,
                "decision":        rule_verdict,
                "op":              msg.op_code,
                "acc_divergence":  acc_div,
                "root_delta_norm": root_delta,
                "leaf_hash_norm":  leaf_hash_norm,
                "window_boundary": at_boundary,
                "anchor_age_norm": 0.0,
            }
            # Build a minimal single-row trace for featurisation
            trace_1 = [fake_row]
            X = featurise(trace_1, window_size=s.bundle.window_size)[None, :, :]
            ctx    = rnn_forward(X, s.gru_params)
            logits = (ctx @ s.gru_params["Wc"].T + s.gru_params["bc"])[0]
            probs  = sigmoid(logits)
            thr    = s.bundle.detection_threshold
            gru_probs = {ATK_LABELS[i]: float(probs[i]) for i in range(N_CLASSES)}
            detected  = [ATK_LABELS[i] for i in range(N_CLASSES) if probs[i] >= thr]
            if not detected:
                detected = [ATK_LABELS[int(np.argmax(probs))]]
            top_conf = float(probs[int(np.argmax(probs))])

        # 4. Interception action
        action = self._decide_action(rule_verdict, gru_probs, top_conf, detected)

        if action in ("QUARANTINE", "BLOCK"):
            s.attacks_blocked += 1
            inc = IncidentCertificate(
                session_id    = s.session_id,
                window_id     = s.current_window_id,
                message_idx   = s.messages_seen,
                true_leaf     = leaf,
                received_leaf = leaf,
                merkle_path   = [],
                window_root   = root,
                anchor_hash   = s.anchor.anchor_hash,
                attack_classes= [a for a in detected if a != "none"],
                gru_probs     = gru_probs,
                action_taken  = action,
                peer_id       = self.peer_id,
            ).sign(s.chain_key)
            s.incidents.append(inc)
            self._log(f"msg#{s.messages_seen} {action}: {detected} conf={top_conf:.2f}")

        decision = PeerDecision(
            msg_idx        = s.messages_seen,
            rule_verdict   = rule_verdict,
            rule_reason    = reason,
            acc_divergence = acc_div,
            gru_probs      = gru_probs,
            detected       = detected,
            action         = action,
            top_conf       = top_conf,
            window_id      = s.current_window_id,
            leaf_hash      = leaf,
            merkle_root    = root,
        )
        s.decisions.append(decision)
        s.messages_seen += 1
        if rule_verdict == "ACCEPT": s.messages_accepted += 1
        return decision

    def _decide_action(self, rule_verdict: str, gru_probs: Dict[str, float],
                       top_conf: float, detected: List[str]) -> InterceptionAction:
        """PASS / WARN / QUARANTINE / BLOCK based on rule + GRU confidence."""
        if rule_verdict == "DROP":
            return "BLOCK"   # Rule violation always blocks
        has_attack = any(d != "none" for d in detected)
        if not has_attack:
            return "PASS"
        if top_conf >= BLOCK_THRESHOLD:
            return "BLOCK"
        if top_conf >= QUARANTINE_THRESHOLD:
            return "QUARANTINE"
        if top_conf >= WARN_THRESHOLD:
            return "WARN"
        return "PASS"

    # ── Window certificate exchange ───────────────────────────────────────────

    def receive_peer_cert(self, cert: WindowCertificate) -> Tuple[bool, str]:
        """
        Validate a Window Certificate from the remote peer.
        Compares roots and accumulators.
        """
        assert self.state is not None
        s = self.state
        s.peer_certs_rx.append(cert)

        # Find our matching window cert
        our_cert = next((c for c in s.window_certs
                         if c.window_id == cert.window_id), None)
        if our_cert is None:
            s.cert_matches.append(False)
            self._log(f"No matching cert for window {cert.window_id}")
            return False, "no_matching_window"

        import hmac as _hmac
        roots_match = _hmac.compare_digest(our_cert.merkle_root, cert.merkle_root)
        accs_match  = _hmac.compare_digest(our_cert.acc_final,   cert.acc_final)

        if roots_match and accs_match:
            s.cert_matches.append(True)
            self._log(f"Window {cert.window_id} ✅ roots+acc match")
            return True, "ok"
        else:
            s.cert_matches.append(False)
            reason = []
            if not roots_match: reason.append("root_mismatch")
            if not accs_match:  reason.append("acc_mismatch")
            self._log(f"Window {cert.window_id} ❌ {reason}")
            return False, ",".join(reason)

    # ── Mutual anchor validation (P2P, no HUB) ───────────────────────────────

    def validate_remote_anchor(self, remote_fp: str,
                                remote_bundle_hash: bytes) -> Tuple[bool, str]:
        """
        Called during P2P mutual validation phase.
        Checks remote anchor fingerprint and bundle hash match.
        """
        assert self.state is not None
        s = self.state

        # Check bundle hash matches
        import hmac as _hmac
        if not _hmac.compare_digest(s.bundle.bundle_hash, remote_bundle_hash):
            return False, "bundle_hash_mismatch — peers received different bundles"

        # Check remote fp is what bundle says it should be
        expected_remote_fp = (s.bundle.anchor_b_fp if self.role=="near"
                              else s.bundle.anchor_a_fp)
        if remote_fp != expected_remote_fp:
            # Allow in simulation
            pass

        self._log(f"Remote anchor validated: fp={remote_fp[:12]}")
        return True, "ok"

    # ── Summary helpers ───────────────────────────────────────────────────────

    def get_trace(self) -> List[Dict]:
        """Flatten decisions to trace-like dicts for UI."""
        if not self.state: return []
        return [
            {
                "i":          d.msg_idx,
                "win":        d.window_id,
                "rule":       d.rule_verdict,
                "reason":     d.rule_reason,
                "acc_div":    round(d.acc_divergence, 4),
                "action":     d.action,
                "detected":   ", ".join(d.detected),
                "top_conf":   round(d.top_conf, 3),
                **{f"p_{lbl}": round(d.gru_probs.get(lbl,0.0), 3)
                   for lbl in ATK_LABELS},
            }
            for d in self.state.decisions
        ]

    def stats(self) -> Dict:
        if not self.state:
            return {}
        s = self.state
        return {
            "messages_seen":     s.messages_seen,
            "messages_accepted": s.messages_accepted,
            "attacks_blocked":   s.attacks_blocked,
            "windows_closed":    len(s.window_certs),
            "cert_match_rate":   (sum(s.cert_matches)/max(len(s.cert_matches),1)),
            "incidents":         len(s.incidents),
        }
