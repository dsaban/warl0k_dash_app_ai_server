# warlok/hub.py — HUB Governor: parameter factory, pre-training, weight registry
import time, itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
import numpy as np

from crypto import H, hkdf, mac, rand_bytes, rand_hex, RunningAccumulator, merkle_build
from anchor import SolidStateAnchor, make_anchor
from chain  import (ChainParamBundle, ChainMsg, StartGrant, WindowState,
                     NanoBundle, WindowCertificate, IncidentCertificate,
                     build_msg, train_profile, verify_msg, WINDOW_SIZE_DEFAULT)
from model  import (ATK_LABELS, N_CLASSES, featurise, make_multihot,
                     train, weights_hash, serialise_weights, get_registry,
                     RNN_IN_DIM)

# ══════════════════════════════════════════════════════════════════════════════
# Attack mutators (deterministic, used during HUB pre-training)
# ══════════════════════════════════════════════════════════════════════════════

def _atk_reorder(w):
    w=w[:];
    if len(w)>=12: w[10],w[11]=w[11],w[10]
    return w

def _atk_drop(w):    return [m for i,m in enumerate(w) if i!=min(20,len(w)-1)]

def _atk_replay(w):
    out=[]
    for i,m in enumerate(w):
        out.append(m)
        if i==5: out.append(m)
    return out

def _atk_timewarp(w, dt=999999):
    out=[]
    for i,m in enumerate(w):
        if i==7: mm=ChainMsg(**m.__dict__); mm.dt_ms=dt; out.append(mm)
        else: out.append(m)
    return out

def _atk_splice(w):
    out=[]
    for i,m in enumerate(w):
        if i==12: mm=ChainMsg(**m.__dict__); mm.op_code="CONTROL"; out.append(mm)
        else: out.append(m)
    return out

def _apply_attacks(w: list, attack_modes: List[str]) -> list:
    current = w
    for mode in attack_modes:
        if   mode=="reorder":  current=_atk_reorder(current)
        elif mode=="drop":     current=_atk_drop(current)
        elif mode=="replay":   current=_atk_replay(current)
        elif mode=="timewarp": current=_atk_timewarp(current)
        elif mode=="splice":   current=_atk_splice(current)
    return current

# ══════════════════════════════════════════════════════════════════════════════
# Full chain simulator (HUB version, with proof features)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_chain(bundle: ChainParamBundle,
                   attack_modes: List[str] = None,
                   edits: List[Dict] = None,
                   forensic_continue: bool = True) -> Dict:
    """
    Simulate a full chain session using bundle parameters.
    Returns result dict with trace (including proof features).
    """
    from anchor import DuplexWire, P2PTLS
    attack_modes = attack_modes or ["none"]
    edits        = edits        or []

    wire  = DuplexWire()
    pA, pB = bundle.peer_a_id or "peerA", bundle.peer_b_id or "peerB"
    psk   = H(b"mutual-trust-root|" + pA.encode() + b"|" + pB.encode())
    tlsA  = P2PTLS(pA, pB, psk, wire.send_a, lambda: wire.recv_a(0.5))
    tlsB  = P2PTLS(pB, pA, psk, wire.send_b, lambda: wire.recv_b(0.5))

    tlsA.hs1_send(); tlsB.hs1_send()
    for _ in range(12):
        if not tlsA.keys: tlsA.hs2_recv_derive()
        if not tlsB.keys: tlsB.hs2_recv_derive()
        if tlsA.keys and tlsB.keys: break
    if not (tlsA.keys and tlsB.keys):
        return {"ok": False, "reason": "HS1 timeout"}

    tlsA.hs3_send_fin(); tlsB.hs3_send_fin()
    okA=okB=False
    for _ in range(12):
        okA = okA or tlsA.hs4_recv_verify()
        okB = okB or tlsB.hs4_recv_verify()
        if okA and okB: break
    if not (okA and okB):
        return {"ok": False, "reason": "HS2 FIN timeout"}

    posture    = "posture_" + rand_hex(8)
    claim      = {"ok": True, "peer_id": pA, "roles": ["operator"],
                  "session_token": rand_hex(16)}
    sid        = rand_hex(8)
    anc        = H(b"WARLOK_ANCHOR|" + pA.encode() + b"|" + pB.encode() +
                   posture.encode() + b"|pump-controller|READ,WRITE|" +
                   str(int(time.time())).encode() + b"|" + sid.encode())
    pol        = H(b"POLICY|pump-controller|READ,WRITE")
    grant      = StartGrant(ok=True, session_id=sid, window_id_start=0,
                            anchor_state_hash=anc, anchor_policy_hash=pol,
                            signature=H(b"SIGN|" + anc + pol))
    chain_key  = hkdf(H(b"chain|" + grant.anchor_state_hash + grant.anchor_policy_hash),
                      b"chain-key", 32)

    # Accumulator seeds
    acc_init   = H(anc + b"ACC_INIT")
    acc_sender = RunningAccumulator(acc_init)

    steps = bundle.window_size

    def mk_ws():
        return WindowState(
            session_id=grant.session_id, window_id=0,
            expected_next_counter=bundle.counter_init,
            expected_step_idx=0,
            last_ts_ms=int(time.time()*1000),
            prev_mac_chain=H(b"WINDOW_PILOT|" + grant.session_id.encode() + b"|0"),
        )

    op_seq = ["READ" if i%3 else "WRITE" for i in range(steps)]

    # Train profile on clean window
    wsT = mk_ws()
    acc_train = RunningAccumulator(acc_init)
    tw = []
    for i in range(min(bundle.window_size, steps)):
        tw.append(build_msg(chain_key, grant, wsT, op_seq[i],
                            f"op{i}".encode(), acc_train))
    nb = train_profile(pA, grant, tw, bundle.dt_ms_slack, bundle.meas_slack)

    # Build clean send window
    wsS = mk_ws()
    acc_send2 = RunningAccumulator(acc_init)
    sw = []
    for i in range(steps):
        sw.append(build_msg(chain_key, grant, wsS, op_seq[i],
                            f"op{i}".encode(), acc_send2))

    # Apply named attacks
    active = [m for m in attack_modes if m != "none"]
    attacked = _apply_attacks(sw, active) if active else sw[:]

    # Apply manual edits
    tampered_set: Set[int] = set()
    if edits:
        INT_F   = {"dt_ms","step_idx","global_counter","window_id"}
        FLOAT_F = {"os_meas"}
        em: Dict[int,list] = {}
        for e in edits:
            f=str(e["field"]); v=e["value"]
            v=int(v) if f in INT_F else (float(v) if f in FLOAT_F else str(v))
            em.setdefault(int(e["msg_idx"]),[]).append((f,v))
        patched=[]
        for i,msg in enumerate(attacked):
            if i in em:
                mm=ChainMsg(**msg.__dict__)
                for f,v in em[i]: setattr(mm,f,v)
                patched.append(mm); tampered_set.add(i)
            else: patched.append(msg)
        attacked=patched

    # Verify stream + build Merkle window
    wsB = mk_ws()
    acc_verifier = RunningAccumulator(acc_init)
    trace=[]; accepted=0; first_drop=None
    window_leaves: List[bytes] = []
    prev_root = H(b"GENESIS_ROOT")
    cert_list: List[WindowCertificate] = []
    current_window_id = 0

    for idx_m, m in enumerate(attacked):
        tlsA.send_rec("CHAIN", m.to_bytes())
        _, pt = tlsB.recv_rec()
        rx = ChainMsg.from_bytes(pt)

        ok, reason, acc_div = verify_msg(chain_key, nb, wsB, rx, acc_verifier)
        dec = "ACCEPT" if ok else "DROP"
        if dec=="DROP" and first_drop is None: first_drop=reason

        leaf = H(rx.canonical_bytes())
        window_leaves.append(leaf)

        # Window boundary
        at_boundary = (len(window_leaves) == bundle.window_size)
        if at_boundary:
            levels, root = merkle_build(window_leaves)
            cert = WindowCertificate(
                session_id=grant.session_id, window_id=current_window_id,
                merkle_root=root, prev_root=prev_root,
                acc_final=acc_verifier.value,
                messages_seen=len(window_leaves), attacks_blocked=0,
                peer_id=pB,
            ).sign(chain_key)
            cert_list.append(cert)
            root_delta = int.from_bytes(
                bytes(a^b for a,b in zip(root[:4], prev_root[:4])), "big"
            ) / (2**32)
            prev_root = root
            current_window_id += 1
            window_leaves = []
        else:
            root_delta = 0.0

        leaf_hash_norm = int.from_bytes(leaf[:4], "big") / (2**32)

        trace.append({
            "i":               idx_m,
            "win":             rx.window_id,
            "step":            rx.step_idx,
            "ctr":             rx.global_counter,
            "dt_ms":           rx.dt_ms,
            "op":              rx.op_code,
            "meas":            round(rx.os_meas, 6),
            "decision":        dec,
            "reason":          reason,
            "tampered":        "✏️" if idx_m in tampered_set else "",
            "acc_divergence":  round(acc_div, 4),
            "root_delta_norm": round(root_delta, 4),
            "leaf_hash_norm":  round(leaf_hash_norm, 4),
            "window_boundary": at_boundary,
            "anchor_age_norm": 0.0,
            "B_win":           wsB.window_id,
            "B_step":          wsB.expected_step_idx,
            "B_ctr":           wsB.expected_next_counter,
        })
        if ok: accepted += 1
        elif not forensic_continue: break

    return {
        "ok": True, "attack_modes": attack_modes,
        "session_id": grant.session_id,
        "dt_range": nb.dt_ms_range,
        "meas_range": (round(nb.meas_range[0],6), round(nb.meas_range[1],6)),
        "op_allowlist": sorted(nb.op_allowlist),
        "accepted": accepted, "sent": len(trace),
        "dropped_reason": first_drop,
        "tampered_indices": sorted(tampered_set),
        "edits_applied": edits, "trace": trace,
        "window_certs": cert_list,
    }


# ══════════════════════════════════════════════════════════════════════════════
# HUB pre-training dataset builder
# ══════════════════════════════════════════════════════════════════════════════

def build_hub_dataset(bundle: ChainParamBundle,
                      log_cb=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate full training dataset using bundle params.
    Includes single-attack and compound 2-attack traces.
    """
    Xs, Ys = [], []
    single_attacks = [a for a in ATK_LABELS if a != "none"]

    total_jobs = (len(ATK_LABELS) * bundle.n_per_std +
                  len(list(itertools.combinations(single_attacks, 2))) * bundle.n_combo)
    done = 0

    # Single-attack traces
    SEED_MOD = 2**31 - 1
    for ci, atk in enumerate(ATK_LABELS):
        for i in range(bundle.n_per_std):
            np.random.seed((bundle.shared_train_seed * 1000 + ci * 100 + i) % SEED_MOD)
            try:
                meta = simulate_chain(bundle, [atk], forensic_continue=True)
                if not meta.get("ok") or not meta["trace"]: continue
                Xs.append(featurise(meta["trace"]))
                Ys.append(make_multihot([ci]))
            except Exception: pass
            done += 1
            if log_cb and done % 20 == 0:
                log_cb(f"Dataset: {done}/{total_jobs} traces ({atk})")

    # Compound 2-attack traces
    for pair_idx, (a1, a2) in enumerate(itertools.combinations(single_attacks, 2)):
        for i in range(bundle.n_combo):
            np.random.seed((bundle.shared_train_seed * 500 + pair_idx*50 + i) % SEED_MOD)
            try:
                meta = simulate_chain(bundle, [a1, a2], forensic_continue=True)
                if not meta.get("ok") or not meta["trace"]: continue
                Xs.append(featurise(meta["trace"]))
                label_idxs = [ATK_LABELS.index(a1), ATK_LABELS.index(a2)]
                Ys.append(make_multihot(label_idxs))
            except Exception: pass
            done += 1
            if log_cb and done % 20 == 0:
                log_cb(f"Dataset: {done}/{total_jobs} traces ({a1}+{a2})")

    return np.stack(Xs).astype(np.float32), np.stack(Ys).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# HUB Governor class
# ══════════════════════════════════════════════════════════════════════════════

class HUBGovernor:
    """
    Central parameter governor.
    Issues ChainParamBundles, pre-trains GRU, manages weight registry.
    Goes silent once peers start their session.
    """

    HUB_KEY = H(b"WARLOK_HUB_MASTER_KEY_v1")

    def __init__(self):
        self.registry   = get_registry()
        self.bundles:  Dict[str, ChainParamBundle]   = {}
        self.incidents: List[IncidentCertificate]    = []
        self.audit_log: List[str]                    = []

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.audit_log.append(f"[{ts}] {msg}")

    def generate_bundle(self,
                        peer_a_id:    str = "peerA",
                        peer_b_id:    str = "peerB",
                        anchor_a:     Optional[SolidStateAnchor] = None,
                        anchor_b:     Optional[SolidStateAnchor] = None,
                        window_size:  int   = 48,
                        n_per_std:    int   = 30,
                        n_combo:      int   = 20,
                        rnn_hdim:     int   = 96,
                        rnn_epochs:   int   = 60,
                        rnn_lr:       float = 0.006,
                        dt_ms_max:    int   = 150,
                        threshold:    float = 0.35,
                        os_seed:      int   = 42) -> ChainParamBundle:
        """Generate a fully-specified ChainParamBundle for a peer pair."""
        import time as _t
        epoch = int(_t.time()) // 3600

        anchor_a_fp = anchor_a.public_fp if anchor_a else "0"*16
        anchor_b_fp = anchor_b.public_fp if anchor_b else "0"*16
        anchor_a_bytes = anchor_a.anchor_hash if anchor_a else b"\x00"*32
        anchor_b_bytes = anchor_b.anchor_hash if anchor_b else b"\x00"*32

        shared_seed_bytes = H(anchor_a_bytes + anchor_b_bytes +
                              epoch.to_bytes(8,"big"))
        shared_seed_int   = int.from_bytes(shared_seed_bytes[:4], "big")

        acc_salt = H(b"ACC_SALT|" + anchor_a_bytes[:16] + anchor_b_bytes[:16])

        bundle = ChainParamBundle(
            session_epoch       = epoch,
            peer_a_id           = peer_a_id,
            peer_b_id           = peer_b_id,
            window_size         = window_size,
            counter_init        = 1,
            counter_stride      = 1,
            dt_ms_min           = 0,
            dt_ms_max           = dt_ms_max,
            dt_ms_slack         = 10,
            meas_slack          = 0.02,
            op_allowlist        = ["READ", "WRITE"],
            forensic_mode       = True,
            os_seed             = os_seed,
            hw_pcr_hex          = H(b"PCR|" + str(os_seed).encode()).hex()[:16],
            kernel_hash_hex     = H(b"KERN|" + str(os_seed).encode()).hex()[:16],
            process_snap_hex    = H(b"PROC|" + str(os_seed).encode()).hex()[:16],
            enclave_nonce_hex   = H(b"ENC|"  + str(os_seed).encode()).hex()[:16],
            leaf_hash_algo      = "sha256",
            tree_arity          = 2,
            acc_init_salt_hex   = acc_salt.hex()[:32],
            shared_train_seed   = shared_seed_int,
            n_per_std           = n_per_std,
            n_combo             = n_combo,
            rnn_hdim            = rnn_hdim,
            rnn_epochs          = rnn_epochs,
            rnn_lr              = rnn_lr,
            rnn_batch           = 32,
            detection_threshold = threshold,
            feature_dim         = 15,
            anchor_a_fp         = anchor_a_fp,
            anchor_b_fp         = anchor_b_fp,
        )
        bundle.compute_hash(self.HUB_KEY)
        session_key = f"{peer_a_id}:{peer_b_id}:{epoch}"
        self.bundles[session_key] = bundle
        self._log(f"Bundle generated for {peer_a_id}↔{peer_b_id} epoch={epoch}")
        return bundle

    def pretrain(self, bundle: ChainParamBundle,
                 extra_traces: List[Tuple[np.ndarray, List[int]]] = None,
                 log_cb=None) -> Tuple[dict, List[float]]:
        """
        Pre-train GRU on full chain simulation with bundle params.
        Stores weights in registry. Returns (params, losses).
        """
        self._log(f"Pre-training started (seed={bundle.shared_train_seed}, "
                  f"epochs={bundle.rnn_epochs})")

        # Build dataset
        X, Y = build_hub_dataset(bundle, log_cb=log_cb)

        # Add any extra user traces
        if extra_traces:
            extra_X = np.stack([x for x,_ in extra_traces]).astype(np.float32)
            extra_Y = np.stack([make_multihot(li) for _,li in extra_traces]).astype(np.float32)
            X = np.concatenate([X, extra_X], axis=0)
            Y = np.concatenate([Y, extra_Y], axis=0)

        N = X.shape[0]
        p = __import__("model", fromlist=["init_rnn"]).init_rnn(
            bundle.shared_train_seed, bundle.rnn_hdim
        )
        opt = {"t":0,"m":{},"v":{}}
        losses = []
        B = bundle.rnn_batch

        for ep in range(1, bundle.rnn_epochs+1):
            idx = np.random.permutation(N); ep_loss=0.0; nb=0
            for s in range(0, N, B):
                b = idx[s:s+B]
                if len(b) < 2: continue
                from model import rnn_step_multilabel
                ep_loss += rnn_step_multilabel(p, X[b], Y[b], opt, bundle.rnn_lr)
                nb += 1
            avg = ep_loss / max(nb,1); losses.append(avg)
            if log_cb and (ep==1 or ep%max(1,bundle.rnn_epochs//8)==0):
                log_cb(f"epoch {ep:3d}/{bundle.rnn_epochs}  loss={avg:.4f}")

        # Store in registry
        session_key = f"{bundle.peer_a_id}:{bundle.peer_b_id}:{bundle.session_epoch}"
        anchor_fp   = bundle.anchor_a_fp + ":" + bundle.anchor_b_fp
        rec = self.registry.store(session_key, p, losses, anchor_fp, N)
        self._log(f"Weights stored v{rec.version} loss={losses[-1]:.4f} n={N}")
        return p, losses

    def receive_incident(self, cert: IncidentCertificate):
        self.incidents.append(cert)
        self._log(f"Incident received from {cert.peer_id}: "
                  f"{cert.attack_classes} → {cert.action_taken}")

    def get_audit_log(self) -> List[str]:
        return list(reversed(self.audit_log))
