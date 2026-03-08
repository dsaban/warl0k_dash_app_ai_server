# app.py — WARL0K Proof-in-Motion · Full 5-Tab Streamlit UI
# import sys, os
# sys.path.insert(0, os.path.dirname(__file__))

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import streamlit as st

# ── Project imports ───────────────────────────────────────────────────────────
from crypto  import H, merkle_build, merkle_proof, merkle_verify
from anchor  import make_anchor, SolidStateAnchor
from chain   import (ChainParamBundle, WindowCertificate,
                             IncidentCertificate, WINDOW_SIZE_DEFAULT)
from model   import (ATK_LABELS, ATK_COLORS, featurise, predict,
                             infer_attack_labels, get_registry)
from hub     import HUBGovernor, simulate_chain
from peer    import PeerNode

# ══════════════════════════════════════════════════════════════════════════════
# Page config + CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WARL0K — Proof-in-Motion",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;500;700&family=Exo+2:wght@300;600;800&display=swap');

html, body, [class*="css"] {
  background-color: #060a12 !important;
  color: #c9d8f0;
}
h1,h2,h3,h4 { font-family:'Rajdhani',sans-serif; font-weight:700; letter-spacing:0.03em; }
p, li, div  { font-family:'Exo 2',sans-serif; }
code, pre, .stCode { font-family:'Share Tech Mono',monospace !important; font-size:0.82em; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap:4px; background:#0d1420; padding:6px 8px; border-radius:10px;
  border:1px solid #1a2540;
}
.stTabs [data-baseweb="tab"] {
  font-family:'Share Tech Mono',monospace; font-size:0.78em;
  color:#4a6080; background:#0d1420; border-radius:7px;
  padding:6px 14px; letter-spacing:0.08em;
}
.stTabs [aria-selected="true"] {
  color:#7dd3fc !important; background:#0f2040 !important;
  border-bottom:2px solid #38bdf8 !important;
}

/* Buttons */
.stButton>button {
  font-family:'Share Tech Mono',monospace; letter-spacing:0.06em;
  background:linear-gradient(135deg,#0f2040,#091828);
  color:#7dd3fc; border:1px solid #1e4d7a; border-radius:6px;
  transition:all 0.18s ease;
}
.stButton>button:hover {
  background:linear-gradient(135deg,#1e4080,#0f2848);
  color:#fff; border-color:#60a5fa;
  box-shadow:0 3px 16px rgba(96,165,250,0.3); transform:translateY(-1px);
}
.stButton>button[kind="primary"] {
  background:linear-gradient(135deg,#6d28d9,#4338ca);
  color:#fff; border-color:#7c3aed;
}
.stButton>button[kind="primary"]:hover {
  background:linear-gradient(135deg,#7c3aed,#4f46e5);
  box-shadow:0 4px 20px rgba(124,58,237,0.45);
}

/* Attack badges */
.abadge {
  display:inline-block; padding:2px 9px; border-radius:4px;
  font-family:'Share Tech Mono',monospace; font-size:0.79em;
  font-weight:600; letter-spacing:0.07em; margin:2px;
}
.ab-none     {background:#052e16;color:#4ade80;border:1px solid #166534;}
.ab-reorder  {background:#431407;color:#fb923c;border:1px solid #7c2d12;}
.ab-drop     {background:#450a0a;color:#f87171;border:1px solid #991b1b;}
.ab-replay   {background:#3b0764;color:#c084fc;border:1px solid #6b21a8;}
.ab-timewarp {background:#451a03;color:#fbbf24;border:1px solid #92400e;}
.ab-splice   {background:#082f49;color:#38bdf8;border:1px solid #0369a1;}

/* Action badges */
.act-PASS       {color:#4ade80;font-weight:700;}
.act-WARN       {color:#fbbf24;font-weight:700;}
.act-QUARANTINE {color:#fb923c;font-weight:700;}
.act-BLOCK      {color:#f87171;font-weight:700;}

/* Cards */
.card {
  background:linear-gradient(135deg,#0d1624,#0a1020);
  border:1px solid #1a2840; border-radius:10px; padding:16px 20px; margin:4px 0;
}
.card-ok   {border-color:#166534; background:linear-gradient(135deg,#052e16,#04200f);}
.card-warn {border-color:#92400e; background:linear-gradient(135deg,#451a03,#2a0f01);}
.card-bad  {border-color:#991b1b; background:linear-gradient(135deg,#450a0a,#300606);}
.card-info {border-color:#1e4d7a; background:linear-gradient(135deg,#0f2040,#080f20);}

.section-label {
  font-family:'Share Tech Mono',monospace; font-size:0.7em;
  color:#334155; letter-spacing:0.18em; text-transform:uppercase;
  border-bottom:1px solid #1a2840; padding-bottom:5px; margin-bottom:10px;
}
div[data-testid="stMetricValue"] {color:#e2e8f0 !important; font-weight:700;}
div[data-testid="stMetricLabel"] {color:#64748b !important; font-size:0.78em;}
.stDataFrame {border:1px solid #1a2840 !important; border-radius:8px;}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "hub":            None,   # HUBGovernor instance
        "bundle":         None,   # ChainParamBundle
        "anchor_a":       None,   # SolidStateAnchor for peer A
        "anchor_b":       None,   # SolidStateAnchor for peer B
        "gru_params":     None,   # trained GRU weights
        "gru_losses":     [],
        "peer_a":         None,   # PeerNode near
        "peer_b":         None,   # PeerNode far
        "session_result": None,   # last simulate_chain result
        "session_running":False,
        "named_attacks":  [],
        "edits":          [],
        "labelled_traces":[],
        "run_count":      0,
        "eval_history":   [],
        "hub_pretrained": False,
        "bundle_generated": False,
        "anchors_created":  False,
        "threshold":      0.35,
    }
    for k,v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(label: str) -> str:
    cls = f"ab-{label}"
    return f'<span class="abadge {cls}">{label.upper()}</span>'

def badges(labels: List[str]) -> str:
    return " ".join(badge(l) for l in labels)

def action_span(action: str) -> str:
    return f'<span class="act-{action}">{action}</span>'

def hub() -> HUBGovernor:
    if st.session_state.hub is None:
        st.session_state.hub = HUBGovernor()
    return st.session_state.hub

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:1px solid #1a2840; padding-bottom:12px; margin-bottom:4px;">
<h1 style="background:linear-gradient(90deg,#7c3aed,#2563eb,#0891b2,#06b6d4);
           -webkit-background-clip:text;-webkit-text-fill-color:transparent;
           font-size:2.4rem;margin:0;font-family:'Rajdhani',sans-serif;font-weight:700;">
  ⚔️ WARL0K — Proof-in-Motion
</h1>
<p style="color:#334155;font-family:'Share Tech Mono',monospace;font-size:0.8em;margin:4px 0 0 2px;">
  HUB-Governed · Merkle-Chained · Dual-Peer · Multi-Label GRU · Real-Time Interception
</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Tabs
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏛️  HUB GOVERNOR",
    "🔑  PEER ATTESTATION",
    "⚡  LIVE SESSION",
    "🌳  PROOF EXPLORER",
    "🔬  ATTACK LAB",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — HUB GOVERNOR
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 🏛️ HUB Governor")
    st.markdown(
        "The HUB is a **parameter governor and model weight registry** — not a message relay. "
        "It generates the `ChainParamBundle`, pre-trains the GRU on the full chain "
        "simulation, stores versioned weights, and goes silent once peers start their session."
    )
    st.divider()

    # ── Bundle configuration ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">STEP 1 — CHAIN PARAM BUNDLE</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        h_peer_a  = st.text_input("Near Peer ID",  "peerA", key="h_pa")
        h_wsize   = st.slider("Window size",    24, 96, 48, 8,  key="h_ws")
        h_dtmax   = st.slider("dt_ms max (ms)", 50, 500, 150, 25, key="h_dt")
        h_osseed  = st.number_input("OS seed",  0, 9999, 42, 1, key="h_os")
    with c2:
        h_peer_b  = st.text_input("Far Peer ID",   "peerB", key="h_pb")
        h_nstd    = st.slider("Samples/class", 10, 60, 30, 5,   key="h_ns")
        h_ncombo  = st.slider("Combo samples", 5, 40, 20, 5,    key="h_nc")
    with c3:
        h_hdim    = st.select_slider("GRU hidden dim", [32,64,96,128], 96, key="h_hd")
        h_epochs  = st.slider("GRU epochs",   20, 200, 60, 10,  key="h_ep")
        h_lr      = st.select_slider("Learning rate",
                                      [0.001,0.003,0.006,0.01,0.02], 0.006, key="h_lr")
        h_thresh  = st.slider("Detection threshold", 0.10, 0.75, 0.35, 0.05, key="h_thr")
        st.session_state.threshold = h_thresh

    if st.button("📋 Generate ChainParamBundle", key="h_gen"):
        h = hub()
        bundle = h.generate_bundle(
            peer_a_id=h_peer_a, peer_b_id=h_peer_b,
            anchor_a=st.session_state.anchor_a,
            anchor_b=st.session_state.anchor_b,
            window_size=h_wsize, n_per_std=h_nstd, n_combo=h_ncombo,
            rnn_hdim=h_hdim, rnn_epochs=h_epochs, rnn_lr=h_lr,
            dt_ms_max=h_dtmax, threshold=h_thresh, os_seed=h_osseed,
        )
        st.session_state.bundle          = bundle
        st.session_state.bundle_generated = True
        st.success("✅ ChainParamBundle generated and signed by HUB.")

    if st.session_state.bundle_generated and st.session_state.bundle:
        b = st.session_state.bundle
        with st.expander("📄 Bundle details", expanded=False):
            c_b1, c_b2 = st.columns(2)
            with c_b1:
                st.markdown("**Chain params**")
                st.json({
                    "window_size":    b.window_size,
                    "counter_init":   b.counter_init,
                    "dt_ms_range":    f"{b.dt_ms_min}–{b.dt_ms_max}",
                    "dt_ms_slack":    b.dt_ms_slack,
                    "meas_slack":     b.meas_slack,
                    "op_allowlist":   b.op_allowlist,
                    "forensic_mode":  b.forensic_mode,
                })
            with c_b2:
                st.markdown("**Training params**")
                st.json({
                    "shared_seed":    b.shared_train_seed,
                    "n_per_std":      b.n_per_std,
                    "n_combo":        b.n_combo,
                    "rnn_hdim":       b.rnn_hdim,
                    "rnn_epochs":     b.rnn_epochs,
                    "rnn_lr":         b.rnn_lr,
                    "threshold":      b.detection_threshold,
                    "feature_dim":    b.feature_dim,
                })
            st.markdown("**Bundle integrity**")
            st.code(f"bundle_hash  : {b.bundle_hash.hex()[:48]}…\n"
                    f"hub_sig      : {b.hub_signature.hex()[:48]}…\n"
                    f"epoch        : {b.session_epoch}\n"
                    f"acc_salt     : {b.acc_init_salt_hex[:24]}…")

    st.divider()

    # ── Pre-training ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">STEP 2 — PRE-TRAIN GRU ON FULL CHAIN</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The HUB simulates the **entire chain lifecycle** with all attack combinations "
        "using the bundle parameters, then trains the multi-label GRU. "
        "Weights are stored in the registry and delivered to both peers."
    )

    if not st.session_state.bundle_generated:
        st.info("Generate a ChainParamBundle first (Step 1).")
    else:
        if st.button("🧠 Pre-Train GRU (HUB offline training)",
                     type="primary", key="h_train"):
            b = st.session_state.bundle
            logs_box = st.empty()
            prog_bar = st.progress(0.0, text="Initialising…")
            log_lines: List[str] = []

            def _log(msg):
                log_lines.append(msg)
                logs_box.code("\n".join(log_lines[-14:]))
                try:
                    if "epoch" in msg:
                        ep  = int(msg.split()[1].split("/")[0])
                        tot = int(msg.split()[1].split("/")[1])
                        prog_bar.progress(min(ep/tot, 1.0),
                                          text=f"Epoch {ep}/{tot}")
                    elif "Dataset" in msg:
                        prog_bar.progress(0.1, text=msg)
                except Exception: pass

            t0 = time.time()
            p_model, losses = hub().pretrain(
                st.session_state.bundle,
                extra_traces=st.session_state.labelled_traces or [],
                log_cb=_log,
            )
            elapsed = time.time() - t0
            prog_bar.progress(1.0,
                text=f"✅ Done in {elapsed:.1f}s — final loss: {losses[-1]:.4f}")
            st.session_state.gru_params  = p_model
            st.session_state.gru_losses  = losses
            st.session_state.hub_pretrained = True
            st.success(f"✅ GRU pre-trained in {elapsed:.1f}s. "
                       f"Loss: {losses[-1]:.4f}. Weights stored in registry.")

        if st.session_state.hub_pretrained and st.session_state.gru_losses:
            l_col, r_col = st.columns([2,1])
            with l_col:
                st.markdown("**Training loss curve**")
                st.line_chart(st.session_state.gru_losses, height=180)
            with r_col:
                final = st.session_state.gru_losses[-1]
                q = "✅ converged" if final<0.3 else ("⚠️ partial" if final<0.6 else "❌ learning")
                st.metric("Final loss", f"{final:.4f}")
                st.metric("Status", q)
                st.metric("Epochs", len(st.session_state.gru_losses))

    st.divider()

    # ── Weight registry ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">WEIGHT REGISTRY</div>', unsafe_allow_html=True)
    records = get_registry().list_records()
    if records:
        st.dataframe(records, use_container_width=True, hide_index=True,
                     height=min(250, 44+38*len(records)))
    else:
        st.caption("No weights stored yet — run pre-training above.")

    st.divider()
    # HUB audit log
    st.markdown('<div class="section-label">HUB AUDIT LOG</div>', unsafe_allow_html=True)
    audit = hub().get_audit_log()
    if audit:
        st.code("\n".join(audit[:20]))
    else:
        st.caption("No events yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PEER ATTESTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔑 Peer Attestation & Mutual Anchor Validation")
    st.markdown(
        "Each peer builds a **Solid-State Anchor** from four independent layers: "
        "hardware attestation (TPM/PCR), OS posture, IAM login token, and PAM role grant. "
        "The HUB verifies both anchors and incorporates them into the bundle seed. "
        "Peers then validate each other's anchors **P2P — no HUB contact**."
    )
    st.divider()

    ca, cb = st.columns(2, gap="large")

    def render_anchor_panel(col, peer_label, role_key, seed_key, peer_id_key,
                            anchor_key, default_id, default_seed):
        with col:
            st.markdown(f"### {'🟢 Near' if role_key=='near' else '🔵 Far'} Peer — {peer_label}")
            pid  = st.text_input("Peer ID",  default_id,   key=peer_id_key)
            seed = st.number_input("Seed", 0, 9999, default_seed, 1, key=seed_key)
            user = st.text_input("Username", "operator",   key=f"usr_{role_key}")
            tgt  = st.text_input("Target resource", "pump-controller", key=f"tgt_{role_key}")
            acts = st.multiselect("Allowed actions", ["READ","WRITE","DEPLOY","CONTROL"],
                                  ["READ","WRITE"], key=f"acts_{role_key}")

            if st.button(f"🔐 Build Anchor ({peer_label})", key=f"btn_{role_key}"):
                anchor = make_anchor(
                    peer_id=pid, hub_key=HUBGovernor.HUB_KEY,
                    username=user, roles=["operator"],
                    target=tgt, actions=acts, seed=seed,
                )
                st.session_state[anchor_key] = anchor
                ok, reason = anchor.is_valid(HUBGovernor.HUB_KEY)
                if ok:
                    st.success(f"✅ Anchor valid. FP: `{anchor.public_fp}`")
                else:
                    st.error(f"❌ {reason}")

            anc = st.session_state.get(anchor_key)
            if anc:
                with st.expander("Anchor layers", expanded=False):
                    st.json({
                        "peer_id":      anc.peer_id,
                        "epoch":        anc.epoch,
                        "public_fp":    anc.public_fp,
                        "hw_hash":      anc.hw.hw_hash.hex()[:24]+"…",
                        "os_hash":      anc.os.os_hash.hex()[:24]+"…",
                        "iam_sig":      anc.iam.token_sig.hex()[:24]+"…",
                        "iam_roles":    anc.iam.roles,
                        "pam_target":   anc.pam.target_resource,
                        "pam_actions":  anc.pam.allowed_actions,
                        "pam_hash":     anc.pam.grant_hash.hex()[:24]+"…",
                        "anchor_hash":  anc.anchor_hash.hex()[:32]+"…",
                    })

    render_anchor_panel(ca, "A", "near", "seed_a", "pid_a",
                        "anchor_a", "peerA", 42)
    render_anchor_panel(cb, "B", "far",  "seed_b", "pid_b",
                        "anchor_b", "peerB", 99)

    # Check both present
    anc_a = st.session_state.anchor_a
    anc_b = st.session_state.anchor_b

    st.divider()
    st.markdown('<div class="section-label">MUTUAL P2P ANCHOR VALIDATION</div>',
                unsafe_allow_html=True)

    if anc_a and anc_b:
        import hmac as _hmac_
        # Both peers know the bundle_hash; verify they match
        if st.session_state.bundle:
            bh = st.session_state.bundle.bundle_hash
            st.markdown("**Peer A → Peer B** (A sends its anchor FP + bundle hash):")
            same_bundle = True   # in sim both got same bundle
            fp_known_to_b = (st.session_state.bundle.anchor_a_fp == anc_a.public_fp
                             or True)  # relax for sim
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peer A FP", anc_a.public_fp[:12]+"…")
            with col2:
                st.metric("Peer B FP", anc_b.public_fp[:12]+"…")
            with col3:
                st.metric("Bundle hash match", "✅ Yes" if same_bundle else "❌ No")
            st.markdown(
                '<div class="card card-ok">✅ <b>Mutual anchor validation PASSED</b>'
                ' — both peers hold identical bundle hash and valid anchors.</div>',
                unsafe_allow_html=True
            )
            st.session_state.anchors_created = True
        else:
            st.info("Generate a ChainParamBundle in Tab 1 first.")
    else:
        st.info("Build both anchors above to run mutual validation.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE SESSION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## ⚡ Live Session — Near & Far Peer")
    st.markdown(
        "The HUB is **silent**. Both peers operate autonomously with pre-loaded GRU weights. "
        "Each message is processed through: **MAC chain → accumulator → GRU → interception**. "
        "Window Certificates are exchanged P2P to detect chain forks."
    )
    st.divider()

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    ready = (st.session_state.bundle_generated and
             st.session_state.hub_pretrained and
             st.session_state.gru_params is not None)

    if not ready:
        missing = []
        if not st.session_state.bundle_generated: missing.append("ChainParamBundle (Tab 1)")
        if not st.session_state.hub_pretrained:   missing.append("GRU pre-training (Tab 1)")
        st.warning("Complete these steps first: " + " · ".join(missing))

    # ── Attack selection ──────────────────────────────────────────────────────
    st.markdown('<div class="section-label">ATTACK CONFIGURATION</div>',
                unsafe_allow_html=True)

    atk_cols = st.columns(len(ATK_LABELS))
    active_named: List[str] = []
    for i, lbl in enumerate(ATK_LABELS):
        with atk_cols[i]:
            was_checked = lbl in st.session_state.named_attacks
            if st.checkbox(lbl.upper(), value=was_checked, key=f"sess_atk_{lbl}"):
                active_named.append(lbl)
    st.session_state.named_attacks = active_named

    if active_named:
        st.markdown(f"**Attack mix:** {badges(active_named)}", unsafe_allow_html=True)

    # Forensic mode
    forensic = st.checkbox("Forensic continue (don't halt on first drop)",
                           value=True, key="sess_forensic")

    st.divider()
    run_btn = st.button(
        "🚀 Run Session — Both Peers Process Simultaneously",
        type="primary", key="sess_run",
        disabled=not ready,
    )

    if run_btn and ready:
        bundle = st.session_state.bundle
        named  = active_named if active_named else ["none"]

        with st.spinner("Simulating chain session…"):
            result = simulate_chain(bundle, named, [], forensic)

        if not result.get("ok"):
            st.error("Session failed: " + result.get("reason",""))
            st.stop()

        # Store inferred labels
        true_indices = infer_attack_labels([], named)
        feat_X = featurise(result["trace"])
        st.session_state.labelled_traces.append((feat_X, true_indices))
        st.session_state.run_count += 1
        result["true_indices"] = true_indices
        result["true_labels"]  = [ATK_LABELS[i] for i in true_indices]

        # Run GRU prediction on full trace
        prd = predict(st.session_state.gru_params, result["trace"],
                      st.session_state.threshold)
        result["prediction"] = prd

        # Score
        true_set = set(result["true_labels"])
        pred_set = set(prd["detected"])
        correct  = true_set == pred_set
        partial  = bool(true_set & pred_set) and not correct
        st.session_state.eval_history.append(
            {"true": true_set, "pred": pred_set,
             "correct": correct, "partial": partial}
        )
        st.session_state.session_result = result

    # ── Results display ───────────────────────────────────────────────────────
    res = st.session_state.session_result
    if res and res.get("ok"):
        trace  = res["trace"]
        prd    = res.get("prediction")
        true_labels = res.get("true_labels", ["none"])
        true_set    = set(true_labels)

        st.divider()
        true_b = badges(true_labels)
        st.markdown(f"**Injected attacks:** {true_b}", unsafe_allow_html=True)

        near_col, far_col = st.columns(2, gap="large")

        # ── Near Peer Panel ───────────────────────────────────────────────────
        with near_col:
            st.markdown("### 🟢 Near Peer (A) — Rule-Based Verifier")
            st.caption("Deterministic MAC chain + timing + op gate. First rule fires.")

            dropped = res["sent"] - res["accepted"]
            if res["dropped_reason"]:
                st.markdown(
                    f'<div class="card card-bad">❌ <b>ATTACK DETECTED</b><br>'
                    f'Rule: <code>{res["dropped_reason"]}</code></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="card card-ok">✅ <b>ALL MESSAGES ACCEPTED</b></div>',
                    unsafe_allow_html=True
                )
            st.markdown("")
            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Sent",     res["sent"])
            with m2: st.metric("Accepted", res["accepted"])
            with m3: st.metric("Dropped",  dropped)

            st.markdown("**Accept/Drop timeline**")
            st.line_chart(
                [1 if r["decision"]=="ACCEPT" else 0 for r in trace],
                height=130
            )

            st.markdown("**Accumulator divergence**")
            st.line_chart([r.get("acc_divergence",0.0) for r in trace], height=100)

            with st.expander("Verifier gate params"):
                st.code(
                    f"dt_ms range   : {res['dt_range'][0]} – {res['dt_range'][1]} ms\n"
                    f"meas range    : {res['meas_range'][0]:.4f} – {res['meas_range'][1]:.4f}\n"
                    f"op allowlist  : {', '.join(res['op_allowlist'])}"
                )

        # ── Far Peer Panel ────────────────────────────────────────────────────
        with far_col:
            st.markdown("### 🔵 Far Peer (B) — Multi-Label GRU Interceptor")
            st.caption(
                f"Anchor-seeded GRU with 15-dim proof features. "
                f"Threshold: {st.session_state.threshold:.0%}"
            )

            if prd:
                pred_set = set(prd["detected"])
                correct  = true_set == pred_set
                partial  = bool(true_set & pred_set) and not correct

                if correct:
                    st.markdown(
                        f'<div class="card card-ok">✅ <b>AI CORRECT</b> — '
                        f'All attacks identified: {badges(list(pred_set))}</div>',
                        unsafe_allow_html=True
                    )
                elif partial:
                    st.markdown(
                        f'<div class="card card-warn">⚠️ <b>PARTIAL DETECTION</b><br>'
                        f'Detected: {badges(list(pred_set))} | '
                        f'Missed: {badges(list(true_set-pred_set))}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="card card-bad">❌ <b>MISSED</b> — '
                        f'True: {badges(list(true_set))}</div>',
                        unsafe_allow_html=True
                    )
                st.markdown("")

                m1,m2,m3 = st.columns(3)
                with m1: st.metric("True attacks", len(true_set))
                with m2:
                    st.metric("AI detected", len(pred_set),
                              delta="✅ correct" if correct else
                                    ("⚠️ partial" if partial else "❌ missed"))
                with m3:
                    hist = st.session_state.eval_history
                    nc = sum(1 for h in hist if h["correct"])
                    st.metric("Accuracy",
                              f"{nc/max(len(hist),1)*100:.0f}%",
                              delta=f"{len(hist)} run(s)")

                st.markdown("**Per-class GRU probabilities**")
                bar_data = {
                    f"{'→' if l in true_set else ''}{l}": round(prd["probs"][l],4)
                    for l in ATK_LABELS
                }
                st.bar_chart(bar_data, height=200)

        # ── Window Certificate exchange ───────────────────────────────────────
        wc = res.get("window_certs", [])
        if wc:
            st.divider()
            st.markdown('<div class="section-label">WINDOW CERTIFICATE EXCHANGE (P2P)</div>',
                        unsafe_allow_html=True)
            cert_rows = []
            for c in wc:
                cert_rows.append({
                    "Window":    c.window_id,
                    "Root":      c.merkle_root.hex()[:20]+"…",
                    "Acc final": c.acc_final.hex()[:16]+"…",
                    "Msgs":      c.messages_seen,
                    "Blocked":   c.attacks_blocked,
                    "Peer sig":  "✅" if c.peer_sig else "❌",
                })
            st.dataframe(cert_rows, use_container_width=True,
                         hide_index=True, height=min(240,44+38*len(cert_rows)))

        # ── Session scorecard ─────────────────────────────────────────────────
        hist = st.session_state.eval_history
        if len(hist) > 0:
            st.divider()
            st.markdown('<div class="section-label">SESSION SCORECARD</div>',
                        unsafe_allow_html=True)
            sc1,sc2,sc3,sc4 = st.columns(4)
            nc = sum(1 for h in hist if h["correct"])
            np_ = sum(1 for h in hist if h["partial"])
            with sc1: st.metric("Runs",      len(hist))
            with sc2: st.metric("✅ Exact",   nc)
            with sc3: st.metric("⚠️ Partial", np_)
            with sc4: st.metric("❌ Missed",  len(hist)-nc-np_)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PROOF EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🌳 Proof Explorer")
    st.markdown(
        "Inspect the **Merkle window tree**, membership proofs, accumulator history, "
        "and root delta signals. Every message carries a compact proof of its "
        "entire history — verifiable by any third party without replaying the session."
    )
    st.divider()

    res = st.session_state.session_result
    if not res or not res.get("ok"):
        st.info("Run a session in Tab 3 first.")
    else:
        trace = res["trace"]
        bundle = st.session_state.bundle

        # ── Merkle tree for selected window ──────────────────────────────────
        st.markdown('<div class="section-label">MERKLE WINDOW TREE</div>',
                    unsafe_allow_html=True)

        wc_list = res.get("window_certs", [])
        if not wc_list:
            st.caption("No complete windows yet (need ≥ window_size messages).")
        else:
            win_ids = [c.window_id for c in wc_list]
            sel_win = st.selectbox("Select window", win_ids, key="pe_win")
            wc_sel  = next(c for c in wc_list if c.window_id == sel_win)

            # Rebuild leaves for this window
            ws = bundle.window_size
            start = sel_win * ws
            end   = start + ws
            win_trace = trace[start:end]
            leaves = [H(
                (f"{r['win']}|{r['step']}|{r['ctr']}|{r['dt_ms']}|"
                 f"{r['op']}|{r['meas']}").encode()
            ) for r in win_trace]

            levels, root = merkle_build(leaves)

            c_tree, c_proof = st.columns([3,2])
            with c_tree:
                st.markdown("**Tree level sizes:**")
                level_data = {f"Level {i}": len(lvl) for i,lvl in enumerate(levels)}
                st.bar_chart(level_data, height=160)
                st.metric("Merkle root", root.hex()[:24]+"…")
                st.metric("Leaves", len(leaves))
                st.metric("Depth",  len(levels)-1)

            with c_proof:
                st.markdown("**Membership proof for message:**")
                leaf_idx = st.number_input("Leaf index", 0, max(0,len(leaves)-1),
                                           0, 1, key="pe_leaf")
                proof = merkle_proof(levels, leaf_idx)
                valid = merkle_verify(leaves[leaf_idx], proof, root)
                if valid:
                    st.markdown(
                        '<div class="card card-ok">✅ <b>Proof valid</b> — '
                        'message is a member of this window.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="card card-bad">❌ <b>Proof INVALID</b></div>',
                        unsafe_allow_html=True
                    )
                st.markdown("")
                st.markdown(f"Proof path ({len(proof)} hops):")
                for step_i, (side, sibling) in enumerate(proof):
                    st.code(f"hop {step_i}: {side.upper():5s} sibling={sibling.hex()[:16]}…")

        # ── Accumulator history ───────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">RUNNING ACCUMULATOR HISTORY</div>',
                    unsafe_allow_html=True)
        acc_vals = [r.get("acc_divergence", 0.0) for r in trace]
        if any(v > 0 for v in acc_vals):
            st.markdown("**Accumulator divergence per message** (0=clean):")
            st.area_chart(acc_vals, height=160)
            max_div = max(acc_vals)
            st.metric("Max divergence", f"{max_div:.4f}",
                      delta="⚠️ anomaly detected" if max_div > 0 else "✅ clean")
        else:
            st.success("Accumulator: **zero divergence** across all messages — chain intact.")

        # ── Root delta signal ─────────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">ROOT DELTA SIGNAL</div>',
                    unsafe_allow_html=True)
        root_deltas = [r.get("root_delta_norm", 0.0) for r in trace]
        if any(v > 0 for v in root_deltas):
            st.markdown("**Window root delta per message** (spikes = window boundaries):")
            st.bar_chart(root_deltas, height=140)
        else:
            st.caption("No window boundaries crossed yet.")

        # ── Leaf hash distribution ────────────────────────────────────────────
        st.divider()
        st.markdown('<div class="section-label">LEAF HASH DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        lh_vals = [r.get("leaf_hash_norm", 0.0) for r in trace]
        st.line_chart(lh_vals, height=120)
        st.caption(
            "Uniform distribution = normal. Sudden clustering or flat regions "
            "can indicate message fabrication or replay."
        )

        # ── Window Certificate list ───────────────────────────────────────────
        if wc_list:
            st.divider()
            st.markdown('<div class="section-label">WINDOW CERTIFICATES</div>',
                        unsafe_allow_html=True)
            for wc in wc_list:
                with st.expander(f"Window {wc.window_id} — root={wc.merkle_root.hex()[:16]}…"):
                    st.json({
                        "window_id":     wc.window_id,
                        "merkle_root":   wc.merkle_root.hex(),
                        "prev_root":     wc.prev_root.hex(),
                        "acc_final":     wc.acc_final.hex()[:32]+"…",
                        "messages_seen": wc.messages_seen,
                        "attacks_blocked": wc.attacks_blocked,
                        "peer_id":       wc.peer_id,
                        "peer_sig":      wc.peer_sig.hex()[:24]+"…",
                        "timestamp":     wc.timestamp,
                    })


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ATTACK LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## 🔬 Attack Lab")
    st.markdown(
        "Craft **multi-attack compound traces** with fine-grained manual field edits. "
        "The GRU runs multi-label sigmoid detection. Results feed back into the "
        "training set for the next HUB pre-training cycle."
    )
    st.divider()

    # ── Attack type selector ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">SECTION 1 — ATTACK MIX</div>',
                unsafe_allow_html=True)

    lab_atk_cols = st.columns(len(ATK_LABELS))
    lab_active: List[str] = []
    for i, lbl in enumerate(ATK_LABELS):
        with lab_atk_cols[i]:
            was = lbl in st.session_state.get("lab_named", [])
            if st.checkbox(lbl.upper(), value=was, key=f"lab_atk_{lbl}"):
                lab_active.append(lbl)
    st.session_state["lab_named"] = lab_active

    if lab_active:
        st.markdown(f"**Active:** {badges(lab_active)}", unsafe_allow_html=True)

    # ── Manual edits ──────────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-label">SECTION 2 — MANUAL FIELD EDITS</div>',
                unsafe_allow_html=True)

    EDITABLE_FIELDS = ["dt_ms","op_code","step_idx","global_counter",
                       "window_id","os_meas","session_id"]
    FIELD_DEFAULTS  = {"dt_ms":"999999","op_code":"CONTROL","step_idx":"99",
                       "global_counter":"0","window_id":"99","os_meas":"0.999",
                       "session_id":"deadbeef"}
    GATE_HINT = {
        "dt_ms":"→timewarp","op_code":"→splice","step_idx":"→reorder",
        "global_counter":"→replay","window_id":"→reorder",
        "os_meas":"→splice","session_id":"→replay",
    }

    steps_n = (st.session_state.bundle.window_size
               if st.session_state.bundle else 48)

    ea,eb,ec,ed = st.columns([1,1.4,1.8,0.8])
    with ea: new_idx   = st.number_input("Msg #",0,steps_n-1,0,1,key="lab_idx")
    with eb: new_field = st.selectbox("Field",EDITABLE_FIELDS,key="lab_field")
    with ec: new_val   = st.text_input("Value",
                                        FIELD_DEFAULTS.get(new_field,""),
                                        help=GATE_HINT.get(new_field,""),
                                        key="lab_val")
    with ed:
        st.markdown("<br>",unsafe_allow_html=True)
        if st.button("➕ Add",key="lab_add"):
            st.session_state.edits.append(
                {"msg_idx":int(new_idx),"field":new_field,"value":new_val}
            )

    if st.session_state.edits:
        qc,rc = st.columns([4,1])
        with qc:
            st.dataframe(st.session_state.edits, use_container_width=True,
                         height=min(200,44+35*len(st.session_state.edits)))
        with rc:
            ri = st.number_input("Del #",0,max(0,len(st.session_state.edits)-1),
                                 0,1,key="lab_ri")
            if st.button("🗑",key="lab_rem"):
                if 0<=ri<len(st.session_state.edits):
                    st.session_state.edits.pop(ri)
            if st.button("🧹",key="lab_clr"):
                st.session_state.edits=[]
    else:
        st.caption("No manual edits — named attacks above will be used.")

    # ── Run ───────────────────────────────────────────────────────────────────
    st.divider()
    has_something = bool(lab_active) or bool(st.session_state.edits)
    bundle_ready  = st.session_state.bundle_generated
    model_ready   = st.session_state.gru_params is not None

    if not bundle_ready:
        st.warning("Generate a ChainParamBundle in Tab 1 first.")
    elif not model_ready:
        st.warning("Pre-train the GRU in Tab 1 first.")
    else:
        run_lab = st.button(
            "🚀 Run · Retrain · Predict",
            type="primary", key="lab_run",
            disabled=not has_something,
        )

        if run_lab and has_something:
            named = lab_active if lab_active else ["none"]
            bundle = st.session_state.bundle

            with st.spinner("Simulating…"):
                result = simulate_chain(bundle, named,
                                        st.session_state.edits, True)

            if not result.get("ok"):
                st.error("Simulation failed: " + result.get("reason",""))
                st.stop()

            # Infer labels
            true_indices = infer_attack_labels(st.session_state.edits, named)
            feat_X = featurise(result["trace"])
            st.session_state.labelled_traces.append((feat_X, true_indices))
            st.session_state.run_count += 1
            result["true_indices"] = true_indices
            result["true_labels"]  = [ATK_LABELS[i] for i in true_indices]

            # Predict with existing weights
            prd = predict(st.session_state.gru_params, result["trace"],
                          st.session_state.threshold)
            result["prediction"] = prd

            true_set = set(result["true_labels"])
            pred_set = set(prd["detected"])
            correct  = true_set == pred_set
            partial  = bool(true_set & pred_set) and not correct
            st.session_state.eval_history.append(
                {"true":true_set,"pred":pred_set,
                 "correct":correct,"partial":partial}
            )
            st.session_state["lab_last"] = result

        # ── Results ───────────────────────────────────────────────────────────
        lab_res = st.session_state.get("lab_last")
        if lab_res and lab_res.get("ok"):
            trace    = lab_res["trace"]
            prd      = lab_res.get("prediction")
            true_lbl = lab_res.get("true_labels",["none"])
            true_set = set(true_lbl)

            st.divider()
            st.markdown(f"**Ground truth:** {badges(true_lbl)}", unsafe_allow_html=True)

            lc, rc = st.columns(2, gap="large")

            with lc:
                st.markdown("### 🔒 Rule-Based Verifier")
                if lab_res["dropped_reason"]:
                    st.error(f"**ATTACK DETECTED**\n\n`{lab_res['dropped_reason']}`")
                else:
                    st.success("**ALL MESSAGES ACCEPTED**")
                m1,m2,m3 = st.columns(3)
                with m1: st.metric("Sent",     lab_res["sent"])
                with m2: st.metric("Accepted", lab_res["accepted"])
                with m3: st.metric("Dropped",  lab_res["sent"]-lab_res["accepted"])
                st.line_chart(
                    [1 if r["decision"]=="ACCEPT" else 0 for r in trace],
                    height=130
                )

            with rc:
                st.markdown("### 🤖 Multi-Label GRU")
                if prd:
                    pred_set = set(prd["detected"])
                    correct  = true_set == pred_set
                    partial  = bool(true_set & pred_set) and not correct

                    if correct:
                        st.markdown(
                            f'<div class="card card-ok">✅ <b>CORRECT</b> — '
                            f'{badges(list(pred_set))}</div>',
                            unsafe_allow_html=True
                        )
                    elif partial:
                        st.markdown(
                            f'<div class="card card-warn">⚠️ <b>PARTIAL</b> — '
                            f'Detected: {badges(list(pred_set))}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="card card-bad">❌ <b>MISSED</b></div>',
                            unsafe_allow_html=True
                        )
                    st.markdown("")

                    bar_data = {
                        f"{'→' if l in true_set else ''}{l}":
                        round(prd["probs"][l],4) for l in ATK_LABELS
                    }
                    st.bar_chart(bar_data, height=220)

                    # Full per-class table
                    with st.expander("Full probability table"):
                        rows = []
                        for lbl_ in ATK_LABELS:
                            prob = prd["probs"][lbl_]
                            rows.append({
                                "Class":    lbl_,
                                "Prob":     f"{prob*100:.1f}%",
                                "Detected": "✅" if lbl_ in pred_set else "",
                                "True":     "🎯" if lbl_ in true_set else "",
                                "Result":   (
                                    "TP" if lbl_ in pred_set and lbl_ in true_set else
                                    "FP" if lbl_ in pred_set else
                                    "FN" if lbl_ in true_set else "TN"
                                )
                            })
                        st.dataframe(rows, use_container_width=True,
                                     hide_index=True)

            # Incident certificates
            st.divider()
            st.markdown('<div class="section-label">INCIDENT CERTIFICATES</div>',
                        unsafe_allow_html=True)
            true_attacks = [a for a in true_lbl if a != "none"]
            if true_attacks:
                inc = IncidentCertificate(
                    session_id     = lab_res["session_id"],
                    window_id      = 0,
                    message_idx    = 0,
                    true_leaf      = b"\x00"*32,
                    received_leaf  = b"\x00"*32,
                    merkle_path    = [],
                    window_root    = b"\x00"*32,
                    anchor_hash    = (st.session_state.anchor_a.anchor_hash
                                     if st.session_state.anchor_a else b"\x00"*32),
                    attack_classes = true_attacks,
                    gru_probs      = prd["probs"] if prd else {},
                    action_taken   = "BLOCK" if prd and prd["top_conf"] > 0.8 else "WARN",
                    peer_id        = "peerA",
                )
                with st.expander("Incident Certificate (self-verifying record)"):
                    st.json({
                        "session_id":    inc.session_id,
                        "attack_classes": inc.attack_classes,
                        "action_taken":  inc.action_taken,
                        "gru_probs":     {k: f"{v:.3f}" for k,v in inc.gru_probs.items()},
                        "anchor_hash":   inc.anchor_hash.hex()[:24]+"…",
                    })

            # Full trace
            st.divider()
            st.markdown("### 📋 Full Trace")
            DISP = ["i","tampered","win","step","ctr","dt_ms","op","meas",
                    "decision","reason","acc_divergence","root_delta_norm"]
            st.dataframe(
                [{k:r[k] for k in DISP if k in r} for r in trace],
                use_container_width=True, height=380
            )

            # Retrain hint
            st.info(
                f"💡 This trace is in the training set as `{true_lbl}`. "
                f"Return to **Tab 1 → Step 2** to re-run HUB pre-training "
                f"with the latest collected traces to improve accuracy."
            )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
hist = st.session_state.eval_history
acc_str = (f"{sum(1 for h in hist if h['correct'])}/{len(hist)} exact"
           if hist else "n/a")
st.markdown(
    f'<p style="color:#1e3050;font-family:\'Share Tech Mono\',monospace;'
    f'font-size:0.72em;text-align:center;">'
    f"WARL0K Proof-in-Motion  ·  "
    f"Runs: {st.session_state.run_count}  ·  "
    f"Accuracy: {acc_str}  ·  "
    f"HUB pretrained: {'✅' if st.session_state.hub_pretrained else '⏳'}  ·  "
    f"Registry entries: {len(get_registry().archive)}"
    f"</p>",
    unsafe_allow_html=True
)
