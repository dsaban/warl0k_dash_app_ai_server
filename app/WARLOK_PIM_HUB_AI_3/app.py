"""
WARL0K PIM Engine — Streamlit Edition
Dark terminal UI: Neural Core · Chain Verify · Chain Proof
"""

import streamlit as st
import numpy as np
import time
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pim_core as pim

# ════════════════════════════════════════════════════════════════════
# SESSION STATE — only safe pattern: if "k" not in st.session_state
# Must be the VERY FIRST thing after imports, before any st.* call
# ════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "params":       None,
    "session":      None,
    "training":     False,
    "train_done":   False,
    "train_error":  None,
    "loss_p1":      [],
    "loss_p2":      [],
    "train_phase":  0,
    "train_epoch":  0,
    "train_loss":   0.0,
    "train_s":      0.0,
    "log_lines":    [],
    "last_result":  None,
    "bulk_results": None,
    "bench_result": None,
    "enc_result":   None,
    "chain_data":   None,
    "progress":     0.0,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Shorthand helpers (always read/write via st.session_state["key"]) ──
def get(k):          return st.session_state[k]
def put(k, v):       st.session_state[k] = v
def append_log(msg): st.session_state["log_lines"].append(msg)

# ════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must come before any other st.* call)
# ════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WARL0K · PIM Auth Engine",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.npz")
CFG        = pim.CFG
TOTAL_EP   = CFG["EPOCHS_PHASE1"] + CFG["EPOCHS_PHASE2"]

# ── Auto-load saved model once per session ──────────────────────────
if st.session_state["params"] is None and os.path.exists(MODEL_PATH):
    try:
        _p, _ = pim.load_model(MODEL_PATH)
        st.session_state["params"]  = _p
        st.session_state["session"] = pim.PIMSession(_p)
        append_log(f"{time.strftime('%H:%M:%S')} ✓ Model auto-loaded from model.npz")
    except Exception as _e:
        append_log(f"{time.strftime('%H:%M:%S')} ✗ Auto-load failed: {_e}")

# ════════════════════════════════════════════════════════════════════
# CSS — dark terminal theme
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #080b0f !important;
    color: #c8d6e5 !important;
    font-family: 'Rajdhani', sans-serif !important;
}
.stApp { background-color: #080b0f !important; }
.main .block-container {
    padding-top: 0.5rem !important;
    padding-bottom: 1rem !important;
    max-width: 100% !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080b0f; }
::-webkit-scrollbar-thumb { background: #1e2d3d; }
h1,h2,h3 { font-family: 'Share Tech Mono', monospace !important; }

.stButton > button {
    background: transparent !important;
    border: 1px solid #007a40 !important;
    color: #00ff88 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: rgba(0,255,136,0.1) !important;
    box-shadow: 0 0 12px rgba(0,255,136,0.3) !important;
}
.stButton > button:disabled { opacity: 0.35 !important; }

.stNumberInput input, .stTextInput input {
    background-color: #111820 !important;
    border: 1px solid #1e2d3d !important;
    color: #c8d6e5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-baseweb="select"] > div {
    background-color: #111820 !important;
    border: 1px solid #1e2d3d !important;
    color: #c8d6e5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
[data-baseweb="select"] * {
    background-color: #111820 !important;
    color: #c8d6e5 !important;
    font-family: 'Share Tech Mono', monospace !important;
}
.stProgress > div > div > div > div {
    background: linear-gradient(90deg,#007a40,#00ff88) !important;
}
.stProgress > div > div > div {
    background: #111820 !important;
    border: 1px solid #1e2d3d !important;
}
label { color: #3d566e !important; font-family: 'Share Tech Mono', monospace !important;
        font-size: 0.68rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; }
[data-testid="stSidebar"] { background-color: #0d1117 !important; }
[data-testid="column"] { border-right: 1px solid #1e2d3d; padding-right: 0.8rem !important; padding-left: 0.8rem !important; }
[data-testid="column"]:last-child { border-right: none !important; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════════════
def html(s): st.markdown(s, unsafe_allow_html=True)

def panel_header(dot_color, title, live=False):
    live_dot = (f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_color};'
                f'display:inline-block;margin-left:auto;animation:pulse 1.4s ease infinite"></span>'
                if live else "")
    html(f"""<div style="display:flex;align-items:center;gap:10px;padding:8px 0;
                border-bottom:1px solid #1e2d3d;margin-bottom:12px">
        <div style="width:8px;height:8px;border-radius:50%;background:{dot_color}"></div>
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.76rem;
                     letter-spacing:2px;color:#7a9ab5;text-transform:uppercase">{title}</span>
        {live_dot}
    </div>""")

def slabel(txt):
    html(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.62rem;'
         f'color:#3d566e;letter-spacing:2px;text-transform:uppercase;margin:8px 0 5px 0">{txt}</div>')

def mrow(k, v, color="#c8d6e5"):
    html(f"""<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px">
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;color:#3d566e">{k}</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.88rem;font-weight:600;color:{color}">{v}</span>
    </div>""")

def cfgrow(k, v):
    html(f"""<div style="display:flex;justify-content:space-between;padding:3px 7px;
                background:#111820;border:1px solid #1e2d3d;margin-bottom:3px;
                font-family:'Share Tech Mono',monospace;font-size:0.7rem">
        <span style="color:#3d566e">{k}</span><span style="color:#00d4ff">{v}</span>
    </div>""")

def loss_svg(p1, p2):
    if not p1 and not p2:
        html('<div style="height:80px;background:#111820;border:1px solid #1e2d3d;'
             'display:flex;align-items:center;justify-content:center;'
             'font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;color:#3d566e">'
             'LOSS CURVE — TRAIN TO POPULATE</div>')
        return
    all_v = p1 + p2
    mx = max(all_v) if all_v else 1
    W, H = 460, 80
    def line(pts, col):
        if len(pts) < 2: return ""
        coords = " ".join(
            f"{i/(len(pts)-1)*W*(.5 if col=='#00c96a' else .5)+( 0 if col=='#00c96a' else W*.5):.1f},"
            f"{H-(v/mx)*H*.82-4:.1f}" for i, v in enumerate(pts))
        return f'<polyline points="{coords}" fill="none" stroke="{col}" stroke-width="1.5"/>'
    # build coords properly
    def seg(pts, x0, x1):
        if len(pts) < 2: return ""
        coords = " ".join(
            f"{x0+(x1-x0)*i/(len(pts)-1):.1f},{H-(v/mx)*H*.82-4:.1f}"
            for i, v in enumerate(pts))
        return f'<polyline points="{coords}" fill="none" stroke-width="1.5"/>'
    p1_svg = p2_svg = ""
    if len(p1) >= 2:
        c = " ".join(f"{W*.5*i/(len(p1)-1):.1f},{H-(v/mx)*H*.82-4:.1f}" for i,v in enumerate(p1))
        p1_svg = f'<polyline points="{c}" fill="none" stroke="#00c96a" stroke-width="1.5"/>'
    if len(p2) >= 2:
        c = " ".join(f"{W*.5+W*.5*i/(len(p2)-1):.1f},{H-(v/mx)*H*.82-4:.1f}" for i,v in enumerate(p2))
        p2_svg = f'<polyline points="{c}" fill="none" stroke="#00d4ff" stroke-width="1.5"/>'
    grid = "".join(f'<line x1="0" y1="{H-f*H*.82-4:.1f}" x2="{W}" y2="{H-f*H*.82-4:.1f}" stroke="#1e2d3d" stroke-width=".5"/>' for f in [.25,.5,.75])
    html(f"""<div style="background:#111820;border:1px solid #1e2d3d;margin-bottom:10px">
        <svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:{H}px;display:block">
            {grid}{p1_svg}{p2_svg}
            <text x="4" y="13" font-family="Share Tech Mono" font-size="10" fill="#007a40">P1</text>
            <text x="{W//2+4}" y="13" font-family="Share Tech Mono" font-size="10" fill="#0099cc">P2</text>
        </svg>
    </div>""")

def render_log(lines):
    items = ""
    for l in lines[-16:]:
        c = ("#00c96a" if any(x in l for x in ["✓","done","complete","loaded"])
             else "#ff2d55" if any(x in l for x in ["✗","Error","error","failed"])
             else "#ffb800" if any(x in l for x in ["Starting","Dataset","Model:"])
             else "#7a9ab5")
        items += f'<div style="color:{c};line-height:1.65">{l}</div>'
    html(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:0.68rem;
                background:#111820;border:1px solid #1e2d3d;padding:8px 10px;
                max-height:190px;overflow-y:auto;margin-bottom:8px">
        {items or '<span style="color:#3d566e">WARL0K PIM engine ready.</span>'}
    </div>""")

def render_verdict(r):
    ok   = r.get("ok", False)
    tam  = r.get("tamper","none")
    vc   = "#00ff88" if ok else "#ff2d55"
    vg   = "rgba(0,255,136,.28)" if ok else "rgba(255,45,85,.28)"
    bc   = "#007a40" if ok else "#c01f3e"
    bg   = "rgba(0,255,136,.03)" if ok else "rgba(255,45,85,.03)"
    vtxt = "◆ AUTHORIZED" if ok else "◇ DENIED"
    ttag = (f' <span style="font-size:.68rem;color:#ffb800;letter-spacing:2px">'
            f'[{tam.upper()}]</span>') if tam != "none" else ""
    meta = (f'CLAIM: ID={r.get("id_pred","?")} &nbsp;·&nbsp; WIN={r.get("w_pred","?")} '
            f'&nbsp;·&nbsp; SEQ#{r.get("seq","?")} &nbsp;·&nbsp; '
            f'PROOF:{str(r.get("proof",""))[:12]}…')
    html(f"""<div style="border:1px solid {bc};background:{bg};padding:12px;margin-top:10px;
                box-shadow:0 0 14px {vg}">
        <div style="font-family:'Share Tech Mono',monospace;font-size:1.3rem;font-weight:700;
                    letter-spacing:4px;color:{vc};text-shadow:0 0 8px {vg};
                    margin-bottom:8px;padding-bottom:7px;border-bottom:1px solid #1e2d3d">
            {vtxt}{ttag}</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:.66rem;color:#3d566e">{meta}</div>
    </div>""")

def render_gates(gates):
    labels = {"pn":"PN PILOT","p_valid":"P_VALID","id_match":"ID MATCH",
              "w_match":"WIN MATCH","pid":"PID MIN","pw":"PW MIN"}
    cells = ""
    for k, ok in gates.items():
        lbl = labels.get(k, k)
        bc  = "#007a40" if ok else "#c01f3e"
        bg  = "rgba(0,255,136,.05)" if ok else "rgba(255,45,85,.05)"
        tc  = "#00ff88" if ok else "#ff2d55"
        cells += (f'<div style="border:1px solid {bc};background:{bg};padding:5px 3px;'
                  f'text-align:center;font-family:\'Share Tech Mono\',monospace">'
                  f'<div style="font-size:.63rem;color:{tc};letter-spacing:1px">{lbl}</div>'
                  f'<div style="font-size:.95rem;font-weight:700;color:{tc}">{"✓" if ok else "✗"}</div></div>')
    html(f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin:9px 0">{cells}</div>')

def render_scores(r):
    scores = [
        ("P_VALID", r.get("p_valid",0),    "#00ff88" if r.get("p_valid",0)>=.8    else "#ff2d55"),
        ("PID",     r.get("pid",0),         "#00ff88" if r.get("pid",0)>=.7       else "#ffb800"),
        ("PW",      r.get("pw",0),          "#00ff88" if r.get("pw",0)>=.4        else "#ffb800"),
        ("PILOT ρ", r.get("pilot_corr",0),  "#00d4ff" if r.get("pilot_corr",0)>=.02 else "#ff2d55"),
        ("L2(MS)",  r.get("l2ms",-1),       "#c8d6e5"),
        ("LAT µs",  r.get("latency_us",0),  "#00d4ff"),
    ]
    cells = ""
    for lbl, val, col in scores:
        vstr = f"{val:.3f}" if isinstance(val,float) and val>=0 else ("—" if isinstance(val,float) else f"{val:.1f}")
        cells += (f'<div style="background:#0d1117;border:1px solid #1e2d3d;padding:7px 3px;text-align:center">'
                  f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.6rem;color:#3d566e;letter-spacing:1px">{lbl}</div>'
                  f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.95rem;font-weight:600;color:{col}">{vstr}</div></div>')
    html(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-top:7px">{cells}</div>')

def render_bulk(results):
    for name, r in results.items():
        ok  = r.get("ok", False)
        pv  = r.get("p_valid", 0)
        bc  = "#007a40" if ok else "#c01f3e"
        tc  = "#00ff88" if ok else "#ff2d55"
        html(f"""<div style="display:flex;justify-content:space-between;align-items:center;
                     font-family:'Share Tech Mono',monospace;font-size:.68rem;padding:4px 8px;
                     border:1px solid #1e2d3d;border-left:3px solid {bc};
                     background:#0d1117;margin-bottom:3px">
            <span style="color:{tc}">{"✓" if ok else "✗"}</span>
            <span style="color:#7a9ab5;flex:1;margin:0 8px">{name}</span>
            <span style="color:#ffb800">{pv:.3f}</span>
        </div>""")

def render_chain(data):
    if not data:
        html('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.72rem;color:#3d566e">No session active</div>')
        return
    ok  = data.get("valid", False)
    col = "#00ff88" if ok else "#ff2d55"
    dot = "◆" if ok else "◇"
    html(f"""
    <div style="font-family:'Share Tech Mono',monospace;font-size:.78rem;color:{col};
                margin-bottom:7px;letter-spacing:2px">{dot} {"CHAIN INTACT" if ok else "CHAIN BROKEN"}</div>
    <div style="background:#111820;border:1px solid #1e2d3d;padding:9px;font-family:'Share Tech Mono',
                monospace;font-size:.68rem;color:#00d4ff;line-height:1.8;word-break:break-all;margin-bottom:9px">
        <span style="color:#3d566e">events:</span> <span style="color:#00ff88">{data.get("events",0)}</span><br>
        <span style="color:#3d566e">state:</span> {data.get("state","")[:16]}…
    </div>""")
    for ev in reversed(data.get("recent_events",[])[-8:]):
        e  = ev.get("event",{})
        ok2= e.get("ok",False)
        bc = "#007a40" if ok2 else "#c01f3e"
        sc = "#00ff88" if ok2 else "#ff2d55"
        html(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:.66rem;
                     padding:6px 9px;background:#0d1117;border:1px solid #1e2d3d;
                     border-left:3px solid {bc};margin-bottom:3px">
            <div style="display:flex;justify-content:space-between;color:#7a9ab5;margin-bottom:2px">
                <span>#{ev.get("seq","?")}</span>
                <span style="color:{sc}">{"OK" if ok2 else "FAIL"}</span>
            </div>
            <div style="display:flex;gap:10px;color:#c8d6e5;flex-wrap:wrap">
                <span>ID:<b>{e.get("claimed_id","?")}</b></span>
                <span>W:<b>{e.get("expected_w","?")}</b></span>
                <span>pV:<b>{e.get("p_valid",0):.3f}</b></span>
                <span>{e.get("latency_us",0):.0f}µs</span>
            </div>
            <div style="color:#3d566e;margin-top:2px;word-break:break-all">proof:{str(ev.get("proof",""))[:22]}…</div>
        </div>""")

def render_enc(pkg, rt_ok):
    col  = "#00ff88" if rt_ok else "#ff2d55"
    stat = "◆ ROUNDTRIP OK — Encrypt→Decrypt verified" if rt_ok else "◇ ROUNDTRIP FAIL"
    back = "AES-256-GCM" if pim.HAS_CRYPTO else "HMAC-XOR fallback"
    html(f"""<div style="font-family:'Share Tech Mono',monospace;font-size:.7rem;
                margin-bottom:6px;color:{col}">{stat} · {back}</div>
    <div style="font-family:'Share Tech Mono',monospace;font-size:.63rem;word-break:break-all;
                padding:9px;background:#111820;border:1px solid #1e2d3d;color:#7a9ab5;
                line-height:1.7;max-height:155px;overflow-y:auto">
        <span style="color:#ffb800">SALT:</span> {pkg.get("salt","")} <br>
        <span style="color:#ffb800">NONCE:</span> {pkg.get("nonce","")} <br>
        <span style="color:#ffb800">TAG:</span> {pkg.get("tag","")} <br>
        <span style="color:#ffb800">CHAIN_HASH:</span> {pkg.get("chain_hash","")} <br>
        <span style="color:#ffb800">CIPHERTEXT:</span> {pkg.get("ciphertext","")[:64]}…
    </div>""")

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
model_loaded = st.session_state["params"] is not None
crypto_lbl   = "⬡ AES-256-GCM" if pim.HAS_CRYPTO else "⬡ HMAC-FALLBACK"
model_lbl    = "◆ MODEL ONLINE" if model_loaded else "◇ MODEL OFFLINE"
model_bc     = "#007a40" if model_loaded else "#3d566e"
model_tc     = "#00ff88" if model_loaded else "#3d566e"
model_bg     = "rgba(0,255,136,.06)" if model_loaded else "rgba(61,86,110,.1)"

html(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
            padding:10px 16px;border-bottom:1px solid #1e2d3d;
            background:linear-gradient(135deg,#080b0f 60%,#0a1018);
            margin-bottom:14px;position:relative;overflow:hidden">
    <div style="position:absolute;top:0;left:0;right:0;height:2px;
                background:linear-gradient(90deg,transparent,#007a40,#00d4ff,#007a40,transparent)"></div>
    <div style="display:flex;align-items:center;gap:12px">
        <div style="width:38px;height:38px;display:flex;align-items:center;justify-content:center;
                    border:1px solid #007a40;background:rgba(0,255,136,.05);
                    clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%)">
            <span style="color:#00ff88;font-size:1.1rem">⬡</span>
        </div>
        <div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:1.2rem;letter-spacing:4px;
                        color:#00ff88;text-shadow:0 0 12px rgba(0,255,136,.35)">WARL0K</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:.62rem;color:#7a9ab5;
                        letter-spacing:3px">PIM AUTH ENGINE · GRU+ATTN · CHAIN-PROOF</div>
        </div>
    </div>
    <div style="display:flex;gap:10px">
        <span style="font-family:'Share Tech Mono',monospace;font-size:.68rem;padding:3px 9px;
                     border:1px solid #0099cc;color:#00d4ff;background:rgba(0,212,255,.05)">{crypto_lbl}</span>
        <span style="font-family:'Share Tech Mono',monospace;font-size:.68rem;padding:3px 9px;
                     border:1px solid {model_bc};color:{model_tc};background:{model_bg}">{model_lbl}</span>
    </div>
</div>""")

# ════════════════════════════════════════════════════════════════════
# THREE COLUMNS
# ════════════════════════════════════════════════════════════════════
c1, c2, c3 = st.columns(3, gap="small")

# ── COLUMN 1 — NEURAL CORE ───────────────────────────────────────────
with c1:
    panel_header("#00ff88", "Neural Core · Train",
                 live=st.session_state["training"])

    slabel("Model Config")
    g1, g2 = st.columns(2)
    with g1:
        cfgrow("N IDENTITIES",  CFG["N_IDENTITIES"])
        cfgrow("HIDDEN DIM",    CFG["HIDDEN_DIM"])
        cfgrow("MS DIM",        CFG["MS_DIM"])
        cfgrow("EPOCHS P1",     CFG["EPOCHS_PHASE1"])
    with g2:
        cfgrow("N WIN/ID",      CFG["N_WINDOWS_PER_ID"])
        cfgrow("ATTN DIM",      CFG["ATTN_DIM"])
        cfgrow("SEQ LEN",       CFG["SEQ_LEN"])
        cfgrow("EPOCHS P2",     CFG["EPOCHS_PHASE2"])

    st.markdown('<hr style="border-color:#1e2d3d;margin:10px 0">', unsafe_allow_html=True)

    slabel("Training Progress")
    ph = st.session_state["train_phase"]
    mrow("Phase",
         f"Phase {ph}" if ph else "—",
         "#00d4ff" if ph == 2 else "#00ff88")
    mrow("Epoch",
         f"{st.session_state['train_epoch']}/{TOTAL_EP}" if st.session_state["train_epoch"] else "—")
    mrow("Avg Loss",
         f"{st.session_state['train_loss']:.4f}" if st.session_state["train_loss"] else "—",
         "#ffb800")
    mrow("Train Time",
         f"{st.session_state['train_s']:.2f}s" if st.session_state["train_s"] else "—")

    st.progress(float(st.session_state["progress"]))

    slabel("Loss Curve")
    loss_svg(st.session_state["loss_p1"], st.session_state["loss_p2"])

    slabel("Console")
    render_log(st.session_state["log_lines"])

    b1, b2 = st.columns(2)
    with b1:
        train_btn = st.button(
            "⬡ TRAIN" if not st.session_state["training"] else "⟳ TRAINING…",
            disabled=st.session_state["training"],
            key="btn_train",
        )
    with b2:
        load_btn = st.button("↓ LOAD", key="btn_load")

    # ── Train action ──────────────────────────────────────────────
    if train_btn and not st.session_state["training"]:
        st.session_state["training"]    = True
        st.session_state["train_done"]  = False
        st.session_state["train_error"] = None
        st.session_state["loss_p1"]     = []
        st.session_state["loss_p2"]     = []
        st.session_state["train_phase"] = 0
        st.session_state["train_epoch"] = 0
        st.session_state["train_loss"]  = 0.0
        st.session_state["progress"]    = 0.0
        append_log(f"{time.strftime('%H:%M:%S')} Starting training…")

        def _run_train():
            try:
                ds = pim.build_dataset()
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} Dataset: {ds['X'].shape[0]} samples")
                p = pim.init_params()
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} Model: {p.param_bytes()//1024:.1f} KB")
                t0 = time.time()

                def _cb(info):
                    ph_  = info.get("phase", 1)
                    ep_  = info.get("epoch", 0)
                    loss_= info.get("loss",  0.0)
                    base = CFG["EPOCHS_PHASE1"] if ph_ == 2 else 0
                    st.session_state["train_phase"] = ph_
                    st.session_state["train_epoch"] = ep_ + base
                    st.session_state["train_loss"]  = loss_
                    st.session_state["progress"]    = min(0.99, (ep_ + base) / TOTAL_EP)
                    if ph_ == 1:
                        st.session_state["loss_p1"].append(loss_)
                    else:
                        st.session_state["loss_p2"].append(loss_)

                pim.train_phase1(p, ds, cb=_cb)
                pim.train_phase2(p, ds, cb=_cb)

                train_s = time.time() - t0
                st.session_state["train_s"]   = train_s
                st.session_state["progress"]  = 1.0
                st.session_state["params"]    = p
                st.session_state["session"]   = pim.PIMSession(p)
                pim.save_model(MODEL_PATH, p, {"train_s": train_s})
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} Training done in {train_s:.2f}s — saved")
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} ✓ Training complete in {train_s:.2f}s")
                st.session_state["train_done"] = True
            except Exception as ex:
                st.session_state["train_error"] = str(ex)
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} ✗ Error: {ex}")
            finally:
                st.session_state["training"] = False

        threading.Thread(target=_run_train, daemon=True).start()
        st.rerun()

    # ── Load action ───────────────────────────────────────────────
    if load_btn:
        if os.path.exists(MODEL_PATH):
            try:
                _p, _meta = pim.load_model(MODEL_PATH)
                st.session_state["params"]  = _p
                st.session_state["session"] = pim.PIMSession(_p)
                append_log(f"{time.strftime('%H:%M:%S')} ✓ Model loaded — {_p.param_bytes()//1024:.1f} KB")
                st.rerun()
            except Exception as ex:
                append_log(f"{time.strftime('%H:%M:%S')} ✗ Load error: {ex}")
        else:
            append_log(f"{time.strftime('%H:%M:%S')} ✗ No model.npz — train first")

    if st.session_state["training"]:
        time.sleep(0.5)
        st.rerun()

# ── COLUMN 2 — CHAIN VERIFY ──────────────────────────────────────────
with c2:
    panel_header("#ffb800", "Chain Verify · Auth")

    slabel("Verification Input")
    vi1, vi2 = st.columns(2)
    with vi1:
        claimed_id = st.number_input("Claimed ID", 0, CFG["N_IDENTITIES"]-1, 0, key="v_cid")
    with vi2:
        expected_w = st.number_input("Expected Window", 0, CFG["N_WINDOWS_PER_ID"]-1, 3, key="v_ew")

    TAMPER_MAP = {
        "None — Legit chain":              "none",
        "Shuffle (replay attack)":          "shuffle",
        "Truncate (dropped steps)":         "truncate",
        "Wrong window (counter drift)":     "wrong_win",
        "Wrong identity (impersonation)":   "wrong_id",
        "Out-of-range window (hard break)": "oob",
    }
    tamper_lbl = st.selectbox("Tamper Mode", list(TAMPER_MAP.keys()), key="v_tamper")
    tamper     = TAMPER_MAP[tamper_lbl]

    verify_btn = st.button("⬡ VERIFY CHAIN", key="btn_verify",
                           disabled=st.session_state["session"] is None)

    if verify_btn and st.session_state["session"] is not None:
        NI, NW, T = CFG["N_IDENTITIES"], CFG["N_WINDOWS_PER_ID"], CFG["SEQ_LEN"]
        cid = max(0, min(int(claimed_id), NI-1))
        ew  = int(expected_w)
        ms  = pim.MS_ALL[cid]
        toks, meas = pim.generate_os_chain(ms, cid*NW+ew)

        if tamper == "shuffle":
            idx  = np.array(pim.XorShift32(0xABCD).shuffle(list(range(T))))
            toks = toks[idx]; meas = meas[idx]
        elif tamper == "truncate":
            t2 = np.zeros(T, np.int32);   m2 = np.zeros(T, np.float32)
            t2[:T//2] = toks[:T//2];      m2[:T//2] = meas[:T//2]
            toks = t2; meas = m2
        elif tamper == "wrong_win":
            toks, meas = pim.generate_os_chain(ms, cid*NW+(ew+7)%NW)
        elif tamper == "wrong_id":
            oid = (cid+1) % NI
            toks, meas = pim.generate_os_chain(pim.MS_ALL[oid], oid*NW+13)
        elif tamper == "oob":
            ew = 9999

        res = st.session_state["session"].verify(cid, ew, toks, meas, ms)
        res["tamper"] = tamper
        res["gates"]  = {k: bool(v) for k, v in res.get("gates", {}).items()}
        st.session_state["last_result"] = res
        # update chain data
        sess = st.session_state["session"]
        st.session_state["chain_data"] = {
            **sess.chain_status(),
            "recent_events": sess.chain.events[-10:],
        }

    if st.session_state["last_result"]:
        r = st.session_state["last_result"]
        render_verdict(r)
        render_gates(r.get("gates", {}))
        render_scores(r)

    st.markdown('<hr style="border-color:#1e2d3d;margin:10px 0">', unsafe_allow_html=True)

    slabel("Bulk Demo — All Tamper Cases")
    bulk_btn = st.button("▶ RUN ALL CASES", key="btn_bulk",
                         disabled=st.session_state["session"] is None)

    if bulk_btn and st.session_state["session"] is not None:
        NI, NW, T = CFG["N_IDENTITIES"], CFG["N_WINDOWS_PER_ID"], CFG["SEQ_LEN"]
        ms0 = pim.MS_ALL[0]
        tk0, me0 = pim.generate_os_chain(ms0, 0*NW+3)
        bulk = {}
        for name, tam in [("none","none"),("shuffle","shuffle"),("truncate","truncate"),
                          ("wrong_win","wrong_win"),("wrong_id","wrong_id"),("oob","oob")]:
            tk, me, ew_ = tk0.copy(), me0.copy(), 3
            if tam == "shuffle":
                idx = np.array(pim.XorShift32(0xABCD).shuffle(list(range(T))))
                tk = tk[idx]; me = me[idx]
            elif tam == "truncate":
                t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
                t2[:T//2]=tk[:T//2]; m2[:T//2]=me[:T//2]; tk=t2; me=m2
            elif tam == "wrong_win":
                tk, me = pim.generate_os_chain(ms0, 0*NW+(3+7)%NW)
            elif tam == "wrong_id":
                oid = 1 % NI
                tk, me = pim.generate_os_chain(pim.MS_ALL[oid], oid*NW+13)
            elif tam == "oob":
                ew_ = 9999
            r2 = st.session_state["session"].verify(0, ew_, tk, me, ms0)
            r2["gates"] = {k: bool(v) for k, v in r2.get("gates",{}).items()}
            bulk[name] = r2
        st.session_state["bulk_results"] = bulk
        sess = st.session_state["session"]
        st.session_state["chain_data"] = {
            **sess.chain_status(),
            "recent_events": sess.chain.events[-10:],
        }

    if st.session_state["bulk_results"]:
        render_bulk(st.session_state["bulk_results"])

    st.markdown('<hr style="border-color:#1e2d3d;margin:10px 0">', unsafe_allow_html=True)

    slabel("Inference Benchmark")
    bench_btn = st.button("⚡ BENCHMARK 50×", key="btn_bench",
                          disabled=st.session_state["session"] is None)

    if bench_btn and st.session_state["session"] is not None:
        p   = st.session_state["params"]
        ms  = pim.MS_ALL[0]
        tok, mea = pim.generate_os_chain(ms, 0)
        times = []
        for _ in range(50):
            t0_ = time.perf_counter()
            pim.verify_chain(p, tok, mea, 0, 0)
            times.append((time.perf_counter()-t0_)*1e6)
        arr = np.array(times)
        st.session_state["bench_result"] = {
            "mean": arr.mean(), "min": arr.min(),
            "max": arr.max(), "p95": np.percentile(arr, 95),
        }

    if st.session_state["bench_result"]:
        b = st.session_state["bench_result"]
        be1, be2, be3 = st.columns(3)
        with be1: mrow("MEAN", f"{b['mean']:.1f} µs", "#00d4ff")
        with be2: mrow("MIN",  f"{b['min']:.1f} µs",  "#00ff88")
        with be3: mrow("P95",  f"{b['p95']:.1f} µs",  "#ffb800")

# ── COLUMN 3 — CHAIN PROOF ────────────────────────────────────────────
with c3:
    panel_header("#00d4ff", "Chain Proof · Crypto")

    slabel("Chain State")
    render_chain(st.session_state["chain_data"])

    if st.button("↻ REFRESH CHAIN", key="btn_refresh"):
        if st.session_state["session"] is not None:
            sess = st.session_state["session"]
            st.session_state["chain_data"] = {
                **sess.chain_status(),
                "recent_events": sess.chain.events[-10:],
            }
        st.rerun()

    st.markdown('<hr style="border-color:#1e2d3d;margin:10px 0">', unsafe_allow_html=True)
    slabel("Encryption Demo · AES-256-GCM")

    ec1, ec2 = st.columns(2)
    with ec1:
        enc_id = st.number_input("ID",     0, CFG["N_IDENTITIES"]-1,  0, key="enc_id")
    with ec2:
        enc_w  = st.number_input("Window", 0, CFG["N_WINDOWS_PER_ID"]-1, 0, key="enc_w")

    enc_btn = st.button("⬡ ENCRYPT TOKENS", key="btn_enc",
                        disabled=st.session_state["session"] is None)

    if enc_btn and st.session_state["session"] is not None:
        cid_ = int(enc_id); w_ = int(enc_w)
        ms_  = pim.MS_ALL[cid_]
        tok_, mea_ = pim.generate_os_chain(ms_, cid_*CFG["N_WINDOWS_PER_ID"]+w_)
        pkg_ = st.session_state["session"].encrypt_tokens(tok_, mea_)
        try:
            t3, m3 = st.session_state["session"].decrypt_tokens(pkg_)
            rt_ = bool(np.array_equal(t3, tok_) and np.allclose(m3, mea_))
        except Exception:
            rt_ = False
        st.session_state["enc_result"] = {"pkg": pkg_, "rt_ok": rt_}

    if st.session_state["enc_result"]:
        render_enc(
            st.session_state["enc_result"]["pkg"],
            st.session_state["enc_result"]["rt_ok"],
        )
