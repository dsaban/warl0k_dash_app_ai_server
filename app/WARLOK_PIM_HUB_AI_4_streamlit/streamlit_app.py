"""
WARL0K PIM Engine v2 — Streamlit Wrapper
Exact replica of the 4-panel dark terminal UI:
  Panel 1: Neural Core · Train
  Panel 2: Chain Verify · Auth
  Panel 3: Peer Hub · Dual Handshake
  Panel 4: Chain Proof · Crypto
"""

import streamlit as st
import numpy as np
import time
import threading
import os
import sys
import json

# ── 1. set_page_config — MUST be the VERY FIRST st.* call ──────────
st.set_page_config(
    page_title="WARL0K · PIM Auth Engine v2",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── 2. Import pim_core AFTER set_page_config ────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
import pim_core as pim

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.npz")
CFG        = pim.CFG
TOTAL_EP   = CFG["EPOCHS_PHASE1"] + CFG["EPOCHS_PHASE2"]
SHARED_KEY = b"\xca\xfe" * 16  # 32-byte shared session key for demo

# ── 3. Session state — ALL keys before any other st.* call ─────────
_KEYS = {
    "params":        None,
    "session":       None,
    "training":      False,
    "train_done":    False,
    "loss_p1":       [],
    "loss_p2":       [],
    "train_phase":   0,
    "train_epoch":   0,
    "train_loss":    0.0,
    "train_s":       0.0,
    "progress":      0.0,
    "log_lines":     [],
    # verify panel
    "last_result":   None,
    "bulk_results":  None,
    "bench_result":  None,
    # peer hub
    "peers":         {},   # peer_id -> PeerSession
    "handshake_log": [],
    "recon_result":  None,
    "ws_events":     [],
    # chain / crypto
    "chain_data":    None,
    "enc_result":    None,
}
for _k, _v in _KEYS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v if not isinstance(_v, list) else []
        if isinstance(_v, dict) and _k not in st.session_state:
            st.session_state[_k] = {}

# Fix dict keys (list comprehension above doesn't handle dicts)
if not isinstance(st.session_state["peers"], dict):
    st.session_state["peers"] = {}

# ── Auto-load model ──────────────────────────────────────────────────
if st.session_state["params"] is None and os.path.exists(MODEL_PATH):
    try:
        _p, _ = pim.load_model(MODEL_PATH)
        st.session_state["params"]  = _p
        st.session_state["session"] = pim.PIMSession(_p, SHARED_KEY)
        st.session_state["log_lines"].append(
            f"{time.strftime('%H:%M:%S')} ✓ Model auto-loaded — {_p.param_bytes()//1024:.1f} KB")
    except Exception as _e:
        st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} ✗ Auto-load: {_e}")

# ════════════════════════════════════════════════════════════════════
# CSS — exact dark terminal replica
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html,body,[class*="css"]{background:#080b0f!important;color:#c8d6e5!important;font-family:'Rajdhani',sans-serif!important}
.stApp{background:#080b0f!important}
.main .block-container{padding-top:.4rem!important;padding-bottom:.5rem!important;max-width:100%!important}
#MainMenu,footer,header{visibility:hidden!important}
.stDeployButton,[data-testid="stToolbar"]{display:none!important}
::-webkit-scrollbar{width:3px;height:3px}
::-webkit-scrollbar-track{background:#080b0f}
::-webkit-scrollbar-thumb{background:#1e2d3d}
h1,h2,h3{font-family:'Share Tech Mono',monospace!important}

/* Buttons */
.stButton>button{background:transparent!important;border:1px solid #007a40!important;
  color:#00ff88!important;font-family:'Share Tech Mono',monospace!important;
  font-size:.76rem!important;letter-spacing:2px!important;text-transform:uppercase!important;
  width:100%!important;transition:all .15s!important}
.stButton>button:hover:not(:disabled){background:rgba(0,255,136,.1)!important;
  box-shadow:0 0 10px rgba(0,255,136,.25)!important}
.stButton>button:disabled{opacity:.35!important}

/* Inputs */
.stNumberInput input,.stTextInput input{background:#111820!important;border:1px solid #1e2d3d!important;
  color:#c8d6e5!important;font-family:'Share Tech Mono',monospace!important}
[data-baseweb="select"]>div{background:#111820!important;border:1px solid #1e2d3d!important;
  color:#c8d6e5!important;font-family:'Share Tech Mono',monospace!important}
[data-baseweb="select"] *{background:#111820!important;color:#c8d6e5!important;
  font-family:'Share Tech Mono',monospace!important}
[data-baseweb="popover"] *{background:#111820!important;color:#c8d6e5!important;
  font-family:'Share Tech Mono',monospace!important}

/* Progress */
.stProgress>div>div>div>div{background:linear-gradient(90deg,#007a40,#00ff88)!important}
.stProgress>div>div>div{background:#111820!important;border:1px solid #1e2d3d!important}

/* Labels */
label{color:#3d566e!important;font-family:'Share Tech Mono',monospace!important;
  font-size:.66rem!important;letter-spacing:2px!important;text-transform:uppercase!important}

/* Columns */
[data-testid="column"]{border-right:1px solid #1e2d3d;
  padding-right:.6rem!important;padding-left:.6rem!important}
[data-testid="column"]:last-child{border-right:none!important}

/* Sidebar */
[data-testid="stSidebar"]{background:#0d1117!important}

@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# HTML HELPERS
# ════════════════════════════════════════════════════════════════════
def H(s): st.markdown(s, unsafe_allow_html=True)

def panel_hdr(dot_col, title, live=False):
    dot = (f'<span style="width:7px;height:7px;border-radius:50%;background:{dot_col};'
           f'display:inline-block;margin-left:auto;animation:pulse 1.4s ease infinite"></span>'
           if live else "")
    H(f'<div style="display:flex;align-items:center;gap:9px;padding:8px 0;'
      f'border-bottom:1px solid #1e2d3d;margin-bottom:11px">'
      f'<div style="width:8px;height:8px;border-radius:50%;background:{dot_col}"></div>'
      f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.74rem;'
      f'letter-spacing:2px;color:#7a9ab5;text-transform:uppercase">{title}</span>{dot}</div>')

def slabel(t, mt=8):
    H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.6rem;'
      f'color:#3d566e;letter-spacing:2px;text-transform:uppercase;margin:{mt}px 0 4px">{t}</div>')

def mrow(k, v, c="#c8d6e5"):
    H(f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px">'
      f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.67rem;color:#3d566e">{k}</span>'
      f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.85rem;'
      f'font-weight:600;color:{c}">{v}</span></div>')

def cfgrow(k, v):
    H(f'<div style="display:flex;justify-content:space-between;padding:3px 7px;'
      f'background:#111820;border:1px solid #1e2d3d;margin-bottom:3px;'
      f'font-family:\'Share Tech Mono\',monospace;font-size:.68rem">'
      f'<span style="color:#3d566e">{k}</span><span style="color:#00d4ff">{v}</span></div>')

def sep(): H('<hr style="border:none;border-top:1px solid #1e2d3d;margin:9px 0">')

def render_log():
    lines = st.session_state["log_lines"]
    rows = ""
    for l in lines[-14:]:
        c = ("#00c96a" if any(x in l for x in ["✓","done","complete","loaded","saved","ACCEPTED","OK"])
             else "#ff2d55" if any(x in l for x in ["✗","Error","error","failed","REJECTED"])
             else "#ffb800" if any(x in l for x in ["Starting","Dataset","Model:","Reconstructing","Handshake"])
             else "#7a9ab5")
        rows += f'<div style="color:{c};line-height:1.6">{l}</div>'
    H(f'<div style="font-family:/Share Tech Mono/,monospace;font-size:.66rem;'
      f'background:#111820;border:1px solid #1e2d3d;padding:7px 9px;'
      f'max-height:175px;overflow-y:auto;margin-bottom:7px">'
      f'{rows or "<span style=/color:#3d566e/>WARL0K PIM v2 ready.</span>"}</div>')

def loss_svg():
    p1 = st.session_state["loss_p1"]
    p2 = st.session_state["loss_p2"]
    if not p1 and not p2:
        H('<div style="height:72px;background:#111820;border:1px solid #1e2d3d;'
          'display:flex;align-items:center;justify-content:center;'
          'font-family:\'Share Tech Mono\',monospace;font-size:.66rem;color:#3d566e">'
          'LOSS CURVE</div>')
        return
    mx = max((p1+p2) or [1]); W, Hh = 440, 72
    def seg(pts, x0, x1, col):
        if len(pts) < 2: return ""
        c = " ".join(f"{x0+(x1-x0)*i/(len(pts)-1):.1f},{Hh-(v/mx)*Hh*.82-4:.1f}"
                     for i, v in enumerate(pts))
        return f'<polyline points="{c}" fill="none" stroke="{col}" stroke-width="1.5"/>'
    grid = "".join(f'<line x1="0" y1="{Hh-f*Hh*.82-4:.0f}" x2="{W}" '
                   f'y2="{Hh-f*Hh*.82-4:.0f}" stroke="#1e2d3d" stroke-width=".5"/>'
                   for f in (.25,.5,.75))
    H(f'<div style="background:#111820;border:1px solid #1e2d3d;margin-bottom:8px">'
      f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:{Hh}px;display:block">'
      f'{grid}{seg(p1,0,W*.5,"#00c96a")}{seg(p2,W*.5,W,"#00d4ff")}'
      f'<text x="4" y="13" font-family="Share Tech Mono" font-size="10" fill="#007a40">P1</text>'
      f'<text x="{W//2+4}" y="13" font-family="Share Tech Mono" font-size="10" fill="#0099cc">P2</text>'
      f'</svg></div>')

def render_verdict(r):
    ok  = r.get("ok", False)
    tam = r.get("tamper", "none")
    vc  = "#00ff88" if ok else "#ff2d55"
    vg  = "rgba(0,255,136,.22)" if ok else "rgba(255,45,85,.22)"
    bc  = "#007a40" if ok else "#c01f3e"
    bg  = "rgba(0,255,136,.03)" if ok else "rgba(255,45,85,.03)"
    tag = (f' <span style="font-size:.66rem;color:#ffb800">[{tam.upper()}]</span>'
           if tam != "none" else "")
    ctr = r.get("counter","?"); dm = r.get("delta_ms",0)
    meta = (f'ID={r.get("id_pred","?")} · W={r.get("w_pred","?")} · '
            f'CTR:{ctr} Δ{dm:.0f}ms · SEQ#{r.get("seq","?")} · '
            f'PROOF:{str(r.get("proof",""))[:12]}…')
    H(f'<div style="border:1px solid {bc};background:{bg};padding:11px;margin-top:9px;'
      f'box-shadow:0 0 12px {vg}">'
      f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:1.2rem;font-weight:700;'
      f'letter-spacing:4px;color:{vc};margin-bottom:7px;padding-bottom:6px;'
      f'border-bottom:1px solid #1e2d3d">{"◆ AUTHORIZED" if ok else "◇ DENIED"}{tag}</div>'
      f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.63rem;color:#3d566e">{meta}</div>'
      f'</div>')

def render_gates(gates):
    LBL = {"pn":"PN PILOT","p_valid":"P_VALID","id_match":"ID MATCH",
           "w_match":"WIN MATCH","pid":"PID MIN","pw":"PW MIN"}
    cells = ""
    for k, ok in gates.items():
        bc = "#007a40" if ok else "#c01f3e"
        bg = "rgba(0,255,136,.05)" if ok else "rgba(255,45,85,.05)"
        tc = "#00ff88" if ok else "#ff2d55"
        cells += (f'<div style="border:1px solid {bc};background:{bg};padding:5px 2px;'
                  f'text-align:center;font-family:\'Share Tech Mono\',monospace">'
                  f'<div style="font-size:.6rem;color:{tc};letter-spacing:.5px">{LBL.get(k,k)}</div>'
                  f'<div style="font-size:.92rem;font-weight:700;color:{tc}">{"✓" if ok else "✗"}</div>'
                  f'</div>')
    H(f'<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:3px;margin:8px 0">{cells}</div>')

def render_scores(r):
    sc = [
        ("P_VALID", r.get("p_valid",0),   "#00ff88" if r.get("p_valid",0)  >=.8  else "#ff2d55"),
        ("PID",     r.get("pid",0),        "#00ff88" if r.get("pid",0)      >=.7  else "#ffb800"),
        ("PW",      r.get("pw",0),         "#00ff88" if r.get("pw",0)       >=.4  else "#ffb800"),
        ("PILOT ρ", r.get("pilot_corr",0), "#00d4ff" if r.get("pilot_corr",0)>=.02 else "#ff2d55"),
        ("L2(MS)",  r.get("l2ms",-1),      "#c8d6e5"),
        ("LAT µs",  r.get("latency_us",0), "#00d4ff"),
    ]
    cells = ""
    for lbl, val, col in sc:
        v = f"{val:.3f}" if isinstance(val,float) and val>=0 else ("—" if isinstance(val,float) else f"{val:.1f}")
        cells += (f'<div style="background:#0d1117;border:1px solid #1e2d3d;padding:6px 2px;text-align:center">'
                  f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.58rem;color:#3d566e">{lbl}</div>'
                  f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.9rem;font-weight:600;color:{col}">{v}</div>'
                  f'</div>')
    H(f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:3px;margin-top:6px">{cells}</div>')

def render_bulk():
    results = st.session_state["bulk_results"]
    if not results: return
    for name, r in results.items():
        ok = r.get("ok", False)
        pv = r.get("p_valid", 0)
        bc = "#007a40" if ok else "#c01f3e"
        tc = "#00ff88" if ok else "#ff2d55"
        H(f'<div style="display:flex;justify-content:space-between;align-items:center;'
          f'font-family:\'Share Tech Mono\',monospace;font-size:.67rem;padding:4px 7px;'
          f'border:1px solid #1e2d3d;border-left:3px solid {bc};background:#0d1117;margin-bottom:3px">'
          f'<span style="color:{tc}">{"✓" if ok else "✗"}</span>'
          f'<span style="color:#7a9ab5;flex:1;margin:0 8px">{name}</span>'
          f'<span style="color:#ffb800">{pv:.3f}</span></div>')

def render_peer_card(peer_id):
    peers = st.session_state["peers"]
    if peer_id not in peers: return
    peer = peers[peer_id]
    hs_ok = peer.handshake_ok
    bc    = "#007a40" if hs_ok else "#1e2d3d"
    vc    = "#00ff88" if hs_ok else "#ffb800"
    vstxt = "◆ HANDSHAKE OK" if hs_ok else "◇ PENDING"
    recon = peer.ms_recon

    # 4-step pipeline
    s_recon  = "done" if peer.ms_recon is not None else "active"
    s_anchor = "done" if peer.anchor is not None else ("active" if peer.ms_recon else "locked")
    s_verify = "done" if peer.handshake_ok else ("active" if peer.anchor else "locked")
    s_done   = "done" if peer.handshake_ok else "locked"

    def step_cell(label, state):
        c = "#007a40" if state=="done" else ("#ffb800" if state=="active" else "#3d566e")
        bg= "rgba(0,255,136,.06)" if state=="done" else ("rgba(255,184,0,.06)" if state=="active" else "transparent")
        return (f'<div style="border:1px solid {c};background:{bg};padding:4px 2px;'
                f'text-align:center;font-family:\'Share Tech Mono\',monospace;font-size:.6rem;color:{c}">'
                f'{label}</div>')

    steps = (step_cell("1.RECON",s_recon) + step_cell("2.ANCHOR",s_anchor) +
             step_cell("3.VERIFY",s_verify) + step_cell("4.DONE",s_done))

    recon_html = ""
    if recon:
        l2s = recon.get("per_window_l2", [])
        mx  = max(l2s+[0.01])
        TOL = CFG["MS_RECON_TOL"]
        bars = ""
        for i, v in enumerate(l2s[:12]):
            pct = min(100, v/mx*100)
            bc2 = "#007a40" if v <= TOL else "#c01f3e"
            bars += (f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:2px">'
                     f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.58rem;'
                     f'color:#3d566e;width:24px">W{i}</span>'
                     f'<div style="flex:1;height:4px;background:#111820;border:1px solid #1e2d3d">'
                     f'<div style="height:100%;background:{bc2};width:{pct:.0f}%"></div></div>'
                     f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.58rem;'
                     f'color:#7a9ab5;width:38px;text-align:right">{v:.3f}</span></div>')
        if len(l2s) > 12:
            bars += f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.58rem;color:#3d566e">…+{len(l2s)-12} more</div>'
        acc_c = "#00ff88" if recon.get("accepted") else "#ff2d55"
        recon_html = (
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.62rem;'
            f'color:{acc_c};margin:5px 0 3px">MS consensus L2: {recon.get("consensus_l2","?"):.5f} — '
            f'{"ACCEPTED" if recon.get("accepted") else "REJECTED"} '
            f'({recon.get("windows_ok","?")}/{recon.get("windows_used","?")} windows OK)</div>'
            f'<div style="margin-bottom:4px">{bars}</div>'
        )

    H(f'<div style="border:1px solid {bc};background:#0d1117;padding:9px;margin-bottom:6px">'
      f'<div style="display:flex;justify-content:space-between;align-items:center;'
      f'font-family:\'Share Tech Mono\',monospace;font-size:.72rem;margin-bottom:6px">'
      f'<span style="color:#b44fff">⬡ {peer_id}</span>'
      f'<span style="color:#3d566e">ID:{peer.identity_id}</span>'
      f'<span style="color:{vc}">{vstxt}</span></div>'
      f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:3px;margin-bottom:6px">{steps}</div>'
      f'{recon_html}</div>')

def render_chain_panel():
    data = st.session_state["chain_data"]
    if not data:
        H('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
          'color:#3d566e">No session active</div>')
        return
    ok  = data.get("valid", False)
    col = "#00ff88" if ok else "#ff2d55"
    H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.76rem;color:{col};'
      f'margin-bottom:6px;letter-spacing:2px">{"◆ CHAIN INTACT" if ok else "◇ CHAIN BROKEN"}</div>'
      f'<div style="background:#111820;border:1px solid #1e2d3d;padding:8px;'
      f'font-family:\'Share Tech Mono\',monospace;font-size:.66rem;color:#00d4ff;'
      f'line-height:1.8;word-break:break-all;margin-bottom:8px">'
      f'<span style="color:#3d566e">events:</span> <span style="color:#00ff88">{data.get("events",0)}</span><br>'
      f'<span style="color:#3d566e">counter:</span> <span style="color:#ffb800">{data.get("counter",0)}</span><br>'
      f'<span style="color:#3d566e">reason:</span> <span style="color:#7a9ab5">{data.get("reason","ok")}</span><br>'
      f'<span style="color:#3d566e">state:</span> {data.get("state","")[:20]}…</div>')

    for ev in reversed(data.get("recent_events", [])[-8:]):
        e   = ev.get("event", {})
        et  = e.get("event","VERIFY")
        ok2 = e.get("ok", None)
        is_anchor = "ANCHOR" in et or "RECONSTRUCT" in et
        bc  = "#7a2db5" if is_anchor else ("#007a40" if ok2 else "#c01f3e")
        sc  = "#b44fff" if is_anchor else ("#00ff88" if ok2 else "#ff2d55")
        H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.64rem;'
          f'padding:5px 8px;background:#0d1117;border:1px solid #1e2d3d;'
          f'border-left:3px solid {bc};margin-bottom:3px">'
          f'<div style="display:flex;justify-content:space-between;color:#7a9ab5;margin-bottom:2px">'
          f'<span>#{ev.get("seq","?")} [{et}]</span>'
          f'<span style="color:{sc}">CTR:{e.get("counter","?")} Δ{e.get("delta_ms",0):.0f}ms</span></div>'
          f'<div style="display:flex;gap:8px;color:#c8d6e5;flex-wrap:wrap">'
          f'{"<span>ID:<b>"+str(e["claimed_id"])+"</b></span>" if "claimed_id" in e else ""}'
          f'{"<span>W:<b>"+str(e["expected_w"])+"</b></span>" if "expected_w" in e else ""}'
          f'{"<span>pV:<b>"+str(e.get("p_valid","?"))+"</b></span>" if "p_valid" in e else ""}'
          f'{"<span>L2:<b>"+str(e.get("consensus_l2","?"))+"</b></span>" if "consensus_l2" in e else ""}'
          f'{"<span>"+str(e.get("latency_us",0))+"µs</span>" if "latency_us" in e else ""}'
          f'</div>'
          f'<div style="color:#3d566e;margin-top:2px;word-break:break-all">proof:{ev.get("proof","")[:22]}…</div>'
          f'</div>')

def render_enc():
    d = st.session_state["enc_result"]
    if not d: return
    pkg   = d["pkg"]; rt_ok = d["rt_ok"]
    col   = "#00ff88" if rt_ok else "#ff2d55"
    back  = "AES-256-GCM" if pim.HAS_CRYPTO else "HMAC-XOR fallback"
    H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.68rem;'
      f'margin-bottom:5px;color:{col}">{"◆ ROUNDTRIP OK" if rt_ok else "◇ FAIL"} · {back}</div>'
      f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.61rem;word-break:break-all;'
      f'padding:8px;background:#111820;border:1px solid #1e2d3d;color:#7a9ab5;'
      f'line-height:1.7;max-height:145px;overflow-y:auto">'
      f'<span style="color:#ffb800">SALT:</span> {pkg.get("salt","")} <br>'
      f'<span style="color:#ffb800">NONCE:</span> {pkg.get("nonce","")} <br>'
      f'<span style="color:#ffb800">TAG:</span> {pkg.get("tag","")} <br>'
      f'<span style="color:#ffb800">CHAIN_HASH:</span> {pkg.get("chain_hash","")} <br>'
      f'<span style="color:#ffb800">CIPHERTEXT:</span> {(pkg.get("ct") or pkg.get("ciphertext",""))[:64]}…</div>')

def render_ws_feed():
    evts = st.session_state["ws_events"]
    if not evts:
        H('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.65rem;'
          'color:#3d566e">No events yet — run handshake</div>')
        return
    rows = ""
    for ev in reversed(evts[-20:]):
        et  = ev.get("event_type","?")
        pid = ev.get("peer_id","?")
        ctr = ev.get("counter","?")
        dm  = ev.get("delta_ms",0)
        c   = "#b44fff" if "ANCHOR" in et or "RECONSTRUCT" in et else \
              ("#00ff88" if ev.get("ok") else "#ff2d55" if ev.get("ok") is False else "#00d4ff")
        rows += f'<div style="color:{c};line-height:1.6">#{ev.get("seq","?")} [{pid}] {et} ctr:{ctr} Δ{dm:.0f}ms</div>'
    H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.64rem;'
      f'background:#111820;border:1px solid #1e2d3d;padding:8px;'
      f'max-height:155px;overflow-y:auto">{rows}</div>')

def refresh_chain_from_session():
    session = st.session_state["session"]
    if session is None: return
    st.session_state["chain_data"] = session.chain_status()

# ════════════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════════════
_loaded   = st.session_state["params"] is not None
_cbc      = "#0099cc" if pim.HAS_CRYPTO else "#3d566e"
_ctc      = "#00d4ff" if pim.HAS_CRYPTO else "#3d566e"
_mbc      = "#007a40" if _loaded else "#3d566e"
_mtc      = "#00ff88" if _loaded else "#3d566e"
_mbg      = "rgba(0,255,136,.06)" if _loaded else "rgba(61,86,110,.1)"
_clbl     = "⬡ AES-256-GCM" if pim.HAS_CRYPTO else "⬡ HMAC-FALLBACK"
_mlbl     = "◆ MODEL ONLINE" if _loaded else "◇ MODEL OFFLINE"

H(f'<div style="display:flex;align-items:center;justify-content:space-between;'
  f'padding:9px 16px;border-bottom:1px solid #1e2d3d;'
  f'background:linear-gradient(135deg,#080b0f 60%,#0a1018);'
  f'margin-bottom:12px;position:relative;overflow:hidden">'
  f'<div style="position:absolute;top:0;left:0;right:0;height:2px;'
  f'background:linear-gradient(90deg,transparent,#007a40,#00d4ff,#b44fff,#007a40,transparent)"></div>'
  f'<div style="display:flex;align-items:center;gap:11px">'
  f'<div style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;'
  f'border:1px solid #007a40;background:rgba(0,255,136,.05);'
  f'clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%)">'
  f'<span style="color:#00ff88;font-size:1rem">⬡</span></div>'
  f'<div><div style="font-family:\'Share Tech Mono\',monospace;font-size:1.15rem;'
  f'letter-spacing:4px;color:#00ff88;text-shadow:0 0 10px rgba(0,255,136,.35)">WARL0K</div>'
  f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.6rem;color:#7a9ab5;'
  f'letter-spacing:3px">PIM AUTH ENGINE v2 · GRU+ATTN · PEER HUB · CHAIN-PROOF</div></div></div>'
  f'<div style="display:flex;gap:9px">'
  f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.66rem;padding:3px 8px;'
  f'border:1px solid #7a2db5;color:#b44fff;background:rgba(180,79,255,.05)">⬡ PEER HUB v2</span>'
  f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.66rem;padding:3px 8px;'
  f'border:1px solid {_cbc};color:{_ctc};background:rgba(0,212,255,.05)">{_clbl}</span>'
  f'<span style="font-family:\'Share Tech Mono\',monospace;font-size:.66rem;padding:3px 8px;'
  f'border:1px solid {_mbc};color:{_mtc};background:{_mbg}">{_mlbl}</span>'
  f'</div></div>')

# ════════════════════════════════════════════════════════════════════
# 4 COLUMNS
# ════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns([1, 1, 1.15, 1], gap="small")

# ════════════════════════════════ PANEL 1 — NEURAL CORE ══════════════
with c1:
    panel_hdr("#00ff88", "Neural Core · Train", live=st.session_state["training"])

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

    sep()
    slabel("Training Progress")
    _ph = st.session_state["train_phase"]
    mrow("Phase", f"Phase {_ph}" if _ph else "—", "#00d4ff" if _ph==2 else "#00ff88")
    mrow("Epoch", f"{st.session_state['train_epoch']}/{TOTAL_EP}" if st.session_state["train_epoch"] else "—")
    mrow("Avg Loss", f"{st.session_state['train_loss']:.4f}" if st.session_state["train_loss"] else "—", "#ffb800")
    mrow("Train Time", f"{st.session_state['train_s']:.2f}s" if st.session_state["train_s"] else "—")
    st.progress(float(st.session_state["progress"]))

    slabel("Loss Curve")
    loss_svg()

    slabel("Console")
    render_log()

    _b1, _b2 = st.columns(2)
    with _b1:
        _train_btn = st.button(
            "⬡ TRAIN" if not st.session_state["training"] else "⟳ TRAINING…",
            disabled=st.session_state["training"], key="btn_train")
    with _b2:
        _load_btn = st.button("↓ LOAD", key="btn_load")

    # ── Train ──────────────────────────────────────────────────────
    if _train_btn and not st.session_state["training"]:
        st.session_state["training"]    = True
        st.session_state["train_done"]  = False
        st.session_state["loss_p1"]     = []
        st.session_state["loss_p2"]     = []
        st.session_state["train_phase"] = 0
        st.session_state["train_epoch"] = 0
        st.session_state["train_loss"]  = 0.0
        st.session_state["progress"]    = 0.0
        st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} Starting training…")

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
                    ph_ = info.get("phase",1); ep_ = info.get("epoch",0); ls_ = info.get("loss",0.0)
                    base = CFG["EPOCHS_PHASE1"] if ph_==2 else 0
                    st.session_state["train_phase"] = ph_
                    st.session_state["train_epoch"] = ep_ + base
                    st.session_state["train_loss"]  = ls_
                    st.session_state["progress"]    = min(0.99,(ep_+base)/TOTAL_EP)
                    st.session_state["loss_p1" if ph_==1 else "loss_p2"].append(ls_)

                pim.train_phase1(p, ds, cb=_cb)
                pim.train_phase2(p, ds, cb=_cb)
                ts = time.time()-t0
                st.session_state["train_s"]  = ts
                st.session_state["progress"] = 1.0
                st.session_state["params"]   = p
                st.session_state["session"]  = pim.PIMSession(p, SHARED_KEY)
                pim.save_model(MODEL_PATH, p, {"train_s":ts})
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} ✓ Training done {ts:.2f}s — saved")
                st.session_state["train_done"] = True
            except Exception as ex:
                st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} ✗ {ex}")
            finally:
                st.session_state["training"] = False

        threading.Thread(target=_run_train, daemon=True).start()
        st.rerun()

    # ── Load ───────────────────────────────────────────────────────
    if _load_btn:
        if os.path.exists(MODEL_PATH):
            try:
                _lp, _ = pim.load_model(MODEL_PATH)
                st.session_state["params"]  = _lp
                st.session_state["session"] = pim.PIMSession(_lp, SHARED_KEY)
                st.session_state["log_lines"].append(
                    f"{time.strftime('%H:%M:%S')} ✓ Loaded — {_lp.param_bytes()//1024:.1f} KB")
                st.rerun()
            except Exception as ex:
                st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} ✗ {ex}")
        else:
            st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} ✗ No model.npz — train first")

    if st.session_state["training"]:
        time.sleep(0.45); st.rerun()

# ════════════════════════════════ PANEL 2 — CHAIN VERIFY ═════════════
with c2:
    panel_hdr("#ffb800", "Chain Verify · Auth")

    slabel("Verification Input")
    _vi1, _vi2 = st.columns(2)
    with _vi1: _cid = st.number_input("Claimed ID",      0, CFG["N_IDENTITIES"]-1, 0, key="v_cid")
    with _vi2: _ew  = st.number_input("Expected Window", 0, CFG["N_WINDOWS_PER_ID"]-1, 3, key="v_ew")

    _TMAP = {
        "None — Legit chain":              "none",
        "Shuffle (replay attack)":          "shuffle",
        "Truncate (dropped steps)":         "truncate",
        "Wrong window (counter drift)":     "wrong_win",
        "Wrong identity (impersonation)":   "wrong_id",
        "Out-of-range (hard break)":        "oob",
    }
    _tlbl   = st.selectbox("Tamper Mode", list(_TMAP.keys()), key="v_tamper")
    _tamper = _TMAP[_tlbl]

    _ver_btn = st.button("⬡ VERIFY CHAIN", key="btn_verify",
                         disabled=st.session_state["session"] is None)

    if _ver_btn and st.session_state["session"] is not None:
        NI, NW, T = CFG["N_IDENTITIES"], CFG["N_WINDOWS_PER_ID"], CFG["SEQ_LEN"]
        cid = max(0, min(int(_cid), NI-1)); ew = int(_ew)
        ms  = pim.MS_ALL[cid]; toks, meas = pim.generate_os_chain(ms, cid*NW+ew)
        if _tamper == "shuffle":
            idx  = np.array(pim.XorShift32(0xABCD).shuffle(list(range(T))))
            toks = toks[idx]; meas = meas[idx]
        elif _tamper == "truncate":
            t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
            t2[:T//2]=toks[:T//2]; m2[:T//2]=meas[:T//2]; toks=t2; meas=m2
        elif _tamper == "wrong_win":
            toks, meas = pim.generate_os_chain(ms, cid*NW+(ew+7)%NW)
        elif _tamper == "wrong_id":
            oid=(cid+1)%NI; toks,meas=pim.generate_os_chain(pim.MS_ALL[oid],oid*NW+13)
        elif _tamper == "oob": ew = 9999
        res = st.session_state["session"].verify(cid, ew, toks, meas, ms)
        res["tamper"] = _tamper
        res["gates"]  = {k:bool(v) for k,v in res.get("gates",{}).items()}
        st.session_state["last_result"] = res
        refresh_chain_from_session()

    if st.session_state["last_result"]:
        _r = st.session_state["last_result"]
        render_verdict(_r); render_gates(_r.get("gates",{})); render_scores(_r)

    sep()
    slabel("Bulk Demo")
    _bulk_btn = st.button("▶ RUN ALL CASES", key="btn_bulk",
                          disabled=st.session_state["session"] is None)

    if _bulk_btn and st.session_state["session"] is not None:
        NI, NW, T = CFG["N_IDENTITIES"], CFG["N_WINDOWS_PER_ID"], CFG["SEQ_LEN"]
        ms0 = pim.MS_ALL[0]; tk0, me0 = pim.generate_os_chain(ms0, 0*NW+3)
        bulk = {}
        for nm, tm in [("none","none"),("shuffle","shuffle"),("truncate","truncate"),
                       ("wrong_win","wrong_win"),("wrong_id","wrong_id"),("oob","oob")]:
            tk, me, ew_ = tk0.copy(), me0.copy(), 3
            if tm=="shuffle":
                idx=np.array(pim.XorShift32(0xABCD).shuffle(list(range(T)))); tk=tk[idx]; me=me[idx]
            elif tm=="truncate":
                t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
                t2[:T//2]=tk[:T//2]; m2[:T//2]=me[:T//2]; tk=t2; me=m2
            elif tm=="wrong_win":
                tk,me=pim.generate_os_chain(ms0,0*NW+(3+7)%NW)
            elif tm=="wrong_id":
                oid=1%NI; tk,me=pim.generate_os_chain(pim.MS_ALL[oid],oid*NW+13)
            elif tm=="oob": ew_=9999
            r2=st.session_state["session"].verify(0,ew_,tk,me,ms0)
            r2["gates"]={k:bool(v) for k,v in r2.get("gates",{}).items()}
            bulk[nm]=r2
        st.session_state["bulk_results"] = bulk
        refresh_chain_from_session()

    render_bulk()

    sep()
    slabel("Inference Benchmark")
    _bench_btn = st.button("⚡ BENCHMARK 50×", key="btn_bench",
                           disabled=st.session_state["session"] is None)

    if _bench_btn and st.session_state["session"] is not None:
        _bp = st.session_state["params"]
        _bt, _bm = pim.generate_os_chain(pim.MS_ALL[0], 0)
        _ts = []
        for _ in range(50):
            _t0=time.perf_counter(); pim.verify_chain(_bp,_bt,_bm,0,0)
            _ts.append((time.perf_counter()-_t0)*1e6)
        _arr = np.array(_ts)
        st.session_state["bench_result"] = {
            "mean":_arr.mean(),"min":_arr.min(),"max":_arr.max(),"p95":np.percentile(_arr,95)}

    if st.session_state["bench_result"]:
        _b = st.session_state["bench_result"]
        _be1,_be2,_be3 = st.columns(3)
        with _be1: mrow("MEAN", f"{_b['mean']:.1f} µs", "#00d4ff")
        with _be2: mrow("MIN",  f"{_b['min']:.1f} µs",  "#00ff88")
        with _be3: mrow("P95",  f"{_b['p95']:.1f} µs",  "#ffb800")

# ════════════════════════════════ PANEL 3 — PEER HUB ══════════════════
with c3:
    panel_hdr("#b44fff", "Peer Hub · Dual Handshake")

    slabel("Register Peers")
    _pr1, _pr2 = st.columns(2)
    with _pr1: _ppid = st.text_input("Peer ID",     value="PEER_A", key="p_peerid")
    with _pr2: _piid = st.number_input("Identity ID", 0, CFG["N_IDENTITIES"]-1, 0, key="p_identid")
    _init_btn = st.button("⬡ INIT PEER", key="btn_peer_init",
                          disabled=st.session_state["params"] is None)

    if _init_btn and st.session_state["params"] is not None:
        _pid = _ppid.strip() or f"PEER_{len(st.session_state['peers'])+1}"
        _iid = int(_piid)
        _peer = pim.PeerSession(_pid, st.session_state["params"], _iid, SHARED_KEY)
        st.session_state["peers"][_pid] = _peer
        st.session_state["log_lines"].append(
            f"{time.strftime('%H:%M:%S')} Peer {_pid} registered (id={_iid})")
        st.rerun()

    # Active peers badge
    _npeers = len(st.session_state["peers"])
    if _npeers:
        pids = list(st.session_state["peers"].keys())
        H(f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
          f'color:#b44fff;margin:5px 0">◆ {_npeers} peer(s) active: {", ".join(pids)}</div>')
    else:
        H('<div style="font-family:\'Share Tech Mono\',monospace;font-size:.7rem;'
          'color:#3d566e;margin:5px 0">No peers registered</div>')

    # Peer cards
    for _pid_c in list(st.session_state["peers"].keys()):
        render_peer_card(_pid_c)

    sep()
    slabel("Handshake Pipeline")
    _hs1, _hs2 = st.columns(2)
    with _hs1: _hsa = st.text_input("Peer A", value="PEER_A", key="hs_a")
    with _hs2: _hsb = st.text_input("Peer B", value="PEER_B", key="hs_b")

    _hs_btn = st.button("⬡ RUN FULL HANDSHAKE A↔B", key="btn_handshake",
                        disabled=len(st.session_state["peers"]) < 2)

    if _hs_btn:
        _pa_id = _hsa.strip(); _pb_id = _hsb.strip()
        _peers = st.session_state["peers"]
        if _pa_id not in _peers or _pb_id not in _peers:
            st.session_state["log_lines"].append(
                f"{time.strftime('%H:%M:%S')} ✗ Both peers must be registered first")
        else:
            _pa = _peers[_pa_id]; _pb = _peers[_pb_id]
            _log = st.session_state["handshake_log"]
            _wsevts = st.session_state["ws_events"]
            st.session_state["log_lines"].append(
                f"{time.strftime('%H:%M:%S')} Handshake: {_pa_id} ↔ {_pb_id}")
            try:
                # Step 1: Reconstruct both
                _ra = _pa.reconstruct()
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pa_id,"event_type":"MS_RECONSTRUCT",
                                 "counter":_pa.chain.counter,"delta_ms":0,"ok":_ra["accepted"]})
                _rb = _pb.reconstruct()
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pb_id,"event_type":"MS_RECONSTRUCT",
                                 "counter":_pb.chain.counter,"delta_ms":0,"ok":_rb["accepted"]})
                # Step 2: Build anchors
                _pkg_a = _pa.build_anchor()
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pa_id,"event_type":"ANCHOR_BUILT",
                                 "counter":_pa.chain.counter,"delta_ms":0})
                _pkg_b = _pb.build_anchor()
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pb_id,"event_type":"ANCHOR_BUILT",
                                 "counter":_pb.chain.counter,"delta_ms":0})
                # Step 3: Cross-verify
                _va = _pa.verify_remote_anchor(_pkg_b)
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pa_id,"event_type":"ANCHOR_VERIFY",
                                 "counter":_pa.chain.counter,"delta_ms":0,"ok":_va["ok"]})
                _vb = _pb.verify_remote_anchor(_pkg_a)
                _wsevts.append({"seq":len(_wsevts)+1,"peer_id":_pb_id,"event_type":"ANCHOR_VERIFY",
                                 "counter":_pb.chain.counter,"delta_ms":0,"ok":_vb["ok"]})

                _mutual = _pa.handshake_ok and _pb.handshake_ok
                _l2a    = _va.get("l2_distance",0); _l2b = _vb.get("l2_distance",0)
                _log.clear()
                _log.append({"ok":_mutual,
                              "steps":[
                                  {"label":f"{_pa_id}: reconstruct","ok":_ra["accepted"],"l2":_ra["consensus_l2"]},
                                  {"label":f"{_pb_id}: reconstruct","ok":_rb["accepted"],"l2":_rb["consensus_l2"]},
                                  {"label":f"{_pa_id}: build anchor","ok":True},
                                  {"label":f"{_pb_id}: build anchor","ok":True},
                                  {"label":f"{_pa_id} verifies {_pb_id}","ok":_va["ok"],"dist":_l2a},
                                  {"label":f"{_pb_id} verifies {_pa_id}","ok":_vb["ok"],"dist":_l2b},
                              ]})
                msg = f"◆ MUTUAL PROOF ESTABLISHED A→B:{_pa.handshake_ok} B→A:{_pb.handshake_ok}" if _mutual \
                      else f"◇ HANDSHAKE FAILED A:{_pa.handshake_ok} B:{_pb.handshake_ok}"
                st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} {msg}")
            except Exception as _ex:
                st.session_state["log_lines"].append(f"{time.strftime('%H:%M:%S')} ✗ Handshake error: {_ex}")
            st.rerun()

    # Handshake result
    if st.session_state["handshake_log"]:
        _hs_res = st.session_state["handshake_log"][-1]
        _mut    = _hs_res.get("ok", False)
        _bc2    = "#007a40" if _mut else "#c01f3e"
        _tc2    = "#00ff88" if _mut else "#ff2d55"
        rows = ""
        for s in _hs_res.get("steps", []):
            ok2 = s.get("ok",False); c2 = "#00ff88" if ok2 else "#ff2d55"
            extra = ""
            if "l2" in s: extra = f'<span style="color:#ffb800;margin-left:auto">L2={s["l2"]:.4f}</span>'
            if "dist" in s: extra = f'<span style="color:#00d4ff;margin-left:auto">dist={s["dist"]:.4f}</span>'
            rows += (f'<div style="display:flex;gap:7px;font-family:\'Share Tech Mono\',monospace;'
                     f'font-size:.66rem;margin-bottom:2px;align-items:center">'
                     f'<span style="color:{c2}">{"✓" if ok2 else "✗"}</span>'
                     f'<span style="color:#7a9ab5;flex:1">{s["label"]}</span>{extra}</div>')
        H(f'<div style="border:1px solid {_bc2};padding:9px;background:#0d1117;margin-top:6px">'
          f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.9rem;'
          f'font-weight:700;letter-spacing:3px;color:{_tc2};margin-bottom:7px">'
          f'{"◆ MUTUAL PROOF OK" if _mut else "◇ HANDSHAKE FAILED"}</div>'
          f'{rows}</div>')

    sep()
    slabel("48-OS Reconstruction")
    _recon_pid = st.text_input("Peer ID for Reconstruction", value="PEER_A", key="recon_peer")
    _recon_btn = st.button("⬡ RECONSTRUCT MS (48 OS)", key="btn_recon",
                           disabled=_recon_pid not in st.session_state["peers"])

    if _recon_btn and _recon_pid in st.session_state["peers"]:
        _rpeer = st.session_state["peers"][_recon_pid]
        st.session_state["log_lines"].append(
            f"{time.strftime('%H:%M:%S')} Reconstructing MS from 48 OS windows for {_recon_pid}…")
        _rres = _rpeer.reconstruct()
        st.session_state["recon_result"] = {"peer_id": _recon_pid, **_rres}
        st.session_state["ws_events"].append({
            "seq": len(st.session_state["ws_events"])+1,
            "peer_id": _recon_pid, "event_type":"MS_RECONSTRUCT",
            "counter": _rpeer.chain.counter, "delta_ms":0, "ok":_rres["accepted"]})
        st.session_state["log_lines"].append(
            f"{time.strftime('%H:%M:%S')} {_recon_pid} recon: "
            f"L2={_rres['consensus_l2']:.4f} ok={_rres['windows_ok']}/{_rres['windows_used']} "
            f"{'✓ ACCEPTED' if _rres['accepted'] else '✗ REJECTED'}")
        st.rerun()

    if st.session_state["recon_result"]:
        _rr = st.session_state["recon_result"]
        _acc_c = "#00ff88" if _rr.get("accepted") else "#ff2d55"
        H(f'<div style="border:1px solid {"#007a40" if _rr.get("accepted") else "#c01f3e"};'
          f'padding:9px;background:#0d1117;margin-top:6px">'
          f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:.8rem;'
          f'font-weight:700;color:{_acc_c};margin-bottom:6px;letter-spacing:2px">'
          f'◆ RECONSTRUCTION {"ACCEPTED" if _rr.get("accepted") else "REJECTED"}</div>')
        mrow("Consensus L2", f"{_rr.get('consensus_l2',0):.5f}", _acc_c)
        mrow("Mean L2",      f"{_rr.get('mean_l2',0):.5f}")
        mrow("Windows OK",   f"{_rr.get('windows_ok','?')}/{_rr.get('windows_used','?')}", "#00d4ff")
        mrow("Latency",      f"{_rr.get('latency_ms',0):.1f} ms", "#ffb800")
        H('</div>')

    sep()
    slabel("Live Chain Feed")
    render_ws_feed()

# ════════════════════════════════ PANEL 4 — CHAIN PROOF ══════════════
with c4:
    panel_hdr("#00d4ff", "Chain Proof · Crypto")

    slabel("Chain State")
    render_chain_panel()

    if st.button("↻ REFRESH CHAIN", key="btn_refresh"):
        refresh_chain_from_session()
        st.rerun()

    sep()
    slabel("Encryption Demo · AES-256-GCM")
    _ec1, _ec2 = st.columns(2)
    with _ec1: _eid = st.number_input("ID",     0, CFG["N_IDENTITIES"]-1,     0, key="enc_id")
    with _ec2: _ew2 = st.number_input("Window", 0, CFG["N_WINDOWS_PER_ID"]-1, 0, key="enc_w")

    _enc_btn = st.button("⬡ ENCRYPT TOKENS", key="btn_enc",
                         disabled=st.session_state["session"] is None)

    if _enc_btn and st.session_state["session"] is not None:
        _ecid=int(_eid); _ecw=int(_ew2)
        _ems=pim.MS_ALL[_ecid]
        _etok,_emea=pim.generate_os_chain(_ems,_ecid*CFG["N_WINDOWS_PER_ID"]+_ecw)
        _pkg=st.session_state["session"].encrypt_tokens(_etok,_emea)
        try:
            _t3,_m3=st.session_state["session"].decrypt_tokens(_pkg)
            _rt=bool(np.array_equal(_t3,_etok) and np.allclose(_m3,_emea))
        except Exception: _rt=False
        st.session_state["enc_result"]={"pkg":_pkg,"rt_ok":_rt}
        refresh_chain_from_session()

    render_enc()
