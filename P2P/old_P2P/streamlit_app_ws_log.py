# streamlit_app_ws.py
# ------------------------------------
# WARL0K P2P Mock Demo (Streamlit) with:
# - Real AEAD (ChaCha20-Poly1305)
# - Live Hub log via WebSocket (fastapi_hub.py)
# - Scrollable + auto-scrolling Hub log
# - Relay/Tunnel indicator + chunked upload progress
# - Heartbeat-based auto-refresh (no extra deps)
# - Local Mock Mode if hub is unreachable
#
# Run order:
#   uvicorn fastapi_hub:app --reload --host 127.0.0.1 --port 8000
#   streamlit run streamlit_app_ws.py
# ------------------------------------
from __future__ import annotations

import streamlit as st
import requests
import threading
import time
import json
import uuid
import secrets
import hashlib
import base64
from datetime import datetime
from typing import Optional

import asyncio
import websockets

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

DEFAULT_HUB_HTTP = "http://127.0.0.1:8000"
DEFAULT_HUB_WS   = "ws://127.0.0.1:8000/ws"

# ---------- helpers ----------
def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()

def derive_key(session_id: str, seed_hex: str, counter: int, drift: bytes=b"") -> tuple[bytes, int]:
    seed = bytes.fromhex(seed_hex)
    msg = session_id.encode() + counter.to_bytes(8, "big") + drift
    digest = hashlib.sha256(seed + msg).digest()
    key = hashlib.sha256(b"WARLOK|" + digest).digest()[:32]
    fp_hint = int.from_bytes(digest[:4], "big")
    return key, fp_hint

def aead_seal(key: bytes, nonce: bytes, aad: bytes, plaintext: bytes) -> bytes:
    return ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)

def aead_open(key: bytes, nonce: bytes, aad: bytes, ciphertext: bytes) -> Optional[bytes]:
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, aad)
    except Exception:
        return None

# ---- state ----
def init_state():
    ss = st.session_state
    ss.setdefault("hub_events", [])
    ss.setdefault("sid", None)
    ss.setdefault("ms_seed", None)
    ss.setdefault("os_seed", None)
    ss.setdefault("counter", 0)
    ss.setdefault("peerA_auth", secrets.token_bytes(32))
    ss.setdefault("peerB_auth", secrets.token_bytes(32))
    ss.setdefault("peerA_pdid", "did:pA:" + secrets.token_hex(4))
    ss.setdefault("peerB_pdid", "did:pB:" + secrets.token_hex(4))
    ss.setdefault("last_delivery", None)
    ss.setdefault("ws_started", False)
    ss.setdefault("ws_connected", False)
    ss.setdefault("ws_error", "")
    ss.setdefault("use_tunnel", False)
    ss.setdefault("HUB_BASE", DEFAULT_HUB_HTTP)
    ss.setdefault("HUB_WS",   DEFAULT_HUB_WS)
    ss.setdefault("last_refresh", time.time())
    ss.setdefault("local_mock_mode", False)  # fallback if hub unreachable

# ---- hub helpers ----
def hub_health() -> bool:
    try:
        r = requests.get(f"{st.session_state['HUB_BASE']}/health", timeout=3)
        return r.ok
    except Exception:
        return False

def hub_rendezvous():
    r = requests.post(f"{st.session_state['HUB_BASE']}/rendezvous", timeout=5)
    r.raise_for_status()
    return r.json()

def hub_link_register(header: dict, fp_hint: int):
    body = {"header": header, "fp_hint": fp_hint}
    r = requests.post(f"{st.session_state['HUB_BASE']}/link/register", json=body, timeout=5)
    r.raise_for_status()
    return r.json()

def hub_policy_check(sid: str, header: dict, fp_hint: int):
    body = {"sid": sid, "link_id": header["link_id"], "type": header["type"], "meta": header["meta"], "fp_hint": fp_hint}
    r = requests.post(f"{st.session_state['HUB_BASE']}/policy/check", json=body, timeout=5)
    r.raise_for_status()
    return r.json()

def hub_push_event(ev_type: str, payload: dict):
    try:
        body = {"type": ev_type, "payload": payload}
        requests.post(f"{st.session_state['HUB_BASE']}/event", json=body, timeout=5)
    except Exception:
        pass

# ---- link object ----
def make_link_object(link_type: str, to_pdid: str, from_pdid: str, meta: dict, content_ref: str | None, sk_auth: bytes) -> dict:
    core = {
        "link_id": str(uuid.uuid4()),
        "type": link_type,
        "to": to_pdid,
        "from": from_pdid,
        "meta": meta,
        "content_ref": content_ref,
        "policy_ref": "default-v1",
    }
    core_bytes = json.dumps(core, sort_keys=True).encode()
    sig = hashlib.sha256(sk_auth + core_bytes).hexdigest()
    core["proof"] = {"sender_sig": sig}
    return core

def verify_link_object(header: dict, pk_auth: bytes) -> bool:
    temp = header.copy()
    proof = temp.pop("proof", None)
    if not proof or "sender_sig" not in proof:
        return False
    data = json.dumps(temp, sort_keys=True).encode()
    expect = hashlib.sha256(pk_auth + data).hexdigest()
    return secrets.compare_digest(expect, proof["sender_sig"])

# ---- WS background thread ----
def ws_worker():
    while True:
        try:
            asyncio.run(ws_loop())
        except Exception as e:
            st.session_state["ws_connected"] = False
            st.session_state["ws_error"] = str(e)
            time.sleep(1.0)

async def ws_loop():
    url = st.session_state["HUB_WS"]
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            st.session_state["ws_connected"] = True
            st.session_state["ws_error"] = ""
            await ws.send("hello")
            while True:
                msg = await ws.recv()
                event = json.loads(msg)
                st.session_state["hub_events"].append(event)
                if len(st.session_state["hub_events"]) > 1000:
                    st.session_state["hub_events"] = st.session_state["hub_events"][-1000:]
                await asyncio.sleep(0.01)
    except Exception as e:
        st.session_state["ws_connected"] = False
        st.session_state["ws_error"] = str(e)
        await asyncio.sleep(0.5)
        raise

def start_ws():
    if not st.session_state["ws_started"]:
        t = threading.Thread(target=ws_worker, daemon=True)
        t.start()
        st.session_state["ws_started"] = True

# ---- UI helpers ----
def hub_status_pills():
    sid = st.session_state.get("sid")
    use_tunnel = st.session_state.get("use_tunnel", False)
    ws_connected = st.session_state.get("ws_connected", False)
    ws_error = st.session_state.get("ws_error", "")
    mock = st.session_state.get("local_mock_mode", False)
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <span style="background:#eef;border:1px solid #bbf;border-radius:12px;padding:2px 8px;">
            Session: <b>{(sid[:8]+'…') if sid else '—'}</b>
          </span>
          <span style="background:{'#e8ffe8' if ws_connected else '#ffe8e8'};border:1px solid #ddd;border-radius:12px;padding:2px 8px;">
            Hub WS: <b>{'connected' if ws_connected else 'disconnected'}</b>
          </span>
          <span style="background:{'#ffecc7' if use_tunnel else '#e8f7ff'};border:1px solid #ddd;border-radius:12px;padding:2px 8px;">
            Relay/Tunnel: <b>{'ON' if use_tunnel else 'OFF'}</b>
          </span>
          <span style="background:{'#f0f0f0' if not mock else '#dff7df'};border:1px solid #ddd;border-radius:12px;padding:2px 8px;">
            Local Mock Mode: <b>{'ON' if mock else 'OFF'}</b>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ws_error and not ws_connected and not st.session_state["local_mock_mode"]:
        st.warning(f"WebSocket error: {ws_error}")

def render_hub_events_scrollable():
    st.markdown(
        """
        <style>
        .logbox {
            max-height: 420px;
            overflow-y: auto;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 8px;
            background: #0b1021;
            color: #c8d3f5;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            line-height: 1.4;
        }
        .logline { margin: 0 0 6px 0; white-space: pre-wrap; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    events = st.session_state["hub_events"][-300:]
    if not events:
        st.info("⚠️ No events received yet. Try 'New Session' or sending a message.")
        return
    lines = []
    for e in events:
        ts = e.get("ts", "")
        et = e.get("type", "")
        payload = json.dumps(e.get("payload", {}), ensure_ascii=False)
        lines.append(f"[{ts}] <{et}> {payload}")
    inner = "".join(f"<div class='logline'>{l}</div>" for l in lines)
    html = f"<div class='logbox' id='hublog'>{inner}</div>"
    html += """
    <script>
      var logBox = document.getElementById('hublog');
      if (logBox) { logBox.scrollTop = logBox.scrollHeight; }
    </script>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---- Main ----
def main():
    st.set_page_config(page_title="WARL0K P2P (WS + AEAD)", layout="wide")
    init_state()

    # Heartbeat to re-render every second
    now = time.time()
    if now - st.session_state["last_refresh"] > 1.0:
        st.session_state["last_refresh"] = now
        st.rerun()

    # Sidebar config
    with st.sidebar:
        st.header("Hub Connection")
        hub_http = st.text_input("Hub HTTP", value=st.session_state["HUB_BASE"])
        hub_ws   = st.text_input("Hub WS",   value=st.session_state["HUB_WS"])
        if hub_http != st.session_state["HUB_BASE"] or hub_ws != st.session_state["HUB_WS"]:
            st.session_state["HUB_BASE"] = hub_http
            st.session_state["HUB_WS"]   = hub_ws
            st.session_state["ws_started"] = False  # restart worker
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Reconnect WS"):
                st.session_state["ws_started"] = False
        with c2:
            if st.button("Test /health"):
                ok = hub_health()
                st.write(f"Hub health: {ok}")
        st.caption("Tip: use 127.0.0.1 (not localhost) to avoid host/CORS issues.")
        st.divider()
        st.checkbox("Force Local Mock Mode (no hub)", key="local_mock_mode")

    # Start WS only if not in mock mode
    if not st.session_state["local_mock_mode"]:
        start_ws()

    st.title("WARL0K P2P Mock — WS Hub, AEAD, Scroll Log, Tunnel Indicator")
    st.caption("Hub sees headers/policy outcomes only; content stays encrypted. Log refreshes every second.")

    top = st.columns([1,1,1.6])
    with top[0]:
        if st.button("🔄 New Session via Hub" if not st.session_state["local_mock_mode"] else "🔄 New Session (Local Mock)"):
            try:
                if st.session_state["local_mock_mode"]:
                    # Create a purely local session & synthesize a hub event
                    st.session_state["sid"] = str(uuid.uuid4())
                    st.session_state["ms_seed"] = secrets.token_hex(32)
                    st.session_state["os_seed"] = secrets.token_hex(32)
                    st.session_state["counter"] = 0
                    st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "rendezvous", "payload": {"sid": st.session_state['sid']}})
                    st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "seeds_issued", "payload": {"sid": st.session_state['sid'], "ms_seed": st.session_state['ms_seed'], "os_seed": st.session_state['os_seed']}})
                    st.success(f"Local session: {st.session_state['sid'][:8]}…")
                else:
                    rv = hub_rendezvous()
                    st.session_state["sid"] = rv["sid"]
                    st.session_state["ms_seed"] = rv["ms_seed"]
                    st.session_state["os_seed"] = rv["os_seed"]
                    st.session_state["counter"] = 0
                    st.success(f"Session: {rv['sid'][:8]}…")
            except Exception as e:
                st.error(f"Hub error: {e}")
    with top[1]:
        use_tunnel = st.toggle("Use Relay/Tunnel (TURN) simulation", value=st.session_state["use_tunnel"])
        if use_tunnel != st.session_state["use_tunnel"]:
            st.session_state["use_tunnel"] = use_tunnel
            sid = st.session_state.get("sid")
            if sid and not st.session_state["local_mock_mode"]:
                hub_push_event("tunnel_status", {"sid": sid, "relay": use_tunnel})
        if st.button("Send test event to Hub", disabled=st.session_state["local_mock_mode"]):
            sid = st.session_state.get("sid") or "no-session"
            hub_push_event("debug_test", {"sid": sid, "msg": "hello from streamlit"})
    with top[2]:
        hub_status_pills()
        st.write(f"`HTTP {st.session_state['HUB_BASE']}`  |  `WS {st.session_state['HUB_WS']}`")

    st.divider()

    colA, colHub, colB = st.columns([1.1, 1.2, 1.1])

    # -------- Peer A --------
    with colA:
        st.subheader("Peer A")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Start a session (Hub or Local Mock).")
        else:
            ctr = st.session_state["counter"]
            KA, fpA = derive_key(sid, st.session_state["ms_seed"], ctr)
            st.caption(f"K_{ctr} (A) derived. fp_hint={fpA}")

            # MESSAGE
            with st.expander("Send MESSAGE"):
                msg = st.text_area("Message text", "Hello from A to B (real AEAD)")
                if st.button("Send ▶️", key="msgA"):
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct = aead_seal(KA, nonce, aad, msg.encode())
                    header = make_link_object("msg", st.session_state["peerB_pdid"], st.session_state["peerA_pdid"],
                                              {"chars": len(msg), "created_at": datetime.utcnow().isoformat()+"Z"},
                                              None, st.session_state["peerA_auth"])
                    try:
                        if st.session_state["local_mock_mode"]:
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "link_register", "payload": {"header": header, "fp_hint": fpA}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "policy_decision", "payload": {"header": {"link_id": header["link_id"], "type": "msg"}, "ticket": {"ok": True, "reason": "ok", "anomaly": False, "ticket": {"sid": sid, "link_id": header["link_id"], "ttl": 120}}}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "path_select", "payload": {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]}})
                            st.session_state["last_delivery"] = {"type":"msg","nonce":nonce,"aad":aad,"ct":ct,"header":header,"counter":ctr}
                            st.session_state["counter"] += 1
                            st.success("Sent (local mock)")
                        else:
                            hub_link_register(header, fpA)
                            hub_push_event("path_select", {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]})
                            ticket = hub_policy_check(sid, header, fpA)
                            if ticket["ok"]:
                                st.session_state["last_delivery"] = {"type":"msg","nonce":nonce,"aad":aad,"ct":ct,"header":header,"counter":ctr}
                                st.session_state["counter"] += 1
                                st.success("Sent")
                            else:
                                st.error(f"Blocked: {ticket['reason']}")
                    except Exception as e:
                        st.error(f"Hub error: {e}")

            # FILE (chunked)
            with st.expander("Send FILE (chunked with progress)"):
                up = st.file_uploader("Choose a file", key="fileA_ws")
                chunk_kb = st.slider("Chunk size (KB)", 32, 1024, 128)
                if up and st.button("Send ▶️", key="fileA_btn"):
                    data = up.read(); total = len(data)
                    aad = f"{sid}|{ctr}".encode()
                    nonce_full = secrets.token_bytes(12)
                    ct_full = aead_seal(KA, nonce_full, aad, data)
                    content_ref = hashlib.sha256(ct_full).hexdigest()
                    header = make_link_object("file", st.session_state["peerB_pdid"], st.session_state["peerA_pdid"],
                                              {"size": total, "filename": up.name, "mime": up.type, "created_at": datetime.utcnow().isoformat()+"Z"},
                                              content_ref, st.session_state["peerA_auth"])
                    try:
                        if st.session_state["local_mock_mode"]:
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "link_register", "payload": {"header": header, "fp_hint": fpA}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "path_select", "payload": {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "policy_decision", "payload": {"header": {"link_id": header["link_id"], "type": "file"}, "ticket": {"ok": True, "reason": "ok", "anomaly": False, "ticket": {"sid": sid, "link_id": header["link_id"], "ttl": 120}}}})
                            sent = 0; pb = st.progress(0, text="Uploading (mock)")
                            chunk = chunk_kb * 1024
                            while sent < total:
                                end = min(sent + chunk, total)
                                sent = end
                                pb.progress(min(1.0, sent/total))
                                st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "upload_progress", "payload": {"sid": sid, "link_id": header["link_id"], "sent": sent, "total": total, "relay": st.session_state["use_tunnel"]}})
                                time.sleep(0.02)
                            st.session_state["last_delivery"] = {"type":"file","nonce":nonce_full,"aad":aad,"ct":ct_full,"header":header,"counter":ctr}
                            st.session_state["counter"] += 1
                            st.success("File sent (mock)")
                        else:
                            hub_link_register(header, fpA)
                            hub_push_event("path_select", {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]})
                            ticket = hub_policy_check(sid, header, fpA)
                            if not ticket["ok"]:
                                st.error(f"Blocked: {ticket['reason']}")
                            else:
                                sent = 0; pb = st.progress(0, text="Uploading")
                                chunk = chunk_kb * 1024
                                while sent < total:
                                    end = min(sent + chunk, total)
                                    part = data[sent:end]
                                    nonce = secrets.token_bytes(12)
                                    _ct_part = aead_seal(KA, nonce, aad, part)
                                    sent = end
                                    pb.progress(min(1.0, sent/total))
                                    hub_push_event("upload_progress", {"sid": sid, "link_id": header["link_id"], "sent": sent, "total": total, "relay": st.session_state["use_tunnel"]})
                                    time.sleep(0.02)
                                st.session_state["last_delivery"] = {"type":"file","nonce":nonce_full,"aad":aad,"ct":ct_full,"header":header,"counter":ctr}
                                st.session_state["counter"] += 1
                                st.success("File sent")
                    except Exception as e:
                        st.error(f"Hub error: {e}")

            # PAYMENT
            with st.expander("Send PAYMENT"):
                amt = st.number_input("Amount", min_value=0.0, value=42.0, step=1.0, key="amtA")
                asset = st.selectbox("Asset", ["USD","EUR","USDC","WBTC"], key="assetA")
                memo = st.text_input("Memo", "Thanks!", key="memoA")
                if st.button("Send ▶️", key="payA_btn"):
                    invoice = {"invoice_id": str(uuid.uuid4()), "to": st.session_state["peerB_pdid"], "from": st.session_state["peerA_pdid"],
                               "amount": amt, "asset": asset, "memo": memo, "created_at": datetime.utcnow().isoformat()+"Z",
                               "expiry": int(time.time()) + 3600}
                    payload = json.dumps(invoice).encode()
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct = aead_seal(KA, nonce, aad, payload)
                    header = make_link_object("pay", st.session_state["peerB_pdid"], st.session_state["peerA_pdid"],
                                              {"amount": amt, "asset": asset, "memo": memo, "created_at": invoice["created_at"]},
                                              None, st.session_state["peerA_auth"])
                    try:
                        if st.session_state["local_mock_mode"]:
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "link_register", "payload": {"header": header, "fp_hint": fpA}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "path_select", "payload": {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]}})
                            st.session_state["hub_events"].append({"ts": datetime.utcnow().isoformat()+"Z", "type": "policy_decision", "payload": {"header": {"link_id": header["link_id"], "type": "pay"}, "ticket": {"ok": True, "reason": "ok", "anomaly": False, "ticket": {"sid": sid, "link_id": header["link_id"], "ttl": 120}}}})
                            st.session_state["last_delivery"] = {"type":"pay","nonce":nonce,"aad":aad,"ct":ct,"header":header,"counter":ctr}
                            st.session_state["counter"] += 1
                            st.success("Sent (mock)")
                        else:
                            hub_link_register(header, fpA)
                            hub_push_event("path_select", {"sid": sid, "link_id": header["link_id"], "relay": st.session_state["use_tunnel"]})
                            ticket = hub_policy_check(sid, header, fpA)
                            if ticket["ok"]:
                                st.session_state["last_delivery"] = {"type":"pay","nonce":nonce,"aad":aad,"ct":ct,"header":header,"counter":ctr}
                                st.session_state["counter"] += 1
                                st.success("Sent")
                            else:
                                st.error(f"Blocked: {ticket['reason']}")
                    except Exception as e:
                        st.error(f"Hub error: {e}")

    # -------- Hub column --------
    with colHub:
        st.subheader("WARL0K Hub (live feed)")
        hub_status_pills()
        render_hub_events_scrollable()

    # -------- Peer B --------
    with colB:
        st.subheader("Peer B")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Start a session (Hub or Local Mock).")
        else:
            last = st.session_state["last_delivery"]
            if last:
                ctr = last["counter"]
                KB, fpB = derive_key(sid, st.session_state["ms_seed"], ctr)
                st.caption(f"K_{ctr} (B) derived. fp_hint={fpB}")
                header = last["header"]
                with st.expander("Link Header"):
                    st.json(header)
                ok_header = verify_link_object(header, st.session_state["peerA_auth"])
                st.write(f"Header signature valid: {'✅' if ok_header else '❌'}")

                pt = aead_open(KB, last["nonce"], last["aad"], last["ct"])
                if pt is None:
                    st.error("AEAD tag invalid — drop")
                else:
                    if last["type"] == "msg":
                        st.success("Message received")
                        st.code(pt.decode(errors="ignore"))
                    elif last["type"] == "file":
                        st.success("File received")
                        received_ref = hashlib.sha256(last["ct"]).hexdigest()
                        st.write(f"ContentRef matched: {'✅' if received_ref == header['content_ref'] else '❌'}")
                        st.download_button("Download file", data=pt, file_name=header['meta'].get("filename","file.bin"))
                    elif last["type"] == "pay":
                        st.success("Payment request received")
                        inv = json.loads(pt.decode())
                        st.json(inv)
                        if st.button("Ack Payment (mock)"):
                            st.session_state["hub_events"].append({
                                "ts": datetime.utcnow().isoformat()+"Z",
                                "type": "payment_receipt_local",
                                "payload": {"invoice_id": inv["invoice_id"], "tx_ref": "psp:DEMO-" + secrets.token_hex(6)}
                            })

    st.divider()
    st.caption("If the hub is unreachable, toggle Local Mock Mode in the sidebar to demo the flow offline.")

if __name__ == "__main__":
    main()
