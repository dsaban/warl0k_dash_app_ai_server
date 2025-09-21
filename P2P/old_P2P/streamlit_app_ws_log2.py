# streamlit_app_ws.py
# ------------------------------------
# WARL0K P2P Mock Demo (Streamlit) with:
# - Real AEAD (ChaCha20-Poly1305)
# - WS live log + HTTP polling fallback
# - Scrollable live feed
# - "Session complete" + "Refresh session log" (reads hub JSONL file tail)
#
# Run:
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
from datetime import datetime
from typing import Optional

import asyncio
import websockets
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

DEFAULT_HUB_HTTP = "http://127.0.0.1:8000"
DEFAULT_HUB_WS   = "ws://127.0.0.1:8000/ws"

# ---------- crypto helpers ----------
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

# ---------- app state ----------
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

    ss.setdefault("HUB_BASE", DEFAULT_HUB_HTTP)
    ss.setdefault("HUB_WS",   DEFAULT_HUB_WS)
    ss.setdefault("last_refresh", time.time())

    ss.setdefault("use_tunnel", False)
    ss.setdefault("ws_started", False)
    ss.setdefault("ws_connected", False)
    ss.setdefault("ws_error", "")
    ss.setdefault("fallback_polling", False)
    ss.setdefault("poll_thread_started", False)
    ss.setdefault("poll_cursor", 0)

    ss.setdefault("session_log_text", "")  # fetched JSONL tail
    ss.setdefault("session_done", False)

# ---------- hub helpers ----------
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
        requests.post(f"{st.session_state['HUB_BASE']}/event", json={"type": ev_type, "payload": payload}, timeout=5)
    except Exception:
        pass

def hub_mark_session_complete(sid: str):
    try:
        requests.post(f"{st.session_state['HUB_BASE']}/session/complete", json={"sid": sid}, timeout=5)
    except Exception:
        pass

def hub_fetch_session_log(sid: str, lines: int = 500) -> str:
    try:
        r = requests.get(f"{st.session_state['HUB_BASE']}/logs/session/{sid}", params={"lines": lines}, timeout=5)
        if r.ok:
            return r.text
    except Exception:
        pass
    return ""

# ---------- link object ----------
def make_link_object(link_type: str, to_pdid: str, from_pdid: str, meta: dict, content_ref: str | None, sk_auth: bytes, sid: str | None) -> dict:
    if sid:
        meta = dict(meta)
        meta["sid"] = sid  # thread sid into header for easier per-session logs
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

# ---------- WS + HTTP polling ----------
def ws_worker():
    while True:
        try:
            asyncio.run(ws_loop())
        except Exception as e:
            st.session_state["ws_connected"] = False
            st.session_state["ws_error"] = str(e)
            st.session_state["fallback_polling"] = True
            time.sleep(1.0)

async def ws_loop():
    url = st.session_state["HUB_WS"]
    try:
        async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:
            st.session_state["ws_connected"] = True
            st.session_state["ws_error"] = ""
            st.session_state["fallback_polling"] = False
            await ws.send("hello")
            while True:
                msg = await ws.recv()
                event = json.loads(msg)
                if isinstance(event, dict) and "id" in event:
                    st.session_state["poll_cursor"] = max(st.session_state["poll_cursor"], event["id"])
                st.session_state["hub_events"].append(event)
                if len(st.session_state["hub_events"]) > 1000:
                    st.session_state["hub_events"] = st.session_state["hub_events"][-1000:]
                await asyncio.sleep(0.01)
    except Exception as e:
        st.session_state["ws_connected"] = False
        st.session_state["ws_error"] = str(e)
        raise

def start_ws():
    if not st.session_state["ws_started"]:
        t = threading.Thread(target=ws_worker, daemon=True)
        t.start()
        st.session_state["ws_started"] = True

def poll_worker():
    while True:
        try:
            if not st.session_state.get("fallback_polling", False):
                time.sleep(0.5)
                continue
            base = st.session_state["HUB_BASE"]
            since = st.session_state.get("poll_cursor", 0)
            r = requests.get(f"{base}/events", params={"since": since}, timeout=5)
            if r.ok:
                data = r.json()
                for ev in data.get("events", []):
                    st.session_state["hub_events"].append(ev)
                    st.session_state["poll_cursor"] = max(st.session_state["poll_cursor"], ev.get("id", since))
                if len(st.session_state["hub_events"]) > 1000:
                    st.session_state["hub_events"] = st.session_state["hub_events"][-1000:]
            time.sleep(1.0)
        except Exception:
            time.sleep(1.0)

def start_polling():
    if not st.session_state["poll_thread_started"]:
        t = threading.Thread(target=poll_worker, daemon=True)
        t.start()
        st.session_state["poll_thread_started"] = True

# ---------- UI helpers ----------
def hub_status_pills():
    sid = st.session_state.get("sid")
    ws_connected = st.session_state.get("ws_connected", False)
    ws_error = st.session_state.get("ws_error", "")
    fb = st.session_state.get("fallback_polling", False)
    st.markdown(
        f"""
        <div style="display:flex; gap:8px; flex-wrap:wrap;">
          <span style="background:#eef;border:1px solid #bbf;border-radius:12px;padding:2px 8px;">
            Session: <b>{(sid[:8]+'…') if sid else '—'}</b>
          </span>
          <span style="background:{'#e8ffe8' if ws_connected else '#ffe8e8'};border:1px solid #ddd;border-radius:12px;padding:2px 8px;">
            Hub WS: <b>{'connected' if ws_connected else 'disconnected'}</b>
          </span>
          <span style="background:{'#e8ffe8' if fb else '#f5f5f5'};border:1px solid #ddd;border-radius:12px;padding:2px 8px;">
            HTTP Polling: <b>{'ON' if fb else 'OFF'}</b>
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if ws_error and not ws_connected:
        st.warning(f"WebSocket error: {ws_error}")

def render_hub_events_scrollable():
    st.markdown(
        """
        <style>
        .logbox {max-height: 420px; overflow-y: auto; border: 1px solid #e5e7eb; border-radius: 8px;
                 padding: 8px; background: #0b1021; color: #c8d3f5; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                 font-size: 12px; line-height: 1.4;}
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
    html = "<div class='logbox' id='hublog'>" + "".join(f"<div class='logline'>{l}</div>" for l in lines) + "</div>"
    html += "<script>var x=document.getElementById('hublog'); if(x){x.scrollTop=x.scrollHeight;}</script>"
    st.markdown(html, unsafe_allow_html=True)

# ---------- Main ----------
def main():
    st.set_page_config(page_title="WARL0K P2P (WS + Log Files)", layout="wide")
    init_state()

    # Heartbeat to re-render every second
    now = time.time()
    if now - st.session_state["last_refresh"] > 1.0:
        st.session_state["last_refresh"] = now
        st.rerun()

    # Sidebar connection settings
    with st.sidebar:
        st.header("Hub Connection")
        hub_http = st.text_input("Hub HTTP", value=st.session_state["HUB_BASE"])
        hub_ws   = st.text_input("Hub WS",   value=st.session_state["HUB_WS"])
        if hub_http != st.session_state["HUB_BASE"] or hub_ws != st.session_state["HUB_WS"]:
            st.session_state["HUB_BASE"] = hub_http
            st.session_state["HUB_WS"]   = hub_ws
            st.session_state["ws_started"] = False
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Reconnect WS"):
                st.session_state["ws_started"] = False
                st.session_state["ws_connected"] = False
        with c2:
            if st.button("Test /health"):
                st.write(f"Hub health: {hub_health()}")
        with c3:
            if st.button("Force HTTP Polling"):
                st.session_state["fallback_polling"] = True

    # start background workers
    start_ws()
    start_polling()

    st.title("WARL0K P2P Mock — WS + HTTP Fallback + File Logs")
    st.caption("Hub logs are persisted to JSONL files; you can refresh and print a session’s log.")

    top = st.columns([1,1,1.6])
    with top[0]:
        if st.button("🔄 New Session via Hub"):
            try:
                rv = hub_rendezvous()
                st.session_state["sid"] = rv["sid"]
                st.session_state["ms_seed"] = rv["ms_seed"]
                st.session_state["os_seed"] = rv["os_seed"]
                st.session_state["counter"] = 0
                st.session_state["session_done"] = False
                st.success(f"Session: {rv['sid'][:8]}…")
            except Exception as e:
                st.error(f"Hub error: {e}")
    with top[1]:
        # Mark session complete (lets team know to refresh the persisted file)
        if st.button("Mark Session Complete", disabled=not st.session_state.get("sid")):
            sid = st.session_state.get("sid")
            if sid:
                hub_mark_session_complete(sid)
                st.session_state["session_done"] = True
        # Refresh session log (reads per-session JSONL from hub)
        if st.button("🔁 Refresh session log", disabled=not st.session_state.get("sid")):
            sid = st.session_state.get("sid")
            text = hub_fetch_session_log(sid, lines=800) if sid else ""
            st.session_state["session_log_text"] = text or "No log data returned."
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
            st.info("Start a session via Hub.")
        else:
            ctr = st.session_state["counter"]
            KA, fpA = derive_key(sid, st.session_state["ms_seed"], ctr)
            st.caption(f"K_{ctr} (A) derived. fp_hint={fpA}")

            # MESSAGE
            with st.expander("Send MESSAGE"):
                msg = st.text_area("Message text", "Hello from A to B (AEAD)")
                if st.button("Send ▶️", key="msgA"):
                    aad = f"{sid}|{ctr}".encode()
                    nonce = secrets.token_bytes(12)
                    ct = aead_seal(KA, nonce, aad, msg.encode())
                    header = make_link_object(
                        "msg",
                        st.session_state["peerB_pdid"],
                        st.session_state["peerA_pdid"],
                        {"chars": len(msg), "created_at": datetime.utcnow().isoformat()+"Z"},
                        None,
                        st.session_state["peerA_auth"],
                        sid
                    )
                    try:
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

            # FILE
            with st.expander("Send FILE (chunked with progress)"):
                up = st.file_uploader("Choose a file", key="fileA_ws")
                chunk_kb = st.slider("Chunk size (KB)", 32, 1024, 128)
                if up and st.button("Send ▶️", key="fileA_btn"):
                    data = up.read(); total = len(data)
                    aad = f"{sid}|{ctr}".encode()
                    nonce_full = secrets.token_bytes(12)
                    ct_full = aead_seal(KA, nonce_full, aad, data)
                    content_ref = hashlib.sha256(ct_full).hexdigest()
                    header = make_link_object(
                        "file",
                        st.session_state["peerB_pdid"],
                        st.session_state["peerA_pdid"],
                        {"size": total, "filename": up.name, "mime": up.type, "created_at": datetime.utcnow().isoformat()+"Z"},
                        content_ref,
                        st.session_state["peerA_auth"],
                        sid
                    )
                    try:
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
                                # simulate per-chunk transport (AEAD per chunk)
                                _ = aead_seal(KA, secrets.token_bytes(12), aad, part)
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
                    header = make_link_object(
                        "pay",
                        st.session_state["peerB_pdid"],
                        st.session_state["peerA_pdid"],
                        {"amount": amt, "asset": asset, "memo": memo, "created_at": invoice["created_at"]},
                        None,
                        st.session_state["peerA_auth"],
                        sid
                    )
                    try:
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

        st.markdown("### Session Log (from hub file)")
        if st.session_state.get("sid"):
            st.code(st.session_state.get("session_log_text", "") or "No session log yet. Click 'Refresh session log'.")
        else:
            st.info("Start a session to enable per-session log refresh.")

    # -------- Peer B --------
    with colB:
        st.subheader("Peer B")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Start a session via Hub.")
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

    st.divider()
    st.caption("After you click 'Mark Session Complete', use 'Refresh session log' to print the persisted JSONL from the hub.")

if __name__ == "__main__":
    main()
