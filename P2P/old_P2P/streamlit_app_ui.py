# streamlit_p2p_ws.py
# Run: streamlit run streamlit_p2p_ws.py
# Hub: python -m uvicorn fastapi_hub_ws:app --host 127.0.0.1 --port 8000
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple

import requests
import streamlit as st
import websockets
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
import threading

HUB_HTTP_DEFAULT = "http://127.0.0.1:8000"
HUB_WS_DEFAULT = "ws://127.0.0.1:8000/ws"

# ----------------- crypto helpers (dual envelope) -----------------
def kdf(seed_hex: str, context: bytes) -> bytes:
    seed = bytes.fromhex(seed_hex)
    digest = hashlib.sha256(seed + context).digest()
    return hashlib.sha256(b"WARLOK|" + digest).digest()[:32]

def inner_encrypt(ms_seed: str, sid: str, ctr: int, plaintext: bytes) -> Tuple[bytes, bytes, bytes]:
    key = kdf(ms_seed, f"{sid}|{ctr}".encode())
    aad = f"{sid}|{ctr}".encode()
    nonce = secrets.token_bytes(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
    return nonce, aad, ct

def inner_decrypt(ms_seed: str, sid: str, ctr: int, nonce: bytes, aad: bytes, ct: bytes) -> Optional[bytes]:
    key = kdf(ms_seed, f"{sid}|{ctr}".encode())
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, ct, aad)
    except Exception:
        return None

def outer_encrypt(os_seed: str, header_json: bytes, inner_ct: bytes) -> Tuple[bytes, bytes]:
    # policy layer: AAD = header_json
    key = kdf(os_seed, b"POLICY")
    nonce = secrets.token_bytes(12)
    tag = ChaCha20Poly1305(key).encrypt(nonce, inner_ct, header_json)
    return nonce, tag

def outer_decrypt(os_seed: str, header_json: bytes, nonce: bytes, tag: bytes) -> Optional[bytes]:
    key = kdf(os_seed, b"POLICY")
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, tag, header_json)
    except Exception:
        return None

# ----------------- hub helpers -----------------
def hub_post(path: str, body=None, timeout=5):
    url = st.session_state["HUB_HTTP"] + path
    try:
        r = requests.post(url, json=body, timeout=timeout)
        return r
    except Exception as e:
        # bubble up a fake response-like obj carrying the error text
        class _R:
            ok = False
            status_code = 0
            text = f"POST {url} failed: {e}"
            def json(self): return {"error": self.text}
        return _R()

def hub_get(path: str, params=None, timeout=5):
    url = st.session_state["HUB_HTTP"] + path
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r
    except Exception as e:
        class _R:
            ok = False
            status_code = 0
            text = f"GET {url} failed: {e}"
            def json(self): return {"error": self.text}
        return _R()

# ----------------- state -----------------
def init_state():
    ss = st.session_state
    ss.setdefault("HUB_HTTP", HUB_HTTP_DEFAULT)
    ss.setdefault("HUB_WS", HUB_WS_DEFAULT)
    ss.setdefault("sid", None)
    ss.setdefault("ms_seed", None)
    ss.setdefault("os_seed", None)
    ss.setdefault("ctr", 0)

    ss.setdefault("ws_connected", False)
    ss.setdefault("ws_started", False)
    ss.setdefault("ws_error", "")
    ss.setdefault("hub_events", [])
    ss.setdefault("poll_cursor", 0)

    ss.setdefault("last_delivery", None)  # holds envelopes + header
    ss.setdefault("hub_log_text", "")
    ss.setdefault("tick", time.time())

# ----------------- websocket consumer (resilient) -----------------
def ws_worker():
    while True:
        try:
            asyncio.run(ws_loop())
        except Exception as e:
            # mark disconnected and wait a bit before retry
            st.session_state["ws_connected"] = False
            st.session_state["ws_error"] = str(e)
            time.sleep(2.0)  # backoff

async def ws_loop():
    url = st.session_state["HUB_WS"]
    async with websockets.connect(
        url,
        ping_interval=20,   # library-level websocket pings
        ping_timeout=20,
        max_size=None,
    ) as ws:
        st.session_state["ws_connected"] = True
        st.session_state["ws_error"] = ""

        # send one hello
        await ws.send("hello")

        async def rx():
            while True:
                raw = await ws.recv()  # raises on disconnect
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                st.session_state["hub_events"].append(ev)
                if isinstance(ev, dict) and "id" in ev:
                    st.session_state["poll_cursor"] = max(st.session_state["poll_cursor"], ev["id"])

        async def tx():
            # userland heartbeat every 15s to keep intermediaries happy
            while True:
                await asyncio.sleep(15)
                try:
                    await ws.send("ping")
                except Exception:
                    break

        # run until one side errors
        await asyncio.gather(rx(), tx())

def start_ws():
    if not st.session_state["ws_started"]:
        t = threading.Thread(target=ws_worker, daemon=True)
        t.start()
        st.session_state["ws_started"] = True

# ----------------- small UI helpers -----------------
def pills():
    sid = st.session_state.get("sid")
    ws = st.session_state.get("ws_connected", False)
    st.markdown(
        f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <span style="background:#eef;border:1px solid #bbf;border-radius:12px;padding:2px 8px">Session: <b>{(sid[:8]+'…') if sid else '—'}</b></span>
          <span style="background:{'#e8ffe8' if ws else '#ffe8e8'};border:1px solid #ddd;border-radius:12px;padding:2px 8px">WS: <b>{'connected' if ws else 'disconnected'}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.session_state.get("ws_error") and not ws:
        st.warning(f"WebSocket error: {st.session_state['ws_error']}")

def render_live_events():
    st.markdown(
        """
        <style>
          .logbox{max-height:360px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:8px;
                  padding:8px;background:#0b1021;color:#c8d3f5;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
          .logline{margin:0 0 6px 0;white-space:pre-wrap}
        </style>
        """,
        unsafe_allow_html=True,
    )
    events = st.session_state["hub_events"][-300:]
    if not events:
        st.info("No events yet. Start a session and send data.")
        return
    lines = []
    for e in events:
        ts = e.get("ts", "")
        typ = e.get("type", "")
        payload = json.dumps(e.get("payload", {}), ensure_ascii=False)
        lines.append(f"[{ts}] <{typ}> {payload}")
    html = "<div class='logbox' id='hublog'>" + "".join(
        f"<div class='logline'>{l}</div>" for l in lines
    ) + "</div>"
    html += "<script>var x=document.getElementById('hublog'); if(x){x.scrollTop=x.scrollHeight;}</script>"
    st.markdown(html, unsafe_allow_html=True)

def refresh_hub_file_log():
    r = hub_get("/logs", {"lines": 800})
    if r.ok:
        st.session_state["hub_log_text"] = r.text or ""
    else:
        st.session_state["hub_log_text"] = f"Failed to fetch log: HTTP {r.status_code} | {r.text}"

# ----------------- main -----------------
def main():
    st.set_page_config(page_title="P2P + Hub (WS • Dual Envelope • Logs)", layout="wide")
    init_state()

    # Keep the live feed fresh (simple 1s heartbeat)
    now = time.time()
    if now - st.session_state["tick"] > 1.0:
        st.session_state["tick"] = now
        st.rerun()

    # Sidebar: endpoints + health + WS control
    with st.sidebar:
        st.header("Hub endpoints")
        http = st.text_input("Hub HTTP", value=st.session_state["HUB_HTTP"], key="inp_hub_http")
        ws = st.text_input("Hub WS", value=st.session_state["HUB_WS"], key="inp_hub_ws")
        if http != st.session_state["HUB_HTTP"] or ws != st.session_state["HUB_WS"]:
            st.session_state["HUB_HTTP"] = http
            st.session_state["HUB_WS"] = ws
            st.session_state["ws_started"] = False
            st.session_state["ws_connected"] = False
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Reconnect WS", key="btn_ws_reconnect"):
                st.session_state["ws_started"] = False
                st.session_state["ws_connected"] = False
        with c2:
            if st.button("Test /health", key="btn_health"):
                r = hub_get("/health")
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"HTTP {r.status_code} | {r.text}")
        with c3:
            if st.button("Fetch /events once", key="btn_fetch_events"):
                r = hub_get("/events", {"since": st.session_state["poll_cursor"]})
                if r.ok:
                    data = r.json().get("events", [])
                    st.session_state["hub_events"].extend(data)
                    if data:
                        st.session_state["poll_cursor"] = max(st.session_state["poll_cursor"], data[-1]["id"])
                else:
                    st.warning(f"/events HTTP {r.status_code} | {r.text}")
        with c4:
            if st.button("Refresh Hub Log", key="btn_refresh_log_sidebar"):
                refresh_hub_file_log()

    # Start WS after endpoints are configured
    start_ws()

    st.title("Peer ↔ Hub (policy monitor) ↔ Peer — Dual-Envelope E2EE")

    # Diagnostics expander
    with st.expander("Diagnostics", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Probe: /rendezvous POST", key="btn_probe_rv_post"):
                r = hub_post("/rendezvous")
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"POST /rendezvous failed — HTTP {getattr(r,'status_code',0)}\n{getattr(r,'text','')}")
        with c2:
            if st.button("Probe: /rendezvous GET", key="btn_probe_rv_get"):
                r = hub_get("/rendezvous")
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"GET /rendezvous failed — HTTP {getattr(r,'status_code',0)}\n{getattr(r,'text','')}")
        with c3:
            if st.button("Probe: policy msg (5 chars)", key="btn_probe_policy_msg"):
                r = hub_post("/policy/check", {
                    "sid": "diag",
                    "link_id": str(uuid.uuid4()),
                    "type": "msg",
                    "meta": {"chars": 5}
                })
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"Policy msg failed — HTTP {r.status_code} | {r.text}")
        with c4:
            if st.button("Probe: policy file (1MB)", key="btn_probe_policy_file"):
                r = hub_post("/policy/check", {
                    "sid": "diag",
                    "link_id": str(uuid.uuid4()),
                    "type": "file",
                    "meta": {"size": 1_000_000}
                })
                if r.ok:
                    st.json(r.json())
                else:
                    st.error(f"Policy file failed — HTTP {r.status_code} | {r.text}")

    # Top controls
    top = st.columns([1, 1, 1.6])
    with top[0]:
        if st.button("🔄 New Session", key="btn_new_session"):
            # Try POST first
            r = hub_post("/rendezvous")
            if r.ok:
                rv = r.json()
                st.session_state["sid"] = rv["sid"]
                st.session_state["ms_seed"] = rv["ms_seed"]
                st.session_state["os_seed"] = rv["os_seed"]
                st.session_state["ctr"] = 0
                st.session_state["last_delivery"] = None
                st.success(f"Session: {rv['sid'][:8]}… (POST)")
            else:
                # Fallback to GET
                r2 = hub_get("/rendezvous")
                if r2.ok:
                    rv = r2.json()
                    st.session_state["sid"] = rv["sid"]
                    st.session_state["ms_seed"] = rv["ms_seed"]
                    st.session_state["os_seed"] = rv["os_seed"]
                    st.session_state["ctr"] = 0
                    st.session_state["last_delivery"] = None
                    st.success(f"Session: {rv['sid'][:8]}… (GET fallback)")
                else:
                    st.error(f"New Session failed.\nPOST /rendezvous → {getattr(r,'status_code',0)} | {getattr(r,'text','')}\n"
                             f"GET /rendezvous → {getattr(r2,'status_code',0)} | {getattr(r2,'text','')}")
    with top[1]:
        pills()
        st.caption(f"`HTTP {st.session_state['HUB_HTTP']}` | `WS {st.session_state['HUB_WS']}`")
    with top[2]:
        if st.button("Manual Rendezvous (GET)", key="btn_manual_rv_get"):
            r = hub_get("/rendezvous")
            if r.ok:
                rv = r.json()
                st.session_state["sid"] = rv["sid"]
                st.session_state["ms_seed"] = rv["ms_seed"]
                st.session_state["os_seed"] = rv["os_seed"]
                st.session_state["ctr"] = 0
                st.session_state["last_delivery"] = None
                st.success(f"Session: {rv['sid'][:8]}… (GET)")
            else:
                st.error(f"GET /rendezvous failed — HTTP {r.status_code} | {r.text}")
        if st.button("Refresh Hub Log (file tail)", key="btn_refresh_log_top"):
            refresh_hub_file_log()
        st.text_area(
            "Hub JSONL log tail (click 'Refresh Hub Log')",
            value=st.session_state.get("hub_log_text", ""),
            height=150,
            key="txt_log_tail_top",
        )

    st.divider()

    colA, colHub, colB = st.columns([1.1, 1.2, 1.1])

    # ----------------- Peer A (sender) -----------------
    with colA:
        st.subheader("Peer A (sender)")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Click 'New Session' first.")
        else:
            ctr = st.session_state["ctr"]

            # MESSAGE
            with st.expander("Send MESSAGE (dual-envelope)", expanded=False):
                msg = st.text_area("Message", "hi", key="txt_msg")
                if st.button("Send ▶️", key="btn_msg_send"):
                    try:
                        # Inner (payload) AEAD
                        n_in, aad_in, ct_in = inner_encrypt(st.session_state["ms_seed"], sid, ctr, msg.encode())

                        # Header (visible to hub)
                        header = {
                            "link_id": str(uuid.uuid4()),
                            "type": "msg",
                            "meta": {"sid": sid, "chars": len(msg), "created_at": datetime.utcnow().isoformat() + "Z"},
                        }
                        header_bytes = json.dumps(header, sort_keys=True).encode()

                        # Outer (policy) AEAD with header as AAD
                        n_out, tag_out = outer_encrypt(st.session_state["os_seed"], header_bytes, ct_in)

                        # Register + policy check
                        rr = hub_post("/link/register", {"header": header, "policy_tag_len": len(tag_out)})
                        if not rr.ok:
                            st.error(f"/link/register failed — HTTP {rr.status_code} | {rr.text}")
                            st.stop()
                        tt = hub_post(
                            "/policy/check",
                            {"sid": sid, "link_id": header["link_id"], "type": "msg", "meta": {"chars": len(msg)}},
                        )
                        if not tt.ok:
                            st.error(f"/policy/check failed — HTTP {tt.status_code} | {tt.text}")
                            st.stop()
                        ticket = tt.json()
                        if not ticket.get("ok", False):
                            st.error(f"Blocked by policy: {ticket.get('reason')}")
                        else:
                            st.session_state["last_delivery"] = {
                                "kind": "msg",
                                "counter": ctr,
                                "header": header,
                                "inner": {"nonce": n_in, "aad": aad_in, "ct": ct_in},
                                "outer": {"nonce": n_out, "tag": tag_out},
                            }
                            st.session_state["ctr"] += 1
                            hub_post("/event", {"type": "delivered", "payload": {"sid": sid, "link_id": header["link_id"]}})
                            st.success("Message sent")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # FILE
            with st.expander("Send FILE (dual-envelope)", expanded=False):
                up = st.file_uploader("Choose a file", key="up_file")
                chunk_kb = st.slider("Chunk size (KB)", 32, 1024, 128, key="sld_chunk")
                if up and st.button("Upload ▶️", key="btn_file_send"):
                    try:
                        data = up.read()
                        total = len(data)

                        # Inner AEAD over full file
                        n_in, aad_in, ct_in = inner_encrypt(st.session_state["ms_seed"], sid, ctr, data)

                        header = {
                            "link_id": str(uuid.uuid4()),
                            "type": "file",
                            "meta": {
                                "sid": sid,
                                "size": total,
                                "filename": up.name,
                                "mime": up.type,
                                "created_at": datetime.utcnow().isoformat() + "Z",
                            },
                        }
                        header_bytes = json.dumps(header, sort_keys=True).encode()
                        n_out, tag_out = outer_encrypt(st.session_state["os_seed"], header_bytes, ct_in)

                        rr = hub_post("/link/register", {"header": header, "policy_tag_len": len(tag_out)})
                        if not rr.ok:
                            st.error(f"/link/register failed — HTTP {rr.status_code} | {rr.text}")
                            st.stop()
                        tt = hub_post(
                            "/policy/check",
                            {"sid": sid, "link_id": header["link_id"], "type": "file", "meta": {"size": total}},
                        )
                        if not tt.ok:
                            st.error(f"/policy/check failed — HTTP {tt.status_code} | {tt.text}")
                            st.stop()
                        ticket = tt.json()
                        if not ticket.get("ok", False):
                            st.error(f"Blocked by policy: {ticket.get('reason')}")
                        else:
                            # Simulate chunking (progress + hub upload_progress events)
                            sent = 0
                            step = chunk_kb * 1024
                            pb = st.progress(0, text="Uploading…", key="pb_upload")
                            while sent < total:
                                sent = min(sent + step, total)
                                pb.progress(min(1.0, sent / total))
                                hub_post(
                                    "/event",
                                    {"type": "upload_progress",
                                     "payload": {"sid": sid, "link_id": header["link_id"], "sent": sent, "total": total}},
                                )
                                time.sleep(0.02)

                            st.session_state["last_delivery"] = {
                                "kind": "file",
                                "counter": ctr,
                                "header": header,
                                "inner": {"nonce": n_in, "aad": aad_in, "ct": ct_in},
                                "outer": {"nonce": n_out, "tag": tag_out},
                            }
                            st.session_state["ctr"] += 1
                            hub_post("/event", {"type": "delivered", "payload": {"sid": sid, "link_id": header["link_id"]}})
                            st.success("File sent")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ----------------- Hub column -----------------
    with colHub:
        st.subheader("Hub (live WebSocket feed)")
        render_live_events()

        st.markdown("---")
        if st.button("Refresh Hub Log (file tail)", key="btn_refresh_log_hub"):
            refresh_hub_file_log()
        st.text_area("Hub JSONL log tail", value=st.session_state.get("hub_log_text", ""), height=180, key="txt_log_tail_hub")

    # ----------------- Peer B (receiver) -----------------
    with colB:
        st.subheader("Peer B (receiver)")
        sid = st.session_state["sid"]
        last = st.session_state.get("last_delivery")
        if not sid or not last:
            st.info("Waiting for data…")
        else:
            ctr = last["counter"]
            header = last["header"]

            # 1) Policy: unwrap outer using header as AAD
            inner_ct = outer_decrypt(
                st.session_state["os_seed"],
                json.dumps(header, sort_keys=True).encode(),
                last["outer"]["nonce"],
                last["outer"]["tag"],
            )
            if inner_ct is None:
                st.error("Policy envelope invalid — drop")
                return

            # 2) Payload: decrypt inner
            pt = inner_decrypt(
                st.session_state["ms_seed"],
                sid,
                ctr,
                last["inner"]["nonce"],
                last["inner"]["aad"],
                inner_ct,
            )
            if pt is None:
                st.error("Inner AEAD tag invalid — drop")
            else:
                if last["kind"] == "msg":
                    st.success("Message received")
                    st.code(pt.decode(errors="ignore"))
                elif last["kind"] == "file":
                    fname = header["meta"].get("filename", "file.bin")
                    st.success(f"File received: {fname}")
                    st.download_button("Download file", data=pt, file_name=fname, key="btn_dl_file")

if __name__ == "__main__":
    main()
