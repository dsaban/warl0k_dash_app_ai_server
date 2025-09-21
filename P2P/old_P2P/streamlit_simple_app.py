# streamlit_p2p_autohub.py
# One-file demo: starts a FastAPI Hub inside Streamlit (background thread)
# Run: streamlit run streamlit_p2p_autohub.py
from __future__ import annotations

import streamlit as st
import threading, time, json, uuid, secrets, hashlib
from datetime import datetime, timezone
from collections import deque
from typing import Dict, Any, Optional

import requests
from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from uvicorn import Config, Server

from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# ----------------------------
# Config
# ----------------------------
HUB_HOST = "127.0.0.1"
HUB_PORT = 8000
HUB_URL  = f"http://{HUB_HOST}:{HUB_PORT}"

# ----------------------------
# Minimal in-process HUB
# ----------------------------
app = FastAPI(title="Embedded WARL0K Hub")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

EVENTS: deque[Dict[str, Any]] = deque(maxlen=2000)
EVENT_ID = 0

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def emit(evt_type: str, payload: Dict[str, Any]):
    global EVENT_ID
    EVENT_ID += 1
    evt = {"id": EVENT_ID, "ts": now_iso(), "type": evt_type, "payload": payload}
    EVENTS.append(evt)
    return evt

@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}

@app.get("/events")
def events(since: int = Query(0)):
    data = [e for e in EVENTS if e["id"] > since]
    nxt = EVENTS[-1]["id"] if EVENTS else since
    return {"ok": True, "events": data, "next": nxt}

@app.post("/rendezvous")
def rendezvous():
    sid = str(uuid.uuid4())
    ms_seed = secrets.token_hex(32)
    os_seed = secrets.token_hex(32)
    emit("rendezvous", {"sid": sid})
    emit("seeds_issued", {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed})
    return {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed}

@app.post("/link/register")
def link_register(header: Dict[str, Any] = Body(...)):
    emit("link_register", {"header": header})
    return {"ok": True}

@app.post("/policy/check")
def policy_check(body: Dict[str, Any] = Body(...)):
    """
    Expect: { sid, type, meta, link_id }
      - type='msg'  -> meta: {chars:int}
      - type='file' -> meta: {size:int}
    Policy:
      - block messages > 2000 chars
      - block files > 50 MB
    """
    typ = body.get("type")
    meta = body.get("meta", {})
    sid = body.get("sid")
    link_id = body.get("link_id")

    ok, reason = True, "ok"
    if typ == "msg" and int(meta.get("chars", 0)) > 2000:
        ok, reason = False, "message too long"
    elif typ == "file" and int(meta.get("size", 0)) > 50_000_000:
        ok, reason = False, "file too large"

    ticket = {"ok": ok, "reason": reason, "ticket": {"sid": sid, "link_id": link_id} if ok else None}
    emit("policy_decision", {"sid": sid, "link_id": link_id, "type": typ, "ticket": ticket})
    return ticket

@app.post("/event")
def event_push(body: Dict[str, Any] = Body(...)):
    typ = body.get("type", "custom")
    payload = body.get("payload", {})
    emit(typ, payload)
    return {"ok": True}

# ----------------------------
# Launch hub inside a thread
# ----------------------------
def start_hub_once():
    if "hub_started" in st.session_state and st.session_state["hub_started"]:
        return
    st.session_state["hub_started"] = True
    def run():
        server = Server(Config(app=app, host=HUB_HOST, port=HUB_PORT, log_level="error"))
        server.run()
    t = threading.Thread(target=run, daemon=True)
    t.start()
    # Wait briefly for startup
    for _ in range(20):
        try:
            ok = requests.get(f"{HUB_URL}/health", timeout=0.3).ok
            if ok: break
        except Exception:
            time.sleep(0.1)

# ----------------------------
# Client-side helpers
# ----------------------------
def derive_key(session_id: str, seed_hex: str, counter: int) -> bytes:
    seed = bytes.fromhex(seed_hex)
    msg = f"{session_id}|{counter}".encode()
    digest = hashlib.sha256(seed + msg).digest()
    return hashlib.sha256(b"WARLOK|" + digest).digest()[:32]

def seal(key: bytes, aad: bytes, plaintext: bytes):
    nonce = secrets.token_bytes(12)
    ct = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)
    return nonce, ct

def open_(key: bytes, aad: bytes, nonce: bytes, ct: bytes) -> Optional[bytes]:
    try:
        return ChaCha20Poly1305(key).decrypt(nonce, ct, aad)
    except Exception:
        return None

def hub_post(path: str, body=None, timeout=5):
    return requests.post(HUB_URL + path, json=body, timeout=timeout)

def hub_get(path: str, params=None, timeout=5):
    return requests.get(HUB_URL + path, params=params, timeout=timeout)

# ----------------------------
# Streamlit state
# ----------------------------
def init_state():
    ss = st.session_state
    ss.setdefault("sid", None)
    ss.setdefault("ms_seed", None)
    ss.setdefault("os_seed", None)
    ss.setdefault("counter", 0)
    ss.setdefault("poll_cursor", 0)
    ss.setdefault("hub_events", [])
    ss.setdefault("last_delivery", None)
    ss.setdefault("tick", time.time())
    ss.setdefault("hub_started", False)
    ss.setdefault("diag", "")

def poll_hub():
    try:
        r = hub_get("/events", {"since": st.session_state["poll_cursor"]})
        if r.ok:
            data = r.json()
            for ev in data.get("events", []):
                st.session_state["hub_events"].append(ev)
                st.session_state["poll_cursor"] = max(st.session_state["poll_cursor"], ev["id"])
            st.session_state["hub_events"] = st.session_state["hub_events"][-400:]
    except Exception as e:
        st.session_state["diag"] = f"Poll failed: {e}"

# ----------------------------
# UI
# ----------------------------
def main():
    st.set_page_config(page_title="P2P with Embedded Hub (Policy + E2EE)", layout="wide")
    init_state()
    start_hub_once()

    # heartbeat for log refresh
    now = time.time()
    if now - st.session_state["tick"] > 1.0:
        st.session_state["tick"] = now
        poll_hub()
        st.rerun()

    st.title("Peer ↔ Embedded Hub (policy/monitor) ↔ Peer — E2EE msgs & files")

    # Diagnostics banner
    try:
        ok = hub_get("/health", timeout=1).ok
    except Exception as e:
        ok = False
        st.session_state["diag"] = f"/health error: {e}"
    if not ok:
        st.error("Hub not reachable. It should auto-start. If this persists, your Python env blocks local sockets.")
    if st.session_state["diag"]:
        st.caption(f"diag: {st.session_state['diag']}")

    top = st.columns([1,1,1.4])
    with top[0]:
        if st.button("🔄 New Session"):
            try:
                r = hub_post("/rendezvous")
                r.raise_for_status()
                rv = r.json()
                st.session_state["sid"] = rv["sid"]
                st.session_state["ms_seed"] = rv["ms_seed"]
                st.session_state["os_seed"] = rv["os_seed"]
                st.session_state["counter"] = 0
                st.session_state["last_delivery"] = None
                st.success(f"Session: {rv['sid'][:8]}…")
            except Exception as e:
                st.error(f"Hub error: {e}")
    with top[1]:
        sid = st.session_state.get("sid")
        st.write(f"Session: `{sid[:8]}…`" if sid else "Session: —")
        st.caption("Hub sees only headers/meta, not plaintext.")
    with top[2]:
        st.write(f"Embedded Hub URL: `{HUB_URL}`")

    st.divider()

    left, mid, right = st.columns([1.1, 1.2, 1.1])

    # ---- Peer A (sender) ----
    with left:
        st.subheader("Peer A (sender)")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Create a session first.")
        else:
            ctr = st.session_state["counter"]
            key = derive_key(sid, st.session_state["ms_seed"], ctr)
            st.caption(f"Key K_{ctr} derived.")

            # MESSAGE
            with st.expander("Send MESSAGE"):
                msg = st.text_area("Message", "Hello from A to B")
                if st.button("Send ▶️", key="msg_send"):
                    try:
                        aad = f"{sid}|{ctr}".encode()
                        nonce, ct = seal(key, aad, msg.encode())
                        link_id = str(uuid.uuid4())
                        header = {"link_id": link_id, "type": "msg",
                                  "meta": {"sid": sid, "chars": len(msg), "created_at": datetime.utcnow().isoformat()+"Z"}}
                        hub_post("/link/register", {"header": header}).raise_for_status()
                        ticket = hub_post("/policy/check", {"sid": sid, "link_id": link_id, "type": "msg",
                                                            "meta": {"chars": len(msg)}}).json()
                        if not ticket.get("ok", False):
                            st.error(f"Blocked by policy: {ticket.get('reason')}")
                        else:
                            st.session_state["last_delivery"] = {"kind":"msg","nonce":nonce,"aad":aad,"ct":ct,"counter":ctr}
                            st.session_state["counter"] += 1
                            hub_post("/event", {"type":"delivered","payload":{"sid":sid,"link_id":link_id}})
                            st.success("Message sent")
                    except Exception as e:
                        st.error(f"Error: {e}")

            # FILE
            with st.expander("Send FILE (AEAD, chunked)"):
                up = st.file_uploader("Choose a file", key="fileA")
                chunk_kb = st.slider("Chunk size (KB)", 32, 1024, 128)
                if up and st.button("Upload ▶️", key="file_send"):
                    try:
                        data = up.read()
                        total = len(data)
                        aad = f"{sid}|{ctr}".encode()

                        link_id = str(uuid.uuid4())
                        header = {"link_id": link_id, "type": "file",
                                  "meta": {"sid": sid, "size": total, "filename": up.name, "mime": up.type,
                                           "created_at": datetime.utcnow().isoformat()+"Z"}}
                        hub_post("/link/register", {"header": header}).raise_for_status()

                        ticket = hub_post("/policy/check", {"sid":sid,"link_id":link_id,"type":"file","meta":{"size":total}}).json()
                        if not ticket.get("ok", False):
                            st.error(f"Blocked by policy: {ticket.get('reason')}")
                        else:
                            key_file = derive_key(sid, st.session_state["ms_seed"], ctr)
                            nonce_full = secrets.token_bytes(12)
                            ct_full = ChaCha20Poly1305(key_file).encrypt(nonce_full, data, aad)

                            sent = 0
                            pb = st.progress(0, text="Uploading…")
                            step = chunk_kb * 1024
                            while sent < total:
                                end = min(sent + step, total)
                                sent = end
                                pb.progress(min(1.0, sent/total))
                                hub_post("/event", {"type":"upload_progress",
                                                    "payload":{"sid":sid,"link_id":link_id,"sent":sent,"total":total}})
                                time.sleep(0.02)

                            st.session_state["last_delivery"] = {"kind":"file","nonce":nonce_full,"aad":aad,"ct":ct_full,
                                                                 "counter":ctr,"file_meta":header["meta"]}
                            st.session_state["counter"] += 1
                            hub_post("/event", {"type":"delivered","payload":{"sid":sid,"link_id":link_id}})
                            st.success("File sent")
                    except Exception as e:
                        st.error(f"Error: {e}")

    # ---- Hub monitor (center) ----
    with mid:
        st.subheader("Hub monitor (HTTP polling)")
        st.markdown("""
        <style>
        .logbox{max-height:420px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:8px;
                padding:8px;background:#0b1021;color:#c8d3f5;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
        .logline{margin:0 0 6px 0;white-space:pre-wrap}
        </style>""", unsafe_allow_html=True)

        events = st.session_state["hub_events"][-300:]
        if not events:
            st.info("No events yet. Create a session and send a message/file.")
        else:
            lines = []
            for e in events:
                ts = e.get("ts",""); typ = e.get("type",""); payload = json.dumps(e.get("payload",{}), ensure_ascii=False)
                lines.append(f"[{ts}] <{typ}> {payload}")
            html = "<div class='logbox' id='hublog'>" + "".join(f"<div class='logline'>{l}</div>" for l in lines) + "</div>"
            html += "<script>var x=document.getElementById('hublog'); if(x){x.scrollTop=x.scrollHeight;}</script>"
            st.markdown(html, unsafe_allow_html=True)

        st.button("Refresh log now", on_click=poll_hub)

    # ---- Peer B (receiver) ----
    with right:
        st.subheader("Peer B (receiver)")
        sid = st.session_state["sid"]
        if not sid:
            st.info("Create a session first.")
        else:
            last = st.session_state.get("last_delivery")
            if not last:
                st.info("Waiting for data…")
            else:
                ctr = last["counter"]
                keyB = derive_key(sid, st.session_state["ms_seed"], ctr)
                pt = open_(keyB, last["aad"], last["nonce"], last["ct"])
                if pt is None:
                    st.error("Auth failed (bad tag).")
                else:
                    if last["kind"] == "msg":
                        st.success("Message received")
                        st.code(pt.decode(errors="ignore"))
                    elif last["kind"] == "file":
                        meta = last.get("file_meta", {})
                        fname = meta.get("filename", "file.bin")
                        st.success(f"File received: {fname}")
                        st.download_button("Download file", data=pt, file_name=fname)

if __name__ == "__main__":
    main()
