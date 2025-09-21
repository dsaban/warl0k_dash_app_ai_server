# fastapi_hub.py
# ------------------------------------
# WARL0K Mock Hub with REST + WebSocket broadcast
# Run: uvicorn fastapi_hub:app --reload --port 8000
# ------------------------------------

import asyncio
import json
import secrets
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="WARL0K Mock Hub")

# CORS for local Streamlit or other frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected WebSocket clients
clients: Set[WebSocket] = set()
clients_lock = asyncio.Lock()

async def broadcast(event: Dict[str, Any]):
    data = json.dumps(event)
    async with clients_lock:
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.discard(d)

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

@app.post("/rendezvous")
async def rendezvous():
    sid = str(uuid.uuid4())
    ms_seed = secrets.token_hex(32)
    os_seed = secrets.token_hex(32)
    await broadcast({"ts": now_iso(), "type": "rendezvous", "payload": {"sid": sid}})
    await broadcast({"ts": now_iso(), "type": "seeds_issued",
                     "payload": {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed}})
    return {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed, "ttl": 300}

@app.post("/link/register")
async def link_register(header: Dict[str, Any] = Body(...), fp_hint: int = Body(...)):
    await broadcast({"ts": now_iso(), "type": "link_register",
                     "payload": {"header": header, "fp_hint": fp_hint}})
    return {"ok": True}

@app.post("/policy/check")
async def policy_check(body: Dict[str, Any] = Body(...)):
    # body: { "sid", "link_id", "type", "meta", "fp_hint" }
    sid = body.get("sid")
    link_id = body.get("link_id")
    typ = body.get("type")
    meta = body.get("meta", {})
    fp_hint = int(body.get("fp_hint", 1))

    ok = True
    reason = "ok"
    anomaly = (fp_hint == 0)

    if typ == "file":
        size = int(meta.get("size", 0))
        if size > 50_000_000:
            ok, reason = False, "file too large"
    elif typ == "msg":
        chars = int(meta.get("chars", 0))
        if chars > 5000:
            ok, reason = False, "message too long"
    elif typ == "pay":
        amt = float(meta.get("amount", 0.0))
        if amt > 10_000.0:
            ok, reason = False, "amount exceeds policy"

    ticket = {
        "ok": ok,
        "reason": reason,
        "anomaly": anomaly,
        "ticket": {"sid": sid, "link_id": link_id, "ttl": 120} if ok else None
    }
    await broadcast({"ts": now_iso(), "type": "policy_decision",
                     "payload": {"header": {"link_id": link_id, "type": typ}, "ticket": ticket}})
    return ticket

# NEW: generic event endpoint so clients can push “tunnel used” / upload progress, etc.
@app.post("/event")
async def push_event(event: Dict[str, Any] = Body(...)):
    # Expecting { "type": "tunnel_status"|"upload_progress"|..., "payload": {...} }
    event.setdefault("ts", now_iso())
    await broadcast(event)
    return {"ok": True}

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        while True:
            # broadcast-only server; read to keep connection alive
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)
