# fastapi_hub_simple.py
# ------------------------------------
# Minimal WARL0K Hub (HTTP-only):
# - POST /rendezvous -> {sid, ms_seed, os_seed}
# - POST /policy/check (msg/file) -> ticket ok/blocked
# - POST /link/register -> record header (hub sees metadata only)
# - POST /event -> append a hub event (e.g., "delivered", "upload_progress")
# - GET  /events?since=ID -> poll hub log (monitoring)
# - GET  /health
# Run: python -m uvicorn fastapi_hub_simple:app --host 127.0.0.1 --port 8000
# ------------------------------------
from __future__ import annotations
import json
import uuid
import secrets
from collections import deque
from datetime import datetime, timezone
from typing import Dict, Any

from fastapi import FastAPI, Body, Query
from fastapi.middleware.cors import CORSMiddleware

def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

app = FastAPI(title="Minimal WARL0K Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# In-memory event log (HTTP polling)
EVENTS: deque[Dict[str, Any]] = deque(maxlen=2000)
EVENT_ID = 0

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
    # Return all events with id > since
    data = [e for e in EVENTS if e["id"] > since]
    nxt = EVENTS[-1]["id"] if EVENTS else since
    return {"ok": True, "events": data, "next": nxt}

@app.post("/rendezvous")
def rendezvous():
    sid = str(uuid.uuid4())
    ms_seed = secrets.token_hex(32)  # “main secret” seed (placeholder)
    os_seed = secrets.token_hex(32)  # “one-shot”  seed (placeholder)
    emit("rendezvous", {"sid": sid})
    emit("seeds_issued", {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed})
    return {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed}

@app.post("/link/register")
def link_register(header: Dict[str, Any] = Body(...)):
    # Hub only sees headers / metadata — no plaintext
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
    if typ == "msg":
        if int(meta.get("chars", 0)) > 2000:
            ok, reason = False, "message too long"
    elif typ == "file":
        if int(meta.get("size", 0)) > 50_000_000:
            ok, reason = False, "file too large"

    ticket = {"ok": ok, "reason": reason, "ticket": {"sid": sid, "link_id": link_id} if ok else None}
    emit("policy_decision", {"sid": sid, "link_id": link_id, "type": typ, "ticket": ticket})
    return ticket

@app.post("/event")
def event_push(body: Dict[str, Any] = Body(...)):
    # Accepts any event type to show in the hub log: { type, payload }
    typ = body.get("type", "custom")
    payload = body.get("payload", {})
    emit(typ, payload)
    return {"ok": True}
