# fastapi_hub.py
# ------------------------------------
# WARL0K Mock Hub:
# - REST endpoints
# - WebSocket broadcast
# - HTTP polling (/events?since=...)
# - FILE LOGGING (JSONL): global logs + per-session logs
# Endpoints for logs:
#   GET  /logs                      -> tail of global log (JSONL as text)
#   GET  /logs/session/{sid}        -> tail of a session log (JSONL as text)
#   POST /session/complete          -> mark session completed (emits event)
#
# Run: python -m uvicorn fastapi_hub:app --reload --host 127.0.0.1 --port 8000
# ------------------------------------

import asyncio
import json
import secrets
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Set, List, Tuple

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------- config ----------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
GLOBAL_LOG = LOG_DIR / "events.jsonl"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def session_log_path(sid: str) -> Path:
    return LOG_DIR / f"session-{sid}.jsonl"

# --------- app + CORS ----------
app = FastAPI(title="WARL0K Mock Hub")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Event store (ring buffer) =========
EVENTS: deque = deque(maxlen=2000)  # holds {"id", "ts", "type", "payload", ...}
EVENT_ID: int = 0
event_lock = asyncio.Lock()

# ========= WebSocket clients =========
clients: Set[WebSocket] = set()
clients_lock = asyncio.Lock()

async def record_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """Store event in ring buffer and write to files."""
    global EVENT_ID
    async with event_lock:
        EVENT_ID += 1
        full = {"id": EVENT_ID, **evt}
        EVENTS.append(full)

    # --- file logging (global) ---
    try:
        jsonl_append(GLOBAL_LOG, full)
    except Exception as e:
        print(f"[WARN] failed to write global log: {e}")

    # --- file logging (per-session if sid in payload) ---
    sid = full.get("payload", {}).get("sid")
    if sid:
        try:
            jsonl_append(session_log_path(sid), full)
        except Exception as e:
            print(f"[WARN] failed to write session log for {sid}: {e}")

    return full

async def broadcast(evt: Dict[str, Any]):
    data = json.dumps(evt)
    async with clients_lock:
        dead = []
        for ws in list(clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for d in dead:
            clients.discard(d)

async def emit(evt_type: str, payload: Dict[str, Any]):
    evt = {"ts": now_iso(), "type": evt_type, "payload": payload}
    full = await record_event(evt)
    await broadcast(full)

# --------- middleware ----------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"[{now_iso()}] {request.method} {request.url.path}")
    resp = await call_next(request)
    return resp

# --------- health ----------
@app.get("/health")
async def health():
    return {"ok": True, "ts": now_iso()}

# --------- polling fallback ----------
@app.get("/events")
async def events_poll(since: int = Query(0, description="last seen event id")):
    async with event_lock:
        data = [e for e in EVENTS if e["id"] > since]
        next_id = EVENTS[-1]["id"] if EVENTS else since
    return {"ok": True, "events": data, "next": next_id}

# --------- rendezvous / policy / link ----------
@app.post("/rendezvous")
async def rendezvous():
    sid = str(uuid.uuid4())
    ms_seed = secrets.token_hex(32)
    os_seed = secrets.token_hex(32)
    await emit("rendezvous", {"sid": sid})
    await emit("seeds_issued", {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed})
    return {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed, "ttl": 300}

class LinkRegisterBody(BaseModel):
    header: Dict[str, Any]
    fp_hint: int

@app.post("/link/register")
async def link_register(body: LinkRegisterBody):
    # Try to thread sid if present in header meta (optional)
    sid = body.header.get("meta", {}).get("sid")
    payload = {"header": body.header, "fp_hint": body.fp_hint}
    if sid:
        payload["sid"] = sid
    await emit("link_register", payload)
    return {"ok": True}

@app.post("/policy/check")
async def policy_check(body: Dict[str, Any] = Body(...)):
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
    payload = {"header": {"link_id": link_id, "type": typ}, "ticket": ticket}
    if sid:
        payload["sid"] = sid
    await emit("policy_decision", payload)
    return ticket

# --------- generic event push (tunnel status, upload progress, etc.) ----------
@app.post("/event")
async def push_event(event: Dict[str, Any] = Body(...)):
    # Expecting { "type": "...", "payload": {...} }
    if "type" not in event:
        return {"ok": False, "error": "missing type"}
    await emit(event["type"], event.get("payload", {}))
    return {"ok": True}

# --------- session completion marker ----------
class SessionCompleteBody(BaseModel):
    sid: str

@app.post("/session/complete")
async def session_complete(body: SessionCompleteBody):
    sid = body.sid
    await emit("session_complete", {"sid": sid})
    # Optionally could rotate/close logs here if needed
    return {"ok": True, "sid": sid}

# --------- LOG retrieval endpoints (text/plain JSONL tails) ----------
def tail_file(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
    # Efficient tail
    lines: List[str] = []
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        block = -1024
        data = b""
        while len(lines) <= max_lines and -block < size:
            f.seek(block, 2)
            data = f.read(min(-block, size))
            lines = data.splitlines()
            block *= 2
    text = b"\n".join(lines[-max_lines:]).decode("utf-8", errors="ignore")
    return text

@app.get("/logs")
async def get_global_log(lines: int = Query(500, ge=1, le=10000)):
    text = tail_file(GLOBAL_LOG, lines)
    return ResponseWithText(text)

@app.get("/logs/session/{sid}")
async def get_session_log(sid: str, lines: int = Query(500, ge=1, le=10000)):
    path = session_log_path(sid)
    text = tail_file(path, lines)
    return ResponseWithText(text)

# Small helper to return text/plain
from fastapi.responses import PlainTextResponse as ResponseWithText

# --------- WS ----------
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        await ws.send_text(json.dumps({"id": 0, "ts": now_iso(), "type": "ws_hello", "payload": {"msg": "connected"}}))
        while True:
            await ws.receive_text()  # keep alive, ignore data
    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)
