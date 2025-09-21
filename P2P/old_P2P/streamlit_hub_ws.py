# fastapi_hub_ws.py
# Run: python -m uvicorn fastapi_hub_ws:app --host 127.0.0.1 --port 8000

import asyncio
import json
import os
import secrets
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

from fastapi import Body, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

# ----------------- config & logging -----------------
LOG_DIR = Path("../logs")
LOG_DIR.mkdir(exist_ok=True)
GLOBAL_LOG = LOG_DIR / "events.jsonl"

# default: permissive; set POLICY_ENFORCE=1 to actually block
POLICY_ENFORCE = os.getenv("POLICY_ENFORCE", "0") == "1"

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------- app -----------------
app = FastAPI(title="WARL0K Hub (WS + Logs)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

EVENTS: deque = deque(maxlen=2000)
EVENT_ID = 0
clients: Set[WebSocket] = set()
clients_lock = asyncio.Lock()
event_lock = asyncio.Lock()

async def _record(evt: Dict[str, Any]) -> Dict[str, Any]:
    """Assign ID, store in ring buffer, append to file."""
    global EVENT_ID
    async with event_lock:
        EVENT_ID += 1
        full = {"id": EVENT_ID, **evt}
        EVENTS.append(full)
    try:
        jsonl_append(GLOBAL_LOG, full)
    except Exception as e:
        print("[log warn]", e)
    return full

async def _broadcast(evt: Dict[str, Any]) -> None:
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

async def emit(evt_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    evt = {"ts": now_iso(), "type": evt_type, "payload": payload}
    full = await _record(evt)
    await _broadcast(full)
    return full

# ----------------- helpers -----------------
def _tail_text(path: Path, max_lines: int) -> str:
    if not path.exists():
        return ""
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
    return b"\n".join(lines[-max_lines:]).decode("utf-8", errors="ignore")

# ----------------- HTTP endpoints -----------------
@app.get("/health")
def health():
    return {"ok": True, "ts": now_iso(), "enforce": POLICY_ENFORCE}

@app.get("/events")
def events(since: int = Query(0)):
    data = [e for e in EVENTS if e["id"] > since]
    nxt = EVENTS[-1]["id"] if EVENTS else since
    return {"ok": True, "events": data, "next": nxt}

@app.get("/logs", response_class=PlainTextResponse)
def logs(lines: int = Query(500, ge=1, le=10000)):
    return _tail_text(GLOBAL_LOG, lines)

# Rendezvous (GET + POST) — identical behavior on both
async def _do_rendezvous():
    sid = str(uuid.uuid4())
    ms_seed = secrets.token_hex(32)  # inner/payload seed
    os_seed = secrets.token_hex(32)  # outer/policy seed
    await emit("rendezvous", {"sid": sid})
    await emit("seeds_issued", {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed})
    return {"sid": sid, "ms_seed": ms_seed, "os_seed": os_seed, "ttl": 300}

@app.post("/rendezvous")
async def rendezvous_post():
    return await _do_rendezvous()

@app.get("/rendezvous")
async def rendezvous_get():
    return await _do_rendezvous()

@app.post("/link/register")
async def link_register(body: Dict[str, Any] = Body(...)):
    # Expect {header:{ link_id, type, meta, ... }, policy_tag_len?: int}
    await emit("link_register", body)
    return {"ok": True}

@app.post("/policy/check")
async def policy_check(body: Dict[str, Any] = Body(...)):
    """
    Expect: { sid, link_id, type, meta }
      - type='msg'  -> meta: {chars:int}
      - type='file' -> meta: {size:int}
    Policy (dry-run unless POLICY_ENFORCE=1):
      - messages > 2000 chars
      - files > 50 MB
    """
    typ = body.get("type")
    meta = body.get("meta", {})
    sid = body.get("sid")
    link_id = body.get("link_id")

    would_block = False
    reason = "ok"

    if typ == "msg" and int(meta.get("chars", 0)) > 2000:
        would_block, reason = True, "message too long"
    elif typ == "file" and int(meta.get("size", 0)) > 50_000_000:
        would_block, reason = True, "file too large"

    ok = not (POLICY_ENFORCE and would_block)

    ticket = {
        "ok": ok,
        "reason": reason,
        "would_block": would_block,
        "ticket": {"sid": sid, "link_id": link_id, "ttl": 120} if ok else None
    }

    await emit("policy_decision", {
        "sid": sid, "link_id": link_id, "type": typ,
        "enforce": POLICY_ENFORCE, "ticket": ticket
    })
    return ticket

@app.post("/event")
async def event_push(body: Dict[str, Any] = Body(...)):
    # Accepts generic events for monitoring: { type, payload }
    await emit(body.get("type", "custom"), body.get("payload", {}))
    return {"ok": True}

# ----------------- WebSocket with keepalive -----------------
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    async with clients_lock:
        clients.add(ws)
    try:
        # welcome
        await ws.send_text(json.dumps({"id": 0, "ts": now_iso(), "type": "ws_hello", "payload": {"msg": "connected"}}))

        async def recv_loop():
            # drain incoming frames to keep the connection healthy
            while True:
                try:
                    await ws.receive_text()
                except WebSocketDisconnect:
                    break

        async def keepalive():
            # send heartbeat every 20s so intermediaries don't idle-timeout
            while True:
                await asyncio.sleep(20)
                try:
                    await ws.send_text(json.dumps({"id": 0, "ts": now_iso(), "type": "ka", "payload": {}}))
                except WebSocketDisconnect:
                    break

        await asyncio.gather(recv_loop(), keepalive())

    except WebSocketDisconnect:
        pass
    finally:
        async with clients_lock:
            clients.discard(ws)
