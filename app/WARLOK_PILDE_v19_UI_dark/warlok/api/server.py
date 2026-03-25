from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, sys

# ── Path resolution ───────────────────────────────────────────────────────────
# server.py may live at:
#   warlok/server.py          → __file__ parent IS the project root
#   warlok/api/server.py      → __file__ parent is api/, project root is one up
# We search both locations so the file works wherever it is placed.
_here = os.path.dirname(os.path.abspath(__file__))
_candidates = [_here, os.path.dirname(_here)]

def _find_dir(name: str) -> str:
    for base in _candidates:
        path = os.path.join(base, name)
        if os.path.isdir(path):
            return path
    raise RuntimeError(
        f"Could not find '{name}/' directory. Searched:\n" +
        "\n".join(f"  {os.path.join(b, name)}" for b in _candidates)
    )

# Add project root to sys.path so core/hsm/hub imports always resolve
for base in _candidates:
    if base not in sys.path:
        sys.path.insert(0, base)

from hsm.demo_hsm import DemoHSM
from hub.controller import Hub
from core.events import get_events
from core.simulator import run_pipeline
from core.attack import tamper, break_signature

app = FastAPI(title="WARLOK", version="2.0")

hsm = DemoHSM(b"root")
hub = Hub(hsm)

# Mount static files — resolves correctly from any server.py location
static_dir = _find_dir("static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0"}


@app.post("/create_block")
def create(data: dict):
    b = hub.create_block(data.get("parents", []), data.get("payload", ""), data.get("meta", {}))
    return {"hash": b.hash, "status": b.status, "timestamp": b.timestamp}


@app.post("/pipeline")
def pipeline():
    created = run_pipeline(hub)
    return {"created": created, "count": len(created)}


@app.post("/attack/tamper")
def attack_t(data: dict):
    tamper(hub, data.get("hash"))
    return {"status": "tampered", "hash": data.get("hash")}


@app.post("/attack/signature")
def attack_s(data: dict):
    break_signature(hub, data.get("hash"))
    return {"status": "broken", "hash": data.get("hash")}


@app.post("/validate")
def validate(data: dict):
    ok, msg = hub.validate(data.get("hash"))
    return {"valid": ok, "msg": msg, "hash": data.get("hash")}


@app.get("/dag")
def dag():
    return {
        "nodes": {
            h: {
                "parents": b.parents,
                "status": b.status,
                "agent": b.meta.get("agent", ""),
                "timestamp": b.timestamp,
                "hash": b.hash,
            }
            for h, b in hub.dag.nodes.items()
        },
        "count": len(hub.dag.nodes),
    }


@app.get("/events")
def events():
    return {"events": get_events(), "count": len(get_events())}


@app.post("/reset")
def reset():
    global hub, hsm
    from core.events import events as ev_list
    ev_list.clear()
    hsm = DemoHSM(b"root")
    hub = Hub(hsm)
    return {"status": "reset"}
