"""
WARLOK API Server v3 — adds peer mesh, storage, cross-validation endpoints.
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os, sys, json

_here       = os.path.dirname(os.path.abspath(__file__))
_candidates = [_here, os.path.dirname(_here)]

def _find_dir(name):
    for base in _candidates:
        p = os.path.join(base, name)
        if os.path.isdir(p): return p
    raise RuntimeError(f"Cannot find '{name}/' in {_candidates}")

for base in _candidates:
    if base not in sys.path: sys.path.insert(0, base)

from demo_hsm   import DemoHSM
from hub.controller import Hub
from core.events    import get_events
from core.simulator import run_pipeline
from core.attack    import tamper, break_signature
from node      import PeerNode
from mesh      import Mesh
from storage.store  import MemoryStore

app = FastAPI(title="WARLOK", version="3.0")

# ── Global state ──────────────────────────────────────────────────────────────
ROOT_KEY = b"warlok-root-secret"

hsm  = DemoHSM(ROOT_KEY)
hub  = Hub(hsm)
mesh = Mesh()

# Boot 3 demo peers
for _pid in ["peer-alpha", "peer-beta", "peer-gamma"]:
    mesh.add_peer(PeerNode(_pid, ROOT_KEY))

# Hub's own storage
hub_store = MemoryStore(peer_id="hub")

static_dir = _find_dir("static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── UI ────────────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return FileResponse(os.path.join(static_dir, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0", "hsm": hsm.status(),
            "peers": mesh.peers}


# ── DAG / blocks ──────────────────────────────────────────────────────────────
@app.post("/create_block")
def create(data: dict):
    b = hub.create_block(data.get("parents", []),
                          data.get("payload", ""),
                          data.get("meta", {}))
    return {"hash": b.hash, "status": b.status,
            "timestamp": b.timestamp, "epoch": b.meta.get("_epoch")}

@app.post("/pipeline")
def pipeline():
    created = run_pipeline(hub)
    return {"created": created, "count": len(created)}

@app.post("/validate")
def validate(data: dict):
    ok, msg = hub.validate(data.get("hash"))
    return {"valid": ok, "msg": msg, "hash": data.get("hash")}

@app.get("/dag")
def dag():
    return {"nodes": {
        h: {"parents": b.parents, "status": b.status,
            "agent": b.meta.get("agent", ""),
            "timestamp": b.timestamp, "hash": b.hash,
            "epoch": b.meta.get("_epoch", 0)}
        for h, b in hub.dag.nodes.items()
    }, "count": len(hub.dag.nodes)}

@app.get("/events")
def events():
    return {"events": get_events(), "count": len(get_events())}

@app.get("/hsm/status")
def hsm_status(): return hsm.status()


# ── Attacks ───────────────────────────────────────────────────────────────────
@app.post("/attack/tamper")
def attack_t(data: dict):
    tamper(hub, data.get("hash"))
    return {"status": "tampered", "hash": data.get("hash")}

@app.post("/attack/signature")
def attack_s(data: dict):
    break_signature(hub, data.get("hash"))
    return {"status": "broken", "hash": data.get("hash")}


# ── Peer mesh ─────────────────────────────────────────────────────────────────
@app.get("/peers")
def list_peers(): return mesh.status()

@app.get("/peers/{peer_id}")
def peer_status(peer_id: str):
    peer = mesh.get_peer(peer_id)
    if not peer: raise HTTPException(404, f"Peer {peer_id} not found")
    return peer.status()

@app.post("/peers/task")
def dispatch_task(data: dict):
    """
    Send a signed task from one peer to another.
    Body: { "from": "peer-alpha", "to": "peer-beta",
            "task": {"type": "...", ...}, "parents": [] }
    """
    try:
        result = mesh.dispatch_task(
            sender_id=data.get("from", "peer-alpha"),
            target_id=data.get("to",   "peer-beta"),
            task=data.get("task", {"type": "echo", "data": "hello"}),
            parents=data.get("parents", [])
        )
        return result
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/peers/replicate")
def replicate_block(data: dict):
    """
    Replicate a block from a source peer to others.
    Body: { "source": "peer-alpha", "hash": "...", "targets": [] }
    """
    try:
        result = mesh.replicate_block(
            source_id=data.get("source", "peer-alpha"),
            block_hash=data["hash"],
            targets=data.get("targets")  # None = all peers
        )
        return {"replicated": result}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))

@app.post("/peers/telemetry")
def send_telemetry(data: dict):
    """
    Store telemetry data on a peer.
    Body: { "from": "peer-alpha", "to": "peer-beta",
            "key": "sensor:01", "data": "hex_string_or_text" }
    """
    raw = data.get("data", "")
    try:
        payload_bytes = bytes.fromhex(raw) if len(raw) % 2 == 0 else raw.encode()
    except ValueError:
        payload_bytes = raw.encode()
    result = mesh.send_telemetry(
        sender_id=data.get("from", "peer-alpha"),
        target_id=data.get("to",   "peer-beta"),
        key=data.get("key", f"telemetry:{len(payload_bytes)}b"),
        data=payload_bytes
    )
    return result

@app.post("/peers/cross_validate")
def cross_validate(data: dict):
    """
    Ask all peers to validate a block hash.
    Body: { "requestor": "peer-alpha", "hash": "..." }
    """
    try:
        return mesh.cross_validate(
            requestor_id=data.get("requestor", "peer-alpha"),
            block_hash=data["hash"]
        )
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


# ── Storage ───────────────────────────────────────────────────────────────────
@app.get("/storage/hub")
def hub_storage_stats(): return hub_store.stats()

@app.post("/storage/hub/put")
def hub_store_put(data: dict):
    key   = data.get("key", "manual")
    value = data.get("value", "").encode()
    receipt = hub_store.put(key, value, meta=data.get("meta", {}))
    return receipt.to_dict()

@app.get("/storage/hub/get/{key}")
def hub_store_get(key: str):
    rec = hub_store.get(key)
    if not rec: raise HTTPException(404, f"Key '{key}' not found")
    return rec.to_dict()

@app.get("/storage/hub/keys")
def hub_store_keys(prefix: str = ""):
    return {"keys": hub_store.list_keys(prefix), "prefix": prefix}

@app.get("/storage/{peer_id}")
def peer_storage_stats(peer_id: str):
    peer = mesh.get_peer(peer_id)
    if not peer: raise HTTPException(404, f"Peer {peer_id} not found")
    return peer.store.stats()

@app.get("/storage/{peer_id}/keys")
def peer_storage_keys(peer_id: str, prefix: str = ""):
    peer = mesh.get_peer(peer_id)
    if not peer: raise HTTPException(404)
    return {"keys": peer.store.list_keys(prefix), "peer_id": peer_id}

@app.get("/storage/{peer_id}/get/{key}")
def peer_store_get(peer_id: str, key: str):
    peer = mesh.get_peer(peer_id)
    if not peer: raise HTTPException(404)
    rec = peer.store.get(key)
    if not rec: raise HTTPException(404, f"Key '{key}' not in {peer_id}")
    return rec.to_dict()


# ── Reset ─────────────────────────────────────────────────────────────────────
@app.post("/reset")
def reset():
    global hub, hsm, mesh, hub_store
    from core.events import events as ev_list
    ev_list.clear()
    hsm       = DemoHSM(ROOT_KEY)
    hub       = Hub(hsm)
    hub_store = MemoryStore(peer_id="hub")
    mesh      = Mesh()
    for _pid in ["peer-alpha", "peer-beta", "peer-gamma"]:
        mesh.add_peer(PeerNode(_pid, ROOT_KEY))
    return {"status": "reset", "peers": mesh.peers}
