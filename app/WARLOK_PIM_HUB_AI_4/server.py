"""
WARL0K PIM Server v2 — FastAPI + Gunicorn Hub
Dual-peer mutual handshake, 48-OS MS reconstruction,
AES anchor transfer, time-delta chain with counters
"""

import os, sys, json, time, asyncio, threading, uuid
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
import pim_core as pim

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.npz")

# ═══════════════════════════════════════════════════════════════════
# GLOBAL STATE — shared across Gunicorn workers via in-process dict
# (for multi-worker Gunicorn use Redis/memcached; for single-worker this is fine)
# ═══════════════════════════════════════════════════════════════════
_lock = threading.Lock()

_state = {
    "params":   None,   # Params — shared model
    "session":  None,   # PIMSession — legacy session
    "training": False,
    "progress": [],
    "done":     False,
    "error":    None,
    "train_s":  0.0,
}

# Peer hub: peer_id -> PeerSession
_peers: dict = {}

def _push(msg: dict):
    with _lock: _state["progress"].append(json.dumps(msg))

def _make_session(p):
    s = pim.PIMSession(p)
    _state["params"] = p; _state["session"] = s
    return s

# ═══════════════════════════════════════════════════════════════════
# LIFESPAN
# ═══════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.exists(MODEL_PATH):
        try:
            p, _ = pim.load_model(MODEL_PATH)
            _make_session(p)
            print(f"[startup] Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"[startup] Could not load model: {e}")
    print("WARL0K PIM v2 — http://localhost:5051  |  docs: /docs")
    yield

app = FastAPI(title="WARL0K PIM Engine v2", version="2.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ═══════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════
class VerifyReq(BaseModel):
    claimed_id: int = Field(0, ge=0)
    expected_w: int = Field(0, ge=0)
    tamper:     str = Field("none")

class EncryptReq(BaseModel):
    claimed_id: int = Field(0, ge=0)
    expected_w: int = Field(0, ge=0)

class BenchReq(BaseModel):
    n: int = Field(50, ge=1, le=500)

class PeerInitReq(BaseModel):
    peer_id:     str = Field(..., description="Unique peer identifier")
    identity_id: int = Field(0, ge=0, description="Which identity this peer claims")
    session_key: Optional[str] = Field(None, description="Hex session key (optional, generated if absent)")

class AnchorExchangeReq(BaseModel):
    peer_id:        str = Field(..., description="Peer submitting its anchor")
    remote_peer_id: str = Field(..., description="Peer whose anchor to verify against")

class PeerVerifyReq(BaseModel):
    peer_id:    str
    claimed_id: int = Field(0, ge=0)
    expected_w: int = Field(0, ge=0)
    tamper:     str = Field("none")

# ═══════════════════════════════════════════════════════════════════
# STATIC / INDEX
# ═══════════════════════════════════════════════════════════════════
@app.get("/", response_class=HTMLResponse)
async def index():
    with open(os.path.join(STATIC_DIR, "index.html"), encoding="utf-8") as f:
        return HTMLResponse(f.read())

# ═══════════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/status")
async def status():
    with _lock:
        loaded   = _state["params"] is not None
        training = _state["training"]
        done     = _state["done"]
        err      = _state["error"]
    return {
        "model_loaded":  loaded,
        "training":      training,
        "done":          done,
        "error":         err,
        "has_crypto":    pim.HAS_CRYPTO,
        "model_on_disk": os.path.exists(MODEL_PATH),
        "cfg":           pim.CFG,
        "active_peers":  list(_peers.keys()),
        "peer_count":    len(_peers),
    }

# ═══════════════════════════════════════════════════════════════════
# TRAIN (SSE stream)
# ═══════════════════════════════════════════════════════════════════
@app.get("/api/train/stream")
async def train_stream():
    async def gen():
        sent = 0
        while True:
            with _lock:
                msgs = _state["progress"][sent:]
                done = _state["done"]; err = _state["error"]
            for m in msgs:
                yield f"data: {m}\n\n"; sent += 1
            if done or err:
                yield f"data: {json.dumps({'type':'done','error':err,'train_s':_state['train_s']})}\n\n"
                break
            await asyncio.sleep(0.15)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.post("/api/train")
async def start_train():
    with _lock:
        if _state["training"]: raise HTTPException(400, "Already training")
        _state["training"]=True; _state["done"]=False
        _state["error"]=None;    _state["progress"].clear()

    def run():
        try:
            ds = pim.build_dataset()
            _push({"type":"log","msg":f"Dataset: {ds['X'].shape[0]} samples"})
            p = pim.init_params()
            _push({"type":"log","msg":f"Model: {p.param_bytes()//1024:.1f} KB"})
            t0 = time.time()
            pim.train_phase1(p, ds, cb=lambda i: _push({"type":"progress",**i}))
            pim.train_phase2(p, ds, cb=lambda i: _push({"type":"progress",**i}))
            ts = time.time()-t0
            _make_session(p)
            pim.save_model(MODEL_PATH, p, {"train_s":ts})
            _push({"type":"log","msg":f"Done in {ts:.2f}s — saved"})
            with _lock: _state["train_s"]=ts; _state["done"]=True
        except Exception as e:
            with _lock: _state["error"]=str(e); _state["done"]=True
        finally:
            with _lock: _state["training"]=False

    threading.Thread(target=run, daemon=True).start()
    return {"status":"started"}

@app.post("/api/load")
async def load_model_ep():
    if not os.path.exists(MODEL_PATH): raise HTTPException(404, "No saved model — train first")
    try:
        p, meta = pim.load_model(MODEL_PATH)
        _make_session(p)
        return {"ok":True,"meta":meta,"param_kb":p.param_bytes()//1024}
    except Exception as e: raise HTTPException(500, str(e))

# ═══════════════════════════════════════════════════════════════════
# LEGACY VERIFY / ENCRYPT / CHAIN / BENCHMARK
# ═══════════════════════════════════════════════════════════════════
@app.post("/api/verify")
async def verify(body: VerifyReq):
    with _lock: session = _state["session"]
    if session is None: raise HTTPException(400, "Model not loaded")
    NI=pim.CFG["N_IDENTITIES"]; NW=pim.CFG["N_WINDOWS_PER_ID"]; T=pim.CFG["SEQ_LEN"]
    cid=max(0,min(body.claimed_id,NI-1)); ew=body.expected_w
    ms=pim.MS_ALL[cid]; toks,meas=pim.generate_os_chain(ms,cid*NW+ew)
    if body.tamper=="shuffle":
        idx=np.array(pim.XorShift32(0xABCD).shuffle(list(range(T)))); toks=toks[idx]; meas=meas[idx]
    elif body.tamper=="truncate":
        t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
        t2[:T//2]=toks[:T//2]; m2[:T//2]=meas[:T//2]; toks=t2; meas=m2
    elif body.tamper=="wrong_win":
        toks,meas=pim.generate_os_chain(ms,cid*NW+(ew+7)%NW)
    elif body.tamper=="wrong_id":
        oid=(cid+1)%NI; toks,meas=pim.generate_os_chain(pim.MS_ALL[oid],oid*NW+13)
    elif body.tamper=="oob": ew=9999
    result=session.verify(cid,ew,toks,meas,ms)
    result["tamper"]=body.tamper
    result["gates"]={k:bool(v) for k,v in result.get("gates",{}).items()}
    return result

@app.post("/api/encrypt")
async def encrypt_ep(body: EncryptReq):
    with _lock: session=_state["session"]
    if session is None: raise HTTPException(400,"Model not loaded")
    NI=pim.CFG["N_IDENTITIES"]; NW=pim.CFG["N_WINDOWS_PER_ID"]
    cid=max(0,min(body.claimed_id,NI-1)); w=max(0,min(body.expected_w,NW-1))
    toks,meas=pim.generate_os_chain(pim.MS_ALL[cid],cid*NW+w)
    pkg=session.encrypt_tokens(toks,meas)
    try: t2,m2=session.decrypt_tokens(pkg); rt=bool(np.array_equal(t2,toks) and np.allclose(m2,meas))
    except: rt=False
    return {"encrypted":pkg,"roundtrip_ok":rt,"has_crypto":pim.HAS_CRYPTO}

@app.get("/api/chain")
async def chain_ep():
    with _lock: session=_state["session"]
    if session is None: raise HTTPException(400,"No session")
    return session.chain_status()

@app.post("/api/benchmark")
async def benchmark(body: BenchReq):
    with _lock: session=_state["session"]
    if session is None: raise HTTPException(400,"Model not loaded")
    toks,meas=pim.generate_os_chain(pim.MS_ALL[0],0)
    times=[]
    for _ in range(body.n):
        t0=time.perf_counter(); pim.verify_chain(session.p,toks,meas,0,0)
        times.append((time.perf_counter()-t0)*1e6)
    arr=np.array(times)
    return {"n":body.n,"mean_us":float(arr.mean()),"min_us":float(arr.min()),
            "max_us":float(arr.max()),"p95_us":float(np.percentile(arr,95))}

# ═══════════════════════════════════════════════════════════════════
# ══ PEER HUB — Dual-peer mutual handshake endpoints ══
# ═══════════════════════════════════════════════════════════════════

@app.post("/api/peer/init")
async def peer_init(body: PeerInitReq):
    """
    Register a new peer in the hub.
    Creates a PeerSession with identity_id and optional shared session_key.
    Both peers must call this with the SAME session_key to share a key.
    """
    with _lock: params = _state["params"]
    if params is None: raise HTTPException(400, "Model not loaded — train or load first")

    NI = pim.CFG["N_IDENTITIES"]
    if not (0 <= body.identity_id < NI):
        raise HTTPException(400, f"identity_id must be in [0, {NI-1}]")

    key = bytes.fromhex(body.session_key) if body.session_key else None
    peer = pim.PeerSession(
        peer_id=body.peer_id,
        params=params,
        identity_id=body.identity_id,
        session_key=key,
    )
    with _lock:
        _peers[body.peer_id] = peer

    return {
        "ok":          True,
        "peer_id":     body.peer_id,
        "identity_id": body.identity_id,
        "session_key": peer.key.hex(),   # return so other peer can use same key
        "model_hash":  peer.model_hash[:16],
        "msg":         "Peer registered. Call /api/peer/reconstruct next.",
    }


@app.post("/api/peer/reconstruct")
async def peer_reconstruct(peer_id: str):
    """
    Run 48-OS chain inference for this peer's identity.
    Produces the MS reconstruction used as the anchor.
    This is the core proof step: 48 independent OS chains reconstruct the same MS.
    """
    with _lock: peer = _peers.get(peer_id)
    if peer is None: raise HTTPException(404, f"Peer '{peer_id}' not found")

    def run_recon():
        result = peer.reconstruct()
        return result

    # Run in thread (CPU-heavy)
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_recon)

    return {
        "peer_id":       peer_id,
        "identity_id":   peer.identity_id,
        "accepted":      result["accepted"],
        "consensus_l2":  result["consensus_l2"],
        "mean_l2":       result["mean_l2"],
        "min_l2":        result["min_l2"],
        "max_l2":        result["max_l2"],
        "windows_ok":    result["windows_ok"],
        "windows_used":  result["windows_used"],
        "latency_ms":    result["latency_ms"],
        "per_window_l2": result["per_window_l2"],
        "per_window_pv": result["per_window_pv"],
        "chain_seq":     peer.chain.seq,
    }


@app.post("/api/peer/anchor/build")
async def peer_anchor_build(peer_id: str):
    """
    Build the AES-256-GCM encrypted anchor from this peer's MS reconstruction.
    The anchor (commitment + encrypted payload) is ready to send to the remote peer.
    """
    with _lock: peer = _peers.get(peer_id)
    if peer is None: raise HTTPException(404, f"Peer '{peer_id}' not found")
    if peer.ms_recon is None: raise HTTPException(400, "Run /api/peer/reconstruct first")

    pkg = peer.build_anchor()
    return {
        "peer_id":    peer_id,
        "commitment": pkg["commitment"],
        "nonce":      pkg["nonce"],
        "ts":         pkg["ts"],
        "encrypted":  pkg["encrypted"],
        "chain_seq":  peer.chain.seq,
        "msg":        "Anchor built. Exchange with remote peer via /api/peer/anchor/verify",
    }


@app.post("/api/peer/anchor/verify")
async def peer_anchor_verify(body: AnchorExchangeReq):
    """
    Peer `peer_id` verifies the anchor submitted by `remote_peer_id`.
    Both peers must have called reconstruct() and anchor/build first.
    On success: handshake_ok=True — mutual proof of shared MS established.
    """
    with _lock:
        peer        = _peers.get(body.peer_id)
        remote_peer = _peers.get(body.remote_peer_id)

    if peer is None:        raise HTTPException(404, f"Peer '{body.peer_id}' not found")
    if remote_peer is None: raise HTTPException(404, f"Remote peer '{body.remote_peer_id}' not found")
    if remote_peer.anchor is None:
        raise HTTPException(400, f"Remote peer '{body.remote_peer_id}' has not built anchor yet")

    remote_pkg = remote_peer.anchor.to_dict()
    result = peer.verify_remote_anchor(remote_pkg)

    return {
        "peer_id":       body.peer_id,
        "remote_peer_id":body.remote_peer_id,
        "handshake_ok":  result["ok"],
        "l2_distance":   result.get("l2_distance", None),
        "reason":        result.get("reason", "?"),
        "chain_seq":     peer.chain.seq,
        "counter":       peer.chain.counter,
    }


@app.get("/api/peer/handshake/status")
async def peer_handshake_status(peer_id: str):
    """
    Full handshake status for a peer: chain integrity, time-deltas, counters, events.
    """
    with _lock: peer = _peers.get(peer_id)
    if peer is None: raise HTTPException(404, f"Peer '{peer_id}' not found")
    return peer.chain_status()


@app.post("/api/peer/verify")
async def peer_verify(body: PeerVerifyReq):
    """
    Run a standard PIM chain verification as this peer.
    Records to the peer's ChainProof with counter + time-delta.
    """
    with _lock: peer = _peers.get(body.peer_id)
    if peer is None: raise HTTPException(404, f"Peer '{body.peer_id}' not found")

    NI=pim.CFG["N_IDENTITIES"]; NW=pim.CFG["N_WINDOWS_PER_ID"]; T=pim.CFG["SEQ_LEN"]
    cid=max(0,min(body.claimed_id,NI-1)); ew=body.expected_w
    ms=pim.MS_ALL[cid]; toks,meas=pim.generate_os_chain(ms,cid*NW+ew)
    if body.tamper=="shuffle":
        idx=np.array(pim.XorShift32(0xABCD).shuffle(list(range(T)))); toks=toks[idx]; meas=meas[idx]
    elif body.tamper=="truncate":
        t2=np.zeros(T,np.int32); m2=np.zeros(T,np.float32)
        t2[:T//2]=toks[:T//2]; m2[:T//2]=meas[:T//2]; toks=t2; meas=m2
    elif body.tamper=="wrong_win":
        toks,meas=pim.generate_os_chain(ms,cid*NW+(ew+7)%NW)
    elif body.tamper=="wrong_id":
        oid=(cid+1)%NI; toks,meas=pim.generate_os_chain(pim.MS_ALL[oid],oid*NW+13)
    elif body.tamper=="oob": ew=9999

    result=peer.verify(cid,ew,toks,meas,ms)
    result["tamper"]=body.tamper
    result["gates"]={k:bool(v) for k,v in result.get("gates",{}).items()}
    return result


@app.get("/api/peer/hub")
async def hub_status():
    """All active peers + their handshake states."""
    with _lock: peers_snap = dict(_peers)
    out = {}
    for pid, peer in peers_snap.items():
        cs = peer.chain.status()
        out[pid] = {
            "peer_id":      pid,
            "identity_id":  peer.identity_id,
            "recon_done":   peer.ms_recon is not None,
            "anchor_done":  peer.anchor is not None,
            "handshake_ok": peer.handshake_ok,
            "chain_events": cs["events"],
            "chain_valid":  cs["valid"],
            "counter":      cs["counter"],
        }
    return {"peers": out, "count": len(out)}


@app.delete("/api/peer/{peer_id}")
async def peer_delete(peer_id: str):
    with _lock:
        if peer_id not in _peers: raise HTTPException(404, f"Peer '{peer_id}' not found")
        del _peers[peer_id]
    return {"ok": True, "peer_id": peer_id, "msg": "Peer removed from hub"}


# ═══════════════════════════════════════════════════════════════════
# WEBSOCKET — live chain events feed
# ═══════════════════════════════════════════════════════════════════
_ws_clients: list = []
_ws_lock = threading.Lock()

@app.websocket("/ws/chain")
async def ws_chain(ws: WebSocket):
    """
    WebSocket feed: streams new chain events from ALL peers in real-time.
    Client receives JSON: {peer_id, event, proof, seq, counter, delta_ms}
    """
    await ws.accept()
    with _ws_lock: _ws_clients.append(ws)
    try:
        last_seqs = {}   # peer_id -> last seq seen
        while True:
            await asyncio.sleep(0.3)
            with _lock: peers_snap = dict(_peers)
            msgs = []
            for pid, peer in peers_snap.items():
                last = last_seqs.get(pid, 0)
                new  = [e for e in peer.chain.events if e["seq"] > last]
                for e in new:
                    msgs.append({"peer_id":pid, **e})
                    last_seqs[pid] = e["seq"]
            for m in msgs:
                await ws.send_json(m)
    except WebSocketDisconnect:
        with _ws_lock:
            if ws in _ws_clients: _ws_clients.remove(ws)


# ═══════════════════════════════════════════════════════════════════
# GUNICORN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    workers = int(os.environ.get("WORKERS", 1))
    port    = int(os.environ.get("PORT", 5051))
    if workers > 1:
        # Gunicorn mode
        import subprocess, sys
        cmd = [
            sys.executable, "-m", "gunicorn",
            "server:app",
            f"--workers={workers}",
            "--worker-class=uvicorn.workers.UvicornWorker",
            f"--bind=0.0.0.0:{port}",
            "--timeout=300",
            "--graceful-timeout=30",
            "--log-level=info",
        ]
        subprocess.run(cmd)
    else:
        uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
