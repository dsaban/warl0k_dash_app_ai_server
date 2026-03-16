"""
WARL0K PIM Web Server — FastAPI edition
REST API + SSE training stream + encrypted chain verification
"""

import os, sys, json, time, asyncio, threading
from contextlib import asynccontextmanager
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(__file__))
import pim_core as pim

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.npz")

# ── Global state ──────────────────────────────────────────────────────────
_state = {
    "params":   None,
    "session":  None,
    "training": False,
    "progress": [],
    "done":     False,
    "error":    None,
    "train_s":  0.0,
}
_lock = threading.Lock()


def _push(msg: dict):
    with _lock:
        _state["progress"].append(json.dumps(msg))

def _make_session(p):
    s = pim.PIMSession(p)
    _state["params"]  = p
    _state["session"] = s
    return s


# ── Lifespan context manager ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    if os.path.exists(MODEL_PATH):
        try:
            p, _ = pim.load_model(MODEL_PATH)
            _make_session(p)
            print(f"[startup] Loaded model from {MODEL_PATH}")
        except Exception as exc:
            print(f"[startup] Could not load model: {exc}")
    print("WARL0K PIM FastAPI server ready — http://localhost:5050")
    yield
    # shutdown — nothing to clean up


app = FastAPI(title="WARL0K PIM Engine", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Pydantic models ───────────────────────────────────────────────────────
class VerifyRequest(BaseModel):
    claimed_id: int = Field(0, ge=0)
    expected_w: int = Field(0, ge=0)
    tamper: str = Field("none")

class EncryptRequest(BaseModel):
    claimed_id: int = Field(0, ge=0)
    expected_w: int = Field(0, ge=0)

class BenchmarkRequest(BaseModel):
    n: int = Field(50, ge=1, le=500)


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    path = os.path.join(STATIC_DIR, "index.html")
    with open(path, encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


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
    }


@app.get("/api/train/stream")
async def train_stream():
    async def generator():
        sent = 0
        while True:
            with _lock:
                msgs = _state["progress"][sent:]
                done = _state["done"]
                err  = _state["error"]
            for m in msgs:
                yield f"data: {m}\n\n"
                sent += 1
            if done or err:
                payload = json.dumps({
                    "type": "done",
                    "error": err,
                    "train_s": _state["train_s"],
                })
                yield f"data: {payload}\n\n"
                break
            await asyncio.sleep(0.15)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/train")
async def start_train():
    with _lock:
        if _state["training"]:
            raise HTTPException(status_code=400, detail="Already training")
        _state["training"] = True
        _state["done"]     = False
        _state["error"]    = None
        _state["progress"].clear()

    def run():
        try:
            ds = pim.build_dataset()
            _push({"type": "log", "msg": f"Dataset: {ds['X'].shape[0]} samples"})
            p = pim.init_params()
            _push({"type": "log", "msg": f"Model: {p.param_bytes()//1024:.1f} KB"})
            t0 = time.time()
            pim.train_phase1(p, ds, cb=lambda info: _push({"type": "progress", **info}))
            pim.train_phase2(p, ds, cb=lambda info: _push({"type": "progress", **info}))
            train_s = time.time() - t0
            _make_session(p)
            pim.save_model(MODEL_PATH, p, {"train_s": train_s})
            _push({"type": "log", "msg": f"Training done in {train_s:.2f}s — model saved"})
            with _lock:
                _state["train_s"] = train_s
                _state["done"]    = True
        except Exception as exc:
            with _lock:
                _state["error"] = str(exc)
                _state["done"]  = True
        finally:
            with _lock:
                _state["training"] = False

    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.post("/api/load")
async def load_model():
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="No saved model — train first")
    try:
        p, meta = pim.load_model(MODEL_PATH)
        _make_session(p)
        return {"ok": True, "meta": meta, "param_kb": p.param_bytes() // 1024}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/verify")
async def verify(body: VerifyRequest):
    with _lock:
        session = _state["session"]
    if session is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    NI = pim.CFG["N_IDENTITIES"]; NW = pim.CFG["N_WINDOWS_PER_ID"]; T = pim.CFG["SEQ_LEN"]
    cid = max(0, min(body.claimed_id, NI - 1))
    ew  = body.expected_w
    ms  = pim.MS_ALL[cid]
    toks, meas = pim.generate_os_chain(ms, cid * NW + ew)

    if body.tamper == "shuffle":
        idx = np.array(pim.XorShift32(0xABCD).shuffle(list(range(T))))
        toks = toks[idx]; meas = meas[idx]
    elif body.tamper == "truncate":
        t2 = np.zeros(T, np.int32); m2 = np.zeros(T, np.float32)
        t2[:T//2] = toks[:T//2];    m2[:T//2] = meas[:T//2]
        toks = t2; meas = m2
    elif body.tamper == "wrong_win":
        ww = (ew + 7) % NW
        toks, meas = pim.generate_os_chain(ms, cid * NW + ww)
    elif body.tamper == "wrong_id":
        oid = (cid + 1) % NI
        toks, meas = pim.generate_os_chain(pim.MS_ALL[oid], oid * NW + 13)
    elif body.tamper == "oob":
        ew = 9999

    result = session.verify(cid, ew, toks, meas, ms)
    result["tamper"] = body.tamper
    result["gates"]  = {k: bool(v) for k, v in result.get("gates", {}).items()}
    return result


@app.post("/api/encrypt")
async def encrypt_demo(body: EncryptRequest):
    with _lock:
        session = _state["session"]
    if session is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    NI = pim.CFG["N_IDENTITIES"]; NW = pim.CFG["N_WINDOWS_PER_ID"]
    cid = max(0, min(body.claimed_id, NI - 1))
    w   = max(0, min(body.expected_w,  NW - 1))
    toks, meas = pim.generate_os_chain(pim.MS_ALL[cid], cid * NW + w)
    pkg = session.encrypt_tokens(toks, meas)
    try:
        t2, m2 = session.decrypt_tokens(pkg)
        rt_ok  = bool(np.array_equal(t2, toks) and np.allclose(m2, meas))
    except Exception:
        rt_ok = False
    return {"encrypted": pkg, "roundtrip_ok": rt_ok, "has_crypto": pim.HAS_CRYPTO}


@app.get("/api/chain")
async def chain_status():
    with _lock:
        session = _state["session"]
    if session is None:
        raise HTTPException(status_code=400, detail="No session active")
    return {**session.chain_status(), "recent_events": session.chain.events[-10:]}


@app.post("/api/benchmark")
async def benchmark(body: BenchmarkRequest):
    with _lock:
        session = _state["session"]
    if session is None:
        raise HTTPException(status_code=400, detail="Model not loaded")

    toks, meas = pim.generate_os_chain(pim.MS_ALL[0], 0)
    times = []
    for _ in range(body.n):
        t0 = time.perf_counter()
        pim.verify_chain(session.p, toks, meas, 0, 0)
        times.append((time.perf_counter() - t0) * 1e6)
    arr = np.array(times)
    return {
        "n": body.n,
        "mean_us": float(arr.mean()), "min_us": float(arr.min()),
        "max_us":  float(arr.max()),  "p95_us": float(np.percentile(arr, 95)),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=5050, reload=False)
