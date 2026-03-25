
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub

app = FastAPI()
hsm = DemoHSM(b"root")
hub = Hub(hsm)

@app.exception_handler(Exception)
async def handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"status":"error","message":str(exc)})

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/create_block")
def create(data: dict):
    h = hub.create_block(data.get("parents",[]), data.get("payload",""), data.get("metadata",{}))
    return {"status":"ok","hash":h}

@app.post("/validate")
def validate(data: dict):
    ok,msg = hub.validate(data.get("hash"))
    return {"status":"ok","valid":ok,"message":msg}

@app.get("/dag")
def dag():
    return {"status":"ok","nodes":{h:{"parents":b.parents} for h,b in hub.dag.nodes.items()}}
