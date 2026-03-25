
from fastapi import FastAPI
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub
from core.events import get_events
from core.simulator import run_pipeline
from core.attack import tamper, break_signature

app=FastAPI()
hsm=DemoHSM(b"root")
hub=Hub(hsm)

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/create_block")
def create(data:dict):
    b=hub.create_block(data.get("parents",[]),data.get("payload",""),{})
    return {"hash":b.hash}

@app.post("/pipeline")
def pipeline():
    return {"created":run_pipeline(hub)}

@app.post("/attack/tamper")
def attack_t(data:dict):
    tamper(hub,data.get("hash"))
    return {"status":"tampered"}

@app.post("/attack/signature")
def attack_s(data:dict):
    break_signature(hub,data.get("hash"))
    return {"status":"broken"}

@app.post("/validate")
def validate(data:dict):
    ok,msg=hub.validate(data.get("hash"))
    return {"valid":ok,"msg":msg}

@app.get("/dag")
def dag():
    return {"nodes":{
        h:{
            "parents":b.parents,
            "status":b.status,
            "agent":b.meta.get("agent","")
        } for h,b in hub.dag.nodes.items()
    }}

@app.get("/events")
def events():
    return {"events":get_events()}
