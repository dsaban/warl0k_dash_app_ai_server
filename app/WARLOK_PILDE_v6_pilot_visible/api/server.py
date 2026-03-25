
from fastapi import FastAPI
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub
from core.events import get_events

app=FastAPI()
hsm=DemoHSM(b"root")
hub=Hub(hsm)

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/create_block")
def create(data:dict):
    return {"hash":hub.create_block(data.get("parents",[]),data.get("payload",""),data.get("metadata",{}))}

@app.post("/validate")
def validate(data:dict):
    ok,msg=hub.validate(data.get("hash"))
    return {"valid":ok,"msg":msg}

@app.get("/dag")
def dag():
    return {"nodes":{h:{"parents":b.parents} for h,b in hub.dag.nodes.items()}}

@app.get("/events")
def events():
    return {"events":get_events()}
