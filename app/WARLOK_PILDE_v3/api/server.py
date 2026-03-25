
from fastapi import FastAPI
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub

app = FastAPI()

hsm = DemoHSM(b"root")
hub = Hub(hsm)

@app.post("/create_block")
def create_block(data: dict):
    parents = data.get("parents", [])
    payload = data.get("payload", "")
    metadata = data.get("metadata", {})

    h = hub.create_block(parents, payload, metadata)
    return {"hash": h}

@app.post("/validate")
def validate(data: dict):
    h = data.get("hash")
    ok, msg = hub.validate(h)
    return {"valid": ok, "message": msg}
