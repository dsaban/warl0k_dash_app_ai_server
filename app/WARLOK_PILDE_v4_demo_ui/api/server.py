
from fastapi import FastAPI
from hsm.demo_hsm import DemoHSM
from hub.controller import Hub

app = FastAPI()

hsm = DemoHSM(b"root")
hub = Hub(hsm)

@app.post("/create_block")
def create_block(data: dict):
    h = hub.create_block(data.get("parents", []), data.get("payload",""), data.get("metadata", {}))
    return {"hash": h}

@app.post("/validate")
def validate(data: dict):
    return {"result": hub.validate(data.get("hash"))}

@app.get("/dag")
def dag():
    return {
        h: {"parents": b.parents}
        for h, b in hub.dag.nodes.items()
    }
