
import time
from core.events import log_event
agents=["Collector","Analyzer","Decision","Executor"]
def run_pipeline(hub):
    prev=None
    created=[]
    for agent in agents:
        payload=f"{agent}-{time.time()}"
        b=hub.create_block([prev] if prev else [],payload,{"agent":agent})
        prev=b.hash
        created.append(b.hash)
        log_event("PIPELINE_STEP",{"agent":agent,"hash":b.hash})
    return created
