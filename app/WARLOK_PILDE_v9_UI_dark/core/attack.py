
from core.events import log_event
def tamper(hub,h):
    b=hub.dag.get(h)
    if b:
        b.payload+=b"tampered"
        b.status="INVALID"
        log_event("ATTACK","tamper")
def break_signature(hub,h):
    b=hub.dag.get(h)
    if b:
        b.signature=b"bad"
        b.status="INVALID"
        log_event("ATTACK","signature")
