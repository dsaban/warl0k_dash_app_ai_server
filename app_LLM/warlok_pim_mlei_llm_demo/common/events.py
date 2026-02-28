# common/events.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class Event:
    ts_utc: str
    actor: str              # GVORN_HUB / SESSION_GATEWAY / EXECUTION_GUARD / CLOUD_PLANNER / ATTACK / UI
    phase: str              # HUB / HANDSHAKE / STEP / REPLY / AUDIT
    kind: str               # semantic event name (stable)
    step: int               # 0 for handshake+hub, 1..N for steps
    status: str             # INFO / ALLOW / BLOCK / OK / DROP / ATTACK / RUN
    msg: str = ""
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d["data"] is None:
            d["data"] = {}
        return d

def status_color(status: str) -> str:
    s = (status or "").upper()
    if s in ("OK", "ALLOW"):
        return "green"
    if s in ("DROP",):
        return "purple"
    if s in ("BLOCK",):
        return "red"
    if s in ("ATTACK",):
        return "orange"
    return "gray"
