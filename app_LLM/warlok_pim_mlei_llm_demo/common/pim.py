from dataclasses import dataclass
from typing import Dict, Any, Tuple

from .util import now_ts, sha256_hex, canon_json

@dataclass
class PIMState:
    session_id: str
    counter: int = 0
    last_hash: str = "GENESIS"

def build_pim_envelope(state: PIMState, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Envelope fields are integrity-protected through hash chaining.
    """
    ts = now_ts()
    counter = state.counter + 1

    core = {
        "sid": state.session_id,
        "ctr": counter,
        "ts": ts,
        "prev": state.last_hash,
        "payload": payload,
    }
    h = sha256_hex(canon_json(core))
    env = dict(core)
    env["h"] = h
    return env

def advance_state(state: PIMState, env: Dict[str, Any]) -> None:
    state.counter = env["ctr"]
    state.last_hash = env["h"]

def verify_pim_envelope(
    state: PIMState,
    env: Dict[str, Any],
    max_skew_s: float,
) -> Tuple[bool, str]:
    """
    Checks:
      - session id matches
      - counter is strictly incrementing by 1
      - prev hash matches local last_hash
      - timestamp within skew window
      - computed hash matches env["h"]
    """
    if env.get("sid") != state.session_id:
        return False, "PIM: session_id mismatch"

    exp_ctr = state.counter + 1
    if env.get("ctr") != exp_ctr:
        return False, f"PIM: counter mismatch (expected {exp_ctr}, got {env.get('ctr')})"

    if env.get("prev") != state.last_hash:
        return False, "PIM: prev-hash mismatch"

    ts = float(env.get("ts", 0.0))
    skew = abs(now_ts() - ts)
    if skew > max_skew_s:
        return False, f"PIM: timestamp skew too large ({skew:.3f}s)"

    # recompute hash
    core = {
        "sid": env["sid"],
        "ctr": env["ctr"],
        "ts": env["ts"],
        "prev": env["prev"],
        "payload": env["payload"],
    }
    h2 = sha256_hex(canon_json(core))
    if h2 != env.get("h"):
        return False, "PIM: hash mismatch"

    return True, "OK"
