from dataclasses import dataclass
from typing import Dict, Any

from common.pim import PIMState, verify_pim_envelope, advance_state, build_pim_envelope
from common.nano_gate import NanoGate
from common.protocol import SecureChannel
from db.csv_store import CsvStore
from config import CFG

@dataclass
class FarPeer:
    sid: str
    channel: SecureChannel
    pim: PIMState
    gate: NanoGate
    db: CsvStore

    def __init__(self, sid: str, channel: SecureChannel):
        self.sid = sid
        self.channel = channel
        self.pim = PIMState(session_id=sid)
        self.gate = NanoGate.train_synthetic(seed=11)  # far tuning
        self.db = CsvStore(CFG.db_path)

    def recv_and_execute(self, blob: Dict[str, Any]) -> Dict[str, Any]:
        env = self.channel.open(blob)

        ok, reason = verify_pim_envelope(self.pim, env, max_skew_s=CFG.max_skew_s)
        if not ok:
            return self._reply_error(f"FAR DROP: {reason}")

        payload = env["payload"]

        allowed, score, decision = self.gate.decide(payload, CFG.far_threshold)
        if not allowed:
            advance_state(self.pim, env)  # still advance to keep continuity (policy choice)
            return self._reply_error(f"FAR BLOCK by gate score={score:.3f} ({decision})")

        # Execute allowed tool
        result = self._exec_tool(payload)
        advance_state(self.pim, env)

        reply_payload = {"tool": "result", "text": "ok", "args": {"result": result}}
        reply_env = build_pim_envelope(self.pim, reply_payload)
        advance_state(self.pim, reply_env)
        return self.channel.seal(reply_env)

    def _exec_tool(self, payload: Dict[str, Any]) -> Any:
        tool = payload.get("tool")
        args = payload.get("args") or {}

        if tool == "read_db":
            limit = int(args.get("limit", 5))
            return {"rows": self.db.read_rows(limit=limit)}

        if tool == "write_db":
            row = args.get("row") or {}
            # final server-side policy: must contain required fields
            for k in ["id", "task", "result", "ts"]:
                if k not in row:
                    return {"error": f"missing field {k}"}
            self.db.append_row(row)
            return {"written": True, "row": row}

        if tool == "summarize":
            rows = args.get("rows") or []
            return {"summary": f"{len(rows)} rows, last_id={rows[-1].get('id') if rows else 'n/a'}"}

        if tool == "llm_query":
            q = str(args.get("q", ""))
            return {"answer": f"(mock) model answered safely for: {q[:80]}"}

        return {"error": f"unknown tool {tool}"}

    def _reply_error(self, msg: str) -> Dict[str, Any]:
        reply_payload = {"tool": "error", "text": msg, "args": {}}
        reply_env = build_pim_envelope(self.pim, reply_payload)
        advance_state(self.pim, reply_env)
        return self.channel.seal(reply_env)
