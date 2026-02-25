from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Callable

from common.pim import PIMState, build_pim_envelope, advance_state, verify_pim_envelope
from common.nano_gate import NanoGate
from common.protocol import SecureChannel
from cloud.llm_cloud_mock import llm_agent_plan
from config import CFG

@dataclass
class NearPeer:
    sid: str
    channel: SecureChannel
    pim: PIMState
    gate: NanoGate

    def __init__(self, sid: str, channel: SecureChannel):
        self.sid = sid
        self.channel = channel
        self.pim = PIMState(session_id=sid)
        self.gate = NanoGate.train_synthetic(seed=9)  # near tuning

    def login_and_auth(self, user: str, pwd: str) -> bool:
        # Demo-only auth
        return (user == "demo" and pwd == "demo")

    def run_task_flow(
        self,
        far_peer_execute: Callable[[Dict[str, Any]], Dict[str, Any]],
        task_prompt: str,
        inject: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        1) Ask mock LLM for a plan (tool calls)
        2) For each tool call:
           - near gate checks
           - PIM envelope + encryption
           - optional attacker injection (before send)
           - far executes + replies
           - near verifies reply PIM
        """
        transcript: List[Dict[str, Any]] = []

        # initial DB preview is empty at near side; far reads authoritative state
        plan = llm_agent_plan(task_prompt, db_preview=[])

        for step_idx, tool_call in enumerate(plan, start=1):
            allowed, score, decision = self.gate.decide(tool_call, CFG.near_threshold)
            transcript.append({"who": "NEAR", "event": "GATE", "step": step_idx, "decision": decision, "score": round(score, 3), "tool": tool_call.get("tool")})

            if not allowed:
                transcript.append({"who": "NEAR", "event": "BLOCKED_LOCALLY", "reason": f"near gate score={score:.3f}"})
                continue

            env = build_pim_envelope(self.pim, tool_call)
            advance_state(self.pim, env)

            blob = self.channel.seal(env)

            # attacker injection point: can tamper with *payload* (won't match PIM hash) OR with tool_call before envelope
            if inject is not None:
                # Simulate attacker that can alter near-side agent layer *before* sealing
                injected_call = inject(tool_call)
                transcript.append({"who": "ATTACK", "event": "INJECT", "step": step_idx, "mutated_tool": injected_call.get("tool")})

                # Rebuild envelope with injected command
                env2 = build_pim_envelope(self.pim, injected_call)
                advance_state(self.pim, env2)
                blob = self.channel.seal(env2)

            reply_blob = far_peer_execute(blob)
            reply_env = self.channel.open(reply_blob)

            ok, reason = verify_pim_envelope(self.pim, reply_env, max_skew_s=CFG.max_skew_s)
            transcript.append({"who": "NEAR", "event": "REPLY_VERIFY", "step": step_idx, "ok": ok, "reason": reason, "reply_tool": reply_env.get("payload", {}).get("tool")})

            if not ok:
                transcript.append({"who": "NEAR", "event": "DROP_REPLY", "reason": reason})
                continue

            advance_state(self.pim, reply_env)
            transcript.append({"who": "NEAR", "event": "REPLY", "payload": reply_env["payload"]})

        return transcript
