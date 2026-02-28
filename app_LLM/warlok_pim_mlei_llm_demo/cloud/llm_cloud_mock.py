# cloud/llm_cloud_mock.py
from typing import Dict, Any, List, Optional
from common.util import now_ts

def _pad(plan: List[Dict[str, Any]], min_steps: int) -> List[Dict[str, Any]]:
    need = max(0, int(min_steps) - len(plan))
    for k in range(need):
        plan.append({
            "tool": "proof_tick",
            "text": "Proof-only tick (no-op): completes OS window without touching assets",
            "args": {"k": k + 1},
            "ts": now_ts(),
        })
    return plan

def llm_agent_plan(task_prompt: str, db_preview: List[Dict[str, Any]], *, os_window_size: Optional[int] = None) -> List[Dict[str, Any]]:
    p = (task_prompt or "").lower()
    target = int(os_window_size) if os_window_size else 0

    if "task 1" in p:
        plan = [
            {"tool": "read_db", "text": "Read last rows for summary", "args": {"limit": 5}, "ts": now_ts()},
            {"tool": "summarize", "text": "Summarize rows safely", "args": {"rows": "__PREV_OUTPUT__"}, "ts": now_ts()},
        ]
        return _pad(plan, target)

    if "task 2" in p:
        plan = [
            {"tool": "write_db", "text": "Append validated result row",
             "args": {"row": {"id": "r2", "task": "task2", "result": "ok", "ts": str(now_ts())}},
             "ts": now_ts()},
        ]
        return _pad(plan, target)

    plan = [{"tool": "llm_query", "text": "General query (no asset write)", "args": {"q": task_prompt}, "ts": now_ts()}]
    return _pad(plan, target)
# from typing import Dict, Any, List, Optional
# from common.util import now_ts
#
#
# def _pad_with_proof_ticks(plan: List[Dict[str, Any]], min_steps: int) -> List[Dict[str, Any]]:
#     """
#     Append harmless no-op steps so we can complete the OS window in the demo.
#     These steps still go through PIM + AEAD + EG verification, but do not touch the DB.
#     """
#     if min_steps <= 0:
#         return plan
#
#     # Count only tool-call steps
#     n = len(plan)
#     need = max(0, int(min_steps) - n)
#
#     for k in range(need):
#         plan.append(
#             {
#                 "tool": "proof_tick",
#                 "text": "Proof-only tick (no-op): advance OS slice / PIM chain without executing actions",
#                 "args": {"k": k + 1},
#                 "ts": now_ts(),
#             }
#         )
#     return plan
#
#
# def llm_agent_plan(
#     task_prompt: str,
#     db_preview: List[Dict[str, Any]],
#     *,
#     os_window_size: Optional[int] = None,
# ) -> List[Dict[str, Any]]:
#     """
#     Returns a list of "agent tool calls" that must pass gates & PIM.
#
#     Enhancement:
#     - If os_window_size is provided, the plan is padded with 'proof_tick' no-op calls
#       so the OS→MS streaming inference can reach ms_done=True and produce ms_ok.
#     """
#     prompt = (task_prompt or "").lower()
#     target_steps = int(os_window_size) if os_window_size else 0
#
#     if "task 1" in prompt:
#         plan = [
#             {"tool": "read_db", "text": "Read last rows for summary", "args": {"limit": 5}, "ts": now_ts()},
#             # IMPORTANT: summarize uses __PREV_OUTPUT__ in your app logic, so it summarizes read_db output safely.
#             {"tool": "summarize", "text": "Summarize DB rows (safe summary)", "args": {"rows": "__PREV_OUTPUT__"}, "ts": now_ts()},
#         ]
#         return _pad_with_proof_ticks(plan, target_steps)
#
#     if "task 2" in prompt:
#         # "write_db" is allowed but must be validated by MLEI policy
#         plan = [
#             {
#                 "tool": "write_db",
#                 "text": "Append a validated result row",
#                 "args": {"row": {"id": "r2", "task": "task2", "result": "ok", "ts": str(now_ts())}},
#                 "ts": now_ts(),
#             }
#         ]
#         # Pad so OS window can complete even on Task 2
#         return _pad_with_proof_ticks(plan, target_steps)
#
#     # default: ask LLM
#     plan = [
#         {"tool": "llm_query", "text": "General reasoning (no DB write)", "args": {"q": task_prompt}, "ts": now_ts()},
#     ]
#     return _pad_with_proof_ticks(plan, target_steps)
# # from typing import Dict, Any, List
# # from common.util import now_ts
# #
# # def llm_agent_plan(task_prompt: str, db_preview: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
# #     """
# #     Returns a list of "agent tool calls" that must pass gates & PIM.
# #     """
# #     prompt = (task_prompt or "").lower()
# #
# #     if "task 1" in prompt:
# #         return [
# #             {"tool": "read_db", "text": "Read last rows for summary", "args": {"limit": 5}, "ts": now_ts()},
# #             {"tool": "summarize", "text": "Summarize DB rows (safe summary)", "args": {"rows": db_preview}, "ts": now_ts()},
# #         ]
# #
# #     if "task 2" in prompt:
# #         # "write_db" is allowed but must be validated by MLEI policy
# #         return [
# #             {"tool": "write_db", "text": "Append a validated result row", "args": {"row": {"id":"r2","task":"task2","result":"ok","ts": str(now_ts())}}, "ts": now_ts()},
# #         ]
# #
# #     # default: ask LLM
# #     return [
# #         {"tool": "llm_query", "text": "General reasoning (no DB write)", "args": {"q": task_prompt}, "ts": now_ts()},
# #     ]
