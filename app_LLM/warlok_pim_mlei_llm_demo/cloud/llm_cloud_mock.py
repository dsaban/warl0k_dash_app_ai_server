from typing import Dict, Any, List
from common.util import now_ts

def llm_agent_plan(task_prompt: str, db_preview: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Returns a list of "agent tool calls" that must pass gates & PIM.
    """
    prompt = (task_prompt or "").lower()

    if "task 1" in prompt:
        return [
            {"tool": "read_db", "text": "Read last rows for summary", "args": {"limit": 5}, "ts": now_ts()},
            {"tool": "summarize", "text": "Summarize DB rows (safe summary)", "args": {"rows": db_preview}, "ts": now_ts()},
        ]

    if "task 2" in prompt:
        # "write_db" is allowed but must be validated by MLEI policy
        return [
            {"tool": "write_db", "text": "Append a validated result row", "args": {"row": {"id":"r2","task":"task2","result":"ok","ts": str(now_ts())}}, "ts": now_ts()},
        ]

    # default: ask LLM
    return [
        {"tool": "llm_query", "text": "General reasoning (no DB write)", "args": {"q": task_prompt}, "ts": now_ts()},
    ]
