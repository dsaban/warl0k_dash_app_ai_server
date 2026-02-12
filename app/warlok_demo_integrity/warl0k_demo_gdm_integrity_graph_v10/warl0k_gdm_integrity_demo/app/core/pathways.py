import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "care"
TASKS_CSV = DATA_DIR / "care_tasks.csv"
RULES_JSONL = DATA_DIR / "care_pathways.jsonl"

def _now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _due_iso(days: int) -> str:
    return (datetime.utcnow().date() + timedelta(days=days)).isoformat()

def load_rules() -> List[Dict[str, Any]]:
    rules = []
    if RULES_JSONL.exists():
        with open(RULES_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rules.append(json.loads(line))
    return rules

def load_tasks() -> pd.DataFrame:
    if TASKS_CSV.exists():
        return pd.read_csv(TASKS_CSV)
    return pd.DataFrame(columns=[
        "task_id","pid","sid","task_type","title","status","priority","owner","due_date",
        "created_at","evidence_claim_ids","notes"
    ])

def save_tasks(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TASKS_CSV, index=False)

def upsert_tasks_from_results(pid: str, results: List[Dict[str, Any]]) -> pd.DataFrame:
    rules = load_rules()
    tasks = load_tasks()
    if tasks.empty:
        tasks = pd.DataFrame(columns=[
            "task_id","pid","sid","task_type","title","status","priority","owner","due_date",
            "created_at","evidence_claim_ids","notes"
        ])

    created = 0
    for res in results:
        sid = str(res.get("sid",""))
        status = str(res.get("status","UNKNOWN"))
        related_claim_ids = res.get("related_claim_ids") or []

        for rule in rules:
            trig = rule.get("trigger", {})
            if trig.get("sid") and trig.get("sid") != sid:
                continue
            if status not in (trig.get("status_in") or []):
                continue

            task_def = rule.get("task", {})
            task_type = str(task_def.get("task_type","TASK"))
            title = str(task_def.get("title","Task"))
            priority = str(task_def.get("priority","MED"))
            owner = str(task_def.get("owner","Clinic"))
            due_days = int(rule.get("due_days", 7))
            due_date = _due_iso(due_days)

            # Deduplicate: active same (pid,sid,task_type,due_date)
            if not tasks.empty:
                dup = tasks[
                    (tasks["pid"].astype(str)==pid) &
                    (tasks["sid"].astype(str)==sid) &
                    (tasks["task_type"].astype(str)==task_type) &
                    (tasks["due_date"].astype(str)==due_date) &
                    (~tasks["status"].astype(str).str.upper().isin(["DONE","CANCELLED"]))
                ]
                if len(dup) > 0:
                    continue

            task_id = f"T{len(tasks)+1:04d}"
            tasks = pd.concat([tasks, pd.DataFrame([{
                "task_id": task_id,
                "pid": pid,
                "sid": sid,
                "task_type": task_type,
                "title": title,
                "status": "OPEN",
                "priority": priority,
                "owner": owner,
                "due_date": due_date,
                "created_at": _now_iso(),
                "evidence_claim_ids": "|".join([str(x) for x in related_claim_ids[:20]]),
                "notes": ""
            }])], ignore_index=True)
            created += 1

    if created:
        save_tasks(tasks)
    return tasks
