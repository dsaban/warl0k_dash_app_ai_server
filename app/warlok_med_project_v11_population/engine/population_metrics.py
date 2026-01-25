# engine/population_metrics.py
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .population_store import (
    list_latest_by_patient_episode,
    latest_actions_map,
    attach_action_status_to_items,
)


def _in_timerange(ts: str, start: Optional[str], end: Optional[str]) -> bool:
    # assumes ISO-ish strings; for MVP we do lexical filtering if provided
    if start and ts < start:
        return False
    if end and ts > end:
        return False
    return True


def compute_kpis(
    root,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    severity_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute top-level KPI counts from latest record per episode.
    """
    latest = list_latest_by_patient_episode(root)
    actions = latest_actions_map(root)

    kpis = Counter()
    total_eps = 0

    for key, r in latest.items():
        ts = str(r.get("timestamp", ""))
        if not _in_timerange(ts, start_ts, end_ts):
            continue
        total_eps += 1

        alerts = r.get("alerts") or []
        patient_id = r.get("patient_id")
        episode_id = r.get("episode_id")
        alerts = attach_action_status_to_items(alerts, patient_id, episode_id, actions)

        for a in alerts:
            sev = a.get("severity", "warning")
            if severity_filter and sev not in severity_filter:
                continue
            if a.get("status") == "resolved":
                continue
            name = a.get("name") or a.get("alert") or "ALERT"
            kpis[f"alert::{name}"] += 1
            kpis[f"severity::{sev}"] += 1

        # derived states are lower urgency; still count
        states = r.get("derived_states") or []
        for s in states:
            name = s.get("name") or s.get("state") or "STATE"
            kpis[f"state::{name}"] += 1

    return {
        "episodes_total": total_eps,
        "counts": dict(kpis)
    }


def build_care_gap_queue(
    root,
    start_ts: Optional[str] = None,
    end_ts: Optional[str] = None,
    trigger_name_filter: Optional[List[str]] = None,
    severity_filter: Optional[List[str]] = None,
    status_filter: Optional[List[str]] = None,
    max_rows: int = 500
) -> List[Dict[str, Any]]:
    """
    Create prioritized actionable queue of alerts across population.
    Only latest record per patient/episode is used.
    """
    latest = list_latest_by_patient_episode(root)
    actions = latest_actions_map(root)

    rows: List[Dict[str, Any]] = []
    for _, r in latest.items():
        ts = str(r.get("timestamp", ""))
        if not _in_timerange(ts, start_ts, end_ts):
            continue

        patient_id = r.get("patient_id")
        episode_id = r.get("episode_id")

        snapshot = r.get("snapshot") or {}
        ga = (snapshot.get("pregnancy_episode") or {}).get("gestational_age_weeks")
        pp_weeks = (snapshot.get("episode_status") or {}).get("postpartum_weeks")
        bmi = (snapshot.get("risk_factors") or {}).get("bmi")
        fg = (snapshot.get("labs") or {}).get("fasting_glucose_mgdl")

        alerts = r.get("alerts") or []
        alerts = attach_action_status_to_items(alerts, patient_id, episode_id, actions)

        for a in alerts:
            name = a.get("name") or a.get("alert") or "ALERT"
            sev = a.get("severity", "warning")
            status = a.get("status", "open")

            if trigger_name_filter and name not in trigger_name_filter:
                continue
            if severity_filter and sev not in severity_filter:
                continue
            if status_filter and status not in status_filter:
                continue

            # use evidence count as a weak confidence proxy (MVP)
            ev = a.get("evidence_claims") or []
            ev_n = len(ev)
            best_score = max([e.get("score", 0.0) for e in ev], default=0.0)

            rows.append({
                "severity": sev,
                "trigger": name,
                "patient_id": patient_id,
                "episode_id": episode_id,
                "timestamp": ts,
                "status": status,
                "evidence_n": ev_n,
                "evidence_best_score": round(float(best_score), 3),
                "gest_weeks": ga,
                "postpartum_weeks": pp_weeks,
                "bmi": bmi,
                "fasting_glucose": fg,
                "rationale": a.get("rationale", ""),
                "alert_key": a.get("alert_key", ""),
                "last_action_ts": a.get("last_action_ts", "")
            })

    # prioritize: severity desc, evidence desc, timestamp asc (older first)
    sev_rank = {"high": 3, "warning": 2, "info": 1}
    rows.sort(key=lambda x: (-sev_rank.get(x["severity"], 0), -x["evidence_n"], x["timestamp"]))
    return rows[:max_rows]


def patient_timeline(root, patient_id: str, episode_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Returns all records for a patient/episode ordered by time ascending.
    """
    from .population_store import list_records  # avoid cycles
    recs = list_records(root, limit=limit)
    out = [r for r in recs if r.get("patient_id") == patient_id and r.get("episode_id") == episode_id]
    out.sort(key=lambda r: str(r.get("timestamp", "")))
    return out
