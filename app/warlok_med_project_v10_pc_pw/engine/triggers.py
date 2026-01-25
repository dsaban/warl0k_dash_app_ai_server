# engine/triggers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from .domain_pack import DomainPack


def load_triggers(domain: DomainPack) -> Dict[str, Any]:
    """
    Loads triggers json from pack (manifest files.triggers).
    """
    files = (domain.manifest.get("files") or {})
    rel = files.get("triggers", "gdm_triggers.json")
    p = domain.pack_dir / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(errors="ignore"))


def _get(snapshot: Dict[str, Any], dotted: str, default=None) -> Any:
    cur: Any = snapshot
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _exists(snapshot: Dict[str, Any], dotted: str) -> bool:
    cur: Any = snapshot
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    return cur is not None


def _cmp(op: str, left: Any, right: Any) -> bool:
    if op == "exists":
        # exists should be handled separately, but keep safe
        return left is not None
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if left is None:
        return False
    try:
        if op == ">":
            return float(left) > float(right)
        if op == ">=":
            return float(left) >= float(right)
        if op == "<":
            return float(left) < float(right)
        if op == "<=":
            return float(left) <= float(right)
    except Exception:
        return False
    return False


def _eval_cond(snapshot: Dict[str, Any], cond: Dict[str, Any]) -> bool:
    field = cond.get("field", "")
    op = cond.get("op", "==")
    val = cond.get("value", None)

    if op == "exists":
        return _exists(snapshot, field)

    left = _get(snapshot, field, None)
    return _cmp(op, left, val)


def _eval_all(snapshot: Dict[str, Any], conds: List[Dict[str, Any]]) -> bool:
    for c in (conds or []):
        if not _eval_cond(snapshot, c):
            return False
    return True


def _eval_none(snapshot: Dict[str, Any], conds: List[Dict[str, Any]]) -> bool:
    # True if none of the conditions are true
    for c in (conds or []):
        if _eval_cond(snapshot, c):
            return False
    return True


def run_triggers(snapshot: Dict[str, Any], domain: DomainPack) -> Dict[str, Any]:
    """
    Evaluates pack triggers against snapshot.
    Returns dict:
      { "derived_states": [...], "alerts": [...], "debug": {...} }
    """
    spec = load_triggers(domain)
    rules = spec.get("rules") or []
    derived_states: List[Dict[str, Any]] = []
    alerts: List[Dict[str, Any]] = []

    fired = 0
    for r in rules:
        when_all = r.get("when_all") or []
        when_none = r.get("when_none") or []
        if not _eval_all(snapshot, when_all):
            continue
        if when_none and not _eval_none(snapshot, when_none):
            continue

        fired += 1
        common = {
            "rule_id": r.get("id"),
            "rule_name": r.get("name"),
            "evidence_edges_any": r.get("evidence_edges_any", []) or []
        }

        if r.get("emit_state"):
            s = dict(r["emit_state"])
            s.update(common)
            derived_states.append(s)

        if r.get("emit_alert"):
            a = dict(r["emit_alert"])
            a.update(common)
            alerts.append(a)

    return {
        "derived_states": derived_states,
        "alerts": alerts,
        "debug": {
            "rules_total": len(rules),
            "rules_fired": fired,
            "version": spec.get("version", "")
        }
    }
