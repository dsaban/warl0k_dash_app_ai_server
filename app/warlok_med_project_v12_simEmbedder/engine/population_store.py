# engine/population_store.py
from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .domain_pack import DomainPack
from .patient_record import PatientRecord
from .patient_match_bm25 import load_patient_layer, extract_fields_from_notes
from .patient_state import normalize_patient_snapshot
from .triggers import run_triggers
from .trigger_evidence import attach_evidence_to_triggers


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update((s or "").encode("utf-8", errors="ignore"))
    return h.hexdigest()


def ensure_population_dirs(root: Path) -> Dict[str, Path]:
    data_dir = root / "data"
    patients_dir = data_dir / "patients"
    patients_dir.mkdir(parents=True, exist_ok=True)
    records_path = patients_dir / "records.jsonl"
    actions_path = patients_dir / "actions.jsonl"
    return {"patients_dir": patients_dir, "records_path": records_path, "actions_path": actions_path}


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
            if limit and len(out) >= limit:
                break
    return out


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _alert_key(item: Dict[str, Any]) -> str:
    # stable key for dedup / actions
    rid = item.get("rule_id", "")
    name = item.get("name") or item.get("alert") or item.get("state") or ""
    return f"{rid}::{name}".strip(":")


def _choose_episode_id(patient_id: str, extra_structured: Optional[Dict[str, Any]]) -> str:
    # Optional pack/EHR can supply episode_id; else fallback to a deterministic placeholder
    if isinstance(extra_structured, dict):
        eid = extra_structured.get("episode_id") or extra_structured.get("pregnancy_episode", {}).get("episode_id")
        if eid:
            return str(eid)
    # fallback (single episode)
    return f"{patient_id}::EPISODE-1"


def ingest_patient_record(
    root: Path,
    index_dir: Path,
    domain: DomainPack,
    patient_id: str,
    visit_id: str,
    notes: str,
    extra_structured: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    per_item_topk: int = 3,
) -> Dict[str, Any]:
    """
    Ingests one patient record into population store:
      - extract fields via patient_layer.json
      - build PSG snapshot via patient_state_spec.json
      - run triggers via triggers.json
      - attach evidence claims using claims.jsonl (BM25) per trigger edges
      - persist into data/patients/records.jsonl

    Returns the stored record.
    """
    paths = ensure_population_dirs(root)
    ts = timestamp or _utc_now_iso()
    episode_id = _choose_episode_id(patient_id, extra_structured)

    patient_layer = load_patient_layer(domain)
    if not patient_layer:
        raise FileNotFoundError("patient_layer.json missing or not referenced in pack manifest (files.patient_layer).")

    rec = PatientRecord(
        patient_id=patient_id,
        visit_id=visit_id,
        timestamp=ts,
        notes=notes or "",
        structured=extra_structured or {}
    )
    rec.extracted = extract_fields_from_notes(rec.notes, patient_layer)

    snapshot = normalize_patient_snapshot(rec, domain, extra_structured=extra_structured)

    tr = run_triggers(snapshot, domain)

    # Build patient-context query text for evidence ranking (BM25)
    default_terms = patient_layer.get("default_query_terms") or []
    query_text = " | ".join([rec.notes] + [f"{k}:{v}" for k, v in (rec.extracted or {}).items()] + list(map(str, default_terms)))

    tr = attach_evidence_to_triggers(
        index_dir=index_dir,
        triggers_out=tr,
        query_text=query_text,
        per_item_topk=per_item_topk
    )

    record = {
        "patient_id": patient_id,
        "episode_id": episode_id,
        "visit_id": visit_id,
        "timestamp": ts,
        "notes_hash": _sha256_text(notes or ""),
        "snapshot": snapshot,
        "derived_states": tr.get("derived_states") or [],
        "alerts": tr.get("alerts") or [],
        "pack_meta": {
            "pack_name": getattr(domain, "name", None) or domain.manifest.get("name", domain.pack_dir.name),
            "pack_dir": str(domain.pack_dir),
            "pack_version": domain.manifest.get("version", ""),
            "triggers_version": (domain.manifest.get("files", {}) or {}).get("triggers", ""),
        },
        "debug": {
            "trigger_debug": tr.get("debug") or {},
            "extracted": rec.extracted or {},
        }
    }

    _append_jsonl(paths["records_path"], record)
    return record


def list_records(root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    paths = ensure_population_dirs(root)
    return _read_jsonl(paths["records_path"], limit=limit)


def list_latest_by_patient_episode(root: Path, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Dedup: return latest record per (patient_id, episode_id).
    """
    records = list_records(root, limit=limit)
    latest: Dict[str, Dict[str, Any]] = {}
    for r in records:
        key = f"{r.get('patient_id')}::{r.get('episode_id')}"
        # timestamps are ISO strings; lexical order works if consistent
        if key not in latest or str(r.get("timestamp", "")) > str(latest[key].get("timestamp", "")):
            latest[key] = r
    return latest


def write_action(
    root: Path,
    patient_id: str,
    episode_id: str,
    alert_key: str,
    status: str,
    note: str = "",
    user: str = ""
) -> Dict[str, Any]:
    """
    Append-only actions log (acknowledged/resolved/open) for a given alert_key.
    """
    paths = ensure_population_dirs(root)
    action = {
        "timestamp": _utc_now_iso(),
        "patient_id": patient_id,
        "episode_id": episode_id,
        "alert_key": alert_key,
        "status": status,
        "note": note,
        "user": user
    }
    _append_jsonl(paths["actions_path"], action)
    return action


def latest_actions_map(root: Path, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns latest action per (patient_id, episode_id, alert_key).
    """
    paths = ensure_population_dirs(root)
    actions = _read_jsonl(paths["actions_path"], limit=limit)
    latest: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        key = f"{a.get('patient_id')}::{a.get('episode_id')}::{a.get('alert_key')}"
        if key not in latest or str(a.get("timestamp", "")) > str(latest[key].get("timestamp", "")):
            latest[key] = a
    return latest


def attach_action_status_to_items(
    items: List[Dict[str, Any]],
    patient_id: str,
    episode_id: str,
    actions_map: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Adds status/last_action_ts to each alert/state using actions_map.
    """
    out = []
    for it in items or []:
        ak = _alert_key(it)
        key = f"{patient_id}::{episode_id}::{ak}"
        a = actions_map.get(key)
        it2 = dict(it)
        it2["alert_key"] = ak
        it2["status"] = (a.get("status") if a else "open")
        it2["last_action_ts"] = (a.get("timestamp") if a else "")
        it2["last_action_note"] = (a.get("note") if a else "")
        out.append(it2)
    return out
