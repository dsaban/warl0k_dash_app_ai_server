# engine/patient_state.py
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .domain_pack import DomainPack
from .patient_record import PatientRecord


def load_patient_state_spec(domain: DomainPack) -> Dict[str, Any]:
    """
    Loads patient_state_spec.json from the pack (manifest files.patient_state_spec).
    Returns {} if missing.
    """
    files = (domain.manifest.get("files") or {})
    rel = files.get("patient_state_spec", "patient_state_spec.json")
    p = domain.pack_dir / rel
    if not p.exists():
        return {}
    return json.loads(p.read_text(errors="ignore"))


def _deep_set(d: Dict[str, Any], dotted: str, value: Any) -> None:
    cur = d
    parts = dotted.split(".")
    for k in parts[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[parts[-1]] = value


def _deep_get(d: Dict[str, Any], dotted: str, default=None) -> Any:
    cur: Any = d
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge_dict(dst[k], v)
        else:
            dst[k] = v
    return dst


def normalize_patient_snapshot(
    rec: PatientRecord,
    domain: DomainPack,
    extra_structured: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Creates a normalized snapshot dict aligned with patient_state_spec.json canonical_fields.
    - Uses rec.extracted (from patient_layer field_extractors)
    - Optionally merges extra_structured (UI or EHR fields)
    - Does NOT do diagnosis. Just creates a state container for triggers.
    """
    spec = load_patient_state_spec(domain)
    canonical = spec.get("canonical_fields") or {}

    # Minimal skeleton (keep stable keys for triggers)
    snapshot: Dict[str, Any] = {
        "risk_factors": {},
        "pregnancy_episode": {},
        "labs": {},
        "diagnosis": {},         # clinician-recorded (optional)
        "treatment": {},         # clinician-recorded (optional)
        "episode_status": {      # derived from encounters (optional)
            "postpartum": False,
            "postpartum_weeks": 0
        },
        "postpartum": {
            "ogtt_done": False
        }
    }

    # Merge user-provided structured payloads if any
    if isinstance(rec.structured, dict):
        _merge_dict(snapshot, rec.structured)
    if isinstance(extra_structured, dict):
        _merge_dict(snapshot, extra_structured)

    # Map extracted flat fields into canonical buckets (risk_factors/labs/pregnancy_episode)
    extracted = rec.extracted or {}

    # Build a reverse lookup from canonical_fields to target dotted path
    # canonical_fields format in spec:
    # canonical_fields: { pregnancy_episode: {gestational_age_weeks:{...}}, labs:{...}, risk_factors:{...}}
    for bucket_name, fields in canonical.items():
        if not isinstance(fields, dict):
            continue
        for field_name in fields.keys():
            # if extracted has exact key, place it under bucket.field
            if field_name in extracted:
                _deep_set(snapshot, f"{bucket_name}.{field_name}", extracted[field_name])

    # A couple of safe, purely-parsing helpers (optional):
    # Try to parse gestational age like "26+4 weeks" or "26 weeks" if not already present.
    if _deep_get(snapshot, "pregnancy_episode.gestational_age_weeks") is None:
        ga = _extract_gestational_age_weeks(rec.notes or "")
        if ga is not None:
            _deep_set(snapshot, "pregnancy_episode.gestational_age_weeks", ga)

    # Try to detect postpartum mention (very conservative)
    if "postpartum" in (rec.notes or "").lower():
        snapshot["episode_status"]["postpartum"] = True

    return snapshot


def _extract_gestational_age_weeks(text: str) -> Optional[float]:
    """
    Parse patterns like:
    - '26+4 weeks' -> 26.57
    - '26 weeks'   -> 26.0
    - '26w4d'      -> 26.57
    """
    t = text.lower()

    m = re.search(r"\b(\d{1,2})\s*\+\s*(\d)\s*(?:weeks|w)\b", t)
    if m:
        w = float(m.group(1))
        d = float(m.group(2))
        return round(w + (d / 7.0), 2)

    m = re.search(r"\b(\d{1,2})\s*w\s*(\d)\s*d\b", t)
    if m:
        w = float(m.group(1))
        d = float(m.group(2))
        return round(w + (d / 7.0), 2)

    m = re.search(r"\b(\d{1,2})\s*(?:weeks|w)\b", t)
    if m:
        return float(m.group(1))

    return None
