from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class PatientRecord:
    patient_id: str
    visit_id: str
    timestamp: str  # ISO string
    structured: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    extracted: Dict[str, Any] = field(default_factory=dict)
    matched_concepts: List[Dict[str, Any]] = field(default_factory=list)
    matched_claims: List[Dict[str, Any]] = field(default_factory=list)
