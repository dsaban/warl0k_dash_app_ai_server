from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class QTypeSpec:
    key: str
    description: str
    required_tags: List[str]
    integrity_checklist: List[str]


class Ontology:
    """Fixed taxonomy for qtypes + tags.

    - QTypes and tags are **loaded from config** and treated as fixed.
    - No inference-time mutation.
    """

    def __init__(self, qtypes: Dict[str, QTypeSpec], tags: Dict[str, List[str]]):
        self.qtypes = qtypes
        self.tags = tags

    @staticmethod
    def load(path: str | Path) -> "Ontology":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        qtypes: Dict[str, QTypeSpec] = {}
        for k, v in data.get("qtypes", {}).items():
            qtypes[k] = QTypeSpec(
                key=k,
                description=v.get("description", ""),
                required_tags=list(v.get("required_tags", [])),
                integrity_checklist=list(v.get("integrity_checklist", [])),
            )

        tags: Dict[str, List[str]] = {k: list(v) for k, v in data.get("tags", {}).items()}

        # Guardrails
        if "unknown" not in qtypes:
            qtypes["unknown"] = QTypeSpec("unknown", "Fallback", [], [])

        return Ontology(qtypes=qtypes, tags=tags)

    def validate_qtype(self, qtype: str) -> str:
        return qtype if qtype in self.qtypes else "unknown"

    def tag_aliases(self) -> List[Tuple[str, str]]:
        """Returns (tag, alias) pairs."""
        out: List[Tuple[str, str]] = []
        for t, aliases in self.tags.items():
            for a in aliases:
                out.append((t, a))
        return out
