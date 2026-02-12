
import pandas as pd
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path

@dataclass
class LexiconEntry:
    entity_id: str
    canonical_name: str
    type: str
    synonyms: List[str]
    codes: str = ""

class LexiconIndex:
    def __init__(self, entries: List[LexiconEntry]):
        self.entries = entries
        # precompile regex patterns for synonyms (word-boundary-ish)
        self.patterns = []
        for e in entries:
            syns = [s.strip() for s in e.synonyms if s.strip()]
            # longer synonyms first to avoid partial matches
            syns = sorted(syns, key=len, reverse=True)
            if not syns:
                continue
            # Escape except keep simple tokens
            parts = []
            for s in syns:
                parts.append(re.escape(s))
            pat = re.compile(r"(?i)\b(" + "|".join(parts) + r")\b")
            self.patterns.append((e, pat))

    @staticmethod
    def load(csv_path: str):
        p = Path(csv_path)
        if not p.exists():
            return None
        df = pd.read_csv(p)
        entries = []
        for _, r in df.iterrows():
            syn = str(r.get("synonyms","") if pd.notna(r.get("synonyms","")) else "")
            syns = [x.strip() for x in syn.split("|") if x.strip()]
            entries.append(LexiconEntry(
                entity_id=str(r.get("entity_id","")),
                canonical_name=str(r.get("canonical_name","")),
                type=str(r.get("type","")),
                synonyms=syns,
                codes=str(r.get("codes","")) if "codes" in df.columns else ""
            ))
        return LexiconIndex(entries)

    def extract(self, text: str, max_entities: int = 12) -> List[Dict[str, Any]]:
        if not text:
            return []
        found = {}
        for e, pat in self.patterns:
            m = pat.search(text)
            if m:
                found[e.entity_id] = {
                    "entity_id": e.entity_id,
                    "canonical_name": e.canonical_name,
                    "type": e.type,
                    "match": m.group(0),
                }
        # heuristic: prefer diverse types + stable order
        out = list(found.values())
        out.sort(key=lambda x: (x["type"], x["canonical_name"]))
        return out[:max_entities]
