import csv
from typing import List, Dict, Any
from pathlib import Path

FIELDNAMES = ["id", "task", "result", "ts"]

class CsvStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                w.writeheader()

    def read_rows(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self.path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-limit:]

    def append_row(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
