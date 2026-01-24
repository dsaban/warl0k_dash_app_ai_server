
import json, re
from typing import Any, Dict, List

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+"," ",text).strip()
    return text

def softmax(xs):
    import math
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x-m) for x in xs]
    s = sum(ex) or 1.0
    return [e/s for e in ex]
