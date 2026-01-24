import re
from typing import List, Tuple

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def split_sentences(text: str) -> List[str]:
    text = (text or "").replace("\n", " ")
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) >= 20:
            out.append(p)
    return out

def window_text_span(text: str, sent: str) -> Tuple[int, int]:
    i = (text or "").find(sent)
    if i == -1:
        return (0, min(len(text or ""), 200))
    return (i, min(len(text), i + len(sent)))
