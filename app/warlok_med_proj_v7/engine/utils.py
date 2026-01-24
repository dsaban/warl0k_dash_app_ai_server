from __future__ import annotations
import re
from typing import List, Iterable

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def split_sents(text: str) -> List[str]:
    text = norm_space(text)
    # decent academic sentence splitting
    parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\[])", text)
    return [p.strip() for p in parts if p.strip()]

def tokens(s: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", s.lower())

def uniq(seq: Iterable[str]) -> List[str]:
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out
