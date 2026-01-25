import re

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def tokens(s: str):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

# from __future__ import annotations
# import re
# from typing import List, Iterable, Dict, Any
#
# def norm_space(s: str) -> str:
#     return re.sub(r"\s+", " ", (s or "").strip())
#
# def tokens(s: str) -> List[str]:
#     return re.findall(r"[a-z0-9]+", (s or "").lower())
#
# def split_sents(text: str) -> List[str]:
#     text = norm_space(text)
#     if not text:
#         return []
#     # academic-ish splitter
#     parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9\[])", text)
#     return [p.strip() for p in parts if p.strip()]
#
# def uniq(seq: Iterable[str]) -> List[str]:
#     seen=set(); out=[]
#     for x in seq:
#         if x not in seen:
#             seen.add(x); out.append(x)
#     return out
#
# def safe_json_load(path) -> Dict[str, Any]:
#     import json
#     from pathlib import Path
#     path = Path(path)
#     return json.loads(path.read_text(errors="ignore"))
#
# def safe_json_dump(path, obj, indent=2):
#     import json
#     from pathlib import Path
#     path = Path(path)
#     path.parent.mkdir(parents=True, exist_ok=True)
#     path.write_text(json.dumps(obj, ensure_ascii=False, indent=indent))
