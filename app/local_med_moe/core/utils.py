# core/utils.py
import re

STOP = {
    "the","a","an","and","or","to","of","in","on","for","with","is","are","was","were","be","been",
    "it","this","that","as","at","by","from","into","over","under","after","before","than","then",
    "also","can","could","should","would","may","might","we","you","they","he","she","i","our","their"
}

def normalize_text(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize(s: str):
    s = normalize_text(s)
    toks = re.findall(r"[a-z0-9\-]+", s)
    return [t for t in toks if t and t not in STOP]
