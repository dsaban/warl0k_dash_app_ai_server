from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ChunkNode:
    chunk_id: str
    file: str
    text: str
    token_count: int
    span: Tuple[int, int]
    section: str

def chunk_by_tokens(text: str, target_tokens: int = 220, overlap_tokens: int = 60) -> List[Tuple[str, Tuple[int,int]]]:
    raw = text or ""
    words = raw.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + target_tokens)
        ch_words = words[i:j]
        ch_text = " ".join(ch_words).strip()
        start = raw.find(ch_text)
        if start == -1:
            start = 0
        end = min(len(raw), start + len(ch_text))
        chunks.append((ch_text, (start, end)))
        if j >= len(words):
            break
        i = max(0, j - overlap_tokens)
    return chunks

def detect_section_headers(text: str) -> List[Tuple[int, str]]:
    """
    Detect simple headers like:
      "Risk Factors"
      "Pathophysiology of GDM"
    Returns list of (char_index, header_text).
    """
    headers = []
    lines = (text or "").splitlines()
    pos = 0
    for ln in lines:
        s = ln.strip()
        if 0 < len(s) <= 80 and s.isprintable():
            # heuristic: title-case-ish or ends without punctuation and not too long
            if s.lower() == s:
                pass
            else:
                if not any(s.endswith(x) for x in [".", "!", "?", ";", ":"]):
                    # avoid very short junk
                    if len(s.split()) <= 8:
                        headers.append((pos, s))
        pos += len(ln) + 1
    return headers

def section_for_span(headers: List[Tuple[int, str]], span_start: int) -> str:
    if not headers:
        return ""
    best = ""
    best_pos = -1
    for p, h in headers:
        if p <= span_start and p >= best_pos:
            best_pos = p
            best = h
    return best
