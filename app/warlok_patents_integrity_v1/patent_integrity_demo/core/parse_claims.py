from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Claim:
    patent_id: str
    claim_no: int
    text: str
    is_independent: bool

def guess_patent_id(text: str, fallback: str) -> str:
    m = re.search(r"\b(US\d{6,}[A-Z]\d?)\b", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z]{2}\d{6,}[A-Z]?\d?)\b", text)
    if m:
        return m.group(1)
    return fallback

def extract_claim_blocks(text: str) -> List[Tuple[int, str]]:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(
        r"(?:^|\n)\s*(?:Claim\s+)?(\d{1,3})\s*[\.\):]\s*(.+?)(?=(?:\n\s*(?:Claim\s+)?\d{1,3}\s*[\.\):])|\Z)",
        re.IGNORECASE | re.DOTALL
    )
    blocks = []
    for m in pattern.finditer(t):
        n = int(m.group(1))
        body = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(body) >= 40:
            blocks.append((n, body))
    if not blocks:
        blocks = [(1, re.sub(r"\s+", " ", t).strip()[:6000])]
    best = {}
    for n, body in blocks:
        if n not in best or len(body) > len(best[n]):
            best[n] = body
    return sorted(best.items(), key=lambda x: x[0])

def is_independent_claim(text: str) -> bool:
    if re.match(r"^\s*(the|a)\s+.*\bclaim\s+\d+", text, re.IGNORECASE):
        return False
    return True

def parse_claims_from_text(text: str, file_stub: str) -> List[Claim]:
    pid = guess_patent_id(text, file_stub)
    blocks = extract_claim_blocks(text)
    out: List[Claim] = []
    for n, body in blocks:
        out.append(Claim(patent_id=pid, claim_no=n, text=body, is_independent=is_independent_claim(body)))
    return out
