from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class AtomicClaim:
    patent_id: str
    claim_no: int
    atomic_id: str
    text: str
    is_independent: bool
    evidence: Dict

_SPLIT_RE = re.compile(
    r"\b(?:comprising|wherein|including|in response to|such that|configured to|adapted to|generating|verifying|transmitting|receiving|determining|authenticating|deriving)\b",
    re.IGNORECASE
)

def atomicize_claim(claim_text: str) -> List[str]:
    parts = _SPLIT_RE.split(claim_text)
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip(" ;,")
        if len(p) >= 30:
            out.append(p)
    if len(out) < 2:
        return [re.sub(r"\s+", " ", claim_text).strip()]
    return out[:20]

def build_atomic_claims(patent_id: str, claim_no: int, claim_text: str, is_independent: bool, source_name: str) -> List[AtomicClaim]:
    atoms = atomicize_claim(claim_text)
    out: List[AtomicClaim] = []
    for i, a in enumerate(atoms, start=1):
        atomic_id = f"{patent_id}-C{claim_no}-A{i}"
        evidence = {"source": source_name, "where": f"Claim {claim_no}", "snippet": a[:400]}
        out.append(AtomicClaim(patent_id=patent_id, claim_no=claim_no, atomic_id=atomic_id, text=a,
                              is_independent=is_independent, evidence=evidence))
    return out
