from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from core.ingest import Chunk
from core.ontology import Ontology


_WORD_RE = re.compile(r"[A-Za-z0-9\-αβγδμ]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    snippet: str
    tags: List[str]
    entities: List[str]


class TfidfIndex:
    def __init__(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.df: Dict[str, int] = {}
        self.tf: List[Dict[str, int]] = []
        self.N = len(chunks)
        for c in chunks:
            toks = tokenize(c.text)
            freq: Dict[str, int] = {}
            for t in toks:
                freq[t] = freq.get(t, 0) + 1
            self.tf.append(freq)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def score(self, query: str, chunk_idx: int) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        freq = self.tf[chunk_idx]
        s = 0.0
        for t in q:
            if t not in freq:
                continue
            df = self.df.get(t, 0)
            idf = math.log((self.N + 1) / (df + 1)) + 1.0
            tf = 1.0 + math.log(freq[t])
            s += tf * idf
        return s


def classify_qtype(question: str, ontology: Ontology) -> str:
    q = question.lower()
    # Fixed, rule-based routing (no dynamic qtype creation)
    if "pedersen" in q or "macrosomia" in q or ("maternal" in q and "fetal" in q and "growth" in q):
        return "fetal_growth_mechanism"
    if "long-term" in q or "postpartum" in q or "cardiometabolic" in q:
        return "maternal_long_term_risk"
    if "clinical model" in q or "transition" in q or ("insulin resistance" in q and "diabetes" in q):
        return "progression_model"
    if "cgm" in q or "smbg" in q or "monitor" in q or "management" in q:
        return "management_monitoring"
    return "unknown"


def infer_query_tags(question: str, ontology: Ontology) -> List[str]:
    q = question.lower()
    hits = []
    for tag, aliases in ontology.tags.items():
        for a in aliases:
            if a.lower() in q:
                hits.append(tag)
                break
    return sorted(set(hits))


def retrieve(
    question: str,
    chunks: List[Chunk],
    chunk_tags: Dict[str, List[str]],
    chunk_entities: Dict[str, List[str]],
    ontology: Ontology,
    top_k: int = 6,
) -> List[RetrievalHit]:
    idx = TfidfIndex(chunks)
    qtags = set(infer_query_tags(question, ontology))

    hits: List[RetrievalHit] = []
    for i, c in enumerate(chunks):
        base = idx.score(question, i)
        tags = chunk_tags.get(c.id, [])
        ents = chunk_entities.get(c.id, [])

        # Boost by tag overlap (makes the graph explainable)
        overlap = len(qtags.intersection(tags))
        score = base * (1.0 + 0.25 * overlap)

        if score <= 0:
            continue
        snippet = c.text[:220] + ("…" if len(c.text) > 220 else "")
        hits.append(RetrievalHit(chunk_id=c.id, score=score, snippet=snippet, tags=tags, entities=ents))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:top_k]
