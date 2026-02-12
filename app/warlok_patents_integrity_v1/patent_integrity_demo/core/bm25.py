from __future__ import annotations
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-/]*")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

@dataclass
class BM25Index:
    docs: List[str]
    doc_tokens: List[List[str]]
    idf: Dict[str, float]
    avgdl: float
    k1: float = 1.5
    b: float = 0.75

    @classmethod
    def build(cls, docs: List[str], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        doc_tokens = [tokenize(d) for d in docs]
        N = len(doc_tokens)
        df: Dict[str, int] = {}
        dl_sum = 0
        for toks in doc_tokens:
            dl_sum += len(toks)
            seen = set(toks)
            for t in seen:
                df[t] = df.get(t, 0) + 1
        avgdl = (dl_sum / N) if N else 0.0
        idf: Dict[str, float] = {}
        for t, dft in df.items():
            idf[t] = math.log(1 + (N - dft + 0.5) / (dft + 0.5))
        return cls(docs=docs, doc_tokens=doc_tokens, idf=idf, avgdl=avgdl, k1=k1, b=b)

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        scores = [0.0] * len(self.docs)
        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            denom_const = self.k1 * (1 - self.b + self.b * (dl / (self.avgdl or 1.0)))
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                term_idf = self.idf.get(term, 0.0)
                s += term_idf * (f * (self.k1 + 1)) / (f + denom_const)
            scores[i] = s
        return scores

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        scores = self.score(query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:top_k] if s > 0]
