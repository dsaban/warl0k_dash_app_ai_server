from __future__ import annotations
import math
import re
from collections import Counter
from typing import List, Tuple, Dict


_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]

class BM25:
    def __init__(self, docs: List[str], k1: float = 1.6, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.toks = [tokenize(d) for d in docs]
        self.lens = [len(t) for t in self.toks]
        self.avgdl = (sum(self.lens) / max(1, len(self.lens)))
        self.df: Dict[str, int] = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1
        self.N = len(docs)

    def score(self, query: str) -> List[float]:
        q = tokenize(query)
        qtf = Counter(q)
        scores = [0.0] * self.N
        for i, doc_toks in enumerate(self.toks):
            tf = Counter(doc_toks)
            dl = self.lens[i]
            for w, qn in qtf.items():
                df = self.df.get(w, 0)
                if df == 0:
                    continue
                # BM25 IDF
                idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                f = tf.get(w, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * (dl / max(1e-9, self.avgdl)))
                scores[i] += idf * (f * (self.k1 + 1) / denom)
        return scores

    def topk(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = self.score(query)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        # return [(i, float(scores[i])) for i in idx if scores[i] > 0.0]
    
        return [(i, float(scores[i])) for i in idx]

