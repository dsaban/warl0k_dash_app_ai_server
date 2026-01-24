import math
from collections import Counter, defaultdict
from typing import List, Tuple
from .utils import norm

def tokenize(text: str) -> List[str]:
    t = norm(text)
    out = []
    buf = []
    for ch in t:
        if ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                out.append("".join(buf))
                buf = []
    if buf:
        out.append("".join(buf))
    return out

class BM25:
    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.tok_docs = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.doc_lens = [len(td) for td in self.tok_docs]
        self.avgdl = (sum(self.doc_lens) / max(1, self.N))

        self.df = defaultdict(int)
        for td in self.tok_docs:
            for term in set(td):
                self.df[term] += 1

        self.idf = {}
        for term, df in self.df.items():
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

        self.tf = [Counter(td) for td in self.tok_docs]

    def score_one(self, query: str, idx: int) -> float:
        q = tokenize(query)
        tf = self.tf[idx]
        dl = self.doc_lens[idx]
        score = 0.0
        for term in q:
            if term not in tf:
                continue
            f = tf[term]
            idf = self.idf.get(term, 0.0)
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            score += idf * (f * (self.k1 + 1.0)) / (denom + 1e-9)
        return float(score)

    def topk(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        scores = [(i, self.score_one(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]
