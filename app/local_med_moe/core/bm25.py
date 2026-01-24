# core/bm25.py
import math
from .utils import tokenize

class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.tokens = [tokenize(d) for d in docs]
        self.df = {}
        self.doc_len = [len(t) for t in self.tokens]
        self.avgdl = sum(self.doc_len) / max(1, len(self.doc_len))

        for ts in self.tokens:
            seen = set(ts)
            for w in seen:
                self.df[w] = self.df.get(w, 0) + 1

        self.N = len(docs)

    def idf(self, w):
        df = self.df.get(w, 0)
        return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

    def score(self, query, idx):
        q = tokenize(query)
        if not q:
            return 0.0
        ts = self.tokens[idx]
        if not ts:
            return 0.0

        tf = {}
        for w in ts:
            tf[w] = tf.get(w, 0) + 1

        score = 0.0
        dl = self.doc_len[idx]
        for w in q:
            if w not in tf:
                continue
            f = tf[w]
            idf = self.idf(w)
            denom = f + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            score += idf * (f * (self.k1 + 1) / (denom + 1e-9))
        return score

    def search(self, query, top_k=20):
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
