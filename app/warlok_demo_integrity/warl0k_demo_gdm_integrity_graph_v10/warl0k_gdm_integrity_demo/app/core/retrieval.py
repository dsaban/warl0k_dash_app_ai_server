import re, math, numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict

_STOP = set("the a an and or to of in on for with without by from as is are was were be been being this that these those it its".split())

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%\.\/\- ]+", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP]
    return toks

class Retriever:
    """
    Hybrid retrieval:
      - BM25 for lexical precision
      - TF-IDF cosine for semantic-ish match (SBERT-lite)
    """
    def __init__(self, passages: List[Dict[str, Any]]):
        self.passages = passages
        self._build()

    def _build(self):
        self.toks = [_tokenize(p["text"]) for p in self.passages]
        self.N = len(self.passages)
        self.df = Counter()
        for toks in self.toks:
            for w in set(toks):
                self.df[w] += 1
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)

        # TF-IDF matrix (sparse-ish via dict of weights)
        vocab = {w:i for i,w in enumerate(self.df.keys())}
        self.vocab = vocab
        self.idf = np.zeros(len(vocab), dtype=np.float32)
        for w,i in vocab.items():
            self.idf[i] = math.log((self.N + 1) / (self.df[w] + 1)) + 1.0

        self.tfidf_rows = []
        self.norms = np.zeros(self.N, dtype=np.float32)
        for i, toks in enumerate(self.toks):
            tf = Counter(toks)
            row = {}
            for w,c in tf.items():
                if w in vocab:
                    j = vocab[w]
                    val = (c / len(toks)) * self.idf[j]
                    row[j] = float(val)
            norm = math.sqrt(sum(v*v for v in row.values())) if row else 0.0
            self.tfidf_rows.append(row)
            self.norms[i] = norm

    def _bm25(self, query_toks: List[str], idx: int, k1=1.5, b=0.75) -> float:
        tf = Counter(self.toks[idx])
        dl = len(self.toks[idx])
        score = 0.0
        for w in query_toks:
            if w not in tf: 
                continue
            df = self.df.get(w, 0)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            score += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * dl / self.avgdl))
        return score

    def _tfidf_cos(self, query_toks: List[str], idx: int) -> float:
        # build query vector
        tf = Counter(query_toks)
        qrow = {}
        for w,c in tf.items():
            if w in self.vocab:
                j = self.vocab[w]
                qrow[j] = (c / len(query_toks)) * float(self.idf[j])
        qnorm = math.sqrt(sum(v*v for v in qrow.values())) if qrow else 0.0
        if qnorm == 0.0 or self.norms[idx] == 0.0:
            return 0.0

        drow = self.tfidf_rows[idx]
        # dot product over smaller dict
        if len(qrow) < len(drow):
            dot = sum(v * drow.get(j, 0.0) for j,v in qrow.items())
        else:
            dot = sum(v * qrow.get(j, 0.0) for j,v in drow.items())
        return float(dot / (qnorm * float(self.norms[idx])))

    def search(self, query: str, k: int = 8, w_bm25: float = 0.6, w_tfidf: float = 0.4) -> List[Dict[str, Any]]:
        qtoks = _tokenize(query)
        scored = []
        for i in range(self.N):
            s1 = self._bm25(qtoks, i)
            s2 = self._tfidf_cos(qtoks, i)
            score = w_bm25 * s1 + w_tfidf * s2
            if score > 0:
                scored.append((score, s1, s2, i))
        scored.sort(reverse=True)
        out = []
        for rank,(score,s1,s2,i) in enumerate(scored[:k], start=1):
            text = self.passages[i]["text"]
            out.append({
                "rank": rank,
                "score": score,
                "bm25": s1,
                "tfidf": s2,
                "pid": self.passages[i]["pid"],
                "doc": self.passages[i]["doc"],
                "text": text,
                "highlights": self._highlight(text, qtoks),
            })
        return out

    def _highlight(self, text: str, qtoks: List[str]) -> str:
        # crude keyword highlight: wrap matched tokens with ** **
        out = text
        for w in sorted(set(qtoks), key=len, reverse=True):
            if len(w) < 3: 
                continue
            out = re.sub(rf"(?i)\b{re.escape(w)}\b", lambda m: f"**{m.group(0)}**", out)
        return out
