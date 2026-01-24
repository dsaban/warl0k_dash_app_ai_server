# core/retriever.py
from typing import List, Dict, Optional
from .bm25 import BM25
from .embeddings import cosine_sim
from .utils import normalize_text
from .qtype import QTypeProfile

def coverage_score(qt: QTypeProfile, text: str) -> float:
    if qt.name == "generic":
        return 1.0
    t = " " + normalize_text(text) + " "
    for w in qt.required_all:
        if f" {w} " not in t:
            return 0.0
    if not qt.required_any:
        return 1.0
    hit = 0
    for w in qt.required_any:
        if f" {w} " in t:
            hit += 1
    return hit / (len(qt.required_any) + 1e-9)

def forbidden_penalty(qt: QTypeProfile, text: str) -> float:
    if not qt.forbidden:
        return 0.0
    t = " " + normalize_text(text) + " "
    hits = 0
    for w in qt.forbidden:
        if f" {w} " in t:
            hits += 1
    return min(0.7, 0.12 * hits)

class HybridRetriever:
    def __init__(self, chunks: List[str], vec512: List, bm25_k1=1.5, bm25_b=0.75):
        self.chunks = chunks
        self.vec512 = vec512
        self.bm25 = BM25(chunks, k1=bm25_k1, b=bm25_b)

    @staticmethod
    def _minmax(xs: List[float]) -> List[float]:
        if not xs:
            return xs
        mn, mx = min(xs), max(xs)
        if abs(mx - mn) < 1e-9:
            return [0.0 for _ in xs]
        return [(x - mn) / (mx - mn) for x in xs]

    def search(self, query: str, q_vec512, top_k: int, cfg, qtype: Optional[QTypeProfile] = None) -> List[Dict]:
        cands = self.bm25.search(query, top_k=max(top_k * 6, 18))
        bm25_scores = [s for _, s in cands]
        bm25_norm = self._minmax(bm25_scores)

        cos_scores = [cosine_sim(q_vec512, self.vec512[idx]) for idx, _ in cands]
        cos_norm = self._minmax(cos_scores)

        if qtype is None:
            cov_scores = [1.0] * len(cands)
            pen_scores = [0.0] * len(cands)
        else:
            cov_scores = [coverage_score(qtype, self.chunks[idx]) for idx, _ in cands]
            pen_scores = [forbidden_penalty(qtype, self.chunks[idx]) for idx, _ in cands]
        cov_norm = self._minmax(cov_scores)

        out = []
        for k, (idx, s_b) in enumerate(cands):
            comb = (cfg.alpha_bm25 * bm25_norm[k] + cfg.alpha_cos * cos_norm[k] + cfg.alpha_cov * cov_norm[k])
            comb = float(max(0.0, comb - pen_scores[k]))
            out.append({
                "idx": idx,
                "score": comb,
                "bm25": float(s_b),
                "cos": float(cos_scores[k]),
                "coverage": float(cov_scores[k]),
                "penalty": float(pen_scores[k]),
                "chunk": self.chunks[idx],
            })
        out.sort(key=lambda d: d["score"], reverse=True)
        return out[:top_k]
