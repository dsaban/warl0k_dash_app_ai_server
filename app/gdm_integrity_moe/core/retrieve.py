from typing import List, Tuple, Dict, Any

from .bm25 import BM25
from .scoring import score_breakdown
from .schema import QTYPE_REQUIREMENTS

class DualBM25Retriever:
    """
    Chunk retrieval: BM25 on chunk.text
    Claim ranking: BM25 on claim.support_text within top chunks
    Both use: schema boosts - penalties with breakdown.
    """
    def __init__(self, chunks, claims):
        self.chunks = chunks
        self.claims = claims

        self.chunk_bm25 = BM25([c.text for c in chunks])
        self.claim_bm25 = BM25([getattr(cl, "support_text", cl.sentence) for cl in claims])

        self.claims_by_chunk: Dict[str, List[int]] = {}
        for i, cl in enumerate(claims):
            self.claims_by_chunk.setdefault(cl.chunk_id, []).append(i)

    def search(self, question: str, qtype: str, top_chunks: int = 8, max_claims: int = 24):
        req = QTYPE_REQUIREMENTS.get(qtype, {})
        required_entities = req.get("required_entities_all", []) + req.get("required_entities_any", [])

        # 1) retrieve chunk candidates then rescore with breakdown total
        raw_chunks = self.chunk_bm25.topk(question, k=max(10, top_chunks * 4))
        chunk_scored: List[Tuple[int, Dict[str, float]]] = []
        for idx, bm in raw_chunks:
            ch = self.chunks[idx]
            bd = score_breakdown(qtype, ch.text, bm, required_entities, section=getattr(ch, "section", ""))
            chunk_scored.append((idx, bd))
        chunk_scored.sort(key=lambda x: x[1]["total"], reverse=True)
        chunk_hits = chunk_scored[:top_chunks]

        # 2) gather claim pool from top chunks
        pool = []
        seen = set()
        for ci, _bd in chunk_hits:
            chunk_id = self.chunks[ci].chunk_id
            for cl_i in self.claims_by_chunk.get(chunk_id, []):
                if cl_i in seen:
                    continue
                seen.add(cl_i)
                pool.append(cl_i)

        # 3) score claims with BM25 then apply breakdown total
        claim_scored: List[Tuple[int, Dict[str, float]]] = []
        for cl_i in pool:
            bm = self.claim_bm25.score_one(question, cl_i)
            cl = self.claims[cl_i]
            txt = getattr(cl, "support_text", cl.sentence)
            bd = score_breakdown(qtype, txt, bm, required_entities, section="")
            claim_scored.append((cl_i, bd))
        claim_scored.sort(key=lambda x: x[1]["total"], reverse=True)
        claim_hits = claim_scored[:max_claims]

        return chunk_hits, claim_hits
