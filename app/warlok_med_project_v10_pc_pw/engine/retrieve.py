from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter

from .utils import tokens
from .graph import Claim
from .domain_pack import DomainPack


def load_claims(index_dir: Path) -> List[Claim]:
    claims: List[Claim] = []
    for line in (index_dir / "claims.jsonl").read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        claims.append(Claim(**obj))
    return claims


def score_claim(query_toks: set, claim: Claim) -> Tuple[float, float, float]:
    """
    Simple BM25-ish: overlap + length normalization + confidence.
    """
    ctoks = set(tokens(claim.text))
    overlap = len(query_toks & ctoks)
    if overlap == 0:
        return (0.0, 0.0, 0.0)
    # penalize overly long claims slightly
    ln = max(1, len(ctoks))
    overlap_norm = overlap / (ln ** 0.35)
    conf = float(getattr(claim, "confidence", 0.0) or 0.0)
    return (overlap_norm, conf, overlap)


def retrieve(index_dir: Path, question: str, domain: DomainPack, frame: Dict[str, Any], k: int = 50) -> List[Claim]:
    all_claims = load_claims(index_dir)
    q = set(tokens(question))

    # base rank
    scored = []
    for c in all_claims:
        base, conf, raw_overlap = score_claim(q, c)
        if base > 0:
            scored.append((base, conf, raw_overlap, c))

    scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    seeds = [c for _, _, _, c in scored[: max(k, 80)]]

    # frame coverage boost: prioritize claims that hit required edges with strong evidence
    req = set(frame.get("required_edges") or [])
    boosted = []
    for c in seeds:
        ev = (c.edge_evidence or {})
        hit_edges = req & set(c.edge_types)
        hit_strength = sum(ev.get(e, 1) for e in hit_edges)
        boosted.append((len(hit_edges), hit_strength, c.confidence, c))

    boosted.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    return [c for _, _, _, c in boosted[:k]]

# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import List, Dict, Any
# from .utils import tokens
# from .graph import Claim
# from .domain_pack import DomainPack
#
#
# def load_claims(index_dir: Path) -> List[Claim]:
#     claims = []
#     for line in (index_dir / "claims.jsonl").read_text(errors="ignore").splitlines():
#         if not line.strip():
#             continue
#         obj = json.loads(line)
#         claims.append(Claim(**obj))
#     return claims
#
#
# def bm25ish_rank(claims: List[Claim], query: str, k: int = 40) -> List[Claim]:
#     q = set(tokens(query))
#     scored = []
#     for c in claims:
#         t = set(tokens(c.text))
#         score = len(q & t)
#         if score > 0:
#             scored.append((score, c.confidence, c))
#     scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
#     return [c for _, _, c in scored[:k]]
#
#
# def retrieve(index_dir: Path, question: str, domain: DomainPack, frame: Dict[str, Any], k: int = 40) -> List[Claim]:
#     all_claims = load_claims(index_dir)
#     seeds = bm25ish_rank(all_claims, question, k=k)
#
#     # frame-edge relevance boost
#     req = set(frame.get("required_edges") or [])
#     boosted = []
#     for c in seeds:
#         hit = len(set(c.edge_types) & req)
#         boosted.append((hit, c.confidence, c))
#     boosted.sort(key=lambda x: (x[0], x[1]), reverse=True)
#     return [c for _, _, c in boosted]
#
# # import math
# # from dataclasses import dataclass
# # from typing import List, Dict, Any
# # from .utils import tokens
# # from .graph import Graph, Claim
# #
# # class BM25Lite:
# #     def __init__(self, docs: List[str], k1=1.2, b=0.75):
# #         self.tokens = [tokens(d) for d in docs]
# #         self.df = {}
# #         for ts in self.tokens:
# #             for t in set(ts):
# #                 self.df[t] = self.df.get(t, 0) + 1
# #         self.N = len(docs)
# #         self.avgdl = sum(len(ts) for ts in self.tokens) / max(1, self.N)
# #         self.k1 = k1
# #         self.b = b
# #
# #     def score(self, q: str, i: int) -> float:
# #         qt = tokens(q)
# #         dt = self.tokens[i]
# #         if not dt:
# #             return 0.0
# #         freq = {}
# #         for t in dt:
# #             freq[t] = freq.get(t, 0) + 1
# #         dl = len(dt)
# #         s = 0.0
# #         for t in qt:
# #             df = self.df.get(t, 0)
# #             if not df:
# #                 continue
# #             idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
# #             tf = freq.get(t, 0)
# #             denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
# #             s += idf * (tf * (self.k1 + 1)) / max(1e-9, denom)
# #         return s
# #
# #     def topk(self, q: str, k=40) -> List[int]:
# #         scored = [(i, self.score(q, i)) for i in range(self.N)]
# #         scored.sort(key=lambda x: x[1], reverse=True)
# #         return [i for i, s in scored[:k] if s > 0]
# #
# # @dataclass
# # class Retrieval:
# #     seed: List[Claim]
# #     expanded: List[Claim]
# #     debug: Dict[str, Any]
# #
# # def retrieve(graph: Graph, question: str, frame_like: Dict[str, Any],
# #              bm25_k=40, hops=2, limit=400) -> Retrieval:
# #
# #     bm = BM25Lite([c.text for c in graph.claims])
# #     idxs = bm.topk(question, bm25_k)
# #     seed = [graph.claims[i] for i in idxs]
# #     seed.sort(key=lambda c: c.confidence, reverse=True)
# #
# #     allowed = set(frame_like.get("allowed_node_types", []))
# #     blocked = set(frame_like.get("blocked_node_types", []))
# #
# #     # STRONG filtering: reject if ANY blocked type is present
# #     def claim_ok(c: Claim) -> bool:
# #         if not c.nodes:
# #             return True
# #         types = [graph.node_type(n) for n in c.nodes]
# #         if any(t in blocked for t in types):
# #             return False
# #         if allowed and not any(t in allowed for t in types):
# #             return False
# #         return True
# #
# #     out = []
# #     seen = set()
# #     for c in seed:
# #         if c.id not in seen and claim_ok(c):
# #             out.append(c)
# #             seen.add(c.id)
# #
# #     frontier = list(out)
# #     for _ in range(hops):
# #         if not frontier or len(out) >= limit:
# #             break
# #         nodes = set()
# #         for c in frontier:
# #             for n in c.nodes:
# #                 if graph.node_type(n) in allowed:
# #                     nodes.add(n)
# #         new = []
# #         for n in nodes:
# #             for c in graph.claims_by_node.get(n, []):
# #                 if c.id in seen:
# #                     continue
# #                 if not claim_ok(c):
# #                     continue
# #                 new.append(c)
# #                 seen.add(c.id)
# #
# #         new.sort(key=lambda c: (c.confidence, 1 if c.edge_types else 0, len(c.edge_types)), reverse=True)
# #         room = max(0, limit - len(out))
# #         out.extend(new[:room])
# #         frontier = new[:room]
# #
# #     return Retrieval(seed, out[:limit], {
# #         "seed_n": len(seed),
# #         "expanded_n": len(out[:limit]),
# #         "bm25_k": bm25_k,
# #         "hops": hops,
# #         "limit": limit
# #     })
