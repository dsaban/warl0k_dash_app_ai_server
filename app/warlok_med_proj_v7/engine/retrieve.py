import math
from dataclasses import dataclass
from typing import List, Dict
from .utils import tokens
from .graph import Graph, Claim
from .frames import FrameSpec

class BM25Lite:
    def __init__(self, docs: List[str], k1=1.2, b=0.75):
        self.tokens = [tokens(d) for d in docs]
        self.df = {}
        for ts in self.tokens:
            for t in set(ts):
                self.df[t] = self.df.get(t, 0) + 1
        self.N = len(docs)
        self.avgdl = sum(len(ts) for ts in self.tokens) / max(1, self.N)
        self.k1 = k1
        self.b = b

    def score(self, q: str, i: int) -> float:
        qt = tokens(q)
        dt = self.tokens[i]
        if not dt:
            return 0.0
        freq = {}
        for t in dt:
            freq[t] = freq.get(t, 0) + 1
        dl = len(dt)
        s = 0.0
        for t in qt:
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            tf = freq.get(t, 0)
            denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
            s += idf * (tf * (self.k1 + 1)) / max(1e-9, denom)
        return s

    def topk(self, q: str, k=40) -> List[int]:
        scored = [(i, self.score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [i for i, s in scored[:k] if s > 0]

@dataclass
class Retrieval:
    seed: List[Claim]
    expanded: List[Claim]
    debug: Dict

def retrieve(graph: Graph, question: str, frame: FrameSpec,
             bm25_k=40, hops=2, limit=400) -> Retrieval:

    bm = BM25Lite([c.text for c in graph.claims])
    idxs = bm.topk(question, bm25_k)
    seed = [graph.claims[i] for i in idxs]
    seed.sort(key=lambda c: c.confidence, reverse=True)
    
    allowed = set(frame.allowed_node_types)
    blocked = set(frame.blocked_node_types)

    # def claim_ok(c: Claim) -> bool:
    #     if not c.nodes:
    #         return True
    #     types = [graph.node_type(n) for n in c.nodes]
    #     return not (types and all(t in blocked for t in types if t != "unknown"))
    
    def claim_ok(c: Claim) -> bool:
        if not c.nodes:
            return True
        types = [graph.node_type(n) for n in c.nodes]
        # reject if any blocked type is present
        if any(t in blocked for t in types):
            return False
        # also require at least one allowed type (optional but helps)
        if allowed and not any(t in allowed for t in types):
            return False
        return True
    
    out = []
    seen = set()
    for c in seed:
        if c.id not in seen and claim_ok(c):
            out.append(c)
            seen.add(c.id)

    frontier = list(out)
    for _ in range(hops):
        if not frontier or len(out) >= limit:
            break
        nodes = set()
        for c in frontier:
            for n in c.nodes:
                if graph.node_type(n) in allowed:
                    nodes.add(n)
        new = []
        for n in nodes:
            for c in graph.claims_by_node.get(n, []):
                if c.id in seen:
                    continue
                if not claim_ok(c):
                    continue
                new.append(c)
                seen.add(c.id)
        # new.sort(key=lambda c: (1 if c.edge_types else 0, len(c.edge_types)), reverse=True)
        new.sort(key=lambda c: (c.confidence, 1 if c.edge_types else 0, len(c.edge_types)), reverse=True)
        
        room = max(0, limit - len(out))
        out.extend(new[:room])
        frontier = new[:room]

    return Retrieval(seed, out[:limit], {
        "seed_n": len(seed),
        "expanded_n": len(out[:limit]),
        "bm25_k": bm25_k,
        "hops": hops,
        "limit": limit
    })
