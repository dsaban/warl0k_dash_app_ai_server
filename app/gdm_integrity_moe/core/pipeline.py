from __future__ import annotations

import math
from typing import Dict, Any, List

from .qtype import detect_qtype
from .experts import EXPERTS
from .gates import evaluate_all
from .moe_router import MoERouter
from .ebm_ranker import EBMRanker
from .learned_rerank import load_reranker, RerankModel

class IntegrityMoEPipeline:
    def __init__(self, chunks, claims, retriever, router=None, ebm=None, rerank_path: str = ""):
        self.chunks = chunks
        self.claims = claims
        self.retriever = retriever
        self.router = router or MoERouter()
        self.ebm = ebm or EBMRanker()

        self.reranker: RerankModel | None = load_reranker(rerank_path) if rerank_path else None
        self.alpha_router = 0.25
        self.beta_energy = 1.0
        self.rerank_lambda = 0.35

    def infer(self, question: str, top_chunks: int = 8, max_claims: int = 24) -> Dict[str,Any]:
        qtype = detect_qtype(question)

        chunk_hits, claim_hits = self.retriever.search(question, qtype, top_chunks=top_chunks, max_claims=max_claims)

        # Optional learned rerank: adjust claim total score (keeps breakdown visible)
        if self.reranker is not None:
            rescored = []
            for cl_i, bd in claim_hits:
                cl = self.claims[cl_i]
                supp = getattr(cl, "support_text", cl.sentence)
                rr = self.reranker.score_pair(question, supp)
                bd = dict(bd)
                bd["rerank"] = float(rr)
                bd["total"] = float(bd["total"] + self.rerank_lambda * rr)
                rescored.append((cl_i, bd))
            rescored.sort(key=lambda x: x[1]["total"], reverse=True)
            claim_hits = rescored

        # Ranked list for experts: [(claim_idx, score)]
        ranked = [(i, bd["total"]) for (i, bd) in claim_hits]

        router_out = self.router.route(question, qtype)

        candidates = []
        for name, fn in EXPERTS.items():
            candidates.append(fn(self.claims, ranked))

        scored = []
        for cand in candidates:
            answer_text = "\n".join([s["text"] for s in cand.sentences]).strip()
            gate = evaluate_all(qtype, cand.sentences, cand.used_entities, cand.used_edges, answer_text)
            ebm_out = self.ebm.score(cand, gate)

            rw = float(router_out.weights.get(cand.expert_name, router_out.weights.get("generic", 0.0)))
            total = float(self.beta_energy * ebm_out.energy - self.alpha_router * (math.log(rw + 1e-6)))

            scored.append({
                "expert": cand.expert_name,
                "router_weight": rw,
                "energy": float(ebm_out.energy),
                "total": total,
                "gates": gate,
                "features": ebm_out.feat,
                "answer_text": answer_text,
                "sentences": cand.sentences,
                "used_entities": cand.used_entities,
                "used_edges": cand.used_edges,
                "blueprint": cand.blueprint,
            })

        scored.sort(key=lambda x: x["total"])
        best = scored[0] if scored else None
        final = best["gates"] if best else {"decision":"ABSTAIN","flags":["no_candidates"]}

        # UI-friendly serialization
        chunk_hits_ui = [{"chunk_idx": i, "breakdown": bd} for (i, bd) in chunk_hits]
        claim_hits_ui = [{"claim_idx": i, "breakdown": bd} for (i, bd) in claim_hits]

        return {
            "qtype": qtype,
            "router": {"weights": router_out.weights, "conf": router_out.conf, "logits": router_out.logits},
            "chunk_hits": chunk_hits_ui,
            "claim_hits": claim_hits_ui,
            "candidates": scored,
            "best": best,
            "final": final,
        }
