# core/pipeline.py
import numpy as np
from .config import AppConfig
from .chunking import split_into_chunks
from .embeddings import HashingEmbedder, Projector
from .retriever import HybridRetriever, coverage_score
from .qtype import detect_qtype
from .role import role_alignment
from .schema import fill_schema
from .composer import compose_from_schema
from .entailment import entailment_check_classification
from .memory import GatedRNNMemory
from .moe import MoE
from .ebm import EnergyModel
from .integrity import integrity_decision

class LocalMedMoEPipeline:
    def __init__(self, docs_text: str, cfg: AppConfig, weights_path: str = None):
        self.cfg = cfg
        self.docs_text = docs_text or ""
        self.chunks = split_into_chunks(self.docs_text, cfg.chunk_size, cfg.chunk_overlap)

        self.embedder = HashingEmbedder(dim=cfg.hash_dim)
        self.projector = Projector(out_dim=cfg.mem_dim, in_dim=cfg.hash_dim, seed=5)

        self.chunk_vec512 = [self.embedder.embed(c) for c in self.chunks]
        self.retriever = HybridRetriever(self.chunks, self.chunk_vec512)

        self.mem = GatedRNNMemory(d=cfg.mem_dim, max_steps=cfg.mem_steps_max)

        self.moe = MoE(d_in=cfg.mem_dim, n_experts=cfg.n_experts, hidden=cfg.expert_hidden)
        self.ebm = EnergyModel(d=cfg.mem_dim)

        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path: str):
        d = np.load(path, allow_pickle=False)
        self.moe.Wr = d["Wr"].astype(np.float32)
        self.moe.br = d["br"].astype(np.float32)
        self.ebm.W = d["ebmW"].astype(np.float32)
        self.ebm.u = d["ebmU"].astype(np.float32)
        self.ebm.b = d["ebmB"].astype(np.float32).item() if d["ebmB"].shape else d["ebmB"].astype(np.float32)

    def _qvec512(self, text: str) -> np.ndarray:
        return self.embedder.embed(text)

    def _qvec(self, text: str) -> np.ndarray:
        return self.projector.proj(self._qvec512(text))

    def _drift_chunk(self, qtype, chunk: str) -> bool:
        if not qtype.forbidden:
            return False
        t = " " + chunk.lower() + " "
        forbid_hits = sum(1 for w in qtype.forbidden if f" {w} " in t)
        req_hits = sum(1 for w in qtype.required_any if f" {w} " in t) if qtype.required_any else 0
        return (forbid_hits >= 2 and req_hits == 0)

    def _retrieve(self, qtype, queries):
        merged = {}
        plan_debug = []
        for sq in queries:
            hits = self.retriever.search(sq, self._qvec512(sq), top_k=max(2, self.cfg.top_k // 2), cfg=self.cfg, qtype=qtype)
            plan_debug.append({"sub_query": sq, "hits": hits})
            for h in hits:
                merged[h["idx"]] = max(merged.get(h["idx"], 0.0), float(h["score"]))

        items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: self.cfg.top_k]
        evidence = [self.chunks[idx] for idx, _ in items]

        # drift filter
        ev2 = [ch for ch in evidence if not self._drift_chunk(qtype, ch)]
        evidence = ev2 if ev2 else evidence

        cov0 = coverage_score(qtype, evidence[0]) if evidence else 0.0
        return items, evidence, cov0, plan_debug

    def infer(self, user_query: str) -> dict:
        qtype = detect_qtype(user_query)

        # ---- plan (qtype aware) ----
        if qtype.name == "placental_hormones_pathologic_gdm":
            plan_queries = [
                user_query + " placental hormones hPL progesterone cortisol growth hormone",
                user_query + " anti-insulin effect insulin resistance pregnancy",
                user_query + " beta-cell compensation increased insulin secretion",
                user_query + " pathological in GDM inadequate compensation hyperglycemia",
            ]
        elif qtype.name == "maternal_beta_cell_failure":
            plan_queries = [
                user_query + " pregnancy insulin resistance placental hormones",
                user_query + " beta-cell compensation increased insulin secretion",
                user_query + " inadequate compensation beta-cell dysfunction failure in some women",
                "why compensation fails in some women beta-cell reserve dysfunction",
            ]
        elif qtype.name == "maternal_progression_model":
            plan_queries = [
                user_query + " pregnancy insulin resistance stress test",
                user_query + " beta-cell insulin secretion compensation failure",
                user_query + " postpartum progression type 2 overt diabetes",
            ]
        elif qtype.name == "maternal_inflammation_cardiometabolic":
            plan_queries = [
                user_query + " obesity inflammation cytokines adipose insulin resistance",
                user_query + " endothelial dysfunction dyslipidemia hypertension",
                user_query + " postpartum long-term cardiometabolic cardiovascular risk type 2",
            ]
        elif qtype.name == "classification":
            plan_queries = [
                user_query + " classified considered defined described as represents prediabetes",
                "Find exact wording that classifies the condition as prediabetes (not just increased risk).",
            ]
        else:
            plan_queries = [user_query]

        # ---- retrieval pass 1 ----
        merged_items, evidence, cov0, plan_debug = self._retrieve(qtype, plan_queries)

        # ---- re-retrieve with anchors if low intent coverage ----
        reretry_debug = []
        if (qtype.name != "generic" and cov0 < self.cfg.min_intent_cov and qtype.anchors and self.cfg.max_reretries > 0):
            merged2, evidence2, cov2, dbg2 = self._retrieve(qtype, list(qtype.anchors))
            reretry_debug = dbg2
            if cov2 > cov0:
                merged_items, evidence, cov0 = merged2, evidence2, cov2

        # ---- role alignment ----
        rs = role_alignment(evidence, qtype.target_role)
        role_ok = True if qtype.target_role == "generic" else (rs.alignment >= self.cfg.min_role_align)

        # ---- schema fill (dedup + critical coverage) ----
        schema = fill_schema(qtype.schema_id, evidence)

        # Deep qtypes require critical coverage
        if qtype.name in ("maternal_beta_cell_failure", "placental_hormones_pathologic_gdm"):
            schema_ok = (schema.get("critical_coverage", 0.0) >= 0.75)
        else:
            schema_ok = (schema.get("coverage", 0.0) >= self.cfg.min_schema_coverage) if qtype.schema_id != "generic" else True

        # ---- entailment (classification) ----
        entailment = None
        if qtype.needs_entailment:
            entailment = entailment_check_classification("gdm is a prediabetic state", evidence_chunks=evidence, min_score=0.18)

        # ---- failure requirement gate ----
        failure_ok = True
        if qtype.needs_failure_explanation:
            # For placental_hormones_pathologic_gdm, failure is "Pathological transition in GDM"
            if qtype.name == "placental_hormones_pathologic_gdm":
                failure_ok = any(x["slot"] == "Pathological transition in GDM" for x in schema.get("filled", []))
            else:
                failure_ok = any(x["slot"] == "Failure in susceptible women" for x in schema.get("filled", []))

        # ---- hormone hard gate (prevents generic-like answers) ----
        hormone_ok = True
        if qtype.name == "placental_hormones_pathologic_gdm":
            hormone_ok = any(x["slot"] == "Placental hormones → insulin resistance" for x in schema.get("filled", []))

        # If missing, try one extra retrieval focused on hormones/failure
        if qtype.name == "placental_hormones_pathologic_gdm" and (not hormone_ok or not failure_ok):
            merged3, evidence3, cov3, dbg3 = self._retrieve(qtype, [
                "placental hormones hPL progesterone cortisol growth hormone anti-insulin insulin resistance",
                "GDM inadequate beta-cell compensation hyperglycemia pathological insulin resistance",
            ])
            reretry_debug.extend(dbg3)
            schema3 = fill_schema(qtype.schema_id, evidence3)
            hormone_ok3 = any(x["slot"] == "Placental hormones → insulin resistance" for x in schema3.get("filled", []))
            failure_ok3 = any(x["slot"] == "Pathological transition in GDM" for x in schema3.get("filled", []))
            if hormone_ok3 or failure_ok3:
                merged_items, evidence, cov0 = merged3, evidence3, cov3
                schema = schema3
                hormone_ok = hormone_ok3
                failure_ok = failure_ok3
                rs = role_alignment(evidence, qtype.target_role)
                role_ok = (rs.alignment >= self.cfg.min_role_align)
                schema_ok = (schema.get("critical_coverage", 0.0) >= 0.75)

        # ---- memory ingest ----
        self.mem.reset()
        self.mem.step(self._qvec("USER: " + user_query))
        for pq in plan_queries[:3]:
            self.mem.step(self._qvec("PLAN: " + pq))
        for ch in evidence[:2]:
            self.mem.step(self._qvec("DOC: " + ch))

        mem_state = self.mem.h.copy()
        attn_ctx = self.mem.attend(mem_state)["context"]

        # ---- MoE + EBM candidates ----
        qv = self._qvec(user_query)
        x = (qv + 0.6 * mem_state + 0.4 * attn_ctx).astype(np.float32)

        moe_out = self.moe.forward(x)
        w = moe_out["weights"]
        router_conf = moe_out["router_conf"]
        expert_vecs = moe_out["expert_vecs"]

        top_idx = list(np.argsort(-w)[: self.cfg.n_candidates])
        energies, candidates = [], []
        ev = self._qvec(evidence[0] if evidence else "")

        for k in top_idx:
            av = (0.7 * expert_vecs[int(k)] + 0.3 * qv).astype(np.float32)
            E = self.ebm.energy(qv, av, ev)
            energies.append(float(E))
            candidates.append({"expert": int(k), "weight": float(w[int(k)]), "energy": float(E)})

        candidates.sort(key=lambda d: d["energy"])
        best = candidates[0] if candidates else {"expert": -1, "weight": 0.0, "energy": 9.9}

        # ---- gates + integrity ----
        gates = {
            "intent_ok": (qtype.name == "generic") or (cov0 >= self.cfg.min_intent_cov),
            "role_ok": role_ok,
            "schema_ok": schema_ok,
            "failure_ok": failure_ok,
            "hormone_ok": hormone_ok,
        }

        integrity = integrity_decision(self.cfg, router_conf, energies, gates, entailment=entailment)

        # Force abstain if required critical gates fail
        if qtype.needs_failure_explanation and not failure_ok:
            integrity["decision"] = "ABSTAIN"
            if "missing_failure_explanation" not in integrity["flags"]:
                integrity["flags"].append("missing_failure_explanation")

        if qtype.name == "placental_hormones_pathologic_gdm" and not hormone_ok:
            integrity["decision"] = "ABSTAIN"
            if "missing_placental_hormone_mechanism" not in integrity["flags"]:
                integrity["flags"].append("missing_placental_hormone_mechanism")

        # ---- answer ----
        composed = compose_from_schema(qtype.name, schema)

        if integrity["decision"] == "ABSTAIN":
            final_answer = (
                "I can’t support a complete answer to this question from the retrieved text.\n\n"
                f"Why:\n"
                f"- qtype={qtype.name}\n"
                f"- intent_top_coverage={cov0:.3f} (min {self.cfg.min_intent_cov:.3f})\n"
                f"- role_alignment(target={rs.target})={rs.alignment:.3f} (min {self.cfg.min_role_align:.3f})\n"
                f"- schema_coverage={schema.get('coverage',0):.2f} | critical={schema.get('critical_coverage',0):.2f}\n"
                f"- hormone_ok={hormone_ok} | failure_ok={failure_ok}\n"
                f"- router_conf={router_conf:.3f} (min {self.cfg.router_conf_min:.3f})\n"
                f"- flags={integrity['flags']}\n"
            )
        else:
            final_answer = composed

        return {
            "final_answer": final_answer,
            "qtype": qtype.name,
            "plan": plan_queries,
            "plan_debug": plan_debug,
            "reretry_debug": reretry_debug,
            "retrieval": [{"idx": idx, "score": float(score), "chunk": self.chunks[idx]} for idx, score in merged_items],
            "intent": {"top_coverage": float(cov0)},
            "role": {"target": rs.target, "maternal": rs.maternal, "offspring": rs.offspring, "monitoring": rs.monitoring, "alignment": rs.alignment},
            "schema": schema,
            "entailment": entailment,
            "moe": {"weights": w.tolist(), "router_conf": float(router_conf), "top_experts": top_idx},
            "candidates": candidates,
            "best": best,
            "integrity": integrity,
        }

# # core/pipeline.py
# import numpy as np
# from .config import AppConfig
# from .chunking import split_into_chunks
# from .embeddings import HashingEmbedder, Projector
# from .retriever import HybridRetriever, coverage_score
# from .qtype import detect_qtype
# from .role import role_alignment
# from .schema import fill_schema
# from .composer import compose_from_schema
# from .entailment import entailment_check_classification
# from .memory import GatedRNNMemory
# from .moe import MoE
# from .ebm import EnergyModel
# from .integrity import integrity_decision
#
# class LocalMedMoEPipeline:
#     def __init__(self, docs_text: str, cfg: AppConfig, weights_path: str = None):
#         self.cfg = cfg
#         self.docs_text = docs_text or ""
#         self.chunks = split_into_chunks(self.docs_text, cfg.chunk_size, cfg.chunk_overlap)
#
#         self.embedder = HashingEmbedder(dim=cfg.hash_dim)
#         self.projector = Projector(out_dim=cfg.mem_dim, in_dim=cfg.hash_dim, seed=5)
#
#         self.chunk_vec512 = [self.embedder.embed(c) for c in self.chunks]
#         self.retriever = HybridRetriever(self.chunks, self.chunk_vec512)
#
#         self.mem = GatedRNNMemory(d=cfg.mem_dim, max_steps=cfg.mem_steps_max)
#
#         self.moe = MoE(d_in=cfg.mem_dim, n_experts=cfg.n_experts, hidden=cfg.expert_hidden)
#         self.ebm = EnergyModel(d=cfg.mem_dim)
#
#         if weights_path:
#             self.load_weights(weights_path)
#
#     def load_weights(self, path: str):
#         d = np.load(path, allow_pickle=False)
#         self.moe.Wr = d["Wr"].astype(np.float32)
#         self.moe.br = d["br"].astype(np.float32)
#         self.ebm.W = d["ebmW"].astype(np.float32)
#         self.ebm.u = d["ebmU"].astype(np.float32)
#         self.ebm.b = d["ebmB"].astype(np.float32).item() if d["ebmB"].shape else d["ebmB"].astype(np.float32)
#
#     def _qvec512(self, text: str) -> np.ndarray:
#         return self.embedder.embed(text)
#
#     def _qvec(self, text: str) -> np.ndarray:
#         return self.projector.proj(self._qvec512(text))
#
#     def _drift_chunk(self, qtype, chunk: str) -> bool:
#         if not qtype.forbidden:
#             return False
#         t = " " + chunk.lower() + " "
#         forbid_hits = sum(1 for w in qtype.forbidden if f" {w} " in t)
#         req_hits = sum(1 for w in qtype.required_any if f" {w} " in t) if qtype.required_any else 0
#         return (forbid_hits >= 2 and req_hits == 0)
#
#     def _retrieve(self, qtype, queries):
#         merged = {}
#         plan_debug = []
#         for sq in queries:
#             hits = self.retriever.search(sq, self._qvec512(sq), top_k=max(2, self.cfg.top_k // 2), cfg=self.cfg, qtype=qtype)
#             plan_debug.append({"sub_query": sq, "hits": hits})
#             for h in hits:
#                 merged[h["idx"]] = max(merged.get(h["idx"], 0.0), float(h["score"]))
#
#         items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: self.cfg.top_k]
#         evidence = [self.chunks[idx] for idx, _ in items]
#         # drift filter
#         ev2 = [ch for ch in evidence if not self._drift_chunk(qtype, ch)]
#         evidence = ev2 if ev2 else evidence
#         cov0 = coverage_score(qtype, evidence[0]) if evidence else 0.0
#         return items, evidence, cov0, plan_debug
#
#     def infer(self, user_query: str) -> dict:
#         qtype = detect_qtype(user_query)
#
#         # ---- plan (qtype aware) ----
#         if qtype.name == "maternal_beta_cell_failure":
#             plan_queries = [
#                 user_query + " pregnancy insulin resistance placental hormones",
#                 user_query + " beta-cell compensation increased insulin secretion",
#                 user_query + " inadequate compensation beta-cell dysfunction failure in some women",
#                 "why compensation fails in some women beta-cell reserve dysfunction",
#             ]
#         elif qtype.name == "maternal_progression_model":
#             plan_queries = [
#                 user_query + " pregnancy insulin resistance stress test",
#                 user_query + " beta-cell insulin secretion compensation failure",
#                 user_query + " postpartum progression type 2 overt diabetes",
#             ]
#         elif qtype.name == "maternal_inflammation_cardiometabolic":
#             plan_queries = [
#                 user_query + " obesity inflammation cytokines adipose insulin resistance",
#                 user_query + " endothelial dysfunction dyslipidemia hypertension",
#                 user_query + " postpartum long-term cardiometabolic cardiovascular risk type 2",
#             ]
#         elif qtype.name == "classification":
#             plan_queries = [
#                 user_query + " classified considered defined described as represents prediabetes",
#                 "Find exact wording that classifies the condition as prediabetes (not just increased risk).",
#             ]
#         else:
#             plan_queries = [user_query]
#
#         # ---- retrieval pass 1 ----
#         merged_items, evidence, cov0, plan_debug = self._retrieve(qtype, plan_queries)
#
#         # ---- re-retrieve with anchors if low intent coverage ----
#         reretry_debug = []
#         if (qtype.name != "generic" and cov0 < self.cfg.min_intent_cov and qtype.anchors and self.cfg.max_reretries > 0):
#             merged2, evidence2, cov2, dbg2 = self._retrieve(qtype, list(qtype.anchors))
#             reretry_debug = dbg2
#             if cov2 > cov0:
#                 merged_items, evidence, cov0 = merged2, evidence2, cov2
#
#         # ---- role alignment ----
#         rs = role_alignment(evidence, qtype.target_role)
#         role_ok = True if qtype.target_role == "generic" else (rs.alignment >= self.cfg.min_role_align)
#
#         # ---- schema fill (dedup + critical coverage) ----
#         schema = fill_schema(qtype.schema_id, evidence)
#
#         # Require stronger coverage for deep mechanism questions
#         if qtype.name == "maternal_beta_cell_failure":
#             schema_ok = (schema["critical_coverage"] >= 0.75)
#         else:
#             schema_ok = (schema["coverage"] >= self.cfg.min_schema_coverage) if qtype.schema_id != "generic" else True
#
#         # ---- entailment (classification) ----
#         entailment = None
#         if qtype.needs_entailment:
#             entailment = entailment_check_classification("gdm is a prediabetic state", evidence_chunks=evidence, min_score=0.18)
#
#         # ---- failure requirement gate (NEW) ----
#         failure_ok = True
#         if qtype.needs_failure_explanation:
#             # must have the critical failure slot filled
#             failure_ok = any(x["slot"] == "Failure in susceptible women" for x in schema.get("filled", []))
#             if not failure_ok and qtype.anchors:
#                 # last attempt: failure anchors
#                 merged3, evidence3, cov3, dbg3 = self._retrieve(qtype, [
#                     "inadequate beta-cell compensation failure insufficient insulin secretion",
#                     "beta-cell dysfunction limited beta-cell reserve cannot compensate pregnancy insulin resistance",
#                 ])
#                 reretry_debug.extend(dbg3)
#                 schema3 = fill_schema(qtype.schema_id, evidence3)
#                 failure_ok3 = any(x["slot"] == "Failure in susceptible women" for x in schema3.get("filled", []))
#                 # accept improvement if it fixes failure slot
#                 if failure_ok3:
#                     merged_items, evidence, cov0 = merged3, evidence3, cov3
#                     schema = schema3
#                     failure_ok = True
#                     # recompute role
#                     rs = role_alignment(evidence, qtype.target_role)
#                     role_ok = (rs.alignment >= self.cfg.min_role_align)
#
#         # ---- memory ingest ----
#         self.mem.reset()
#         self.mem.step(self._qvec("USER: " + user_query))
#         for pq in plan_queries[:3]:
#             self.mem.step(self._qvec("PLAN: " + pq))
#         for ch in evidence[:2]:
#             self.mem.step(self._qvec("DOC: " + ch))
#
#         mem_state = self.mem.h.copy()
#         attn_ctx = self.mem.attend(mem_state)["context"]
#
#         # ---- MoE + EBM candidates ----
#         qv = self._qvec(user_query)
#         x = (qv + 0.6 * mem_state + 0.4 * attn_ctx).astype(np.float32)
#
#         moe_out = self.moe.forward(x)
#         w = moe_out["weights"]
#         router_conf = moe_out["router_conf"]
#         expert_vecs = moe_out["expert_vecs"]
#
#         top_idx = list(np.argsort(-w)[: self.cfg.n_candidates])
#         energies, candidates = [], []
#         ev = self._qvec(evidence[0] if evidence else "")
#
#         for k in top_idx:
#             av = (0.7 * expert_vecs[int(k)] + 0.3 * qv).astype(np.float32)
#             E = self.ebm.energy(qv, av, ev)
#             energies.append(float(E))
#             candidates.append({"expert": int(k), "weight": float(w[int(k)]), "energy": float(E)})
#
#         candidates.sort(key=lambda d: d["energy"])
#         best = candidates[0] if candidates else {"expert": -1, "weight": 0.0, "energy": 9.9}
#
#         # ---- gates + integrity ----
#         gates = {
#             "intent_ok": (qtype.name == "generic") or (cov0 >= self.cfg.min_intent_cov),
#             "role_ok": role_ok,
#             "schema_ok": schema_ok,
#             "failure_ok": failure_ok,
#         }
#
#         integrity = integrity_decision(self.cfg, router_conf, energies, gates, entailment=entailment)
#
#         # if failure missing, force abstain
#         if not failure_ok:
#             integrity["decision"] = "ABSTAIN"
#             if "missing_failure_explanation" not in integrity["flags"]:
#                 integrity["flags"].append("missing_failure_explanation")
#
#         # ---- build answer: composer (NEW) ----
#         composed = compose_from_schema(qtype.name, schema)
#
#         if integrity["decision"] == "ABSTAIN":
#             final_answer = (
#                 "I can’t support a complete answer to this question from the retrieved text.\n\n"
#                 f"Why:\n"
#                 f"- qtype={qtype.name}\n"
#                 f"- intent_top_coverage={cov0:.3f} (min {self.cfg.min_intent_cov:.3f})\n"
#                 f"- role_alignment(target={rs.target})={rs.alignment:.3f} (min {self.cfg.min_role_align:.3f})\n"
#                 f"- schema_coverage={schema.get('coverage',0):.2f} | critical={schema.get('critical_coverage',0):.2f}\n"
#                 f"- failure_required={qtype.needs_failure_explanation} | failure_ok={failure_ok}\n"
#                 f"- router_conf={router_conf:.3f} (min {self.cfg.router_conf_min:.3f})\n"
#                 f"- flags={integrity['flags']}\n"
#             )
#         else:
#             final_answer = composed
#
#         return {
#             "final_answer": final_answer,
#             "qtype": qtype.name,
#             "plan": plan_queries,
#             "plan_debug": plan_debug,
#             "reretry_debug": reretry_debug,
#             "retrieval": [{"idx": idx, "score": float(score), "chunk": self.chunks[idx]} for idx, score in merged_items],
#             "intent": {"top_coverage": float(cov0)},
#             "role": {"target": rs.target, "maternal": rs.maternal, "offspring": rs.offspring, "monitoring": rs.monitoring, "alignment": rs.alignment},
#             "schema": schema,
#             "entailment": entailment,
#             "moe": {"weights": w.tolist(), "router_conf": float(router_conf), "top_experts": top_idx},
#             "candidates": candidates,
#             "best": best,
#             "integrity": integrity,
#         }
#
# # # core/pipeline.py
# # import numpy as np
# # from .config import AppConfig
# # from .chunking import split_into_chunks
# # from .embeddings import HashingEmbedder, Projector
# # from .retriever import HybridRetriever, coverage_score
# # from .qtype import detect_qtype
# # from .role import role_alignment
# # from .schema import fill_schema
# # from .entailment import entailment_check_classification
# # from .memory import GatedRNNMemory
# # from .moe import MoE
# # from .ebm import EnergyModel
# # from .integrity import integrity_decision
# #
# # def _minmax(xs):
# #     if not xs:
# #         return xs
# #     mn, mx = min(xs), max(xs)
# #     if abs(mx - mn) < 1e-9:
# #         return [0.0 for _ in xs]
# #     return [(x - mn) / (mx - mn) for x in xs]
# #
# # class LocalMedMoEPipeline:
# #     """
# #     Single stable class. No more broken imports.
# #     """
# #     def __init__(self, docs_text: str, cfg: AppConfig, weights_path: str = None):
# #         self.cfg = cfg
# #         self.docs_text = docs_text or ""
# #         self.chunks = split_into_chunks(self.docs_text, cfg.chunk_size, cfg.chunk_overlap)
# #
# #         self.embedder = HashingEmbedder(dim=cfg.hash_dim)
# #         self.projector = Projector(out_dim=cfg.mem_dim, in_dim=cfg.hash_dim, seed=5)
# #
# #         # Precompute chunk vectors
# #         self.chunk_vec512 = [self.embedder.embed(c) for c in self.chunks]
# #
# #         self.retriever = HybridRetriever(self.chunks, self.chunk_vec512)
# #         self.mem = GatedRNNMemory(d=cfg.mem_dim, max_steps=cfg.mem_steps_max)
# #
# #         self.moe = MoE(d_in=cfg.mem_dim, n_experts=cfg.n_experts, hidden=cfg.expert_hidden)
# #         self.ebm = EnergyModel(d=cfg.mem_dim)
# #
# #         if weights_path:
# #             self.load_weights(weights_path)
# #
# #     def load_weights(self, path: str):
# #         d = np.load(path, allow_pickle=False)
# #         # expected keys
# #         self.moe.Wr = d["Wr"].astype(np.float32)
# #         self.moe.br = d["br"].astype(np.float32)
# #         self.ebm.W = d["ebmW"].astype(np.float32)
# #         self.ebm.u = d["ebmU"].astype(np.float32)
# #         self.ebm.b = d["ebmB"].astype(np.float32).item() if d["ebmB"].shape else d["ebmB"].astype(np.float32)
# #
# #     def _qvec512(self, text: str) -> np.ndarray:
# #         return self.embedder.embed(text)
# #
# #     def _qvec(self, text: str) -> np.ndarray:
# #         return self.projector.proj(self._qvec512(text))
# #
# #     def _drift_chunk(self, qtype, chunk: str) -> bool:
# #         if not qtype.forbidden:
# #             return False
# #         t = " " + chunk.lower() + " "
# #         forbid_hits = sum(1 for w in qtype.forbidden if f" {w} " in t)
# #         req_hits = sum(1 for w in qtype.required_any if f" {w} " in t) if qtype.required_any else 0
# #         return (forbid_hits >= 2 and req_hits == 0)
# #
# #     def infer(self, user_query: str) -> dict:
# #         qtype = detect_qtype(user_query)
# #
# #         # ---- plan (simple, qtype aware) ----
# #         plan = []
# #         if qtype.name == "maternal_progression_model":
# #             plan = [
# #                 {"type":"mechanism", "sub_query": user_query + " pregnancy insulin resistance stress test"},
# #                 {"type":"beta_cell", "sub_query": user_query + " beta-cell insulin secretion compensation failure"},
# #                 {"type":"postpartum", "sub_query": user_query + " postpartum progression type 2 overt diabetes"},
# #             ]
# #         elif qtype.name == "maternal_inflammation_cardiometabolic":
# #             plan = [
# #                 {"type":"mechanism", "sub_query": user_query + " obesity inflammation cytokines adipose insulin resistance"},
# #                 {"type":"pathway", "sub_query": user_query + " endothelial dysfunction dyslipidemia hypertension"},
# #                 {"type":"outcome", "sub_query": user_query + " postpartum long-term cardiometabolic cardiovascular risk type 2"},
# #             ]
# #         elif qtype.name == "classification":
# #             plan = [
# #                 {"type":"class", "sub_query": user_query + " classified considered defined described as represents"},
# #                 {"type":"exact", "sub_query": "Find exact wording that classifies the condition as prediabetes (not just increased risk)."},
# #             ]
# #         else:
# #             plan = [{"type":"default", "sub_query": user_query}]
# #
# #         # ---- retrieval (merge over plan steps) ----
# #         merged = {}
# #         plan_debug = []
# #         for st in plan:
# #             sq = st["sub_query"]
# #             hits = self.retriever.search(sq, self._qvec512(sq), top_k=max(2, self.cfg.top_k // 2), cfg=self.cfg, qtype=qtype)
# #             plan_debug.append({"step": st, "hits": hits})
# #             for h in hits:
# #                 merged[h["idx"]] = max(merged.get(h["idx"], 0.0), float(h["score"]))
# #
# #         if not merged:
# #             hits = self.retriever.search(user_query, self._qvec512(user_query), top_k=self.cfg.top_k, cfg=self.cfg, qtype=qtype)
# #             plan_debug.append({"step": {"type":"fallback","sub_query": user_query}, "hits": hits})
# #             for h in hits:
# #                 merged[h["idx"]] = float(h["score"])
# #
# #         merged_items = sorted(merged.items(), key=lambda x: x[1], reverse=True)[: self.cfg.top_k]
# #         evidence = [self.chunks[idx] for idx, _ in merged_items]
# #
# #         # ---- drift filter ----
# #         ev2 = [ch for ch in evidence if not self._drift_chunk(qtype, ch)]
# #         evidence = ev2 if ev2 else evidence
# #
# #         # ---- re-retrieve anchors if low coverage ----
# #         cov0 = coverage_score(qtype, evidence[0]) if evidence else 0.0
# #         reretry_debug = []
# #         if (qtype.name != "generic" and cov0 < self.cfg.min_intent_cov and self.cfg.max_reretries > 0 and qtype.anchors):
# #             merged2 = {}
# #             for a in qtype.anchors:
# #                 hits = self.retriever.search(a, self._qvec512(a), top_k=max(2, self.cfg.top_k // 2), cfg=self.cfg, qtype=qtype)
# #                 reretry_debug.append({"anchor": a, "hits": hits})
# #                 for h in hits:
# #                     merged2[h["idx"]] = max(merged2.get(h["idx"], 0.0), float(h["score"]))
# #             if merged2:
# #                 items2 = sorted(merged2.items(), key=lambda x: x[1], reverse=True)[: self.cfg.top_k]
# #                 evidence2 = [self.chunks[idx] for idx, _ in items2]
# #                 cov2 = coverage_score(qtype, evidence2[0]) if evidence2 else 0.0
# #                 if cov2 > cov0:
# #                     merged_items = items2
# #                     evidence = evidence2
# #                     cov0 = cov2
# #
# #         # ---- role alignment ----
# #         rs = role_alignment(evidence, qtype.target_role)
# #         role_ok = True if qtype.target_role == "generic" else (rs.alignment >= self.cfg.min_role_align)
# #
# #         # ---- schema fill ----
# #         schema = fill_schema(qtype.schema_id, evidence)
# #         schema_ok = (schema["coverage"] >= self.cfg.min_schema_coverage) if qtype.schema_id != "generic" else True
# #
# #         # ---- entailment (classification) ----
# #         entailment = None
# #         if qtype.needs_entailment:
# #             entailment = entailment_check_classification("gdm is a prediabetic state", evidence_chunks=evidence, min_score=0.18)
# #
# #         # ---- memory ingest ----
# #         self.mem.reset()
# #         self.mem.step(self._qvec("USER: " + user_query))
# #         for st in plan[:3]:
# #             self.mem.step(self._qvec("PLAN: " + st["sub_query"]))
# #         for ch in evidence[:2]:
# #             self.mem.step(self._qvec("DOC: " + ch))
# #
# #         mem_state = self.mem.h.copy()
# #         attn_ctx = self.mem.attend(mem_state)["context"]
# #
# #         # ---- MoE + EBM candidates ----
# #         qv = self._qvec(user_query)
# #         x = (qv + 0.6 * mem_state + 0.4 * attn_ctx).astype(np.float32)
# #
# #         moe_out = self.moe.forward(x)
# #         w = moe_out["weights"]
# #         router_conf = moe_out["router_conf"]
# #         expert_vecs = moe_out["expert_vecs"]
# #
# #         # choose top experts
# #         top_idx = list(np.argsort(-w)[: self.cfg.n_candidates])
# #         energies, candidates = [], []
# #         ev = self._qvec(evidence[0] if evidence else "")
# #
# #         for k in top_idx:
# #             av = (0.7 * expert_vecs[int(k)] + 0.3 * qv).astype(np.float32)
# #             E = self.ebm.energy(qv, av, ev)
# #             energies.append(float(E))
# #             candidates.append({"expert": int(k), "weight": float(w[int(k)]), "energy": float(E)})
# #
# #         candidates.sort(key=lambda d: d["energy"])
# #         best = candidates[0] if candidates else {"expert": -1, "weight": 0.0, "energy": 9.9}
# #
# #         # ---- gates + integrity ----
# #         gates = {
# #             "intent_ok": (qtype.name == "generic") or (cov0 >= self.cfg.min_intent_cov),
# #             "role_ok": role_ok,
# #             "schema_ok": schema_ok,
# #         }
# #         integrity = integrity_decision(self.cfg, router_conf, energies, gates, entailment=entailment)
# #
# #         # ---- build answer (schema-first) ----
# #         def schema_answer(schema_obj):
# #             lines = [f"QType: {qtype.name}"]
# #             for it in schema_obj["filled"]:
# #                 lines.append(f"- {it['slot']}: {it['sentence']}")
# #             if schema_obj["missing"]:
# #                 lines.append("\nNot supported by provided text:")
# #                 for m in schema_obj["missing"]:
# #                     lines.append(f"- {m['slot']}: {m['guidance']}")
# #             return "\n".join(lines)
# #
# #         final_answer = schema_answer(schema)
# #         if integrity["decision"] == "ABSTAIN":
# #             final_answer = (
# #                 "I can’t support an answer to this question from the retrieved text.\n\n"
# #                 f"Why:\n"
# #                 f"- qtype={qtype.name}\n"
# #                 f"- intent_top_coverage={cov0:.3f} (min {self.cfg.min_intent_cov:.3f})\n"
# #                 f"- role_alignment(target={rs.target})={rs.alignment:.3f} (min {self.cfg.min_role_align:.3f})\n"
# #                 f"- schema_coverage={schema['coverage']:.2f} (min {self.cfg.min_schema_coverage:.2f})\n"
# #                 f"- router_conf={router_conf:.3f} (min {self.cfg.router_conf_min:.3f})\n"
# #                 f"- flags={integrity['flags']}\n"
# #             )
# #
# #         return {
# #             "final_answer": final_answer,
# #             "qtype": qtype.name,
# #             "plan": plan,
# #             "plan_debug": plan_debug,
# #             "reretry_debug": reretry_debug,
# #             "retrieval": [{"idx": idx, "score": float(score), "chunk": self.chunks[idx]} for idx, score in merged_items],
# #             "intent": {"top_coverage": float(cov0)},
# #             "role": {"target": rs.target, "maternal": rs.maternal, "offspring": rs.offspring, "monitoring": rs.monitoring, "alignment": rs.alignment},
# #             "schema": schema,
# #             "entailment": entailment,
# #             "moe": {"weights": w.tolist(), "router_conf": float(router_conf), "top_experts": top_idx},
# #             "candidates": candidates,
# #             "best": best,
# #             "integrity": integrity,
# #         }
