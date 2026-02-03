from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

from .domain_pack import DomainPack
from .retrieve import load_claims


def _dedupe(items: List[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for it in items:
        key = it["q"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
        if len(out) >= max_total:
            break
    return out


def generate_questions(
    index_dir: Path,
    out_path: Path,
    domain: DomainPack,
    max_total: int = 8000,
    max_mined: int = 4000,
    max_2hop: int = 2000
) -> Dict[str, Any]:
    qt = domain.question_templates or {}
    out: List[Dict[str, Any]] = []

    # 1) Pack templates (frames + edges)
    for fid, templates in (qt.get("by_frame") or {}).items():
        for t in templates:
            out.append({"q": t, "source": "template_frame", "frame": fid})

    for edge, templates in (qt.get("by_edge") or {}).items():
        for t in templates:
            out.append({"q": t, "source": "template_edge", "edge": edge})

    for t in (qt.get("chain_2hop") or []):
        out.append({"q": t, "source": "template_2hop"})

    # 2) Mine from claims (high confidence, edge-bearing)
    claims = load_claims(index_dir)
    edge_counts = Counter()
    for c in claims:
        for e in c.edge_types:
            edge_counts[e] += 1

    mined = 0
    for c in sorted(claims, key=lambda x: x.confidence, reverse=True):
        if mined >= max_mined:
            break
        if not c.edge_types:
            continue
        e0 = c.edge_types[0]
        # Create multiple variants to scale without hallucination
        out.append({"q": f"Explain the mechanism captured by {e0} using a 4–6 step causal chain.", "source": "mined_edge", "edge": e0})
        out.append({"q": f"What is the evidence-based rationale for {e0}, and what intermediate steps link cause to outcome?", "source": "mined_edge", "edge": e0})
        mined += 2

    # 3) 2-hop co-occurrence chains (edges that tend to appear together)
    edge_to_edges = Counter()
    for c in claims:
        es = list(dict.fromkeys(c.edge_types))
        for i in range(len(es)):
            for j in range(i + 1, len(es)):
                edge_to_edges[(es[i], es[j])] += 1

    # pick top pairs
    pairs = edge_to_edges.most_common(max_2hop)
    for (e1, e2), _ in pairs:
        out.append({
            "q": f"Build a stepwise explanation connecting {e1} → {e2}, citing evidence for each step.",
            "source": "2hop_pair",
            "edges": [e1, e2]
        })

    out = _dedupe(out, max_total)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out), encoding="utf-8")

    return {
        "questions": len(out),
        "mined": mined,
        "edges_present": len(edge_counts),
        "top_edges": edge_counts.most_common(10),
        "out_path": str(out_path)
    }

# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Any, List
# from .domain_pack import DomainPack
#
#
# def generate_questions(index_dir: Path, out_path: Path, domain: DomainPack, max_total: int = 3000) -> Dict[str, Any]:
#     qt = domain.question_templates or {}
#     out: List[Dict[str, Any]] = []
#
#     # frame templates
#     for fid, templates in (qt.get("by_frame") or {}).items():
#         for t in templates:
#             out.append({"q": t, "source": "by_frame", "frame": fid})
#
#     # edge templates
#     for edge, templates in (qt.get("by_edge") or {}).items():
#         for t in templates:
#             out.append({"q": t, "source": "by_edge", "edge": edge})
#
#     # 2-hop templates
#     for t in (qt.get("chain_2hop") or []):
#         out.append({"q": t, "source": "chain_2hop"})
#
#     out = out[:max_total]
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out), encoding="utf-8")
#
#     return {"questions": len(out), "out_path": str(out_path)}
#
# # from __future__ import annotations
# # import json
# # from pathlib import Path
# # from collections import Counter, defaultdict
# # from typing import Dict, List, Any, Tuple
# #
# # from .graph import Graph
# # from .domain_pack import DomainPack
# # from .utils import tokens, uniq
# #
# # def _dedupe(qs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
# #     seen=set(); out=[]
# #     for it in qs:
# #         k = it["q"].strip().lower()
# #         if k in seen:
# #             continue
# #         seen.add(k)
# #         out.append(it)
# #     return out
# #
# # def _cap(qs: List[Dict[str, Any]], max_total: int) -> List[Dict[str, Any]]:
# #     return qs[:max_total] if len(qs) > max_total else qs
# #
# # def generate_questions(index_dir: Path, out_path: Path, domain: DomainPack,
# #                        max_total: int = 8000,
# #                        max_per_edge: int = 400,
# #                        max_per_frame: int = 800,
# #                        max_mined: int = 2500,
# #                        max_2hop: int = 1500) -> Dict[str, Any]:
# #     g = Graph.load(index_dir)
# #
# #     # coverage stats
# #     edge_counts = Counter()
# #     # frame_ids = [f.id for f in domain.frames.values()]
# #     # frames may be list[dict] or dict[id->FrameSpec/dict]
# #     if isinstance(domain.frames, dict):
# #         frame_ids = list(domain.frames.keys())
# #     else:
# #         frame_ids = [f["id"] for f in domain.frames]
# #
# #     for c in g.claims:
# #         for e in c.edge_types:
# #             edge_counts[e] += 1
# #
# #     qs: List[Dict[str, Any]] = []
# #
# #     # 1) Frame templates
# #     for fid, templates in (domain.question_templates.get("by_frame", {}) or {}).items():
# #         if fid not in frame_ids:
# #             continue
# #         for i, t in enumerate(templates[:max_per_frame]):
# #             qs.append({"q": t, "source": "frame_template", "frame": fid, "rank": i})
# #
# #     # 2) Edge templates (allocate budget based on doc coverage)
# #     by_edge = domain.question_templates.get("by_edge", {}) or {}
# #     for e, templates in by_edge.items():
# #         if edge_counts.get(e, 0) <= 0:
# #             continue
# #         budget = min(max_per_edge, max(30, 15 * edge_counts[e]))
# #         for i in range(budget):
# #             t = templates[i % len(templates)]
# #             # small style variants
# #             style = i % 4
# #             if style == 1:
# #                 q = f"In mechanistic steps, {t[0].lower()}{t[1:]}"
# #             elif style == 2:
# #                 q = f"{t} Use a concise 4–6 step causal chain."
# #             elif style == 3:
# #                 q = f"{t} Include key intermediates and directionality."
# #             else:
# #                 q = t
# #             qs.append({"q": q, "source": "edge_template", "edge": e})
# #
# #     # 3) Mine questions from actual evidence sentences (high confidence + has edges)
# #     # Create “why/how/what/when” prompts that reflect the text
# #     mined = 0
# #     for c in sorted(g.claims, key=lambda x: x.confidence, reverse=True):
# #         if mined >= max_mined:
# #             break
# #         if not c.edge_types:
# #             continue
# #         # avoid very long sentence mining
# #         if len(tokens(c.text)) > 55:
# #             continue
# #         # make a question that forces the same edge family
# #         # (simple, stable, no LLM)
# #         edges = c.edge_types
# #         primary = edges[0]
# #         if primary in by_edge and by_edge[primary]:
# #             # already templated heavily; mine less
# #             if mined % 3 != 0:
# #                 continue
# #         # heuristic question forms
# #         if "why" in c.text.lower() or "because" in c.text.lower():
# #             q = f"Explain the causal mechanism described: {primary}. What is the reasoning chain?"
# #         elif any(k in c.text.lower() for k in ["recommend", "should", "guideline", "suggest"]):
# #             q = f"What guideline recommendation is supported by the evidence for {primary}, and what is the rationale?"
# #         elif any(k in c.text.lower() for k in ["risk", "associated", "elevated"]):
# #             q = f"Based on evidence, how is {primary} linked to downstream risk, and what is the causal explanation?"
# #         else:
# #             q = f"Explain the mechanism captured by {primary} using the evidence in the documents."
# #         qs.append({"q": q, "source": "mined_claim", "edge": primary, "evidence_id": c.id})
# #         mined += 1
# #
# #     # 4) 2-hop chain questions from co-occurrence (edge1 -> edge2 via shared nodes)
# #     edge_to_nodes = defaultdict(Counter)
# #     node_to_edges = defaultdict(Counter)
# #     for c in g.claims:
# #         for e in c.edge_types:
# #             for n in c.nodes:
# #                 edge_to_nodes[e][n] += 1
# #                 node_to_edges[n][e] += 1
# #
# #     pair_scores = Counter()
# #     edges_present = [e for e in edge_counts.keys() if edge_counts[e] > 0]
# #     for e1 in edges_present:
# #         for n, w in edge_to_nodes[e1].most_common(30):
# #             for e2, w2 in node_to_edges[n].most_common(15):
# #                 if e2 == e1:
# #                     continue
# #                 pair_scores[(e1, e2)] += min(w, w2)
# #
# #     chain_templates = domain.question_templates.get("chain_2hop", []) or [
# #         "Explain the causal chain: {e1} → {e2}. Provide intermediate steps and why it matters clinically.",
# #         "Build a stepwise explanation connecting {e1} to {e2} using evidence.",
# #     ]
# #     for (e1, e2), _ in pair_scores.most_common(max_2hop):
# #         for t in chain_templates[:2]:
# #             qs.append({"q": t.format(e1=e1, e2=e2), "source": "2hop", "edges": [e1, e2]})
# #
# #     qs = _dedupe(qs)
# #     qs = _cap(qs, max_total)
# #
# #     out_path.parent.mkdir(parents=True, exist_ok=True)
# #     out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in qs))
# #
# #     return {
# #         "questions": len(qs),
# #         "edges_present": len(edges_present),
# #         "mined": mined,
# #         "frames": len(frame_ids),
# #         "top_edges": edge_counts.most_common(10),
# #     }
