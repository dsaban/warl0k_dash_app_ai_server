from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List
from .domain_pack import DomainPack
from .retrieve import retrieve


@dataclass
class AnswerResult:
    markdown: str
    debug: Dict[str, Any]
    frame: Dict[str, Any]


def answer(index_dir: Path, question: str, domain: DomainPack) -> AnswerResult:
    fid = domain.route_frame(question)
    frame = domain.frames.get(fid) or next(iter(domain.frames.values()))

    cand = retrieve(index_dir, question, domain, frame, k=40)

    required = list(frame.get("required_edges") or [])
    min_cov = int(frame.get("min_required_covered", 1))
    min_steps = int(frame.get("min_steps", 3))
    max_steps = int(frame.get("max_steps", 6))

    covered = set()
    steps = []
    used = []

    for c in cand:
        # runtime infer can add edges
        inf_edges, _ = domain.infer_edges_with_strength(c.text)
        edges = set(c.edge_types) | set(inf_edges)

        add = False
        for e in required:
            if e not in covered and e in edges:
                add = True
                break
        if add or len(steps) < min_steps:
            steps.append(c)
            used.append(c.id)
            covered |= (edges & set(required))
        if len(steps) >= max_steps:
            break
        if len(covered) >= min_cov and len(steps) >= min_steps:
            break

    soft = False
    if len(covered) < min_cov:
        soft = True
        # fallback: only pick evidence that hits required edges
        req = set(required)
        filt = []
        for c in cand:
            inf_edges, _ = domain.infer_edges_with_strength(c.text)
            edges = set(c.edge_types) | set(inf_edges)
            if edges & req:
                filt.append(c)
        steps = (filt[:max_steps] if filt else cand[:max_steps])

    # drift: based on mapped node types from edges_meta
    blocked = set(frame.get("blocked_node_types") or [])
    drift_hits = 0
    for c in steps:
        types = set(domain.edge_types_to_node_types(c.edge_types))
        if blocked and (types & blocked):
            drift_hits += 1
    drift_ratio = (drift_hits / max(1, len(steps)))

    md = []
    md.append(f"**Question**\n\n{question}\n")
    md.append(f"**Frame {fid}: {frame.get('name','')}**\n")
    
    if soft:
        md.append("_Chain incomplete (soft fallback used — missing required edge coverage)._  \n")
    md.append("**Evidence pack:**\n")
    for i, c in enumerate(steps, 1):
        md.append(
            f"{i}. {c.text}\n\n"
            f"   - evidence: `{c.id}` | doc: `{c.doc}` | conf: `{c.confidence:.3f}` | edges: `{', '.join(c.edge_types)}`\n"
        )

    debug = {
        "soft_fallback_used": soft,
        "covered_edges": sorted(list(covered)),
        "missing_edges": [e for e in required if e not in covered],
        "drift_ratio": drift_ratio,
        "retrieval": {"seed_n": 40, "returned": len(cand)},
        "used_claims": used,
    }

    return AnswerResult(markdown="\n".join(md), debug=debug, frame={"id": fid, "name": frame.get("name", "")})

# from dataclasses import dataclass
# from typing import Dict, Any, List, Tuple
# from pathlib import Path
#
# from .graph import Graph, Claim
# from .retrieve import retrieve
# from .domain_pack import DomainPack
#
# def drift_ratio(graph: Graph, used: List[Claim], frame: Dict[str, Any]) -> float:
#     blocked = set(frame.get("blocked_node_types", []))
#     allowed = set(frame.get("allowed_node_types", [])) | {"threshold_or_timing"}
#     tot = drift = 0
#     for c in used:
#         for n in c.nodes:
#             t = graph.node_type(n)
#             tot += 1
#             if t in blocked or (allowed and t not in allowed and t != "unknown"):
#                 drift += 1
#     return drift / max(1, tot)
#
# @dataclass
# class Answer:
#     frame: Dict[str, Any]
#     markdown: str
#     debug: Dict[str, Any]
#
# def answer(index_dir: Path, question: str, domain: DomainPack) -> Answer:
#     g = Graph.load(index_dir)
#     frame = domain.route_frame(question)
#
#     r = retrieve(
#         graph=g,
#         question=question,
#         frame_like=frame,   # see retrieve.py patch below
#         bm25_k=int(domain.scoring.get("bm25_k", 40)),
#         hops=int(domain.scoring.get("hops", 2)),
#         limit=int(domain.scoring.get("retrieval_limit", 400)),
#     )
#
#     required = list(frame.get("required_edges", []))
#     required_set = set(required)
#     min_cov = int(frame.get("min_required_covered", 2))
#     min_steps = int(frame.get("min_steps", 3))
#     max_steps = int(frame.get("max_steps", 6))
#
#     covered = set()
#     steps: List[Tuple[Claim, set]] = []
#     used: List[Claim] = []
#
#     # rank by confidence first
#     expanded_sorted = sorted(r.expanded, key=lambda c: c.confidence, reverse=True)
#
#     for c in expanded_sorted:
#         edges = set(c.edge_types) | set(domain.infer_edges(c.text))
#         new = [e for e in edges if e in required_set and e not in covered]
#         if not new:
#             continue
#         used.append(c)
#         covered.update(new)
#         steps.append((c, edges))
#         if len(steps) >= max_steps:
#             break
#         if len(covered) >= min_cov and len(steps) >= min_steps:
#             break
#
#     soft = False
#     if len(steps) < min_steps or len(covered) < min_cov:
#         # edge-only fallback (no "top N random")
#         soft = True
#         cand = []
#         for c in expanded_sorted:
#             edges = set(c.edge_types) | set(domain.infer_edges(c.text))
#             if edges & required_set:
#                 cand.append((c, edges))
#         steps = cand[:max_steps]
#         used = [c for c, _ in steps]
#
#     dr = drift_ratio(g, used, frame)
#
#     md = [f"### Frame {frame['id']}: {frame['name']}", ""]
#     md.append("**Chain assembled:**" if not soft else "**Evidence pack (partial):**")
#
#     for i, (c, edges) in enumerate(steps, 1):
#         md.append(
#             f"**Step {i}.** {c.text}  \n"
#             f"_evidence_: `{c.id}` | _doc_: `{c.doc}` | _conf_: `{c.confidence:.3f}` | _edges_: `{', '.join(sorted(edges))}`\n"
#         )
#
#     return Answer(
#         frame={"id": frame["id"], "name": frame["name"]},
#         markdown="\n".join(md),
#         debug={
#             "soft_fallback_used": soft,
#             "covered_edges": sorted(list(covered)),
#             "missing_edges": sorted([e for e in required if e not in covered]),
#             "drift_ratio": dr,
#             "retrieval": r.debug
#         }
#     )
