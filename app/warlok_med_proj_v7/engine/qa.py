from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from pathlib import Path
from .frames import FRAMES, FrameSpec
from .graph import Graph, Claim
from .retrieve import retrieve
from .edge_infer import infer_edges

def route(question: str) -> FrameSpec:
    q = question.lower()
    if "macrosomia" in q or "pedersen" in q:
        return FRAMES["B"]
    if "postpartum" in q or "follow-up" in q:
        return FRAMES["D"]
    if "guideline" in q or "criteria" in q:
        return FRAMES["E"]
    if "risk" in q or "ethnicity" in q:
        return FRAMES["C"]
    if "treatment" in q or "monitor" in q:
        return FRAMES["F"]
    if any(k in q for k in
           ["reframed", "chronic", "transient", "rather than", "long-term", "postpartum risk", "future diabetes"]):
        return FRAMES["G"]  # or FRAMES["D"] if you extend D instead
    if any(k in q for k in
           ["reframed", "chronic", "transient", "rather than", "long-term", "postpartum risk", "future diabetes"]):
        return FRAMES["D"]  # or a new Frame G
    
    return FRAMES["A"]

def drift_ratio(graph: Graph, used: List[Claim], frame: FrameSpec) -> float:
    blocked = set(frame.blocked_node_types)
    allowed = set(frame.allowed_node_types + ["threshold_or_timing"])
    tot = drift = 0
    for c in used:
        for n in c.nodes:
            t = graph.node_type(n)
            tot += 1
            if t in blocked or (t not in allowed and t != "unknown"):
                drift += 1
    return drift / max(1, tot)

@dataclass
class Answer:
    frame: Dict[str, Any]
    markdown: str
    debug: Dict[str, Any]

def answer(index_dir: Path, question: str) -> Answer:
    g = Graph.load(index_dir)
    frame = route(question)
    r = retrieve(g, question, frame)

    required = frame.required_edges
    covered = set()
    steps = []
    used = []

    # for c in r.expanded:
    for c in sorted(r.expanded, key=lambda x: x.confidence, reverse=True):
        edges = set(c.edge_types) | set(infer_edges(c.text))
        new = [e for e in edges if e in required and e not in covered]
        if not new:
            continue
        used.append(c)
        covered.update(new)
        steps.append((c, edges))
        if len(steps) >= frame.max_steps:
            break
        if len(covered) >= frame.min_required_covered and len(steps) >= frame.min_steps:
            break

    soft = False
    if len(steps) < frame.min_steps:
        soft = True
        steps = [(c, set(c.edge_types) | set(infer_edges(c.text))) for c in r.expanded[:frame.max_steps]]
        used = [c for c, _ in steps]

    dr = drift_ratio(g, used, frame)

    md = [f"### Frame {frame.id}: {frame.name}", "",
          "**Chain assembled:**" if not soft else "**Evidence pack (partial):**"]
    for i, (c, edges) in enumerate(steps, 1):
        md.append(
            f"**Step {i}.** {c.text}  \n"
            f"_evidence_: `{c.id}` | _doc_: `{c.doc}` | _conf_: `{c.confidence:.3f}` | _edges_: `{', '.join(edges)}`\n"
        )
        
        # md.append(
        #     f"**Step {i}.** {c.text}  \n"
        #     f"_evidence_: `{c.id}` | _doc_: `{c.doc}` | _edges_: `{', '.join(edges)}`\n"
        # )
    
    

    return Answer(
        frame={"id": frame.id, "name": frame.name},
        markdown="\n".join(md),
        debug={
            "soft_fallback_used": soft,
            "covered_edges": list(covered),
            "drift_ratio": dr,
            "retrieval": r.debug
        }
    )
