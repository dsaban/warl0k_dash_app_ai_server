
from typing import Dict, Any, List, Tuple

from streamlit_agraph import Node, Edge, Config

STATUS_COLOR = {
    "OK":"#16A34A",
    "DUE":"#F59E0B",
    "OVERDUE":"#DC2626",
    "UNKNOWN":"#9CA3AF",
    "NOT_APPLICABLE":"#CBD5E1",
}

def build_patient_graph_agraph(patient: Dict[str, Any], results: List[Dict[str, Any]]) -> Tuple[List[Node], List[Edge], Config, Dict[str, Dict[str, Any]]]:
    """
    Returns (nodes, edges, config, node_meta_by_id) suitable for streamlit-agraph.
    Node ids are stable strings like PATIENT:P001, CHECK:S001, CLAIM:c0094, EVID:p00123, ENTITY:OGTT.
    """
    node_meta: Dict[str, Dict[str, Any]] = {}

    pid = str(patient.get("pid", "P?"))
    pname = str(patient.get("name", ""))
    patient_id = f"PATIENT:{pid}"
    nodes: List[Node] = []
    edges: List[Edge] = []

    nodes.append(Node(
        id=patient_id,
        label=f"{pid}\n{pname}".strip(),
        size=28,
        color="#111827",
        font={"color":"#FFFFFF"}
    ))
    node_meta[patient_id] = {"kind":"patient", "patient": patient}

    episode_id = f"EPISODE:{pid}:PREG"
    nodes.append(Node(
        id=episode_id,
        label="Pregnancy Episode",
        size=20,
        color="#374151",
        font={"color":"#FFFFFF"}
    ))
    node_meta[episode_id] = {"kind":"episode", "patient": patient}
    edges.append(Edge(source=patient_id, target=episode_id, label="HAS"))

    # Facts (compact)
    facts = []
    if patient.get("risk_level") is not None:
        facts.append(("Risk", str(patient.get("risk_level"))))
    if patient.get("gestational_age_weeks") is not None:
        facts.append(("GA(w)", str(patient.get("gestational_age_weeks"))))
    if patient.get("postpartum_weeks") is not None:
        facts.append(("PP(w)", str(patient.get("postpartum_weeks"))))
    if patient.get("history_gdm"):
        facts.append(("Hx", "GDM"))
    for k, v in facts[:5]:
        fid = f"FACT:{pid}:{k}"
        nodes.append(Node(id=fid, label=f"{k}: {v}", size=14, color="#6B7280", font={"color":"#FFFFFF"}))
        node_meta[fid] = {"kind":"fact", "key":k, "value":v}
        edges.append(Edge(source=episode_id, target=fid, label="HAS"))

    # Events
    for idx, e in enumerate(patient.get("events", [])):
        et = str(e.get("type", "event"))
        name = str(e.get("name", ""))[:30]
        dt = str(e.get("date",""))
        eid = f"EVENT:{pid}:{idx}"
        label = et
        if name:
            label += f"\n{name}"
        if dt:
            label += f"\n{dt}"
        nodes.append(Node(id=eid, label=label, size=14, color="#2563EB", font={"color":"#FFFFFF"}))
        node_meta[eid] = {"kind":"event", "event": e}
        edges.append(Edge(source=episode_id, target=eid, label="OBS"))

    # Checks + actions + optional linked claims/evidence placeholders
    for r in results:
        sid = str(r.get("sid","S?"))
        title = str(r.get("title",""))
        status = str(r.get("status","UNKNOWN"))
        check_id = f"CHECK:{sid}"
        ccol = STATUS_COLOR.get(status, "#9CA3AF")
        nodes.append(Node(
            id=check_id,
            label=f"{sid}\n{title}\n[{status}]",
            size=18,
            color=ccol,
            font={"color":"#FFFFFF"}
        ))
        node_meta[check_id] = {"kind":"check", "check": r}
        edges.append(Edge(source=episode_id, target=check_id, label="CHECKS"))

        rec = str(r.get("recommendation",""))
        if rec:
            action_id = f"ACTION:{sid}"
            nodes.append(Node(
                id=action_id,
                label=("Action:\n" + (rec[:80] + ("…" if len(rec)>80 else ""))),
                size=14,
                color="#E5E7EB",
                font={"color":"#111827"}
            ))
            node_meta[action_id] = {"kind":"action", "check": r, "recommendation": rec}
            edges.append(Edge(source=check_id, target=action_id, label="NEXT"))

        # Linked claims
        for cid in (r.get("related_claim_ids") or [])[:5]:
            claim_id = f"CLAIM:{cid}"
            if claim_id not in node_meta:
                nodes.append(Node(id=claim_id, label=f"Claim\n{cid}", size=12, color="#0EA5E9", font={"color":"#FFFFFF"}))
                node_meta[claim_id] = {"kind":"claim", "claim_id": cid}
            edges.append(Edge(source=check_id, target=claim_id, label="USES"))

        # Linked evidence passages
        for j, ev in enumerate((r.get("evidence") or [])[:3]):
            evid = str(ev.get("passage_id") or ev.get("id") or f"ev{j}")
            evid_id = f"EVID:{evid}"
            if evid_id not in node_meta:
                nodes.append(Node(id=evid_id, label=f"Evidence\n{evid}", size=11, color="#A855F7", font={"color":"#FFFFFF"}))
                node_meta[evid_id] = {"kind":"evidence", "evidence": ev}
            edges.append(Edge(source=check_id, target=evid_id, label="CITES"))

    config = Config(
        width="100%",
        height=520,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F59E0B",
        collapsible=False,
    )
    return nodes, edges, config, node_meta
