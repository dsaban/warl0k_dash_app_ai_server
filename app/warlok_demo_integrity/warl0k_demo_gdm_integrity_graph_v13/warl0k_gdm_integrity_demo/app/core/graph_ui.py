
from typing import Dict, Any, List, Tuple
import networkx as nx
import numpy as np
import plotly.graph_objects as go

STATUS_COLOR = {
    "OK":"#16A34A",
    "DUE":"#F59E0B",
    "OVERDUE":"#DC2626",
    "UNKNOWN":"#9CA3AF",
    "NOT_APPLICABLE":"#CBD5E1",
}

def build_patient_graph(patient: Dict[str, Any], results: List[Dict[str, Any]]) -> Tuple[go.Figure, Dict[str, Dict[str, Any]]]:
    """Returns (plotly figure, node_meta_by_id). Node ids are strings like PATIENT:P001, CHECK:S001, CLAIM:c0094, EVID:p00103."""
    G = nx.Graph()
    meta = {}

    pid = patient.get("pid","P?")
    patient_node = f"PATIENT:{pid}"
    G.add_node(patient_node)
    meta[patient_node] = {"kind":"patient","label": f"Patient {pid}", "data": patient}

    for r in results:
        check_node = f"CHECK:{r['sid']}"
        G.add_node(check_node)
        meta[check_node] = {"kind":"check","label": f"{r['sid']}\n{r['status']}", "data": r}
        G.add_edge(patient_node, check_node)

        # claims (gold + related)
        for c in r.get("claims", []):
            cid = c.get("claim_id")
            if not cid:
                continue
            claim_node = f"CLAIM:{cid}"
            if claim_node not in meta:
                G.add_node(claim_node)
                meta[claim_node] = {"kind":"claim","label": f"{cid}", "data": c}
            G.add_edge(check_node, claim_node)

            # evidence nodes from claim
            for ev in c.get("evidence", [])[:2]:
                evid = ev.get("pid") or ev.get("passage_id") or ev.get("id")
                if not evid:
                    continue
                ev_node = f"EVID:{evid}"
                if ev_node not in meta:
                    G.add_node(ev_node)
                    meta[ev_node] = {"kind":"evidence","label": f"{evid}", "data": ev}
                G.add_edge(claim_node, ev_node)

        # evidence nodes from retriever
        for ev in (r.get("evidence") or [])[:2]:
            evid = ev.get("pid")
            if not evid:
                continue
            ev_node = f"EVID:{evid}"
            if ev_node not in meta:
                G.add_node(ev_node)
                meta[ev_node] = {"kind":"evidence","label": f"{evid}", "data": ev}
            G.add_edge(check_node, ev_node)

    pos = nx.spring_layout(G, seed=7, k=0.9)

    # edges
    edge_x, edge_y = [], []
    for a,b in G.edges():
        x0,y0 = pos[a]
        x1,y1 = pos[b]
        edge_x += [x0,x1,None]
        edge_y += [y0,y1,None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#94A3B8"),
        hoverinfo='none',
        mode='lines'
    )

    # nodes
    node_ids = list(G.nodes())
    xs = [pos[n][0] for n in node_ids]
    ys = [pos[n][1] for n in node_ids]

    colors = []
    sizes = []
    texts = []
    hover = []
    custom = []

    for nid in node_ids:
        m = meta[nid]
        kind = m["kind"]
        custom.append(nid)
        if kind == "patient":
            colors.append("#111827"); sizes.append(34); texts.append("PATIENT"); hover.append(m["label"])
        elif kind == "check":
            status = m["data"].get("status","UNKNOWN")
            colors.append(STATUS_COLOR.get(status, "#9CA3AF")); sizes.append(26)
            texts.append(m["data"].get("sid","CHECK"))
            hover.append(f"{m['data'].get('title','')} ({status})")
        elif kind == "claim":
            colors.append("#2563EB"); sizes.append(20); texts.append(m["data"].get("claim_id","CLAIM"))
            hover.append(m["data"].get("summary","(claim)"))
        else: # evidence
            colors.append("#6D28D9"); sizes.append(16); texts.append("EVID")
            doc = m["data"].get("doc","")
            hover.append(f"{doc} • {m['data'].get('pid','')}")
    node_trace = go.Scatter(
        x=xs, y=ys,
        mode='markers+text',
        text=texts,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(color=colors, size=sizes, line=dict(width=1, color="#0F172A")),
        customdata=custom
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=520
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)

    return fig, meta
