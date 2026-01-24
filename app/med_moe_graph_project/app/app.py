from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from core.pipeline import LocalMedMoEPipeline


st.set_page_config(page_title="Local Medical MoE + Graph", layout="wide")

@st.cache_resource
def get_pipeline() -> LocalMedMoEPipeline:
    return LocalMedMoEPipeline()

pipe = get_pipeline()

st.title("Local Medical MoE + Graph Explorer")

tab1, tab2, tab3, tab4 = st.tabs(["Ask", "Graph Explorer", "Ontology", "Ingest/Rebuild"]) 

with tab1:
    st.subheader("Ask a question")
    q = st.text_area(
        "Question",
        value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia.",
        height=90,
    )
    colA, colB = st.columns([1, 1])
    with colA:
        top_k = st.slider("Top-K evidence chunks", min_value=3, max_value=12, value=6)
    with colB:
        hops = st.slider("Graph neighborhood hops", min_value=1, max_value=3, value=2)

    if st.button("Infer", type="primary"):
        res = pipe.infer(q, top_k=top_k, neighborhood_hops=hops)

        left, right = st.columns([1.2, 1])
        with left:
            st.markdown("### Answer")
            st.write(res.answer)
            st.caption(f"QType: **{res.qtype}** | Integrity: {'✅' if res.integrity_ok else '⚠️'} {res.integrity_notes}")

            st.markdown("### Ranked evidence")
            for i, h in enumerate(res.hits, start=1):
                st.markdown(f"**{i}. {h.chunk_id}** (score={h.score:.4f})")
                st.write(h.snippet)
                st.caption(f"tags: {', '.join(h.tags) if h.tags else '-'} | entities: {', '.join(h.entities) if h.entities else '-'}")

        with right:
            st.markdown("### Explanation graph signals")
            st.write({
                "detected_qtype": res.qtype,
                "query_tags": res.qtags,
                "neighborhood_nodes": res.graph_neighborhood_nodes,
                "neighborhood_edges": res.graph_edges_count,
            })

            st.markdown("### Chunk → tags/entities table")
            st.dataframe(res.conn_table, use_container_width=True)

with tab2:
    st.subheader("Graph Explorer")
    gpath = pipe.graph_path
    if not gpath.exists():
        st.warning("Graph not found yet. Build it in the Ingest tab.")
    else:
        data = json.loads(Path(gpath).read_text(encoding="utf-8"))
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        col1, col2 = st.columns([1, 1])
        with col1:
            st.write(f"Nodes: {len(nodes)}")
            st.write(f"Edges: {len(edges)}")
            st.download_button(
                "Download graph.json",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name="graph.json",
                mime="application/json",
            )

        st.markdown("#### Filter nodes")
        ntypes = sorted({n.get("ntype") for n in nodes})
        selected = st.multiselect("Node types", options=ntypes, default=ntypes)
        show_n = st.slider("Show first N nodes", 20, 400, 120)
        f_nodes = [n for n in nodes if n.get("ntype") in selected][:show_n]
        st.dataframe(f_nodes, use_container_width=True)

        st.markdown("#### Neighborhood plot (simple)")
        st.caption("This is a lightweight plot for quick intuition; the authoritative view is the edge list above.")

        # Build a small subgraph around the first few chunks
        chunk_nodes = [n for n in nodes if n.get("id", "").startswith("CHUNK::")][:6]
        center_ids = [n["id"] for n in chunk_nodes]
        sub_ids = set(center_ids)
        for e in edges:
            if e.get("src") in center_ids or e.get("dst") in center_ids:
                sub_ids.add(e.get("src"))
                sub_ids.add(e.get("dst"))
        sub_edges = [e for e in edges if e.get("src") in sub_ids and e.get("dst") in sub_ids]

        # circular layout
        sub_ids = list(sub_ids)
        if sub_ids:
            import math
            coords = {}
            for i, nid in enumerate(sub_ids):
                ang = 2 * math.pi * i / max(1, len(sub_ids))
                coords[nid] = (math.cos(ang), math.sin(ang))

            fig = plt.figure()
            ax = plt.gca()
            for e in sub_edges:
                x1, y1 = coords[e["src"]]
                x2, y2 = coords[e["dst"]]
                ax.plot([x1, x2], [y1, y2])
            for nid, (x, y) in coords.items():
                ax.scatter([x], [y])
                ax.text(x, y, nid.split("::")[0], fontsize=8)
            ax.set_axis_off()
            st.pyplot(fig, use_container_width=True)

        st.markdown("#### Filter edges")
        etypes = sorted({e.get("etype") for e in edges})
        e_selected = st.multiselect("Edge types", options=etypes, default=etypes)
        show_e = st.slider("Show first N edges", 20, 800, 200)
        f_edges = [e for e in edges if e.get("etype") in e_selected][:show_e]
        st.dataframe(f_edges, use_container_width=True)

with tab3:
    st.subheader("Ontology (fixed taxonomy)")
    onto = json.loads(Path(pipe.ontology_path).read_text(encoding="utf-8"))
    st.markdown("#### QTypes")
    st.dataframe([
        {"qtype": k, "description": v.get("description", ""), "required_tags": ", ".join(v.get("required_tags", []))}
        for k, v in onto.get("qtypes", {}).items()
    ], use_container_width=True)

    st.markdown("#### Tags")
    st.dataframe([
        {"tag": k, "aliases": ", ".join(v)} for k, v in onto.get("tags", {}).items()
    ], use_container_width=True)

with tab4:
    st.subheader("Ingest / Rebuild graph")
    st.write("If you changed the corpus in `data/docs`, rebuild the graph to refresh chunks/tags/entities.")
    if st.button("Rebuild graph now"):
        pipe.rebuild_graph()
        st.success("Graph rebuilt and saved.")

    st.markdown("#### Project paths")
    st.code(
        f"docs_dir: {pipe.docs_dir}\nontology: {pipe.ontology_path}\nentity_lexicon: {pipe.entity_lexicon_path}\ngraph_path: {pipe.graph_path}"
    )
