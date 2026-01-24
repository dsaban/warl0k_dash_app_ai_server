import streamlit as st
from pathlib import Path
import json

from engine.qa import answer
from engine.pipeline import ensure_dirs, save_uploaded_docs, build_all
from engine.graph import Graph

ROOT = Path(".").resolve()
paths = ensure_dirs(ROOT)
DOCS_DIR = paths["docs_dir"]
INDEX_DIR = paths["index_dir"]
EVAL_DIR = paths["eval_dir"]

st.set_page_config(page_title="GDM ChainGraph v3", layout="wide")
st.title("GDM ChainGraph v3 — UI-driven indexing + causal QA")

tab_upload, tab_build, tab_ask, tab_ontology, tab_claims, tab_eval = st.tabs(
    ["Upload docs", "Build index", "Ask", "Ontology", "Claims", "Eval results"]
)

with tab_upload:
    st.subheader("1) Upload your .txt documents")
    st.write(f"Docs will be saved into: `{DOCS_DIR}`")

    uploaded = st.file_uploader(
        "Upload .txt docs",
        type=["txt"],
        accept_multiple_files=True
    )

    overwrite = st.checkbox("Overwrite files if same name exists", value=True)

    if st.button("Save uploaded docs", type="primary"):
        res = save_uploaded_docs(DOCS_DIR, uploaded, overwrite=overwrite)
        st.success(f"Saved: {len(res['saved'])}, Skipped: {len(res['skipped'])}")
        if res["saved"]:
            st.write("Saved files:")
            for p in res["saved"]:
                st.code(p)
        if res["skipped"]:
            st.write("Skipped:")
            st.json(res["skipped"])

    st.divider()
    st.write("Currently in docs/:")
    files = sorted([p.name for p in DOCS_DIR.glob("*.txt")])
    st.dataframe([{"file": f} for f in files] if files else [{"file": "(none)"}])

with tab_build:
    st.subheader("2) Build / Rebuild index & ontology (from UI)")

    run_eval = st.checkbox("Also generate questions + self-eval", value=True)
    eval_limit = st.slider("Eval question limit", min_value=50, max_value=2000, value=600, step=50)

    if st.button("Build now", type="primary"):
        br = build_all(ROOT, run_eval=run_eval, eval_limit=eval_limit)
        if br.ok:
            st.success(br.message)
            st.json(br.stats)
        else:
            st.error(br.message)

    st.divider()
    st.write("Index files status:")
    st.write(f"- ontology.json: `{(INDEX_DIR/'ontology.json').exists()}`")
    st.write(f"- claims.jsonl: `{(INDEX_DIR/'claims.jsonl').exists()}`")
    st.write(f"- eval results: `{(EVAL_DIR/'results.jsonl').exists()}`")

with tab_ask:
    st.subheader("3) Ask (requires index built)")

    if not (INDEX_DIR / "ontology.json").exists():
        st.warning("Index not built yet. Go to **Build index** tab and click **Build now**.")
    else:
        q = st.text_area(
            "Question",
            height=120,
            value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia."
        )
        if st.button("Answer", type="primary"):
            res = answer(INDEX_DIR, q)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(res.markdown)
            with c2:
                st.subheader("Debug")
                st.json(res.debug)

with tab_ontology:
    st.subheader("Ontology (data/index/ontology.json)")
    ont_path = INDEX_DIR / "ontology.json"
    if not ont_path.exists():
        st.warning("No ontology yet. Build the index first.")
    else:
        ont = json.loads(ont_path.read_text(errors="ignore"))
        st.write("Meta:", ont.get("meta", {}))
        st.write("Node types:", ont.get("node_types", []))
        st.write("Edge types:", ont.get("edge_types", []))
        nodes = ont.get("nodes", [])
        st.dataframe([{"id": n["id"], "type": n["type"], "label": n["label"]} for n in nodes[:800]])

with tab_claims:
    st.subheader("Claims (data/index/claims.jsonl)")
    claims_path = INDEX_DIR / "claims.jsonl"
    if not claims_path.exists():
        st.warning("No claims yet. Build the index first.")
    else:
        g = Graph.load(INDEX_DIR)
        only_edges = st.checkbox("Show only claims with typed edges", value=True)
        rows = []
        for c in g.claims:
            if only_edges and not c.edge_types:
                continue
            rows.append({
                "id": c.id,
                "doc": c.doc,
                "conf": c.confidence,
                "edges": ", ".join(c.edge_types) if c.edge_types else "-",
                "text": c.text
            })
            if len(rows) >= 600:
                break
        st.dataframe(rows)

with tab_eval:
    st.subheader("Eval results (data/eval/results.jsonl)")
    res_path = EVAL_DIR / "results.jsonl"
    if not res_path.exists():
        st.info("No eval results yet. In **Build index**, enable self-eval and run build.")
    else:
        only_incomplete = st.checkbox("Show only incomplete", value=True)
        rows = []
        for line in res_path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if only_incomplete and r.get("complete", False):
                continue
            rows.append({
                "frame": r.get("frame"),
                "question": r.get("q"),
                "missing_edges": ", ".join(r.get("missing_edges", [])),
                "covered_edges": ", ".join(r.get("covered_edges", [])),
                "seed_n": r.get("retrieval", {}).get("seed_n"),
                "expanded_n": r.get("retrieval", {}).get("expanded_n"),
            })
            if len(rows) >= 300:
                break
        st.dataframe(rows)

# import streamlit as st
# from pathlib import Path
# from engine.qa import answer
#
# st.set_page_config(layout="wide")
# st.title("GDM ChainGraph v3")
#
# idx = Path("data/index")
#
# q = st.text_area("Ask a question", height=120)
# if st.button("Run"):
#     res = answer(idx, q)
#     st.markdown(res.markdown)
#     with st.expander("Debug"):
#         st.json(res.debug)
