
import os, json
import streamlit as st
from core.pipeline import GDMPipeline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="GDM MoE+EBM Graph Reasoner", layout="wide")

@st.cache_resource
def load_pipe():
    pipe = GDMPipeline(ROOT)
    wpath = os.path.join(ROOT, "artifacts", "trained_weights.json")
    if os.path.exists(wpath):
        with open(wpath, "r", encoding="utf-8") as f:
            pipe.weights = json.load(f)
    return pipe

pipe = load_pipe()

st.title("GDM Graph-Gated MoE + Energy-Based Model (Python-only)")

tabs = st.tabs(["Ask", "Graph Explorer", "Training", "Data"])

with tabs[0]:
    q = st.text_area("Question", value="Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia.", height=100)
    if st.button("Run inference", type="primary"):
        out = pipe.run(q)
        st.subheader("Answer")
        st.write(out["answer"])

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("QType", out["qtype"])
        c2.metric("MoE score", f"{out['moe_score']:.2f}")
        c3.metric("Energy", f"{out['energy']:.2f}")
        c4.metric("Final", f"{out['final_score']:.2f}")

        st.subheader("Entities detected")
        st.write(out["entities"] or "(none matched lexicon)")

        st.subheader("Question tags (derived)")
        st.write(out["question_tags"] or "(none)")

        st.subheader("Causal path")
        st.write(" → ".join(out["proposed_path"]) if out["proposed_path"] else "(none)")

        st.subheader("Connections (edges on path)")
        for e in out["connections"]:
            st.markdown(f"**{e['from']}** — *{e['relation']}* → **{e['to']}**")
            st.caption("tags: " + ", ".join(e.get("tags", [])))

        st.subheader("Experts (routing + contribution)")
        for ex in out["experts"]:
            with st.expander(f"{ex['name']} | route={ex['route_weight']:.2f} | raw={ex['raw_score']:.2f}"):
                st.write(ex["reasons"] or "(no active features)")

        st.subheader("Evidence (ranked)")
        for r in out["evidence"]:
            with st.container(border=True):
                st.markdown(f"**{r['id']}** • score={r['score']:.2f}")
                st.write(r["text"])
                st.caption("entities: " + ", ".join(r.get("entities", [])))
                st.caption("tags: " + ", ".join(r.get("tags", [])))
                st.caption("source: " + r.get("source", ""))

        st.subheader("Energy breakdown")
        st.json(out["energy_breakdown"])

        st.download_button("Download inference JSON", data=json.dumps(out, indent=2), file_name="inference_output.json", mime="application/json")

with tabs[1]:
    st.subheader("Graph Explorer")
    g = json.load(open(os.path.join(ROOT, "data", "graph_seed.json"), "r", encoding="utf-8"))
    st.caption(f"nodes={len(g['nodes'])} edges={len(g['edges'])}")
    tag_filter = st.text_input("Filter edges by tag contains", value="")
    shown=0
    for a,b,rel,meta in g["edges"]:
        tags = meta.get("tags", [])
        if tag_filter and not any(tag_filter.lower() in t.lower() for t in tags):
            continue
        shown += 1
        st.markdown(f"- **{a}** — *{rel}* → **{b}**")
        st.caption(", ".join(tags))
    st.write(f"Showing {shown} edges")
    st.download_button("Download graph_seed.json", data=json.dumps(g, indent=2), file_name="graph_seed.json", mime="application/json")

with tabs[2]:
    st.subheader("Training (contrastive path alignment)")
    st.write("This training only adjusts tiny weights so GOLD causal paths score above BAD paths.")
    epochs = st.slider("epochs", 1, 100, 25)
    lr = st.slider("lr", 0.01, 1.0, 0.2)
    if st.button("Run training", type="primary"):
        from training.train import train
        wts, updates = train(ROOT, epochs=epochs, lr=lr)
        st.success(f"Training done. updates={updates}. saved artifacts/trained_weights.json")
        st.json({k:wts[k] for k in list(wts.keys())[:30]})

with tabs[3]:
    st.subheader("Locked ontology & datasets")
    onto = json.load(open(os.path.join(ROOT, "config", "ontology.json"), "r", encoding="utf-8"))
    lex = json.load(open(os.path.join(ROOT, "data", "lexicon", "entities.json"), "r", encoding="utf-8"))
    ds  = json.load(open(os.path.join(ROOT, "data", "trainset.json"), "r", encoding="utf-8"))
    st.write("QTypes:", onto["fixed_qtypes"])
    st.write("#entities:", len(lex))
    st.write("#train items:", len(ds))
    st.download_button("Download trainset.json", data=json.dumps(ds, indent=2), file_name="trainset.json", mime="application/json")
