# app.py
import os
import streamlit as st
# libs
from core.config import AppConfig
from core.pipeline import LocalMedMoEPipeline

st.set_page_config(page_title="Local Med MoE (Integrity)", layout="wide")

def load_docs_from_folder(folder="docs"):
    texts = []
    if os.path.isdir(folder):
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith(".txt"):
                with open(os.path.join(folder, fn), "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read())
    return "\n\n".join(texts)

def main():
    st.title("Local Medical MoE — Integrity-Gated QA (NumPy-only)")

    cfg = AppConfig()

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Documents")
        docs_text = load_docs_from_folder("docs")
        up = st.file_uploader("Upload .txt docs (optional)", type=["txt"], accept_multiple_files=True)
        if up:
            parts = []
            for f in up:
                parts.append(f.read().decode("utf-8", errors="ignore"))
            docs_text = "\n\n".join(parts)

        st.caption(f"Docs loaded chars: {len(docs_text)}")
        weights_path = st.text_input("Weights path (optional)", value="trained_weights.npz")

        if st.button("Build / Rebuild pipeline"):
            if not docs_text.strip():
                st.error("No docs loaded. Put .txt files in ./docs or upload.")
            else:
                st.session_state.pipe = LocalMedMoEPipeline(
                    docs_text, cfg,
                    weights_path=weights_path if (weights_path and os.path.exists(weights_path)) else None
                )
                st.success("Pipeline ready.")

        if "pipe" in st.session_state:
            st.caption(f"Indexed chunks: {len(st.session_state.pipe.chunks)}")
        else:
            st.info("Click 'Build / Rebuild pipeline' first.")

    with right:
        st.subheader("Inference")
        q = st.text_area(
            "Ask a question",
            value="In what ways does GDM serve as a clinical model for understanding the transition from insulin resistance to overt diabetes?",
            height=110
        )

        run = st.button("Run inference")
        if run:
            if "pipe" not in st.session_state:
                st.error("Pipeline not built yet.")
            else:
                out = st.session_state.pipe.infer(q)
                st.session_state.last_out = out

        if "last_out" in st.session_state:
            out = st.session_state.last_out
            st.markdown("### Answer")
            st.write(out["final_answer"])

            # Top-level expanders ONLY (no nesting)
            with st.expander("Integrity Decision", expanded=True):
                st.json(out["integrity"])

            with st.expander("QType / Intent / Role / Schema", expanded=False):
                st.json({
                    "qtype": out.get("qtype"),
                    "intent": out.get("intent"),
                    "role": out.get("role"),
                    "schema": {"schema_id": out.get("schema",{}).get("schema_id"),
                               "coverage": out.get("schema",{}).get("coverage"),
                               "missing": out.get("schema",{}).get("missing", [])}
                })
                if out.get("entailment") is not None:
                    st.markdown("**Entailment (classification)**")
                    st.json(out["entailment"])

            with st.expander("MoE / EBM Candidates", expanded=False):
                st.json(out.get("moe", {}))
                st.json(out.get("candidates", []))

            with st.expander("Retrieval (top chunks)", expanded=False):
                for r in out.get("retrieval", []):
                    st.markdown(f"**idx={r['idx']} score={r['score']:.3f}**")
                    st.write(r["chunk"][:1200] + ("..." if len(r["chunk"]) > 1200 else ""))

            with st.expander("Plan Debug (per step hits)", expanded=False):
                # st.json(out.get("plan", []))
                st.write(out.get("plan", []))
                st.json(out.get("plan_debug", [])[:3])  # show first few (can be long)

            with st.expander("Re-retrieve Debug (anchors)", expanded=False):
                st.json(out.get("reretry_debug", []))

if __name__ == "__main__":
    main()
