
## `app1.py`
import streamlit as st
import numpy as np

from core.retrieve import DualBM25Retriever
from core.moe_router import MoERouter
from core.ebm_ranker import EBMRanker
from core.pipeline import IntegrityMoEPipeline

st.set_page_config(page_title="GDM Integrity MoE (BM25)", layout="wide")
st.title("GDM Integrity MoE — BM25 + ClaimGraph + MoE + EBM + Learning")

def load_index(path: str):
    d = np.load(path, allow_pickle=True)

    class Chunk: pass
    class Claim: pass

    chunks = []
    for i in range(len(d["chunk_id"])):
        c = Chunk()
        c.chunk_id = d["chunk_id"][i]
        c.file = d["chunk_file"][i]
        c.text = d["chunk_text"][i]
        c.span = tuple(d["chunk_span"][i])
        c.token_count = int(d["chunk_tok"][i])
        c.section = d["chunk_section"][i] if "chunk_section" in d else ""
        chunks.append(c)

    claims = []
    for i in range(len(d["claim_id"])):
        c = Claim()
        c.claim_id = d["claim_id"][i]
        c.file = d["claim_file"][i]
        c.chunk_id = d["claim_chunk"][i]
        c.sentence = d["claim_sentence"][i]
        c.span = tuple(d["claim_span"][i])
        c.support_text = d["claim_support"][i]
        c.entities = list(d["claim_entities"][i]) if d["claim_entities"][i] is not None else []
        c.edge_hints = list(d["claim_edges"][i]) if d["claim_edges"][i] is not None else []
        claims.append(c)

    return chunks, claims

with st.sidebar:
    st.header("Runtime")

    index_path = st.text_input("Index file", value="data/index.npz")
    router_path = st.text_input("Router weights (optional)", value="models/router_weights.npz")
    ebm_path = st.text_input("EBM weights (optional)", value="models/ebm_weights.npz")
    rerank_path = st.text_input("Claim rerank weights (optional)", value="models/rerank_weights.npz")

    top_chunks = st.slider("Top chunks", 3, 25, 8)
    max_claims = st.slider("Max claims (within chunks)", 8, 120, 24)

    show_debug = st.checkbox("Show debug panels", value=True)
    reload_btn = st.button("Load / Reload")

if "rt" not in st.session_state:
    st.session_state.rt = None

def build_runtime():
    chunks, claims = load_index(index_path)
    retriever = DualBM25Retriever(chunks, claims)

    router = MoERouter()
    if router_path.strip():
        try:
            router.load(router_path.strip())
        except Exception as e:
            st.sidebar.warning(f"Router weights not loaded: {e}")

    ebm = EBMRanker()
    if ebm_path.strip():
        try:
            ebm.load(ebm_path.strip())
        except Exception as e:
            st.sidebar.warning(f"EBM weights not loaded: {e}")

    pipe = IntegrityMoEPipeline(
        chunks=chunks,
        claims=claims,
        retriever=retriever,
        router=router,
        ebm=ebm,
        rerank_path=(rerank_path.strip() if rerank_path.strip() else "")
    )
    return {"chunks": chunks, "claims": claims, "pipe": pipe}

if reload_btn or st.session_state.rt is None:
    try:
        st.session_state.rt = build_runtime()
        st.sidebar.success(f"Loaded chunks={len(st.session_state.rt['chunks'])}, claims={len(st.session_state.rt['claims'])}")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

rt = st.session_state.rt
chunks = rt["chunks"]
claims = rt["claims"]
pipe = rt["pipe"]

q = st.text_area(
    "Question",
    height=110,
    value="Why is ethnicity considered an independent risk factor for GDM, and how is this linked to background population rates of type 2 diabetes?"
)

run = st.button("Infer")

if run:
    out = pipe.infer(q, top_chunks=top_chunks, max_claims=max_claims)

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Result")
        final = out["final"]
        best = out.get("best")

        if final["decision"] == "ALLOW":
            st.success("ALLOW")
            st.write(best["answer_text"] if best else "")
        else:
            st.error("ABSTAIN (integrity gates failed)")
            st.write(final.get("flags", []))
            if best and best.get("answer_text"):
                st.caption("Best draft (debug, not allowed):")
                st.write(best["answer_text"])

        if best:
            st.subheader("Blueprint")
            st.json(best.get("blueprint", {}))

    with right:
        st.subheader("MoE Router votes")
        st.json(out["router"])

        if best:
            st.subheader("Best candidate scoring")
            st.json({
                "expert": best["expert"],
                "router_weight": best["router_weight"],
                "energy": best["energy"],
                "total": best["total"],
                "gates": best["gates"],
                "features": best["features"],
            })

    if show_debug:
        tab1, tab2, tab3 = st.tabs(["Candidates", "Top Chunks", "Top Claims"])

        with tab1:
            st.subheader("Candidates (sorted by total)")
            for i, cand in enumerate(out["candidates"][:8], start=1):
                header = f"{i}. {cand['expert']} | w={cand['router_weight']:.3f} | E={cand['energy']:.3f} | total={cand['total']:.3f} | {cand['gates']['decision']}"
                with st.expander(header, expanded=(i == 1)):
                    st.write("Gates:", cand["gates"])
                    st.write("EBM features:", cand["features"])
                    st.write("Used entities:", cand["used_entities"])
                    st.write("Used edges:", cand["used_edges"])
                    st.write("Answer:")
                    st.write(cand["answer_text"])
                    st.json(cand["sentences"])

        with tab2:
            st.subheader("Top chunks (BM25 + boost - penalty)")
            for item in out["chunk_hits"][:10]:
                ch = chunks[item["chunk_idx"]]
                bd = item["breakdown"]
                title = f"{ch.chunk_id} | {ch.file} | total={bd['total']:.3f} (bm25={bd['bm25']:.3f}, boost={bd['boost']:.3f}, pen={bd['penalty']:.3f})"
                with st.expander(title, expanded=False):
                    st.write(ch.text)
                    st.json(bd)

        with tab3:
            st.subheader("Top claims (BM25 + boost - penalty + optional rerank)")
            for item in out["claim_hits"][:15]:
                cl = claims[item["claim_idx"]]
                bd = item["breakdown"]
                title = f"{cl.claim_id} | {cl.file} | total={bd['total']:.3f} (bm25={bd['bm25']:.3f}, boost={bd['boost']:.3f}, pen={bd['penalty']:.3f})"
                with st.expander(title, expanded=False):
                    st.write("Sentence:")
                    st.write(cl.sentence)
                    st.caption(f"Span: {cl.span[0]}–{cl.span[1]} | chunk={cl.chunk_id}")
                    st.write("Support window:")
                    st.write(cl.support_text)
                    st.write({"entities": cl.entities, "edge_hints": cl.edge_hints})
                    st.json(bd)
