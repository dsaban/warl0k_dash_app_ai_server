import json
from pathlib import Path
import tempfile

import streamlit as st
from pyvis.network import Network

from core.ingest import build_corpus_from_paths
from core.graph_store import build_graph
from core.novelty import novelty_summary, draft_claim_skeleton

st.set_page_config(page_title="Patent Integrity Graph Explorer", layout="wide")

DEFAULT_PATHS = [f"/mnt/data/{i}.txt" for i in range(10)]

@st.cache_data(show_spinner=False)
def load_corpus(paths):
    return build_corpus_from_paths(paths)

def bm25_search(corpus, query, top_k=25):
    bm25 = corpus.get("bm25")
    if bm25 is None:
        return []
    hits = bm25.search(query, top_k=top_k)
    out = []
    for idx, score in hits:
        a = corpus["atomic"][idx]
        out.append((a, score))
    return out

def filter_by_state(records, required_tags):
    out = []
    for r in records:
        tags = r.get("tags", {})
        ok = True
        for cat, wanted in required_tags.items():
            if not wanted:
                continue
            have = set(tags.get(cat, []))
            if have.isdisjoint(wanted):
                ok = False
                break
        if ok:
            out.append(r)
    return out

def render_pyvis(graph_dict):
    net = Network(height="650px", width="100%", bgcolor="#0b1020", font_color="white", directed=False)
    net.barnes_hut()
    for n in graph_dict["nodes"]:
        nid = n["id"]
        label = n["label"]
        ntype = n["type"]
        size = 20 if ntype == "atomic" else 12
        title = json.dumps(n.get("meta", {}), ensure_ascii=False, indent=2)[:1500]
        net.add_node(nid, label=label, title=f"<pre>{title}</pre>", size=size, group=ntype)
    for e in graph_dict["edges"]:
        net.add_edge(e["source"], e["target"], title=e["type"])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        html = Path(f.name).read_text(encoding="utf-8", errors="ignore")
    st.components.v1.html(html, height=700, scrolling=True)

st.title("Patent Integrity Graph Explorer (Demo)")
st.caption("Evidence-locked claim mining • BM25 lexical search • Graph exploration • Novelty/Draft builder")

with st.sidebar:
    st.header("Corpus")
    use_default = st.checkbox("Use default /mnt/data/0..9.txt", value=True)
    if use_default:
        paths = DEFAULT_PATHS
    else:
        folder = st.text_input("Folder with .txt patent files", value=str(Path("data/raw").resolve()))
        folder_p = Path(folder)
        paths = [str(p) for p in sorted(folder_p.glob("*.txt"))]

    st.divider()
    st.header("Query State (filters)")
    mech = st.multiselect("Mechanisms", [
        "CHALLENGE_RESPONSE","NONCE","TIME_WINDOW","COUNTER","DEVICE_FINGERPRINT","PUF",
        "OUT_OF_BAND","PAIRING","ATTESTATION","ANOMALY_SCORING"
    ])
    crypto = st.multiselect("Crypto", ["MAC_AEAD","HASH","KDF","SIGNATURE"])
    constraints = st.multiselect("Constraints", ["MCU_CONSTRAINED","NO_PKI","OFFLINE","BROKERED","OT_GATEWAY"])
    threats = st.multiselect("Threats", ["REPLAY","MITM","CLONING","SPOOFING","HIJACK"])
    required = {"mechanism": set(mech), "crypto": set(crypto), "constraint": set(constraints), "threat": set(threats)}

corpus = load_corpus(paths)
atomic_all = corpus["atomic"]

tabs = st.tabs(["Search", "Evidence Pack", "Knowledge Graph", "Novelty / Draft Builder"])

if "selected_atomic_ids" not in st.session_state:
    st.session_state.selected_atomic_ids = []
if "focused_atomic_id" not in st.session_state:
    st.session_state.focused_atomic_id = None

with tabs[0]:
    st.subheader("Lexical Search (BM25) over Atomic Claims")
    q = st.text_input("Search query", value="m2m authentication device identification replay nonce")
    top_k = st.slider("Top K", 5, 50, 20)
    hits = bm25_search(corpus, q, top_k=top_k)

    filtered_hits = []
    for a, score in hits:
        tags = a.get("tags", {})
        ok = True
        for cat, wanted in required.items():
            if wanted and set(tags.get(cat, [])).isdisjoint(wanted):
                ok = False
                break
        if ok:
            filtered_hits.append((a, score))

    st.write(f"Results: {len(filtered_hits)} (after filters)")
    for a, score in filtered_hits:
        with st.container(border=True):
            st.markdown(f"**{a['patent_id']} — Claim {a['claim_no']} — {a['atomic_id']}**  \nBM25: `{score:.3f}` • Strength: `{a.get('strength',0):.3f}` • Independent: `{a.get('is_independent', False)}`")
            st.write(a["text"])
            t = a.get("tags", {})
            tag_line = " | ".join(
                [*(f"M:{x}" for x in t.get("mechanism", [])),
                 *(f"C:{x}" for x in t.get("crypto", [])),
                 *(f"K:{x}" for x in t.get("constraint", [])),
                 *(f"T:{x}" for x in t.get("threat", []))]
            )
            if tag_line:
                st.caption("Tags: " + tag_line)

            c1, c2, c3 = st.columns([1,1,2])
            with c1:
                if st.button("Focus evidence", key=f"focus-{a['atomic_id']}"):
                    st.session_state.focused_atomic_id = a["atomic_id"]
            with c2:
                if a["atomic_id"] in st.session_state.selected_atomic_ids:
                    if st.button("Remove from novelty", key=f"rm-{a['atomic_id']}"):
                        st.session_state.selected_atomic_ids.remove(a["atomic_id"])
                else:
                    if st.button("Add to novelty", key=f"add-{a['atomic_id']}"):
                        st.session_state.selected_atomic_ids.append(a["atomic_id"])
            with c3:
                ev = a.get("evidence", {})
                st.caption(f"Source: {ev.get('source')} • Where: {ev.get('where')}")

with tabs[1]:
    st.subheader("Evidence Pack (evidence-locked)")
    fid = st.session_state.focused_atomic_id
    if not fid:
        st.info("Pick an AtomicClaim and click **Focus evidence** in the Search tab.")
    else:
        rec = next((x for x in atomic_all if x["atomic_id"] == fid), None)
        if not rec:
            st.warning("Focused AtomicClaim not found.")
        else:
            st.markdown(f"### {rec['patent_id']} — Claim {rec['claim_no']} — {rec['atomic_id']}")
            st.write(rec["text"])
            st.write("**Evidence**")
            st.json(rec.get("evidence", {}))
            st.write("**Tags**")
            st.json(rec.get("tags", {}))
            st.write("**Scope / strength**")
            st.metric("Strength (demo)", rec.get("strength", 0.0))

            st.write("**Nearest neighbors (shared tags)**")
            my_tags = rec.get("tags", {})
            my_set = set(sum([my_tags.get(k, []) for k in my_tags], []))
            scored = []
            for other in atomic_all:
                if other["atomic_id"] == fid:
                    continue
                oset = set(sum([other.get("tags", {}).get(k, []) for k in other.get("tags", {})], []))
                inter = len(my_set.intersection(oset))
                if inter:
                    scored.append((inter, other))
            scored.sort(key=lambda x: x[0], reverse=True)
            for inter, other in scored[:10]:
                with st.container(border=True):
                    st.markdown(f"**{other['patent_id']} C{other['claim_no']} • shared tags: {inter}**")
                    st.write(other["text"][:260] + ("..." if len(other["text"])>260 else ""))

with tabs[2]:
    st.subheader("Knowledge Graph (local view)")
    universe = filter_by_state(atomic_all, required) if any(required.values()) else atomic_all
    st.caption(f"Graph built from {len(universe)} AtomicClaims (filtered by Query State if selected).")
    graph = build_graph(universe)
    render_pyvis(graph)

with tabs[3]:
    st.subheader("Novelty / Draft Builder")
    selected = [x for x in atomic_all if x["atomic_id"] in st.session_state.selected_atomic_ids]
    st.write(f"Selected building blocks: **{len(selected)}**")
    if selected:
        for s in selected:
            with st.container(border=True):
                st.markdown(f"**{s['atomic_id']}** — {s['patent_id']} C{s['claim_no']}")
                st.write(s["text"])
        summ = novelty_summary(selected, atomic_all)
        st.write("### Tag bundle (your invention composition)")
        st.json(summ["bundle"])
        st.write("### Rare tags within this corpus (good novelty hooks)")
        st.json(summ["rare_tags"])
        st.write("### Draft skeleton")
        st.code(draft_claim_skeleton(summ["bundle"]))
    else:
        st.info("Add AtomicClaims to novelty in the Search tab (click **Add to novelty**).")
