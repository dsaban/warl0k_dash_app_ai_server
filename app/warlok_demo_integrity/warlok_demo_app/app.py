
import json
from pathlib import Path
from collections import Counter

import streamlit as st
import matplotlib.pyplot as plt

import core

APP_TITLE = "WARL0K GDM Integrity Demo 1 (Evidence-Locked Q&A)"
BASE_DIR = Path(__file__).parent
CLAIMS_PATH = BASE_DIR / "claims_200.jsonl"
EVAL_PATH = BASE_DIR / "eval_set_100.jsonl"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Local, NumPy-only retrieval + constrained composition + integrity scoring (support + contradiction checks).")

@st.cache_data
def load_claims():
    return core.load_jsonl(CLAIMS_PATH)

@st.cache_data
def load_eval():
    return core.load_jsonl(EVAL_PATH)

@st.cache_resource
def build_index(claims):
    return core.build_claim_matrix(claims)

claims = load_claims()
C = build_index(claims)
eval_rows = load_eval()

tab1, tab2, tab3 = st.tabs(["Ask a question", "Batch benchmark", "Claim browser"])

with tab1:
    colA, colB = st.columns([1,1], gap="large")

    with colA:
        st.subheader("Evidence-locked answer")
        q = st.text_area("Question", value="In a low-risk pregnancy, when is GDM screening usually performed and what test is typically used?", height=90)
        k = st.slider("Top-k evidence claims", min_value=3, max_value=12, value=6, step=1)
        max_sents = st.slider("Max sentences in locked answer", min_value=1, max_value=5, value=3, step=1)

        if st.button("Generate evidence-locked answer", type="primary"):
            top, sims = core.retrieve(claims, C, q, k=k)
            locked = core.compose_answer(top, max_sentences=max_sents)
            st.session_state["last_q"] = q
            st.session_state["locked_answer"] = locked
            st.session_state["locked_top"] = top
            st.session_state["locked_sims"] = [float(x) for x in sims]

        locked = st.session_state.get("locked_answer", "")
        if locked:
            st.markdown("**Locked Answer:**")
            st.write(locked)

            m = core.support_metrics(claims, C, locked)
            label = core.integrity_label(m)
            st.markdown(f"**Integrity (locked):** `{label}`  •  support={m['support_rate']:.3f}  •  contradictions={m['contradictions']}")

            with st.expander("Evidence used (top-k)"):
                top = st.session_state.get("locked_top", [])
                sims = st.session_state.get("locked_sims", [])
                for c, s in zip(top, sims):
                    st.markdown(f"- **{c['claim_id']}** ({s:.3f}) — *{c['doc']} | {c.get('kind','')}*")
                    st.write(c["text"])
                    st.caption("Evidence span:")
                    st.write(c.get("evidence",""))

    with colB:
        st.subheader("Compare a free answer (paste) vs locked")
        free = st.text_area("Paste a free-form answer (e.g., ChatGPT)", value="", height=180, placeholder="Paste any answer here to score it against the claim store…")

        if st.button("Score pasted answer"):
            st.session_state["free_answer"] = free
            m_free = core.support_metrics(claims, C, free)
            st.session_state["m_free"] = m_free
            st.session_state["free_label"] = core.integrity_label(m_free)

        if "m_free" in st.session_state:
            st.markdown("**Free Answer:**")
            st.write(st.session_state.get("free_answer",""))
            m_free = st.session_state["m_free"]
            st.markdown(f"**Integrity (free):** `{st.session_state['free_label']}`  •  support={m_free['support_rate']:.3f}  •  contradictions={m_free['contradictions']}")

            with st.expander("Sentence-by-sentence best evidence match"):
                for sent, claim, sim in m_free["matches"]:
                    st.markdown(f"- **sim={sim:.3f}** → **{claim['claim_id']}** (*{claim['doc']}*)")
                    st.write(sent)
                    st.caption("Best-matching claim:")
                    st.write(claim["text"])
                    st.caption("Evidence span:")
                    st.write(claim.get("evidence",""))

with tab2:
    st.subheader("Batch benchmark on the 100-question eval set")
    st.write("Optionally upload `free_answers.jsonl` with rows like: `{ \"qid\": \"Q001\", \"answer\": \"...\" }`.")
    uploaded = st.file_uploader("Upload free_answers.jsonl (optional)", type=["jsonl","txt"])
    free_map = {}
    if uploaded is not None:
        data = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
        for line in data:
            if not line.strip():
                continue
            obj = json.loads(line)
            free_map[obj["qid"]] = obj.get("answer","")

    report = core.batch_score(eval_rows, claims, C, free_map=free_map)

    locked_labels = Counter([r["locked_label"] for r in report])
    st.markdown("### Locked summary")
    st.json(dict(locked_labels))

    if free_map:
        free_labels = Counter([r.get("free_label","") for r in report if "free_label" in r])
        st.markdown("### Free summary")
        st.json(dict(free_labels))

    def plot_counts(counter, title):
        labels = list(counter.keys())
        values = [counter[k] for k in labels]
        fig = plt.figure()
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel("count")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    plot_counts(locked_labels, "Locked integrity labels")
    if free_map:
        plot_counts(free_labels, "Free integrity labels")

    st.download_button("Download report.json", data=json.dumps(report, indent=2), file_name="report.json", mime="application/json")

with tab3:
    st.subheader("Claim browser (200 claims)")
    query = st.text_input("Search claims (substring match)", value="OGTT")
    show_n = st.slider("Show first N matches", min_value=5, max_value=50, value=15, step=5)

    ql=query.strip().lower()
    matches=[]
    for c in claims:
        blob = (c.get("text","") + " " + c.get("evidence","")).lower()
        if ql in blob:
            matches.append(c)

    st.write(f"Matches: {len(matches)}")
    for c in matches[:show_n]:
        st.markdown(f"**{c['claim_id']}** — *{c['doc']} | {c.get('kind','')}*")
        st.write(c["text"])
        st.caption("Evidence span:")
        st.write(c.get("evidence",""))
        st.divider()
