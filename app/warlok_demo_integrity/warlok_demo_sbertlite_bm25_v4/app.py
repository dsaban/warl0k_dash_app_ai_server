
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
st.caption("Local, NumPy-only retrieval + constrained composition + integrity scoring. Includes a learned classifier trained on synthetic variants.")

@st.cache_data
def load_claims():
    return core.load_jsonl(CLAIMS_PATH)

@st.cache_data
def load_eval():
    return core.load_jsonl(EVAL_PATH)

@st.cache_resource
def build_matchers(claims, eval_rows):
    C = core.build_claim_matrix(claims)

    model_path = (BASE_DIR / "sbert_lite_model.npz")
    if model_path.exists():
        sbert = core.SBERLite.load(model_path)
    else:
        sbert = core.train_sbert_lite_from_eval(eval_rows, claims, epochs=25, batch_size=64, in_dim=4096, out_dim=256, lr=0.05, seed=7)
        sbert.save(model_path)

    claim_embs = sbert.encode_texts([c.get("text","") for c in claims])

    bm25 = core.build_bm25_index(claims)

    learned = core.fit_learned_evaluator(eval_rows, claims, C, sbert_model=sbert, claim_embs=claim_embs)
    return C, learned, sbert, claim_embs, bm25

claims = load_claims()
eval_rows = load_eval()
eval_by_id = {r["qid"]: r for r in eval_rows}
C, learned_model, sbert_model, claim_embs, bm25_index = build_matchers(claims, eval_rows)

if "free_answers_map" not in st.session_state:
    st.session_state["free_answers_map"] = {}

st.sidebar.header("Evaluator")

st.sidebar.header("Matcher (evidence similarity)")
matcher = st.sidebar.radio("Evidence matcher", ["BM25→SBERT rerank (best)", "SBERT-lite (recommended)", "Hasher (baseline)"], index=0)

mode = st.sidebar.radio("Choose evaluator", ["Learned (recommended)", "Heuristic"], index=0)
show_probs = st.sidebar.checkbox("Show probability breakdown (learned)", value=True)

tab_qbank, tab1, tab2, tab3 = st.tabs(["Question bank (Eval 100)", "Ask a question", "Batch benchmark", "Claim browser"])

def render_integrity(answer: str):
    if mode.startswith("Learned"):
        lab, p, m = core.score_answer_learned(learned_model, claims, C, answer)
        st.markdown(f"**Integrity (learned):** `{lab}`  •  support={m['support_rate']:.3f}  •  contradictions={m['contradictions']}")
        if show_probs:
            st.caption("Probabilities (good / neutral / entangled / bad / contradictory)")
            st.write([float(x) for x in p])
        return m
    m = core.support_metrics(claims, C, answer)
    st.markdown(f"**Integrity (heuristic):** `{core.integrity_label_heuristic(m)}`  •  support={m['support_rate']:.3f}  •  contradictions={m['contradictions']}")
    return m

with tab_qbank:
    st.subheader("Question bank: select an eval question and compare vs a pasted GPT answer")
    left, right = st.columns([1, 1], gap="large")

    with left:
        qids = sorted(eval_by_id.keys())
        chosen = st.selectbox("Select question (qid)", qids, index=0)
        row = eval_by_id[chosen]

        st.markdown("**Question:**")
        st.write(row["question"])

        st.markdown("**Evidence-locked answer (baseline from eval set):**")
        st.write(row.get("locked_answer", ""))

        render_integrity(row.get("locked_answer", ""))

        with st.expander("Gold support claims (top-3 IDs) + evidence"):
            st.write("Gold claim IDs:", ", ".join(row.get("gold_claim_ids", [])))
            gold_set = set(row.get("gold_claim_ids", []))
            for c in claims:
                if c.get("claim_id") in gold_set:
                    st.markdown(f"- **{c['claim_id']}** — *{c['doc']} | {c.get('kind','')}*")
                    st.write(c.get("text",""))
                    st.caption("Evidence span:")
                    st.write(c.get("evidence",""))

    with right:
        st.markdown("### Paste GPT/free answer for this question")
        existing = st.session_state["free_answers_map"].get(chosen, "")
        pasted = st.text_area("GPT/free answer", value=existing, height=220, placeholder="Paste the GPT answer here…")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            if st.button("Save answer", type="primary"):
                st.session_state["free_answers_map"][chosen] = pasted
                st.success(f"Saved answer for {chosen}")
        with c2:
            if st.button("Clear answer"):
                st.session_state["free_answers_map"].pop(chosen, None)
                st.info(f"Cleared answer for {chosen}")
        with c3:
            score_now = st.button("Score now")

        if score_now:
            render_integrity(pasted)
            with st.expander("Sentence-by-sentence best evidence match"):
                if matcher.startswith("SBERT") or matcher.startswith("BM25"):
                    st.caption("SBERT-lite best matches (semantic rerank)")
                m_free = core.support_metrics(claims, C, pasted)
                for sent, claim, sim in m_free["matches"]:
                    st.markdown(f"- **sim={sim:.3f}** → **{claim['claim_id']}** (*{claim['doc']}*)")
                    st.write(sent)
                    if matcher.startswith('SBERT') or matcher.startswith('BM25'):
                        best, bs = core.sbert_retrieve(claims, claim_embs, sent, sbert_model, k=1)
                        st.caption(f"SBERT-lite best claim (sim={float(bs[0]):.3f}): {best[0]['claim_id']}")
                        st.write(best[0].get('text',''))
                        st.caption('Evidence span:')
                        st.write(best[0].get('evidence',''))
                    st.caption("Best-matching claim:")
                    st.write(claim.get("text",""))
                    st.caption("Evidence span:")
                    st.write(claim.get("evidence",""))

        st.markdown("### Export saved GPT answers")
        if st.session_state["free_answers_map"]:
            export_rows = [{"qid": qid, "answer": ans} for qid, ans in sorted(st.session_state["free_answers_map"].items())]
            data = "\n".join(json.dumps(r, ensure_ascii=False) for r in export_rows) + "\n"
            st.download_button("Download free_answers.jsonl", data=data, file_name="free_answers.jsonl", mime="application/jsonl")
        else:
            st.caption("No saved answers yet.")

with tab1:
    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        st.subheader("Evidence-locked answer (ad-hoc)")
        q = st.text_area("Question", value="In a low-risk pregnancy, when is GDM screening usually performed and what test is typically used?", height=90)
        k = st.slider("Top-k evidence claims", min_value=3, max_value=12, value=6, step=1, key="k_ad_hoc")
        max_sents = st.slider("Max sentences in locked answer", min_value=1, max_value=5, value=3, step=1, key="ms_ad_hoc")

        if st.button("Generate evidence-locked answer", type="primary", key="gen_ad_hoc"):
            top, sims = (core.hybrid_retrieve_bm25_sbert(claims, bm25_index, claim_embs, q, sbert_model, recall_k=40, k=k) if matcher.startswith('BM25') else (core.sbert_retrieve(claims, claim_embs, q, sbert_model, k=k) if matcher.startswith('SBERT') else core.retrieve(claims, C, q, k=k)))
            locked = core.compose_answer(top, max_sentences=max_sents)
            st.session_state["locked_answer"] = locked
            st.session_state["locked_top"] = top
            st.session_state["locked_sims"] = [float(x) for x in sims]

        locked = st.session_state.get("locked_answer", "")
        if locked:
            st.markdown("**Locked Answer:**")
            st.write(locked)
            render_integrity(locked)

            with st.expander("Evidence used (top-k)"):
                top = st.session_state.get("locked_top", [])
                sims = st.session_state.get("locked_sims", [])
                for c, s in zip(top, sims):
                    st.markdown(f"- **{c['claim_id']}** ({s:.3f}) — *{c['doc']} | {c.get('kind','')}*")
                    st.write(c.get("text",""))
                    st.caption("Evidence span:")
                    st.write(c.get("evidence",""))

    with colB:
        st.subheader("Score a pasted free answer (ad-hoc)")
        free = st.text_area("Paste a free-form answer (e.g., ChatGPT)", value="", height=180, placeholder="Paste any answer here…")
        if st.button("Score pasted answer", key="score_ad_hoc"):
            render_integrity(free)

with tab2:
    st.subheader("Batch benchmark on the 100-question eval set")
    st.write("Optionally upload `free_answers.jsonl` with rows like: `{ \"qid\": \"Q001\", \"answer\": \"...\" }`.")
    uploaded = st.file_uploader("Upload free_answers.jsonl (optional)", type=["jsonl", "txt"])
    free_map = {}
    if uploaded is not None:
        data = uploaded.getvalue().decode("utf-8", errors="ignore").splitlines()
        for line in data:
            if not line.strip():
                continue
            obj = json.loads(line)
            free_map[obj["qid"]] = obj.get("answer","")

    report = core.batch_score(eval_rows, claims, C, free_map=free_map, learned_model=learned_model, sbert_model=sbert_model, claim_embs=claim_embs)

    if mode.startswith("Learned"):
        locked_labels = Counter([r["locked_label_learned"] for r in report])
        st.markdown("### Locked summary (learned)")
        st.json(dict(locked_labels))
        if free_map:
            free_labels = Counter([r.get("free_label_learned","") for r in report if "free_label_learned" in r])
            st.markdown("### Free summary (learned)")
            st.json(dict(free_labels))
    else:
        locked_labels = Counter([r["locked_label_heur"] for r in report])
        st.markdown("### Locked summary (heuristic)")
        st.json(dict(locked_labels))
        if free_map:
            free_labels = Counter([r.get("free_label_heur","") for r in report if "free_label_heur" in r])
            st.markdown("### Free summary (heuristic)")
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

    plot_counts(locked_labels, "Integrity labels")
    if free_map:
        plot_counts(free_labels, "Free integrity labels")

    st.download_button("Download report.json", data=json.dumps(report, indent=2), file_name="report.json", mime="application/json")

with tab3:
    st.subheader("Claim browser (200 claims)")
    query = st.text_input("Search claims (substring match)", value="OGTT")
    show_n = st.slider("Show first N matches", min_value=5, max_value=50, value=15, step=5)

    ql = query.strip().lower()
    matches = []
    for c in claims:
        blob = (c.get("text","") + " " + c.get("evidence","")).lower()
        if ql in blob:
            matches.append(c)

    st.write(f"Matches: {len(matches)}")
    for c in matches[:show_n]:
        st.markdown(f"**{c['claim_id']}** — *{c['doc']} | {c.get('kind','')}*")
        st.write(c.get("text",""))
        st.caption("Evidence span:")
        st.write(c.get("evidence",""))
        st.divider()
