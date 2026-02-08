import streamlit as st
from core.store import DataStore
from core.retrieval import Retriever
from core.qa import QAEvaluator
from core.patient_state import PatientStateEngine

st.set_page_config(page_title="WARL0K GDM Integrity Demo", layout="wide")

@st.cache_resource
def load_store():
    return DataStore.load_default()

store = load_store()
retriever = Retriever(store.passages)
qa_eval = QAEvaluator(store)
ps_engine = PatientStateEngine(store, retriever)

st.sidebar.title("WARL0K • GDM Integrity")
mode = st.sidebar.radio("Mode", ["Evidence Search", "Q/A Validator", "Patient State"], index=2)

st.sidebar.markdown("---")
st.sidebar.caption("Integrity rule: answers must be supported by retrieved evidence (or the system refuses).")

if mode == "Evidence Search":
    st.title("Evidence Search (BM25 + TF‑IDF semantic)")
    q = st.text_input("Query", "low risk pregnancy screening window 24 28 weeks OGTT")
    k = st.slider("Top‑K", 3, 15, 8)

    if st.button("Search"):
        results = retriever.search(q, k=k)
        for r in results:
            with st.expander(f"{r['rank']}. {r['doc']} • {r['pid']} • score={r['score']:.3f}"):
                st.write(r["text"])
                st.code(r["highlights"], language="text")

elif mode == "Q/A Validator":
    st.title("Q/A Validator (locked answers + variants)")
    qs = store.questions
    qids = [q["qid"] for q in qs]
    selected = st.selectbox("Question ID", qids, index=0)
    qobj = next(q for q in qs if q["qid"] == selected)

    st.subheader("Question")
    st.write(qobj["question"])

    st.subheader("Locked answer (gold)")
    st.write(qobj["locked_answer"])

    st.subheader("Try a variant / custom answer")
    variant = st.selectbox("Example", ["locked_answer", "contradictory_example", "bad_example", "neutral_example", "entangled_example", "custom"])
    if variant == "custom":
        answer = st.text_area("Answer to validate", value="", height=140)
    else:
        answer = qobj[variant] if variant != "locked_answer" else qobj["locked_answer"]
        st.info(answer)

    k = st.slider("Evidence Top‑K", 3, 20, 10)
    if st.button("Validate"):
        report = qa_eval.validate(qobj, answer, retriever, k=k)

        col1, col2 = st.columns([1,1])
        with col1:
            st.subheader("Result")
            st.metric("Integrity score", f"{report['score']:.2f}")
            st.write("**Verdict:**", report["verdict"])
            st.write("**Reasons:**")
            for x in report["reasons"]:
                st.write(f"- {x}")

        with col2:
            st.subheader("Evidence used")
            for e in report["evidence"]:
                with st.expander(f"{e['rank']}. {e['doc']} • {e['pid']} • score={e['score']:.3f}"):
                    st.write(e["text"])

        st.subheader("Field checks")
        st.json(report["field_checks"])

elif mode == "Patient State":
    st.title("Patient State (Gold care checks + evidence‑locked recommendations)")
    pats = store.patients
    pid = st.selectbox("Patient", [p["pid"] + " — " + p.get("name","") for p in pats], index=0)
    pid = pid.split(" — ")[0]
    patient = next(p for p in pats if p["pid"] == pid)

    st.subheader("Patient snapshot")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Risk level", str(patient.get("risk_level", "—")))
    c2.metric("GA weeks", str(patient.get("gestational_age_weeks", "—")))
    c3.metric("Postpartum weeks", str(patient.get("postpartum_weeks", "—")))
    c4.metric("History GDM", "Yes" if patient.get("history_gdm") else "No")

    st.markdown("---")
    st.subheader("Checklist")
    results = ps_engine.evaluate_patient(patient, k=8)

    for res in results:
        status = res["status"]
        icon = {"DUE":"🟠","OVERDUE":"🔴","OK":"🟢","UNKNOWN":"⚪","NOT_APPLICABLE":"➖"}.get(status,"⚪")
        with st.expander(f"{icon} {res['sid']} • {res['title']} — {status}"):
            st.write(res["description"])
            st.write("**Recommendation:**", res["recommendation"])
            st.write("**Why (evidence‑locked):**")
            for ev in res["evidence"]:
                st.write(f"- {ev['doc']} • {ev['pid']} (score={ev['score']:.3f})")
                st.caption(ev["text"])
