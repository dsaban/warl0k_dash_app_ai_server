import streamlit as st
import pandas as pd
from core.store import DataStore
from core.retrieval import PolicyRetriever
from core.engine import FinanceIntegrityEngine, STATUS_ICON
from core.viz import render_invoice_graph

st.set_page_config(page_title="WARL0K Finance Integrity Demo", layout="wide")

@st.cache_resource
def load_store():
    return DataStore.load()

store = load_store()
retriever = PolicyRetriever(store.policies)
engine = FinanceIntegrityEngine(store, retriever)

st.sidebar.title("WARL0K • Finance Integrity")
mode = st.sidebar.radio("Mode", ["Demo Overview", "Transactions", "Close Dashboard", "Evidence Library"], index=0)

def persist(name: str, df: pd.DataFrame):
    store.save_table(name, df)
    st.cache_resource.clear()

if mode == "Demo Overview":
    st.title("Finance Close & Audit Integrity (Evidence‑First Demo)")
    st.markdown("""
This demo shows an **Integrity‑First** control plane for AP and close:
- Every check produces **OK/DUE/OVERDUE/BLOCKED** statuses
- Every recommendation is bound to **policy evidence** (retrieved + shown)
- Data is persisted in CSV files (acts as a lightweight database)

Use the sidebar to explore transactions and month‑end close status.
""")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Vendors", len(store.vendors))
    c2.metric("Invoices", len(store.invoices))
    c3.metric("Approvals", len(store.approvals))
    c4.metric("Close tasks", len(store.close_tasks))

elif mode == "Transactions":
    st.title("Transactions (AP Integrity)")
    left, right = st.columns([1,2])

    with left:
        st.subheader("Invoice list")
        df = store.invoices.copy()
        st.dataframe(df, use_container_width=True, hide_index=True)
        inv_id = st.selectbox("Select invoice", df["invoice_id"].tolist(), index=0)

        st.markdown("### Add / Edit invoices (persistent CSV)")
        edited = st.data_editor(df, num_rows="dynamic", use_container_width=True, key="inv_editor")
        if st.button("Save invoices to CSV"):
            persist("invoices", edited)
            st.success("Saved invoices.csv")

        st.markdown("### Approvals (persistent CSV)")
        ap = store.approvals.copy()
        ap_edit = st.data_editor(ap, num_rows="dynamic", use_container_width=True, key="appr_editor")
        if st.button("Save approvals to CSV"):
            persist("approvals", ap_edit)
            st.success("Saved approvals.csv")

    with right:
        inv = store.invoices[store.invoices["invoice_id"] == inv_id].iloc[0].to_dict()
        st.subheader(f"Invoice card: {inv_id}")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Vendor", inv.get("vendor_id",""))
        c2.metric("Amount (USD)", f"{float(inv.get('amount_usd',0)):,.2f}")
        c3.metric("PO", inv.get("po_number","—") if str(inv.get("po_number","")).strip() else "—")
        c4.metric("Close period", inv.get("close_period","—"))

        checks = engine.invoice_checks(inv_id)
        st.subheader("Integrity checklist")
        for c in checks:
            icon = STATUS_ICON.get(c["status"], "⚪")
            with st.expander(f"{icon} {c['cid']} • {c['title']} — {c['status']}"):
                st.write("**Recommendation:**", c["recommendation"])
                st.write("**Policy evidence (retrieved):**")
                for ev in c["evidence"]:
                    st.write(f"- {ev['title']} ({ev['tag']}) • score={ev['score']:.3f}")
                    st.caption(ev["clause"])

        st.subheader("Invoice integrity graph")
        render_invoice_graph(st, inv, checks)

elif mode == "Close Dashboard":
    st.title("Month‑End Close Integrity")
    period = st.selectbox("Close period", sorted(store.close_tasks["close_period"].unique().tolist()), index=0)
    st.subheader(f"Close checklist: {period}")

    tasks = store.close_tasks.copy()
    tasks_edit = st.data_editor(tasks, num_rows="dynamic", use_container_width=True, key="tasks_editor")
    if st.button("Save close tasks to CSV"):
        persist("close_tasks", tasks_edit)
        st.success("Saved close_tasks.csv")

    checks = engine.close_checks(period)
    for c in checks:
        icon = STATUS_ICON.get(c["status"], "⚪")
        with st.expander(f"{icon} {c['cid']} • {c['title']} — {c['status']}", expanded=True):
            st.write("**Recommendation:**", c["recommendation"])
            st.write("**Tasks:**")
            st.dataframe(pd.DataFrame(c.get("details",[])), use_container_width=True, hide_index=True)
            st.write("**Policy evidence:**")
            for ev in c["evidence"]:
                st.write(f"- {ev['title']} ({ev['tag']}) • score={ev['score']:.3f}")
                st.caption(ev["clause"])

elif mode == "Evidence Library":
    st.title("Evidence Library (Policies)")
    st.markdown("Search policies like you search medical evidence: **retrieve → cite → enforce**.")
    q = st.text_input("Search policies", "CFO approval threshold for payments")
    hits = retriever.search(q, k=5)
    for ev in hits:
        with st.expander(f"{ev['policy_id']} • {ev['title']} ({ev['tag']}) • score={ev['score']:.3f}"):
            st.write(ev["clause"])
