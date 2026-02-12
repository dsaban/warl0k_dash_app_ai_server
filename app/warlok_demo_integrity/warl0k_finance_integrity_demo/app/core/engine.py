from typing import Dict, Any, List
import pandas as pd

STATUS_ICON = {"OK":"🟢","DUE":"🟠","OVERDUE":"🔴","UNKNOWN":"⚪","NA":"➖","BLOCKED":"⛔"}

class FinanceIntegrityEngine:
    def __init__(self, store, retriever):
        self.store = store
        self.retriever = retriever

    def invoice_checks(self, invoice_id: str) -> List[Dict[str, Any]]:
        inv_df = self.store.invoices
        row = inv_df[inv_df["invoice_id"] == invoice_id]
        if len(row) == 0:
            return [{"cid":"C000","title":"Invoice not found","status":"UNKNOWN","recommendation":"Verify invoice ID","evidence":[]}]
        inv = row.iloc[0].to_dict()

        vendor = self.store.vendors[self.store.vendors["vendor_id"] == inv["vendor_id"]].iloc[0].to_dict()
        contract = self.store.contracts[self.store.contracts["contract_id"] == vendor["contract_id"]].iloc[0].to_dict()
        appr = self.store.approvals[self.store.approvals["invoice_id"] == invoice_id]

        amount = float(inv.get("amount_usd", 0.0))
        po_required = str(contract.get("requires_po","NO")).upper() == "YES"
        has_po = str(inv.get("po_number","")).strip() != ""
        received = str(inv.get("received_goods","")).upper()

        checks = []

        # Duplicate invoice number per vendor
        dup = inv_df[
            (inv_df["vendor_id"] == inv["vendor_id"]) &
            (inv_df["invoice_number"] == inv["invoice_number"]) &
            (inv_df["invoice_id"] != inv["invoice_id"])
        ]
        status = "OK" if len(dup) == 0 else "BLOCKED"
        checks.append({
            "cid":"C101","title":"Duplicate invoice number","status":status,
            "recommendation":"Investigate duplicate vendor invoice number before posting/payment.",
            "policy_query":"duplicate invoice number vendor investigate",
        })

        # PO required + missing
        status = "BLOCKED" if (po_required and not has_po) else "OK"
        checks.append({
            "cid":"C102","title":"PO requirement","status":status,
            "recommendation":"If contract requires a PO, invoice must reference an approved PO.",
            "policy_query":"three way match PO required invoice must reference approved PO",
        })

        # Receiving confirmation
        status = "BLOCKED" if (po_required and has_po and received in ("NO","")) else "OK"
        checks.append({
            "cid":"C103","title":"Receiving confirmation (3-way match)","status":status,
            "recommendation":"Require receiving confirmation for PO-based invoices prior to payment.",
            "policy_query":"receiving confirmation required prior to payment three way match",
        })

        # Approval threshold
        need_cfo = amount >= 10000.0
        have_mgr = any(appr["approver_role"].astype(str).str.lower().eq("manager"))
        have_cfo = any(appr["approver_role"].astype(str).str.lower().eq("cfo"))
        if need_cfo and not have_cfo:
            status = "DUE"
        elif not have_mgr:
            status = "DUE"
        else:
            status = "OK"
        checks.append({
            "cid":"C104","title":"Approvals present (matrix)","status":status,
            "recommendation":"Ensure approvals meet threshold policy (manager; CFO for high-value payments).",
            "policy_query":"approval matrix CFO threshold 10000 requires CFO approval",
        })

        # GL coding
        gl = str(inv.get("gl_account","")).strip()
        cc = str(inv.get("cost_center","")).strip()
        status = "OK" if (gl and cc) else "DUE"
        checks.append({
            "cid":"C105","title":"GL + cost center coding","status":status,
            "recommendation":"Assign valid GL account and cost center before posting.",
            "policy_query":"GL coding cost center required before posting invoice",
        })

        for c in checks:
            c["evidence"] = self.retriever.search(c["policy_query"], k=3)
        return checks

    def close_checks(self, close_period: str) -> List[Dict[str, Any]]:
        tasks = self.store.close_tasks[self.store.close_tasks["close_period"] == close_period].copy()
        if len(tasks) == 0:
            return [{
                "cid":"CL000","title":"No close tasks configured","status":"UNKNOWN",
                "recommendation":"Add close checklist tasks for this period.",
                "evidence": self.retriever.search("close checklist reconciliations accrual SOX sign off", k=3),
                "details":[]
            }]

        incomplete = tasks[~tasks["status"].astype(str).str.upper().eq("DONE")]
        status = "OK" if len(incomplete) == 0 else "OVERDUE"
        return [{
            "cid":"CL101","title":"Close checklist completion","status":status,
            "recommendation":"Complete reconciliations, accrual review, and SOX sign-off before close approval.",
            "evidence": self.retriever.search("close checklist reconciliations accrual SOX sign off", k=3),
            "details": tasks.to_dict(orient="records")
        }]
