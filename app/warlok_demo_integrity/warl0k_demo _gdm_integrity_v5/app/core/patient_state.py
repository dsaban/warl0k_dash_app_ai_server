from typing import Dict, Any, List

class PatientStateEngine:
    def __init__(self, store, retriever):
        self.store = store
        self.retriever = retriever

    def _has_event(self, patient: Dict[str, Any], event_type: str) -> bool:
        for e in patient.get("events", []):
            if e.get("type") == event_type:
                return True
        return False

    def _eval_check(self, patient: Dict[str, Any], chk: Dict[str, Any]) -> Dict[str, Any]:
        logic = chk["logic"]
        status = "UNKNOWN"

        # NOT_APPLICABLE if required flags not present
        for flag in logic.get("requires_flag_true", []):
            if not patient.get(flag, False):
                return {**chk, "status":"NOT_APPLICABLE"}

        # basic condition types
        if logic["type"] == "window_due":
            ga = patient.get("gestational_age_weeks")
            if ga is None:
                status = "UNKNOWN"
            else:
                in_window = (ga >= logic["ga_min"]) and (ga <= logic["ga_max"])
                missing_event = all(not self._has_event(patient, t) for t in logic.get("requires_event_absent", []))
                if in_window and missing_event:
                    status = "DUE"
                else:
                    status = "OK"

        elif logic["type"] == "overdue":
            ga = patient.get("gestational_age_weeks")
            if ga is None:
                status = "UNKNOWN"
            else:
                missing_event = all(not self._has_event(patient, t) for t in logic.get("requires_event_absent", []))
                if ga > logic["ga_gt"] and missing_event:
                    status = "OVERDUE"
                else:
                    status = "OK"

        elif logic["type"] == "postpartum_due":
            pp = patient.get("postpartum_weeks")
            if pp is None:
                status = "UNKNOWN"
            else:
                in_window = (pp >= logic["pp_min"]) and (pp <= logic["pp_max"])
                missing_event = all(not self._has_event(patient, t) for t in logic.get("requires_event_absent", []))
                if in_window and missing_event:
                    status = "DUE"
                else:
                    status = "OK"

        elif logic["type"] == "plan_check":
            have_required = all(self._has_event(patient, t) for t in logic.get("requires_event_present", []))
            missing_plan = all(not self._has_event(patient, t) for t in logic.get("requires_event_absent", []))
            if have_required and missing_plan:
                status = "DUE"
            else:
                status = "OK"

        return {**chk, "status": status}

    def evaluate_patient(self, patient: Dict[str, Any], k: int = 8) -> List[Dict[str, Any]]:
        out = []
        for chk in self.store.state_checks:
            res = self._eval_check(patient, chk)
            # evidence retrieval: use title + recommendation as query
            query = f"{chk['title']} {chk['recommendation']}"
            ev = self.retriever.search(query, k=k)
            out.append({
                "sid": chk["sid"],
                "title": chk["title"],
                "description": chk["description"],
                "status": res["status"],
                "recommendation": chk["recommendation"],
                "gold_claim_ids": chk.get("gold_claim_ids", []),
                "evidence": ev[:5],
            })
        return out
