import re
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

GA_RE = re.compile(r"(?P<w>\d{1,2})\s*(?:w|wk|weeks?)\s*(?:\+\s*(?P<d>\d{1,2})\s*(?:d|days?))?", re.I)

def parse_ga(s: str) -> Optional[Tuple[int,int]]:
    """Parse gestational age like '26w+3d' -> (26,3)."""
    if not s:
        return None
    m = GA_RE.search(str(s))
    if not m:
        # try '24-28 weeks' -> take first number
        m2 = re.search(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*weeks?", str(s), re.I)
        if m2:
            return (int(m2.group(1)), 0)
        return None
    w = int(m.group('w'))
    d = int(m.group('d') or 0)
    return (w,d)

def ga_to_days(ga: Tuple[int,int]) -> int:
    return ga[0]*7 + ga[1]

def mmol_to_mgdl_glucose(v: float) -> float:
    return v * 18.0

def mgdl_to_mmol_glucose(v: float) -> float:
    return v / 18.0

def normalize_glucose(v: float, unit: str) -> Tuple[float,str]:
    u = (unit or '').strip().lower()
    if u in ['mmol/l','mmol']:
        return float(v), 'mmol/L'
    if u in ['mg/dl','mgdl']:
        return mgdl_to_mmol_glucose(float(v)), 'mmol/L'
    # unknown unit, return as-is but mark
    return float(v), unit or ''

@dataclass
class ActionItem:
    action_id: str
    title: str
    due: str
    owner: str
    why_claim_ids: List[str]
    evidence_refs: List[Dict[str, Any]]
    severity: str = 'info'

class PathwayEngine:
    """Deterministic patient action generator backed by claim IDs."""
    def __init__(self, claims: Dict[str, Dict[str, Any]]):
        self.claims = claims or {}

    def _claim_ev(self, claim_ids: List[str]) -> List[Dict[str, Any]]:
        ev=[]
        for cid in claim_ids:
            c=self.claims.get(cid)
            if not c: 
                continue
            for e in (c.get('evidence') or []):
                ev.append(e)
        return ev[:6]

    def evaluate(self, patient: Dict[str, Any]) -> List[ActionItem]:
        # patient expected schema:
        # {id, name, ga_weeks, ga_days, events:[{type, ga, date, value, unit, ...}]}
        actions: List[ActionItem] = []
        ga = (int(patient.get('ga_weeks') or 0), int(patient.get('ga_days') or 0))
        ga_days = ga_to_days(ga)
        events = patient.get('events') or []

        def has_event(t: str) -> bool:
            return any((e.get('type')==t) for e in events)

        # SCREENING DUE: 24-28w no OGTT ordered/performed
        if ga_days >= ga_to_days((24,0)) and ga_days <= ga_to_days((28,6)) and (not has_event('ogtt_75g_done')):
            claim_ids=['c0094']  # universal screening 24-28 weeks 75g OGTT (seed)
            actions.append(ActionItem(
                action_id='A_SCREEN_24_28',
                title='Order / perform universal GDM screening (75g OGTT)',
                due='Now (24–28 weeks window)',
                owner='OB / Clinic',
                why_claim_ids=claim_ids,
                evidence_refs=self._claim_ev(claim_ids),
                severity='high'
            ))

        # HIGH-RISK EARLY SCREEN: high-risk flag and GA < 15w and not screened
        if patient.get('high_risk') and ga_days < ga_to_days((15,0)) and (not has_event('early_screen_done')):
            claim_ids=['c0162']  # early screening in high-risk earlier than 15 weeks (seed)
            actions.append(ActionItem(
                action_id='A_EARLY_SCREEN',
                title='Early screening for glucose intolerance (high-risk pregnancy)',
                due='Before 15 weeks',
                owner='OB / Endocrine',
                why_claim_ids=claim_ids,
                evidence_refs=self._claim_ev(claim_ids),
                severity='high'
            ))

        # POSTPARTUM TEST: delivered and no postpartum OGTT
        if patient.get('delivered') and (not has_event('postpartum_ogtt_75g_done')):
            claim_ids=['c0171','c0107']  # postpartum OGTT timing (seed)
            actions.append(ActionItem(
                action_id='A_POSTPARTUM_OGTT',
                title='Schedule postpartum 75g 2‑hour OGTT',
                due='4–12 weeks postpartum (or ≥6 weeks per guideline)',
                owner='Primary care / Endocrine',
                why_claim_ids=claim_ids,
                evidence_refs=self._claim_ev(claim_ids),
                severity='high'
            ))

        # GLUCOSE TARGETS (if SMBG present) -> escalate
        # If there are >=3 readings with fasting >5.3 mmol/L or 2h >6.7 mmol/L in last few events, recommend escalation
        smbg = [e for e in events if e.get('type')=='smbg']
        high=0
        for e in smbg[-14:]:
            val=e.get('value')
            unit=e.get('unit','')
            if val is None: 
                continue
            vmm,_=normalize_glucose(float(val), unit)
            when=str(e.get('when') or '').lower()
            if 'fast' in when and vmm > 5.3:
                high += 1
            if ('2h' in when or 'post' in when) and vmm > 6.7:
                high += 1
        if patient.get('gdm_confirmed') and high >= 3 and (not patient.get('on_medication')):
            claim_ids=['c0007']  # insulin thresholds text exists in seed
            actions.append(ActionItem(
                action_id='A_ESCALATE_THERAPY',
                title='Escalate therapy (consider insulin) due to persistent above-target glucose',
                due='Within 1 week',
                owner='Endocrine / OB',
                why_claim_ids=claim_ids,
                evidence_refs=self._claim_ev(claim_ids),
                severity='medium'
            ))

        return actions


def upsert_tasks_from_results(pid: str, results: List[Dict[str, Any]], tasks_csv_path: str = None):
    """Persist supervision tasks derived from patient_state results into a CSV.

    - Creates/updates rows by (pid, sid)
    - Keeps last_seen timestamp
    - Designed for Streamlit demo persistence (file-backed).
    """
    try:
        import pandas as _pd
        import os as _os
        from datetime import datetime as _dt

        if tasks_csv_path is None:
            base = _os.path.join(_os.path.dirname(__file__), '..', 'data', 'tasks')
            _os.makedirs(base, exist_ok=True)
            tasks_csv_path = _os.path.join(base, 'tasks.csv')

        cols = ['pid','sid','title','status','due_date','owner','why_claim_ids','evidence_refs','last_seen']
        if _os.path.exists(tasks_csv_path):
            df = _pd.read_csv(tasks_csv_path)
        else:
            df = _pd.DataFrame(columns=cols)

        now = _dt.utcnow().isoformat(timespec='seconds') + 'Z'

        for r in (results or []):
            sid = str(r.get('sid',''))
            if not sid:
                continue
            title = str(r.get('title',''))
            status = str(r.get('status',''))
            due = str(r.get('due_date','') or r.get('due',''))
            owner = str(r.get('owner',''))
            why = r.get('why_claim_ids') or r.get('why') or []
            if isinstance(why, list):
                why_s = ' '.join([str(x) for x in why])
            else:
                why_s = str(why)
            ev = r.get('evidence') or r.get('evidence_refs') or []
            # compact evidence refs to doc:pid list
            ev_s = []
            for e in ev:
                doc = e.get('doc') if isinstance(e, dict) else ''
                pid2 = e.get('pid') if isinstance(e, dict) else ''
                if doc and pid2:
                    ev_s.append(f"{doc}:{pid2}")
            ev_s = '|'.join(ev_s)

            mask = (df['pid'].astype(str)==str(pid)) & (df['sid'].astype(str)==sid)
            if mask.any():
                df.loc[mask, ['title','status','due_date','owner','why_claim_ids','evidence_refs','last_seen']] = [title,status,due,owner,why_s,ev_s,now]
            else:
                df = _pd.concat([df, _pd.DataFrame([{
                    'pid': str(pid),
                    'sid': sid,
                    'title': title,
                    'status': status,
                    'due_date': due,
                    'owner': owner,
                    'why_claim_ids': why_s,
                    'evidence_refs': ev_s,
                    'last_seen': now
                }])], ignore_index=True)

        # save
        df.to_csv(tasks_csv_path, index=False)
        return df
    except Exception:
        # fail safe for demo
        import pandas as _pd
        return _pd.DataFrame()
