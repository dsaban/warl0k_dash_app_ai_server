
class EnergyModel:
    def __init__(self, required_chain, weights=None):
        self.required_chain = required_chain or []
        self.w = weights or {
            "missing_required_node": 2.0,
            "order_break": 1.5,
            "drift_tag": 0.6,
            "too_short": 0.8
        }

    def energy(self, path, evidence, question_tags):
        pen = {k:0.0 for k in self.w}
        pset=set(path or [])
        for n in self.required_chain:
            if n not in pset:
                pen["missing_required_node"] += 1.0
        idx = {n:(path.index(n) if (path and n in path) else None) for n in self.required_chain}
        last=-1
        for n in self.required_chain:
            i=idx[n]
            if i is None: 
                continue
            if i < last:
                pen["order_break"] += 1.0
            last=i
        qset=set(question_tags or [])
        drift=0.0
        for ev in evidence[:5]:
            et=set(ev.get("tags",[]))
            if et and qset and len(et.intersection(qset))==0:
                drift += 1.0
        pen["drift_tag"]=drift
        if path and len(path) < max(3, len(self.required_chain)-1):
            pen["too_short"]=1.0
        E=0.0
        br={}
        for k,v in pen.items():
            br[k]=self.w[k]*v
            E += br[k]
        return float(E), {k:float(v) for k,v in br.items()}
