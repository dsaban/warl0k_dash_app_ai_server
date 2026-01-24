
from core.ontology import Ontology
from core.lexicon import Lexicon
from core.graph import Graph
from core.retrieval import Retriever
from core.moe.router import Router
from core.ebm.energy import EnergyModel

DEFAULT_WEIGHTS = {
    "has_maternal_hyperglycemia": 2.0,
    "has_placenta_transfer": 2.0,
    "has_fetal_insulin": 2.2,
    "has_macrosomia": 1.6,
    "path_len_norm": 0.5,
    "mentions_mtor": 1.2,
    "mentions_systemA": 0.9,
    "mentions_systemL": 0.9,
    "mentions_placenta_transfer": 0.7,
    "has_obesity": 1.3,
    "has_inflammation": 1.5,
    "has_adipokines": 0.9,
    "has_ir": 1.2,
    "has_endothelium": 1.1,
    "has_preg_ir": 1.0,
    "has_comp": 1.2,
    "has_stress": 0.8,
    "has_failure": 1.1,
    "has_t2d": 1.2,
}

def derive_question_tags(evidence, entities):
    tags=set()
    for ev in evidence[:4]:
        for t in ev.get("tags",[]):
            tags.add(t)
    if "maternal_obesity" in entities:
        tags.add("inflammation")
    if "maternal_hyperglycemia" in entities:
        tags.add("glucose_transfer")
    return sorted(tags)

class GDMPipeline:
    def __init__(self, root: str):
        self.root=root
        self.onto=Ontology(root)
        self.lex=Lexicon(root)
        self.graph=Graph(root)
        self.retr=Retriever(root)
        self.router=Router()
        self.weights=dict(DEFAULT_WEIGHTS)

    def infer_qtype(self, question, entities):
        q=question.lower()
        if "pedersen" in q or "macrosomia" in q or "large for gestational age" in q:
            return "pedersen_macrosomia"
        if "obesity" in q and ("inflammation" in q or "cardiometabolic" in q or "risk" in q):
            return "obesity_inflammation_postpartum_risk"
        if "transition" in q or "overt diabetes" in q or "clinical model" in q:
            return "insulin_resistance_to_diabetes_transition"
        # fallback by overlap
        best=None; best_cov=-1
        for qt, chain in self.onto.required_causal_chains.items():
            cov=len(set(chain).intersection(set(entities)))
            if cov>best_cov:
                best_cov=cov; best=qt
        return best or "pedersen_macrosomia"

    def propose_path(self, qtype):
        return list(self.onto.required_causal_chains.get(qtype, []))

    def compose_answer(self, qtype, evidence):
        ev = evidence[:3]
        ev_lines = "\n".join([f"- {r['text']} ({r.get('source','source')})" for r in ev])
        if qtype=="pedersen_macrosomia":
            return ("Maternal hyperglycemia increases fetal glucose exposure because glucose crosses the placenta. "
                    "The fetal pancreas responds with hyperinsulinemia; fetal insulin is anabolic, promoting adipogenesis and growth, "
                    "leading to fetal macrosomia.\n\nEvidence used:\n"+ev_lines)
        if qtype=="obesity_inflammation_postpartum_risk":
            return ("Obesity drives chronic low-grade inflammation (including altered adipokines) that worsens insulin resistance. "
                    "This contributes to endothelial dysfunction and a persistent adverse vascular/metabolic profile, "
                    "increasing long-term postpartum cardiometabolic risk after a GDM pregnancy.\n\nEvidence used:\n"+ev_lines)
        if qtype=="insulin_resistance_to_diabetes_transition":
            return ("Pregnancy increases insulin resistance. GDM illustrates failure of adequate beta-cell compensation: "
                    "compensatory hypersecretion can progress to beta-cell stress and dysfunction, increasing the risk of overt type 2 diabetes.\n\nEvidence used:\n"+ev_lines)
        return "(No template for this QType.)\n\nEvidence used:\n"+ev_lines

    def run(self, question: str):
        ents = self.lex.extract_entities(question)
        evidence = self.retr.search(question, ents, topk=6)
        qtype = self.infer_qtype(question, ents)
        self.onto.assert_qtype(qtype)
        path = self.propose_path(qtype)
        qtags = derive_question_tags(evidence, ents)

        # MoE scoring
        routed = self.router.route(ents)
        expert_rows=[]
        moe_score=0.0
        for ex,p in routed:
            out = ex.infer(self.weights, ents, evidence, path)
            moe_score += p*out.score
            expert_rows.append({"name":out.name,"route_weight":p,"raw_score":out.score,"reasons":out.reasons})

        # EBM
        ebm = EnergyModel(self.onto.required_causal_chains.get(qtype, []))
        energy, breakdown = ebm.energy(path, evidence, qtags)
        final = moe_score - energy

        # connections
        conns=[]
        for a,b,rel,meta in self.graph.edges_on_path(path):
            conns.append({"from":a,"to":b,"relation":rel,"tags":meta.get("tags",[])})

        return {
            "qtype": qtype,
            "entities": ents,
            "question_tags": qtags,
            "proposed_path": path,
            "connections": conns,
            "evidence": evidence,
            "experts": expert_rows,
            "moe_score": float(moe_score),
            "energy": float(energy),
            "energy_breakdown": breakdown,
            "final_score": float(final),
            "answer": self.compose_answer(qtype, evidence)
        }
