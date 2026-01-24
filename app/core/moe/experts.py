
from dataclasses import dataclass

@dataclass
class ExpertOutput:
    name: str
    score: float
    reasons: list

class Expert:
    name="base"
    activates_on=[]
    def features(self, entities, evidence, path):
        raise NotImplementedError
    def infer(self, weights, entities, evidence, path):
        feats = self.features(entities, evidence, path)
        score=0.0; reasons=[]
        for k,v in feats.items():
            w = weights.get(k,0.0)
            score += w*v
            if v:
                reasons.append(f"{k}={v:.2f} * w={w:.2f}")
        return ExpertOutput(self.name, float(score), reasons)

class PedersenExpert(Expert):
    name="pedersen_macrosomia"
    activates_on=["maternal_hyperglycemia","placental_glucose_transfer","fetal_hyperinsulinemia","fetal_macrosomia"]
    def features(self, entities, evidence, path):
        e=set(entities)
        ev_tags=set(t for r in evidence for t in r.get("tags",[]))
        return {
            "has_maternal_hyperglycemia": 1.0 if "maternal_hyperglycemia" in e else 0.0,
            "has_placenta_transfer": 1.0 if ("placental_glucose_transfer" in e or "placenta" in ev_tags) else 0.0,
            "has_fetal_insulin": 1.0 if ("fetal_hyperinsulinemia" in e or "fetal_insulin" in ev_tags) else 0.0,
            "has_macrosomia": 1.0 if ("fetal_macrosomia" in e or "macrosomia" in ev_tags) else 0.0,
            "path_len_norm": (len(path)/6.0) if path else 0.0
        }

class PlacentaMTORExpert(Expert):
    name="placenta_mtor"
    activates_on=["mTOR_activation","system_A_transport","system_L_transport","placental_glucose_transfer"]
    def features(self, entities, evidence, path):
        e=set(entities)
        ev_ents=set(x for r in evidence for x in r.get("entities",[]))
        return {
            "mentions_mtor": 1.0 if ("mTOR_activation" in e or "mTOR_activation" in ev_ents) else 0.0,
            "mentions_systemA": 1.0 if ("system_A_transport" in e or "system_A_transport" in ev_ents) else 0.0,
            "mentions_systemL": 1.0 if ("system_L_transport" in e or "system_L_transport" in ev_ents) else 0.0,
            "mentions_placenta_transfer": 1.0 if ("placental_glucose_transfer" in e or "placental_glucose_transfer" in ev_ents) else 0.0,
        }

class ObesityInflammationExpert(Expert):
    name="obesity_inflammation"
    activates_on=["maternal_obesity","chronic_low_grade_inflammation","insulin_resistance","endothelial_dysfunction"]
    def features(self, entities, evidence, path):
        e=set(entities)
        ev_tags=set(t for r in evidence for t in r.get("tags",[]))
        return {
            "has_obesity": 1.0 if "maternal_obesity" in e else 0.0,
            "has_inflammation": 1.0 if ("chronic_low_grade_inflammation" in e or "inflammation" in ev_tags) else 0.0,
            "has_adipokines": 1.0 if ("adipokines" in e or "adipokines" in ev_tags) else 0.0,
            "has_ir": 1.0 if ("insulin_resistance" in e or "insulin_resistance" in ev_tags) else 0.0,
            "has_endothelium": 1.0 if ("endothelial_dysfunction" in e or "vascular_risk" in ev_tags) else 0.0,
        }

class BetaCellProgressionExpert(Expert):
    name="beta_cell_progression"
    activates_on=["pregnancy_insulin_resistance","beta_cell_compensation","beta_cell_failure","type2_diabetes"]
    def features(self, entities, evidence, path):
        e=set(entities)
        ev_tags=set(t for r in evidence for t in r.get("tags",[]))
        return {
            "has_preg_ir": 1.0 if "pregnancy_insulin_resistance" in e else 0.0,
            "has_comp": 1.0 if ("beta_cell_compensation" in e or "beta_cell_compensation" in ev_tags) else 0.0,
            "has_stress": 1.0 if "beta_cell_stress" in e else 0.0,
            "has_failure": 1.0 if ("beta_cell_failure" in e or "beta_cell_failure" in ev_tags) else 0.0,
            "has_t2d": 1.0 if ("type2_diabetes" in e or "t2d_risk" in ev_tags) else 0.0,
        }

def build_experts():
    return [PedersenExpert(), PlacentaMTORExpert(), ObesityInflammationExpert(), BetaCellProgressionExpert()]
