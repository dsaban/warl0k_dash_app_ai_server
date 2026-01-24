
from .utils import load_json

class Ontology:
    def __init__(self, root: str):
        self.onto = load_json(f"{root}/config/ontology.json")
        self.fixed_qtypes = set(self.onto["fixed_qtypes"])
        self.tags = set(self.onto["tags"])
        self.relations = {r["type"]: r for r in self.onto["relations"]}
        self.required_causal_chains = self.onto.get("required_causal_chains", {})

    def assert_qtype(self, qtype: str):
        if qtype not in self.fixed_qtypes:
            raise ValueError(f"QType '{qtype}' not allowed (fixed).")
