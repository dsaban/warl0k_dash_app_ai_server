
from .utils import load_json, normalize

class Lexicon:
    def __init__(self, root: str):
        self.entities = load_json(f"{root}/data/lexicon/entities.json")
        self._syn2eid = {}
        for eid, meta in self.entities.items():
            for s in meta.get("syn", []):
                self._syn2eid[normalize(s)] = eid

    def extract_entities(self, text: str):
        qn = normalize(text)
        hits = set()
        syns = sorted(self._syn2eid.keys(), key=len, reverse=True)
        for s in syns:
            if s and s in qn:
                hits.add(self._syn2eid[s])
        return sorted(hits)
