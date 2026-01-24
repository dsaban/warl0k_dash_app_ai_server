
from .utils import load_jsonl, normalize

class Retriever:
    def __init__(self, root: str):
        self.kb = load_jsonl(f"{root}/data/knowledge_base.jsonl")

    def search(self, question: str, entities, topk: int = 6):
        qn = normalize(question)
        e_set = set(entities)
        scored=[]
        for row in self.kb:
            txt = normalize(row["text"])
            e_ov = len(e_set.intersection(set(row.get("entities",[]))))
            kw_ov = sum(1 for w in set(qn.split()) if w in txt)
            score = 2.5*e_ov + 0.05*kw_ov
            r = dict(row); r["score"]=float(score)
            scored.append(r)
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:topk]
