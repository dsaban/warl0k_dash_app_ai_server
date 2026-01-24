
from .experts import build_experts
from core.utils import softmax

class Router:
    def __init__(self):
        self.experts = build_experts()

    def route(self, entities):
        e=set(entities)
        raw=[]
        for ex in self.experts:
            raw.append(float(len(e.intersection(set(ex.activates_on)))))
        probs=softmax(raw)
        out=[]
        for ex,p in zip(self.experts, probs):
            out.append((ex, float(p)))
        out.sort(key=lambda t: t[1], reverse=True)
        return out
