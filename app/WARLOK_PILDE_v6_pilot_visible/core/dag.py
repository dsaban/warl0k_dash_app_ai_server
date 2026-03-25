
class DAG:
    def __init__(self): self.nodes={}
    def add(self,b):
        for p in b.parents:
            if p not in self.nodes: raise Exception(f"Missing parent {p}")
        self.nodes[b.hash]=b
    def get(self,h): return self.nodes.get(h)
