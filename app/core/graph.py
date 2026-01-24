
from .utils import load_json

class Graph:
    def __init__(self, root: str):
        seed = load_json(f"{root}/data/graph_seed.json")
        self.nodes = set(seed["nodes"])
        self.adj = {}
        self.edges = []
        for a,b,rel,meta in seed["edges"]:
            self.add_edge(a,b,rel,meta)

    def add_edge(self,a,b,rel,meta):
        self.nodes.add(a); self.nodes.add(b)
        self.edges.append((a,b,rel,meta))
        self.adj.setdefault(a, []).append((b,rel,meta))

    def neighbors(self,a):
        return self.adj.get(a, [])

    def edges_on_path(self, path):
        out=[]
        for i in range(len(path)-1):
            a,b=path[i],path[i+1]
            for nb,rel,meta in self.neighbors(a):
                if nb==b:
                    out.append((a,b,rel,meta)); break
        return out
