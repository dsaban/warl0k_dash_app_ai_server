
class DAG:
    def __init__(self):
        self.nodes = {}
    def add(self, block):
        for p in block.parents:
            if p not in self.nodes:
                raise Exception("Missing parent")
        self.nodes[block.hash] = block
    def get(self, h):
        return self.nodes.get(h)
