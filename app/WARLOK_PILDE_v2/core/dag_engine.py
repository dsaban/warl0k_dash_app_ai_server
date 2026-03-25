
class DAGEngine:
    def __init__(self):
        self.nodes = {}
    def add_block(self, block):
        for p in block.parents:
            if p not in self.nodes:
                raise Exception(f"Missing parent: {p}")
        self.nodes[block.hash] = block
    def get_block(self, h):
        return self.nodes.get(h)
