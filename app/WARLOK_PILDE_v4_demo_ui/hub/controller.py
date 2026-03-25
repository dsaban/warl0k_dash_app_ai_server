
from core.block import Block
from core.dag import DAG

class Hub:
    def __init__(self, hsm):
        self.hsm = hsm
        self.dag = DAG()
        self.counter = 0

    def create_block(self, parents, payload, metadata):
        self.counter += 1
        b = Block(parents, payload.encode(), metadata, self.counter)
        h = b.compute_hash()
        b.signature = self.hsm.sign(h.encode())
        self.dag.add(b)
        return h

    def validate(self, h):
        b = self.dag.get(h)
        if not b:
            return False
        if not self.hsm.verify(b.hash.encode(), b.signature):
            return False
        for p in b.parents:
            if not self.validate(p):
                return False
        return True
