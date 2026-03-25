
from core.block import Block
from core.dag_engine import DAGEngine
from core.validator import DAGValidator
class Hub:
    def __init__(self, hsm):
        self.hsm = hsm
        self.dag = DAGEngine()
        self.validator = DAGValidator(self.dag, hsm)
    def create(self, parents, payload, meta):
        b = Block(parents, payload, meta)
        h = b.compute_hash()
        b.signature = self.hsm.sign(h.encode())
        self.dag.add_block(b)
        return h
    def validate(self, h):
        return self.validator.validate(h)
