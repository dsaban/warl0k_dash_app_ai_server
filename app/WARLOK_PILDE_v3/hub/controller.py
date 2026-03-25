
from core.block import Block
from core.dag import DAG
from core.validator import Validator
from core.replay import ReplayProtector
from core.storage import Storage

class Hub:
    def __init__(self, hsm):
        self.hsm = hsm
        self.dag = DAG()
        self.validator = Validator(self.dag, hsm)
        self.replay = ReplayProtector()
        self.storage = Storage()
        self.counter = 0

    def create_block(self, parents, payload, metadata):
        self.counter += 1

        if not self.replay.validate(self.counter):
            raise Exception("Replay detected")

        b = Block(parents, payload.encode(), metadata, self.counter)
        h = b.compute_hash()
        b.signature = self.hsm.sign(h.encode())

        self.dag.add(b)
        self.storage.save(b)

        return h

    def validate(self, h):
        return self.validator.validate(h)
