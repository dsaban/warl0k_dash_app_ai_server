
from core.block import Block
from core.dag import DAG

class Hub:
    def __init__(self,hsm):
        self.hsm=hsm
        self.dag=DAG()

    def create_block(self,parents,payload,meta):
        b=Block(parents,payload,meta)
        h=b.compute_hash()
        b.signature=self.hsm.sign(h.encode())
        self.dag.add(b)
        return h

    def validate(self,h):
        b=self.dag.get(h)
        if not b: return False,"Missing"
        if not self.hsm.verify(b.hash.encode(),b.signature):
            return False,"Bad sig"
        for p in b.parents:
            ok,msg=self.validate(p)
            if not ok: return False,msg
        return True,"VALID"
