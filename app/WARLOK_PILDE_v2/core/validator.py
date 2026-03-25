
class DAGValidator:
    def __init__(self, dag, hsm):
        self.dag = dag
        self.hsm = hsm
    def validate(self, h):
        b = self.dag.get_block(h)
        if not b:
            return False, "Missing"
        if not self.hsm.verify(b.hash.encode(), b.signature):
            return False, "Bad sig"
        for p in b.parents:
            ok, msg = self.validate(p)
            if not ok:
                return False, msg
        return True, "VALID"
