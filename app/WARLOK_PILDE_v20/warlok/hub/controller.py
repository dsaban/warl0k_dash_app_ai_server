"""
Hub v2 — per-block HSM signing + epoch metadata on every block.
"""
from core.block  import Block
from core.dag    import DAG
from core.events import log_event

class Hub:
    def __init__(self, hsm):
        self.hsm = hsm
        self.dag = DAG()

    def create_block(self, parents, payload, meta):
        log_event("HUB_CREATE", {"parents": parents, "payload": payload})
        b = Block(parents, payload, meta)
        h = b.compute_hash()
        log_event("HASH", {"hash": h})
        # v2: sign with per-block derived key; stamp epoch on meta
        b.signature = self.hsm.sign(h.encode(), block_hash=h)
        b.meta["_epoch"] = self.hsm.epoch_id
        log_event("SIGNED", {"hash": h, "epoch": self.hsm.epoch_id})
        self.dag.add(b)
        log_event("DAG_ADD", {"hash": h})
        return b

    def validate(self, h):
        b = self.dag.get(h)
        if not b:
            return False, "Missing"
        # Use epoch-aware verify if epoch is stamped on the block
        epoch = b.meta.get("_epoch")
        if epoch is not None:
            ok = self.hsm.verify_epoch(b.hash.encode(), b.signature, b.hash, epoch)
        else:
            ok = self.hsm.verify(b.hash.encode(), b.signature, block_hash=b.hash)
        if not ok:
            return False, "Bad signature"
        for p in b.parents:
            ok, msg = self.validate(p)
            if not ok:
                return False, msg
        return True, "VALID"
