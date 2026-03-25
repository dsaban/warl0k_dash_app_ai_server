
import hashlib,json,time
class Block:
    def __init__(self,parents,payload,meta):
        self.parents=parents
        self.payload=payload.encode() if isinstance(payload,str) else payload
        self.meta=meta
        self.timestamp=time.time()
        self.hash=None
        self.signature=None
        self.status="ACTIVE"
    def compute_hash(self):
        p=b''.join([x.encode() for x in sorted(self.parents)])
        m=json.dumps(self.meta,sort_keys=True).encode()
        self.hash=hashlib.sha256(p+self.payload+m).hexdigest()
        return self.hash
