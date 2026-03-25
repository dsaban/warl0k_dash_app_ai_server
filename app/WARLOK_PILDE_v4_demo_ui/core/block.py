
import hashlib, json, time
class Block:
    def __init__(self, parents, payload, metadata, counter):
        self.parents = parents
        self.payload = payload
        self.metadata = metadata
        self.counter = counter
        self.timestamp = int(time.time())
        self.signature = None
        self.hash = None
    
    def compute_hash(self):
        parent_bytes = b''.join(sorted([p.encode() for p in self.parents]))
        payload_bytes = self.payload if isinstance(self.payload, bytes) else self.payload.encode()
        meta_bytes = json.dumps(self.metadata, sort_keys=True).encode()
        
        combined = parent_bytes + payload_bytes + meta_bytes
        self.hash = hashlib.sha256(combined).hexdigest()
        return self.hash
    
    # def compute_hash(self):
    #     parent_bytes = b''.join(sorted([p.encode() for p in self.parents]))
    #     meta_bytes = json.dumps(self.metadata, sort_keys=True).encode()
    #
    #     combined = parent_bytes + self.payload + meta_bytes
    #     self.hash = hashlib.sha256(combined).hexdigest()
    #     return self.hash
    # def compute_hash(self):
    #     combined = b''.join(sorted(self.parents)) + self.payload + json.dumps(self.metadata).encode()
    #     self.hash = hashlib.sha256(combined).hexdigest()
    #     return self.hash
