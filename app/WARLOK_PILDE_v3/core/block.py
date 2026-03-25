
import time, hashlib, json

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
        parent_bytes = b''.join(sorted(self.parents))
        meta_bytes = json.dumps(self.metadata, sort_keys=True).encode()
        combined = parent_bytes + self.payload + meta_bytes + str(self.counter).encode()
        self.hash = hashlib.sha256(combined).hexdigest()
        return self.hash
