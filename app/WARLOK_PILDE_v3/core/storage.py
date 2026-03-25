
import json, os

class Storage:
    def __init__(self, path='data.json'):
        self.path = path
        if not os.path.exists(path):
            with open(path, 'w') as f:
                json.dump({}, f)

    def save(self, block):
        with open(self.path, 'r') as f:
            data = json.load(f)

        data[block.hash] = {
            "parents": block.parents,
            "payload": block.payload.decode(),
            "metadata": block.metadata,
            "counter": block.counter,
            "signature": block.signature.hex()
        }

        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2)
