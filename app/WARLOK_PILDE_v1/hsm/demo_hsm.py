import hashlib, hmac

class DemoHSM:
    def __init__(self, master):
        self.master = master

    def derive_key(self, label, context):
        return hmac.new(self.master, label+context, hashlib.sha256).digest()

    def sign(self, data):
        return hmac.new(self.master, data, hashlib.sha256).digest()