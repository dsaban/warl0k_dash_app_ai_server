
import hashlib, hmac
class DemoHSM:
    def __init__(self, key):
        self.key = key
    def derive_key(self, label, context):
        return hmac.new(self.key, label+context, hashlib.sha256).digest()
    def sign(self, data):
        return hmac.new(self.key, data, hashlib.sha256).digest()
    def verify(self, data, sig):
        return self.sign(data) == sig
