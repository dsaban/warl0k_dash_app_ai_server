
import hmac,hashlib
class DemoHSM:
    def __init__(self,k): self.k=k
    def sign(self,d): return hmac.new(self.k,d,hashlib.sha256).digest()
    def verify(self,d,s): return self.sign(d)==s
