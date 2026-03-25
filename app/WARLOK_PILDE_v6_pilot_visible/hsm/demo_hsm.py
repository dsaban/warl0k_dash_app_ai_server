
import hmac,hashlib
from core.events import log_event
class DemoHSM:
    def __init__(self,k): self.k=k
    def sign(self,d):
        log_event("HSM_SIGN",{"data":d.hex()})
        return hmac.new(self.k,d,hashlib.sha256).digest()
    def verify(self,d,s):
        log_event("HSM_VERIFY",{"data":d.hex()})
        return self.sign(d)==s
