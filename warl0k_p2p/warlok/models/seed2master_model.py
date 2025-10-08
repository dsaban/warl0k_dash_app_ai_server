import binascii
from ..crypto import hmac_sha256
class Seed2MasterModel:
 def __init__(self, target_device_id, W_bytes, out_len_chars):
  self.target_device_id=target_device_id; self.W=W_bytes; self.out_len=out_len_chars
 def compute_master(self):
  return binascii.hexlify(hmac_sha256(self.W, b"M", self.target_device_id.encode())).decode()[:self.out_len]
