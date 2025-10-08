import binascii
from ..crypto import hmac_sha256

class Seed2MasterModel:
    """
    Peer holds only W (weight vector) derived at hub:
      W = HMAC(seed0_target, "W" || target_device_id)
    Master = HMAC(W, "M" || target_device_id)  [hex-truncated to master_len]
    """
    def __init__(self, target_device_id: str, W_bytes: bytes, out_len_chars: int):
        self.target_device_id = target_device_id
        self.W = W_bytes
        self.out_len = out_len_chars

    def compute_master(self) -> str:
        material = hmac_sha256(self.W, b"M", self.target_device_id.encode())
        return binascii.hexlify(material).decode()[:self.out_len]
