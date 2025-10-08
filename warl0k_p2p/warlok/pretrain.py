import binascii
from .models.sess2master_drnn import Sess2MasterDRNN
from .crypto import hmac_sha256

def hexs(b): return binascii.hexlify(b).decode()

def obf_ticket(W_peer, n, obf_len):
 return hexs(hmac_sha256(W_peer, b"obf", n.to_bytes(8,"big")))[:obf_len]

def master_seedpath(W_peer, peer_id, out_len):
 return hexs(hmac_sha256(W_peer, b"M", peer_id.encode()))[:out_len]

class Pretrainer:
 def __init__(self, adapters, obf_len, master_len, hidden, lr):
  self.adapters=adapters; self.obf_len=obf_len; self.master_len=master_len; self.hidden=hidden; self.lr=lr
 def schedule_window(self, owner_id, peer_id, W_peer, start_n, window, meta=(8,40)):
  target=master_seedpath(W_peer, peer_id, self.master_len)
  for n in range(start_n, start_n+window):
   if self.adapters.exists(peer_id, n): continue
   obf=obf_ticket(W_peer, n, self.obf_len)
   d=Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr)
   d.set_context(peer_id=peer_id, W_bytes=W_peer, target_len_chars=self.master_len)
   d.meta_pretrain(m_samples=meta[0], steps=meta[1], obf_len=self.obf_len)
   d.train_pair(obf, target, epochs=20, check_every=3, patience=2)
   self.adapters.save(peer_id, n, d)
