"""
Per-counter ticketed adapters. DEMO ONLY (pickle, no encryption).
"""
import os, pickle
class TicketedAdapters:
 def __init__(self, dirpath=".adapters"): self.dir=dirpath; os.makedirs(self.dir, exist_ok=True)
 def _peer_dir(self, peer_id): p=os.path.join(self.dir, peer_id); os.makedirs(p, exist_ok=True); return p
 def path(self, peer_id, n): return os.path.join(self._peer_dir(peer_id), f"n_{n:08d}.pkl")
 def exists(self, peer_id, n): return os.path.exists(self.path(peer_id, n))
 def save(self, peer_id, n, drnn):
  with open(self.path(peer_id, n),"wb") as f:
   pickle.dump({"peer_id":drnn.peer_id,"target_len":drnn._target_len,"W":drnn.W,
                "Wxh":drnn.Wxh,"Whh":drnn.Whh,"Why":drnn.Why,"bh":drnn.bh,"by":drnn.by},f)
 def load(self, peer_id, n, ctor):
  with open(self.path(peer_id, n),"rb") as f: s=pickle.load(f)
  d=ctor(); d.peer_id=s["peer_id"]; d._target_len=s["target_len"]; d.W=s["W"]
  d.Wxh=s["Wxh"]; d.Whh=s["Whh"]; d.Why=s["Why"]; d.bh=s["bh"]; d.by=s["by"]; return d
