
from core.block import Block
from core.dag import DAG
from core.events import log_event
class Hub:
 def __init__(self,hsm):
  self.hsm=hsm
  self.dag=DAG()
 def create_block(self,parents,payload,meta):
  log_event("HUB_CREATE",{"parents":parents,"payload":payload})
  b=Block(parents,payload,meta)
  h=b.compute_hash()
  log_event("HASH",{"hash":h})
  b.signature=self.hsm.sign(h.encode())
  log_event("SIGNED",{"hash":h})
  self.dag.add(b)
  log_event("DAG_ADD",{"hash":h})
  return h
 def validate(self,h):
  b=self.dag.get(h)
  if not b: return False,"Missing"
  if not self.hsm.verify(b.hash.encode(),b.signature):
   return False,"Bad signature"
  for p in b.parents:
   ok,msg=self.validate(p)
   if not ok: return False,msg
  return True,"VALID"
