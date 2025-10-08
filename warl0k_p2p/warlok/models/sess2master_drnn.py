import numpy as np, hashlib, binascii
from ..crypto import hmac_sha256
VOCAB=list("0123456789abcdef")

def _onehot(c,vsz):
 v=np.zeros(vsz); v["0123456789abcdef".index(c)]=1.0; return v

class Sess2MasterDRNN:
 def __init__(self, hidden_dim=48, lr=0.05, vocab=VOCAB):
  self.vocab=vocab; self.vsz=len(vocab); self.h=hidden_dim; self.lr=lr
  self.Wxh=self.Whh=self.Why=None; self.bh=self.by=None; self.peer_id=None; self.W=None; self._target_len=None; self._meta_ready=False
 def set_context(self, peer_id, W_bytes, target_len_chars):
  self.peer_id=peer_id; self.W=W_bytes; self._target_len=target_len_chars
  seed=hmac_sha256(W_bytes, b"drnn_init", peer_id.encode())
  rng=np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8],"big"))
  self.Wxh=rng.normal(0,0.02,(self.h,self.vsz)); self.Whh=rng.normal(0,0.02,(self.h,self.h))
  self.Why=rng.normal(0,0.02,(self.vsz,self.h)); self.bh=np.zeros(self.h); self.by=np.zeros(self.vsz)
  self._meta_ready=False
 def meta_pretrain(self, m_samples=8, steps=40, obf_len=16):
  import binascii
  target=binascii.hexlify(hmac_sha256(self.W,b"M",self.peer_id.encode())).decode()[:self._target_len]
  obfs=[binascii.hexlify(hmac_sha256(self.W,b"synthetic_obf", i.to_bytes(4,"big"))).decode()[:obf_len] for i in range(1,m_samples+1)]
  it=0
  for _ in range(steps):
   obf=obfs[it % len(obfs)]; it+=1; self._step_teacher_forcing(obf,target)
  self._meta_ready=True
 def train_pair(self, obf, target, epochs=20, check_every=3, patience=2):
  consec=0; ran=0; early=False
  for ep in range(1,epochs+1):
   self._step_teacher_forcing(obf,target); ran=ep
   if ep%check_every==0 or ep==epochs:
    if self.predict(obf,len(target))==target:
     consec+=1
     if consec>=patience: early=True; break
    else: consec=0
  return {"status":"ok","epochs_run":ran,"early_stopped":early,"meta_used":self._meta_ready}
 def predict(self, obf, out_len):
  h=self._forward_condition(obf); out=""; x=_onehot('0',self.vsz)
  for _ in range(out_len):
   y=self.Why@h + self.by; p=self._softmax(y); idx=int(np.argmax(p)); out+=self.vocab[idx]
   x=_onehot(self.vocab[idx],self.vsz); h=np.tanh(self.Wxh@x + self.Whh@h + self.bh)
  return out
 def _softmax(self,x):
  e=np.exp(x-np.max(x)); return e/(np.sum(e)+1e-9)
 def _onehot_seq(self,s): return [_onehot(c,self.vsz) for c in s]
 def _h0_from_obf(self, obf):
  d=hashlib.sha256(obf.encode()).digest(); import numpy as np
  arr=np.frombuffer((d*((self.h//32)+1))[:self.h],dtype=np.uint8)
  x=(arr.astype(np.float32)/255.0)*2.0 - 1.0; return np.tanh(x)
 def _forward_condition(self, obf):
  h=self._h0_from_obf(obf)
  for x in self._onehot_seq(obf): h=np.tanh(self.Wxh@x + self.Whh@h + self.bh)
  return h
 def _step_teacher_forcing(self, obf, target):
  xs_t=self._onehot_seq(target); hs=[self._forward_condition(obf)]; ps=[]
  for t in range(len(target)):
   y=self.Why@hs[-1] + self.by; p=self._softmax(y); ps.append(p)
   h=np.tanh(self.Wxh@xs_t[t] + self.Whh@hs[-1] + self.bh); hs.append(h)
  t_idx=[self.vocab.index(c) for c in target]
  dWxh=np.zeros_like(self.Wxh); dWhh=np.zeros_like(self.Whh); dWhy=np.zeros_like(self.Why); dbh=np.zeros_like(self.bh); dby=np.zeros_like(self.by)
  dh_next=np.zeros(self.h)
  for t in reversed(range(len(target))):
   dy=ps[t].copy(); dy[t_idx[t]]-=1.0; dby+=dy; dWhy+=np.outer(dy,hs[t+1])
   dh=self.Why.T@dy + dh_next; dt=(1-hs[t+1]**2)*dh; dbh+=dt; dWxh+=np.outer(dt,xs_t[t]); dWhh+=np.outer(dt,hs[t]); dh_next=self.Whh.T@dt
  for d in (dWxh,dWhh,dWhy,dbh,dby):
   np.clip(d,-5,5,out=d)
  self.Wxh-=self.lr*dWxh; self.Whh-=self.lr*dWhh; self.Why-=self.lr*dWhy; self.bh-=self.lr*dbh; self.by-=self.lr*dby
