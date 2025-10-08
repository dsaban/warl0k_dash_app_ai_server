import json, struct
_HDR=struct.Struct('!I')

def send_msg(s,o):
 d=json.dumps(o).encode(); s.sendall(_HDR.pack(len(d))); s.sendall(d)

def _recvall(s,n):
 b=b''
 while len(b)<n:
  c=s.recv(n-len(b))
  if not c: raise ConnectionError('socket closed')
  b+=c
 return b

def recv_msg(s):
 ln=_HDR.unpack(_recvall(s,_HDR.size))[0]
 return json.loads(_recvall(s,ln).decode())
