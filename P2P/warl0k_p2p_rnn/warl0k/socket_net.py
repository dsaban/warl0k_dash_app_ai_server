import json, struct, socket

_HDR = struct.Struct("!I")

def send_msg(sock, obj: dict):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(_HDR.pack(len(data)))
    sock.sendall(data)

def _recvall(sock, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return buf

def recv_msg(sock) -> dict:
    hdr = _recvall(sock, _HDR.size)
    (ln,) = _HDR.unpack(hdr)
    data = _recvall(sock, ln)
    return json.loads(data.decode("utf-8"))
