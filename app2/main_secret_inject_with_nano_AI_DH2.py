#!/usr/bin/env python3
"""
WARL0K demo (pure Python stdlib; no Rust-backed crypto libs):

- TLS transport via ssl (self-signed localhost cert if available)
- Ephemeral DH Group-14 (RFC 3526) for session bootstrap (no seed shipping)
- Wrapper injects ms-TTL proof + MAC-only inner "envelope" (demonstration)
- Server validates proof, opens envelope, enforces policy
- Nano-AI pepper (device-unique) bound into per-message key (HMAC-based simulation)

Run:
  python3 warlok_demo.py
"""

import base64, hashlib, hmac, json, os, secrets, socket, ssl, threading, time, subprocess

CERT_PATH = "cert.pem"
KEY_PATH  = "key.pem"

# -------------------------------
# Utilities (HKDF, b64)
# -------------------------------

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    okm, t, counter = b"", b"", 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([counter]), hashlib.sha256).digest()
        okm += t
        counter += 1
    return okm[:length]

def hkdf(ikm: bytes, salt: bytes, info: bytes, length: int=32) -> bytes:
    return hkdf_expand(hkdf_extract(salt, ikm), info, length)

def b64(x: bytes) -> str:
    return base64.urlsafe_b64encode(x).decode().rstrip("=")

def b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)

# -------------------------------
# TLS helpers (self-signed for localhost)
# -------------------------------

def ensure_self_signed_cert():
    """Try to generate a self-signed cert with openssl (optional)."""
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    print("[SETUP] Generating self-signed TLS cert via openssl (optional) ...")
    try:
        subprocess.check_call([
            "openssl","req","-x509","-nodes","-newkey","rsa:2048",
            "-keyout", KEY_PATH, "-out", CERT_PATH,
            "-subj","/CN=localhost","-days","1"
        ])
    except Exception as e:
            print("[WARN] Could not generate TLS cert:", e)
            print("[WARN] Provide cert.pem/key.pem to enable TLS. The demo will still attempt TLS;")
            print("       if it fails on your environment, you can adapt to plain sockets easily.")

def tls_server_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=CERT_PATH, keyfile=KEY_PATH)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

def tls_client_context():
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx

# -------------------------------
# Pure-Python DH Group-14 (RFC 3526)
# -------------------------------

_GROUP14_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A63A36210000000000090563", 16)
_GROUP14_G = 2

def dh14_gen_keypair():
    priv = secrets.randbelow(_GROUP14_P - 2) + 2
    pub  = pow(_GROUP14_G, priv, _GROUP14_P)
    return priv, pub

def dh14_shared_secret(priv, peer_pub) -> bytes:
    if not (1 < peer_pub < _GROUP14_P - 1):
        raise ValueError("Invalid DH peer public")
    s = pow(peer_pub, priv, _GROUP14_P)
    byte_len = (_GROUP14_P.bit_length() + 7) // 8
    return s.to_bytes(byte_len, "big")

# -------------------------------
# HUB (session orchestration; no seed shipping)
# -------------------------------

class Hub:
    def __init__(self):
        self.sessions = {}

    def new_session(self, device_id: str, route: str, allow_methods=("WRITE","READ")):
        session_id = b64(os.urandom(12))
        policy = {"allow_methods": list(allow_methods), "route": route, "device_id": device_id}
        policy_bytes = json.dumps(policy, sort_keys=True).encode()
        policy_hash = hashlib.sha256(policy_bytes).digest()
        self.sessions[session_id] = {
            "policy": policy, "policy_hash": policy_hash, "created_at": time.time()
        }
        return session_id, policy, policy_hash

# -------------------------------
# "Nano-AI" pepper (device-local, enclaved simulation)
# -------------------------------

class NanoAI:
    """
    Simulate a device-unique, enclaved nano-model vector via HMAC over context.
    In production, this would be a deterministic RNN forward pass over ctx.
    """
    def __init__(self, model_key: bytes):
        self.model_key = model_key  # never leaves device/enclave in real life

    def pepper(self, ctx: bytes, out_len: int = 32) -> bytes:
        prk = hmac.new(self.model_key, ctx, hashlib.sha256).digest()
        return hkdf_expand(prk, b"nano-ai-pepper", out_len)

# -------------------------------
# Inner "envelope" (MAC-only demo)
# -------------------------------

def mac_only_wrap(final_key: bytes, plaintext: bytes, aad: bytes) -> dict:
    tag = hmac.new(final_key, aad + b"|" + plaintext, hashlib.sha256).digest()
    return {"alg":"mac-only","nonce":"", "ct": b64(plaintext), "tag": b64(tag)}

def mac_only_open(final_key: bytes, env: dict, aad: bytes) -> bytes:
    pt  = b64d(env["ct"])
    tag = b64d(env["tag"])
    if hmac.compare_digest(tag, hmac.new(final_key, aad + b"|" + pt, hashlib.sha256).digest()):
        return pt
    raise ValueError("MAC verify failed")

def wrap_selected_fields(obj: dict, final_key: bytes, aad: bytes, fields=("/cmd","/data")) -> dict:
    wrapped = dict(obj)
    for f in fields:
        name = f.strip("/")
        if name in wrapped and wrapped[name] is not None:
            pt = json.dumps(wrapped[name]).encode() if not isinstance(wrapped[name], (str,bytes)) else (
                wrapped[name].encode() if isinstance(wrapped[name], str) else wrapped[name]
            )
            wrapped[name] = {"_warlok_env": mac_only_wrap(final_key, pt, aad)}
    return wrapped

def unwrap_selected_fields(obj: dict, final_key: bytes, aad: bytes, fields=("/cmd","/data")) -> dict:
    unwrapped = dict(obj)
    for f in fields:
        name = f.strip("/")
        if name in unwrapped and isinstance(unwrapped[name], dict) and "_warlok_env" in unwrapped[name]:
            pt = mac_only_open(final_key, unwrapped[name]["_warlok_env"], aad)
            try:
                unwrapped[name] = json.loads(pt.decode())
            except Exception:
                unwrapped[name] = pt.decode(errors="ignore")
    return unwrapped

# -------------------------------
# SERVER (validator)
# -------------------------------

class Server(threading.Thread):
    def __init__(self, hub: Hub, device_id="plc-01", host="127.0.0.1", port=4443):
        super().__init__(daemon=True)
        self.hub = hub
        self.host = host
        self.port = port
        self.device_id = device_id
        ensure_self_signed_cert()
        self.ctx = tls_server_context()
        self.nano_ai = NanoAI(model_key=os.urandom(32))
        self.sessions = {}  # session_id -> session_seed (derived via DH)

        # DH priv/pub for init
        self._dh_priv, self._dh_pub = dh14_gen_keypair()

    def run(self):
        with socket.create_server((self.host, self.port)) as sock:
            with self.ctx.wrap_socket(sock, server_side=True) as ssock:
                print(f"[SERVER] TLS listening on https://{self.host}:{self.port}")
                self._serve(ssock)

    def _serve(self, ssock):
        while True:
            conn, addr = ssock.accept()
            with conn:
                data = conn.recv(16384)
                if not data: continue
                response = self.handle_request(data)
                conn.sendall(response)

    def handle_request(self, raw: bytes) -> bytes:
        try:
            text = raw.decode(errors="ignore")
            head, _, body = text.partition("\r\n\r\n")
            lines = head.split("\r\n")
            req_line = lines[0]
            headers = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k,v = ln.split(":",1)
                    headers[k.strip().lower()] = v.strip()

            method, path, _ = req_line.split(" ", 2)

            # --- INIT handshake (derive session seed via DH) ---
            if headers.get("x-warlok-init","").lower() == "v1":
                proof = json.loads(headers.get("x-warlok-init-meta","{}"))
                session_id = proof.get("session_id","")
                session    = self.hub.sessions.get(session_id)
                if not session:
                    return self._http(401, "Unknown session for init")

                client_pub_b64 = headers.get("x-warlok-clientpub","")
                if not client_pub_b64:
                    return self._http(400, "Missing client pub")

                client_pub_int = int.from_bytes(b64d(client_pub_b64), "big")
                z = dh14_shared_secret(self._dh_priv, client_pub_int)
                server_pub = self._dh_pub.to_bytes((_GROUP14_P.bit_length()+7)//8, "big")

                info = ("WARL0K/session" + session_id + session["policy"]["device_id"]).encode()
                s0 = hkdf(ikm=z, salt=session["policy_hash"], info=info, length=32)
                self.sessions[session_id] = s0

                return self._http(200, "INIT-OK", extra_headers={"X-WARLOK-ServerPub": b64(server_pub)})

            # --- Regular request path ---
            proof_hdr = headers.get("x-warlok-proof","")
            meta_hdr  = headers.get("x-warlok-meta","")
            if not proof_hdr or not meta_hdr:
                return self._http(400, "Missing WARL0K headers")

            proof = json.loads(proof_hdr)
            meta  = json.loads(meta_hdr)

            session_id = proof["session_id"]
            session = self.hub.sessions.get(session_id)
            if not session:
                return self._http(401, "Unknown session")

            if session_id not in self.sessions:
                return self._http(428, "Precondition Required: run DH init first")

            # Policy check
            policy = session["policy"]
            if method.upper() not in policy["allow_methods"] or policy["route"] != meta["route"]:
                return self._http(403, "Policy denied")

            # Per-message key from DH-derived session seed
            seed = self.sessions[session_id]
            ctx_core = (session_id + meta["route"] + meta["device_id"]).encode()
            info = ctx_core + b64d(proof["nonce"]) + proof["counter"].to_bytes(8, "big")
            k_i = hkdf(ikm=seed, salt=session["policy_hash"], info=info, length=32)

            # Nano-AI pepper (device-unique) from context
            ctx_ai = b"|".join([ctx_core, proof["counter"].to_bytes(8,"big"), b64d(proof["nonce"]), session["policy_hash"]])
            nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)

            # Bind into final key
            final_key = hkdf(ikm=k_i + nano_pepper, salt=session["policy_hash"], info=b"bind", length=32)

            # Verify outer proof signature
            payload = "|".join([method, path, body, b64(session["policy_hash"]), str(proof["ttl_ms"]), str(proof["counter"])]).encode()
            expected_sig = hmac.new(final_key, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(expected_sig, b64d(proof["sig"])):
                return self._http(401, "Invalid proof signature")

            # TTL
            if (time.time()*1000) > proof["issued_ms"] + proof["ttl_ms"]:
                return self._http(401, "Expired proof")

            # Open inner MAC-only "envelope" on selected fields
            obj = json.loads(body) if body.strip() else {}
            aad = ("|".join([meta["route"], meta["device_id"], str(proof["counter"])]).encode())
            obj2 = unwrap_selected_fields(obj, final_key, aad, fields=("/cmd","/data"))

            return self._http(200, f"OK (method {method}). Decrypted body: {json.dumps(obj2)}")

        except Exception as e:
            return self._http(500, f"Server error: {e}")

    def _http(self, code: int, msg: str, extra_headers: dict=None) -> bytes:
        reason = {200:"OK",400:"Bad Request",401:"Unauthorized",403:"Forbidden",
                  428:"Precondition Required",500:"Internal Server Error"}[code]
        hdrs = [f"HTTP/1.1 {code} {reason}",
                "Content-Type: text/plain"]
        if extra_headers:
            for k,v in extra_headers.items():
                hdrs.append(f"{k}: {v}")
        body = msg + "\n"
        hdrs.append(f"Content-Length: {len(body)}")
        return ("\r\n".join(hdrs) + "\r\n\r\n" + body).encode()

# -------------------------------
# CLIENT (wrapper)
# -------------------------------

class Client(threading.Thread):
    def __init__(self, session_id: str, policy_hash: bytes,
                 host="127.0.0.1", port=4443, route="/api/write", device_id="plc-01",
                 device_model_key: bytes=None):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.policy_hash = policy_hash
        self.host = host; self.port = port
        self.route = route; self.device_id = device_id
        ensure_self_signed_cert()
        self.ctx = tls_client_context()
        self.counter = 0
        self.nano_ai = NanoAI(model_key=device_model_key or os.urandom(32))
        self.session_seed = None

        # DH priv/pub for init
        self._dh_priv, self._dh_pub = dh14_gen_keypair()

    def run(self):
        time.sleep(0.3)
        self.init_handshake()  # derive session seed via DH (no seed shipping)

        # 1) Valid request: WRITE with inner MAC-only envelope
        body = {"cmd": {"op":"set_speed"}, "data": {"value": 42}, "note": "non-sensitive"}
        self.send_request("WRITE", self.route, body)

        # 2) Blocked by policy: DELETE
        body2 = {"cmd": {"op":"wipe"}, "data": {"confirm": True}}
        self.send_request("DELETE", self.route, body2)

    def _connect(self):
        s = socket.create_connection((self.host, self.port))
        return self.ctx.wrap_socket(s, server_hostname="localhost")

    def init_handshake(self):
        """POST /init with client pub; expect server pub; derive session seed."""
        sock = self._connect()
        with sock:
            meta = {"session_id": self.session_id}
            client_pub_b64 = b64(self._dh_pub.to_bytes((_GROUP14_P.bit_length()+7)//8, "big"))
            req = f"""POST /init HTTP/1.1\r
Host: {self.host}\r
Content-Length: 0\r
X-WARLOK-Init: v1\r
X-WARLOK-Init-Meta: {json.dumps(meta)}\r
X-WARLOK-ClientPub: {client_pub_b64}\r
\r
"""
            sock.sendall(req.encode())
            resp = sock.recv(8192).decode()
            # crude parse
            lines = resp.split("\r\n")
            hdrs = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k,v = ln.split(":",1); hdrs[k.strip().lower()] = v.strip()
            server_pub_b64 = hdrs.get("x-warlok-serverpub","")
            if not server_pub_b64:
                print("[CLIENT] INIT failed:", resp)
                return
            server_pub = int.from_bytes(b64d(server_pub_b64), "big")
            z = dh14_shared_secret(self._dh_priv, server_pub)
            info = ("WARL0K/session" + self.session_id + self.device_id).encode()
            self.session_seed = hkdf(ikm=z, salt=self.policy_hash, info=info, length=32)
            print("[CLIENT] Derived session_seed via DH Group-14")

    def send_request(self, method: str, path: str, body_obj: dict):
        if self.session_seed is None:
            print("[CLIENT] No session seed; did INIT fail?")
            return
        sock = self._connect()
        with sock:
            proof_hdr, meta_hdr, body_txt = self.build_warlok_payload(method, path, body_obj)
            req = f"""{method} {path} HTTP/1.1\r
Host: {self.host}\r
Content-Type: application/json\r
Content-Length: {len(body_txt)}\r
X-WARLOK-Proof: {proof_hdr}\r
X-WARLOK-Meta: {meta_hdr}\r
\r
{body_txt}"""
            sock.sendall(req.encode())
            resp = sock.recv(16384).decode()
            print("[CLIENT] Response:\n", resp)

    def build_warlok_payload(self, method: str, path: str, body_obj: dict):
        self.counter += 1
        nonce = os.urandom(16)
        ttl_ms = 300
        issued_ms = int(time.time()*1000)

        # Per-message key from DH-derived session seed
        ctx_core = (self.session_id + self.route + self.device_id).encode()
        info = ctx_core + nonce + self.counter.to_bytes(8,"big")
        k_i = hkdf(ikm=self.session_seed, salt=self.policy_hash, info=info, length=32)

        # Device-unique nano-AI pepper
        ctx_ai = b"|".join([ctx_core, self.counter.to_bytes(8,"big"), nonce, self.policy_hash])
        nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)

        # Bind into final key
        final_key = hkdf(ikm=k_i + nano_pepper, salt=self.policy_hash, info=b"bind", length=32)

        # Inner MAC-only "envelope" on selected fields
        aad = ("|".join([self.route, self.device_id, str(self.counter)])).encode()
        wrapped_obj = wrap_selected_fields(body_obj, final_key, aad, fields=("/cmd","/data"))
        body_txt = json.dumps(wrapped_obj, separators=(",",":"))

        # Outer proof signature over canonical payload
        payload = "|".join([method, path, body_txt, b64(self.policy_hash), str(ttl_ms), str(self.counter)]).encode()
        sig = hmac.new(final_key, payload, hashlib.sha256).digest()

        proof = {
            "session_id": self.session_id,
            "nonce": b64(nonce),
            "counter": self.counter,
            "ttl_ms": ttl_ms,
            "issued_ms": issued_ms,
            "policy_hash": b64(self.policy_hash),
            "sig": b64(sig)
        }
        meta = {"route": self.route, "device_id": self.device_id}
        return json.dumps(proof), json.dumps(meta), body_txt

# -------------------------------
# Demo runner
# -------------------------------

def main():
    # HUB sets session + policy (no seed shipped)
    hub = Hub()
    session_id, policy, policy_hash = hub.new_session(
        device_id="plc-01", route="/api/write", allow_methods=("WRITE","READ"))
    print("[HUB] session_id:", session_id)
    print("[HUB] policy:", policy)

    # Share a device model key so both sides agree for the demo
    device_model_key = os.urandom(32)

    server = Server(hub=hub, device_id=policy["device_id"], port=4443)
    # Align server nano model key with client (demo only)
    server.nano_ai = NanoAI(model_key=device_model_key)

    client = Client(session_id=session_id,
                    policy_hash=policy_hash, port=4443,
                    route=policy["route"], device_id=policy["device_id"],
                    device_model_key=device_model_key)

    server.start()
    client.start()
    client.join(timeout=5)

if __name__ == "__main__":
    ensure_self_signed_cert()
    main()
