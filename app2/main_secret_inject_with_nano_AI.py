#!/usr/bin/env python3
"""
WARL0K authentication demo (TLS) with nano-AI pepper (device-unique) in the key derivation.

Flow:
- Hub creates a session (session_id, policy, session_seed) and TLS runs as usual.
- Client "wrapper" derives a per-message key k_i from session_seed and context,
  ALSO derives a "nano-AI pepper" from device-local model weights + context,
  then binds them into a final_key that signs the proof header.
- Server "validator" repeats both derivations and validates signature, TTL, policy.

Notes:
- TLS cert is self-signed (generated at runtime via openssl if available).
- The nano-AI pepper here is simulated as an HMAC over context with a device-unique
  "model key" that never leaves the device (what an enclave would protect).
"""

import base64, hashlib, hmac, json, os, secrets, socket, ssl, threading, time, subprocess

CERT_PATH = "cert.pem"
KEY_PATH  = "key.pem"

# -------------------------------
# Utilities
# -------------------------------

def hkdf_extract(salt: bytes, ikm: bytes) -> bytes:
    return hmac.new(salt, ikm, hashlib.sha256).digest()

def hkdf_expand(prk: bytes, info: bytes, length: int) -> bytes:
    okm = b""
    t = b""
    counter = 1
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
# Self-signed TLS cert for localhost
# -------------------------------

def ensure_self_signed_cert():
    if os.path.exists(CERT_PATH) and os.path.exists(KEY_PATH):
        return
    print("[SETUP] Generating self-signed TLS cert via openssl ...")
    try:
        subprocess.check_call([
            "openssl","req","-x509","-nodes","-newkey","rsa:2048",
            "-keyout", KEY_PATH, "-out", CERT_PATH,
            "-subj","/CN=localhost","-days","1"
        ])
    except Exception as e:
        print("[WARN] Could not generate TLS cert:", e)

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
# HUB: session orchestration
# -------------------------------

class Hub:
    def __init__(self):
        self.sessions = {}

    def new_session(self, device_id: str, route: str, allow_methods=("WRITE","READ")):
        session_id = b64(os.urandom(12))
        policy = {"allow_methods": list(allow_methods), "route": route, "device_id": device_id}
        policy_bytes = json.dumps(policy, sort_keys=True).encode()
        policy_hash = hashlib.sha256(policy_bytes).digest()
        session_seed = os.urandom(32)  # In production, derive via ECDH/KEM; demo keeps it simple
        self.sessions[session_id] = {
            "seed": session_seed,
            "policy": policy,
            "policy_hash": policy_hash,
            "created_at": time.time()
        }
        return session_id, session_seed, policy, policy_hash

# -------------------------------
# "Nano-AI" pepper (device-local, enclaved)
# -------------------------------
class NanoAI:
    """
    Simulates an enclaved nano-model producing a deterministic pepper
    from device-unique model weights (key) + per-message context.
    In production, this would be an RNN forward pass; here we use HMAC.
    """
    def __init__(self, model_key: bytes):
        self.model_key = model_key  # device-unique, never leaves device/enclave

    def pepper(self, ctx: bytes, out_len: int = 32) -> bytes:
        # Derive a pseudo-vector (expandable) deterministically from ctx
        prk = hmac.new(self.model_key, ctx, hashlib.sha256).digest()
        return hkdf_expand(prk, b"nano-ai-pepper", out_len)

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
        # Device-unique model key (what an enclave would protect)
        self.nano_ai = NanoAI(model_key=os.urandom(32))

    def run(self):
        with socket.create_server((self.host, self.port)) as sock:
            with self.ctx.wrap_socket(sock, server_side=True) as ssock:
                print(f"[SERVER] TLS listening on https://{self.host}:{self.port}")
                self._serve(ssock)

    def _serve(self, ssock):
        while True:
            conn, addr = ssock.accept()
            with conn:
                data = conn.recv(8192)
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

            # Policy check
            policy = session["policy"]
            if method.upper() not in policy["allow_methods"] or policy["route"] != meta["route"]:
                return self._http(403, "Policy denied")

            # Derive cryptographic key k_i (from hub/session seed)
            seed = session["seed"]
            ctx_core = (session_id + meta["route"] + meta["device_id"]).encode()
            info = ctx_core + b64d(proof["nonce"]) + proof["counter"].to_bytes(8, "big")
            k_i = hkdf(ikm=seed, salt=session["policy_hash"], info=info, length=32)

            # Derive device-unique nano-AI pepper (enclave-based)
            ctx_ai = b"|".join([
                ctx_core,
                proof["counter"].to_bytes(8,"big"),
                b64d(proof["nonce"]),
                session["policy_hash"],
            ])
            nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)

            # Bind both into final_key
            final_key = hkdf(ikm=k_i + nano_pepper, salt=session["policy_hash"], info=b"bind", length=32)

            # Verify signature over canonical payload
            payload = "|".join([
                method, path, body, b64(session["policy_hash"]),
                str(proof["ttl_ms"]), str(proof["counter"])
            ]).encode()
            expected_sig = hmac.new(final_key, payload, hashlib.sha256).digest()

            if not hmac.compare_digest(expected_sig, b64d(proof["sig"])):
                return self._http(401, "Invalid proof signature (nano-AI binding failed)")

            # TTL
            if (time.time()*1000) > proof["issued_ms"] + proof["ttl_ms"]:
                return self._http(401, "Expired proof")

            return self._http(200, f"OK (method {method} allowed). Body: {body}")

        except Exception as e:
            return self._http(500, f"Server error: {e}")

    def _http(self, code: int, msg: str) -> bytes:
        reason = {200:"OK",400:"Bad Request",401:"Unauthorized",403:"Forbidden",500:"Internal Server Error"}[code]
        body = msg + "\n"
        return (f"HTTP/1.1 {code} {reason}\r\nContent-Type: text/plain\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode()

# -------------------------------
# CLIENT (wrapper)
# -------------------------------

class Client(threading.Thread):
    def __init__(self, session_id: str, session_seed: bytes, policy_hash: bytes,
                 host="127.0.0.1", port=4443, route="/api/write", device_id="plc-01",
                 device_model_key: bytes=None):
        super().__init__(daemon=True)
        self.session_id = session_id
        self.session_seed = session_seed
        self.policy_hash = policy_hash
        self.host = host
        self.port = port
        self.route = route
        self.device_id = device_id
        ensure_self_signed_cert()
        self.ctx = tls_client_context()
        self.counter = 0
        # Client-side view of the device's enclaved model (same device-unique key)
        # In a real deployment, the server-side validator and the device share that key.
        self.nano_ai = NanoAI(model_key=device_model_key or os.urandom(32))

    def run(self):
        time.sleep(0.3)
        self.send_request("WRITE", self.route, '{"cmd":"set_speed","value":42}')
        time.sleep(0.1)
        self.send_request("DELETE", self.route, '{"cmd":"wipe"}')  # should be blocked by policy

    def send_request(self, method: str, path: str, body: str):
        sock = socket.create_connection((self.host, self.port))
        sock = self.ctx.wrap_socket(sock, server_hostname="localhost")
        with sock:
            proof_hdr, meta_hdr = self.build_warlok_headers(method, path, body)
            req = f"""{method} {path} HTTP/1.1\r
Host: {self.host}\r
Content-Type: application/json\r
Content-Length: {len(body)}\r
X-WARLOK-Proof: {proof_hdr}\r
X-WARLOK-Meta: {meta_hdr}\r
\r
{body}"""
            sock.sendall(req.encode())
            resp = sock.recv(8192).decode()
            print("[CLIENT] Response:\n", resp)

    def build_warlok_headers(self, method: str, path: str, body: str):
        self.counter += 1
        nonce = os.urandom(16)
        ttl_ms = 300
        issued_ms = int(time.time()*1000)

        # Cryptographic key from session seed
        ctx_core = (self.session_id + self.route + self.device_id).encode()
        info = ctx_core + nonce + self.counter.to_bytes(8,"big")
        k_i = hkdf(ikm=self.session_seed, salt=self.policy_hash, info=info, length=32)

        # Nano-AI pepper (device-unique, enclaved) from context
        ctx_ai = b"|".join([ctx_core, self.counter.to_bytes(8,"big"), nonce, self.policy_hash])
        nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)

        # Bind both into final key
        final_key = hkdf(ikm=k_i + nano_pepper, salt=self.policy_hash, info=b"bind", length=32)

        payload = "|".join([
            method, path, body, b64(self.policy_hash), str(ttl_ms), str(self.counter)
        ]).encode()
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
        return json.dumps(proof), json.dumps(meta)

# -------------------------------
# Demo runner
# -------------------------------

def main():
    hub = Hub()
    session_id, session_seed, policy, policy_hash = hub.new_session(
        device_id="plc-01", route="/api/write", allow_methods=("WRITE","READ"))
    print("[HUB] session_id:", session_id)
    print("[HUB] policy:", policy)

    # For the demo, make client+server share the same device model key
    device_model_key = os.urandom(32)

    server = Server(hub=hub, device_id=policy["device_id"], port=4443)
    # overwrite the server's nano model key so both sides agree for the demo
    server.nano_ai = NanoAI(model_key=device_model_key)
    print("[SERVER] Device model key (simulated enclave):", b64(device_model_key))

    client = Client(session_id=session_id, session_seed=session_seed,
                    policy_hash=policy_hash, port=4443,
                    route=policy["route"], device_id=policy["device_id"],
                    device_model_key=device_model_key)

    server.start()
    client.start()
    client.join(timeout=3)

if __name__ == "__main__":
    main()
