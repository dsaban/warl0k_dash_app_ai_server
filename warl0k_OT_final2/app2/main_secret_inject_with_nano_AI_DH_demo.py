# app1.py
# Streamlit WARL0K demo (no Rust / no external crypto libs):
# - Pure-Python DH Group-14 session bootstrap (no seed shipping)
# - HKDF/HMAC per-message keys + device-unique nano-AI pepper
# - MAC-only inner "envelope" (demonstration of envelope-in-envelope)
# - Three columns: Client | Hub | Server
# - File logging: warlok_streamlit.log

import streamlit as st
import os, json, time, hmac, hashlib, base64, secrets
from datetime import datetime

LOG_PATH = "warlok_streamlit.log"

# ========= Utilities =========
def log(line: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
    msg = f"[{ts}] {line}"
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    st.session_state["log_buf"].append(msg)

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

# ========= Pure-Python DH Group-14 (RFC 3526) =========
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

# ========= Nano-AI pepper (enclave-sim) =========
class NanoAI:
    """Simulate device-unique enclaved vector via HMAC over context with model_key."""
    def __init__(self, model_key: bytes):
        self.model_key = model_key  # never leaves device/enclave in real life

    def pepper(self, ctx: bytes, out_len: int = 32) -> bytes:
        prk = hmac.new(self.model_key, ctx, hashlib.sha256).digest()
        return hkdf_expand(prk, b"nano-ai-pepper", out_len)

# ========= Inner "envelope" (MAC-only demo) =========
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

# ========= Hub / Server / Client stateful sims =========
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
        log(f"[HUB] New session: {session_id} policy={policy}")
        return session_id, policy, policy_hash

class Server:
    def __init__(self, hub: Hub, device_id="plc-01"):
        self.hub = hub
        self.device_id = device_id
        self.nano_ai = NanoAI(model_key=st.session_state["device_model_key"])  # demo-shared
        self.sessions = {}  # session_id -> session_seed
        self._dh_priv, self._dh_pub = dh14_gen_keypair()

    # INIT: receive client_pub, return server_pub, derive s0
    def init_handshake(self, session_id: str, client_pub_b64: str):
        session = self.hub.sessions.get(session_id)
        if not session:
            raise ValueError("Unknown session")
        peer_pub = int.from_bytes(b64d(client_pub_b64), "big")
        z = dh14_shared_secret(self._dh_priv, peer_pub)
        s0 = hkdf(
            ikm=z,
            salt=session["policy_hash"],
            info=("WARL0K/session" + session_id + session["policy"]["device_id"]).encode(),
            length=32
        )
        self.sessions[session_id] = s0
        server_pub_b64 = b64(self._dh_pub.to_bytes((_GROUP14_P.bit_length()+7)//8, "big"))
        log(f"[SERVER] INIT ok: stored session_seed (hidden). Replying server_pub.")
        return server_pub_b64

    # Regular validate
    def handle_request(self, method: str, path: str, body_txt: str, proof: dict, meta: dict):
        session_id = proof["session_id"]
        session = self.hub.sessions.get(session_id)
        if not session:
            return 401, "Unknown session"
        if session_id not in self.sessions:
            return 428, "Precondition Required: run DH init first"

        # Policy check
        policy = session["policy"]
        if method.upper() not in policy["allow_methods"] or policy["route"] != meta["route"]:
            return 403, "Policy denied"

        # Per-message keys
        seed = self.sessions[session_id]
        ctx_core = (session_id + meta["route"] + meta["device_id"]).encode()
        info = ctx_core + b64d(proof["nonce"]) + int(proof["counter"]).to_bytes(8, "big")
        k_i = hkdf(ikm=seed, salt=session["policy_hash"], info=info, length=32)

        ctx_ai = b"|".join([ctx_core, int(proof["counter"]).to_bytes(8,"big"),
                             b64d(proof["nonce"]), session["policy_hash"]])
        nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)

        final_key = hkdf(ikm=k_i + nano_pepper, salt=session["policy_hash"], info=b"bind", length=32)

        payload = "|".join([
            method, path, body_txt, b64(session["policy_hash"]),
            str(proof["ttl_ms"]), str(proof["counter"])
        ]).encode()
        expected_sig = hmac.new(final_key, payload, hashlib.sha256).digest()

        # Verify
        if not hmac.compare_digest(expected_sig, b64d(proof["sig"])):
            return 401, "Invalid proof signature"
        if (time.time()*1000) > int(proof["issued_ms"]) + int(proof["ttl_ms"]):
            return 401, "Expired proof"

        # Open inner envelope on /cmd, /data
        try:
            obj = json.loads(body_txt) if body_txt.strip() else {}
            aad = ("|".join([meta["route"], meta["device_id"], str(proof["counter"])]).encode())
            obj2 = unwrap_selected_fields(obj, final_key, aad, fields=("/cmd","/data"))
        except Exception as e:
            return 400, f"Body unwrap error: {e}"

        # Fine-grained policy could be enforced here.
        return 200, f"OK (method {method}). Decrypted body: {json.dumps(obj2)}"

class Client:
    def __init__(self, session_id: str, policy_hash: bytes, route="/api/write", device_id="plc-01"):
        self.session_id = session_id
        self.policy_hash = policy_hash
        self.route = route
        self.device_id = device_id
        self.nano_ai = NanoAI(model_key=st.session_state["device_model_key"])  # demo-shared
        self.session_seed = None
        self.counter = 0
        self._dh_priv, self._dh_pub = dh14_gen_keypair()

    def init_handshake(self, server: Server):
        client_pub_b64 = b64(self._dh_pub.to_bytes((_GROUP14_P.bit_length()+7)//8, "big"))
        log(f"[CLIENT] INIT → server (session_id={self.session_id}, client_pub)")
        server_pub_b64 = server.init_handshake(self.session_id, client_pub_b64)
        server_pub_int = int.from_bytes(b64d(server_pub_b64), "big")
        z = dh14_shared_secret(self._dh_priv, server_pub_int)
        info = ("WARL0K/session" + self.session_id + self.device_id).encode()
        self.session_seed = hkdf(ikm=z, salt=self.policy_hash, info=info, length=32)
        log("[CLIENT] INIT ack ← server_pub; derived session_seed (hidden).")

    def build_request(self, method: str, path: str, body_obj: dict):
        self.counter += 1
        nonce = os.urandom(16)
        ttl_ms = 400
        issued_ms = int(time.time()*1000)

        ctx_core = (self.session_id + self.route + self.device_id).encode()
        info = ctx_core + nonce + self.counter.to_bytes(8,"big")
        k_i = hkdf(ikm=self.session_seed, salt=self.policy_hash, info=info, length=32)

        ctx_ai = b"|".join([ctx_core, self.counter.to_bytes(8,"big"), nonce, self.policy_hash])
        nano_pepper = self.nano_ai.pepper(ctx_ai, out_len=32)
        final_key = hkdf(ikm=k_i + nano_pepper, salt=self.policy_hash, info=b"bind", length=32)

        # Wrap selected fields
        aad = ("|".join([self.route, self.device_id, str(self.counter)])).encode()
        wrapped = wrap_selected_fields(body_obj, final_key, aad, fields=("/cmd","/data"))
        body_txt = json.dumps(wrapped, separators=(",",":"))

        payload = "|".join([
            method, path, body_txt, b64(self.policy_hash),
            str(ttl_ms), str(self.counter)
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
        return proof, meta, body_txt

# ========= Streamlit UI =========
st.set_page_config(page_title="WARL0K Streamlit Demo (Stdlib Crypto)", layout="wide")
st.title("WARL0K — Wrapper ↔ Hub ↔ Enclave (Stdlib Crypto • Streamlit Demo)")

# Session state
if "log_buf" not in st.session_state:
    st.session_state["log_buf"] = []
if "hub" not in st.session_state:
    st.session_state["hub"] = Hub()
if "device_model_key" not in st.session_state:
    # Demo: share model key so client+server derive the same nano-pepper
    st.session_state["device_model_key"] = os.urandom(32)
if "server" not in st.session_state:
    st.session_state["server"] = Server(st.session_state["hub"], device_id="plc-01")
if "session" not in st.session_state:
    st.session_state["session"] = {"session_id": None, "policy": None, "policy_hash": None, "client": None}

c_client, c_hub, c_server = st.columns(3)

with c_hub:
    st.subheader("HUB")
    route = st.text_input("Route", value="/api/write")
    device_id = st.text_input("Device ID", value="plc-01")
    allow_write = st.checkbox("Allow WRITE", value=True)
    allow_read = st.checkbox("Allow READ", value=True)

    if st.button("1) Create Session"):
        methods = []
        if allow_write: methods.append("WRITE")
        if allow_read: methods.append("READ")
        session_id, policy, policy_hash = st.session_state["hub"].new_session(
            device_id=device_id, route=route, allow_methods=tuple(methods or ["READ"]))
        st.session_state["session"] = {"session_id": session_id, "policy": policy, "policy_hash": policy_hash, "client": None}

    st.write("**Session ID:**", st.session_state["session"]["session_id"])
    st.write("**Policy:**", st.session_state["session"]["policy"])

with c_client:
    st.subheader("CLIENT (Wrapper)")
    if st.button("2) INIT Handshake"):
        s = st.session_state["session"]
        if not s["session_id"]:
            st.warning("Create a session first.")
        else:
            s["client"] = Client(
                session_id=s["session_id"], policy_hash=s["policy_hash"],
                route=s["policy"]["route"], device_id=s["policy"]["device_id"]
            )
            try:
                s["client"].init_handshake(st.session_state["server"])
                st.success("INIT OK — session_seed derived.")
            except Exception as e:
                st.error(f"INIT failed: {e}")

    st.divider()
    st.write("**Send requests**")
    if st.button("3) Send WRITE"):
        s = st.session_state["session"]
        if not s["client"] or s["client"].session_seed is None:
            st.warning("Run INIT first.")
        else:
            body = {"cmd":{"op":"set_speed"}, "data":{"value":42}, "note":"non-sensitive"}
            proof, meta, body_txt = s["client"].build_request("WRITE", route, body)
            log(f"[CLIENT] WRITE → Server: proof={json.dumps(proof)} meta={json.dumps(meta)} body={body_txt}")
            code, msg = st.session_state["server"].handle_request("WRITE", route, body_txt, proof, meta)
            log(f"[SERVER] WRITE result: {code} {msg}")
            st.code(msg, language="json")

    if st.button("4) Send DELETE"):
        s = st.session_state["session"]
        if not s["client"] or s["client"].session_seed is None:
            st.warning("Run INIT first.")
        else:
            body = {"cmd":{"op":"wipe"}, "data":{"confirm":True}}
            proof, meta, body_txt = s["client"].build_request("DELETE", route, body)
            log(f"[CLIENT] DELETE → Server: proof={json.dumps(proof)} meta={json.dumps(meta)} body={body_txt}")
            code, msg = st.session_state["server"].handle_request("DELETE", route, body_txt, proof, meta)
            log(f"[SERVER] DELETE result: {code} {msg}")
            st.code(msg, language="json")

with c_server:
    st.subheader("SERVER (Validator / Enclave)")
    st.write("**Sessions (seed held server-side):**", len(st.session_state["server"].sessions))
    st.write("**Device Model Key:** demo-shared (for reproducible nano-pepper)")

st.divider()
st.subheader("Demo Log")
st.caption(f"Appends to **{LOG_PATH}** (persisted on disk)")
if st.button("Clear Log"):
    st.session_state["log_buf"].clear()
    try:
        if os.path.exists(LOG_PATH): os.remove(LOG_PATH)
    except Exception:
        pass
    st.success("Log cleared.")
st.text_area("Events", value="\n".join(st.session_state["log_buf"]), height=260)
