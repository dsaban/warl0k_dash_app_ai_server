import socket, ssl, threading, yaml
from warlok.net import send_msg, recv_msg
from warlok.crypto import hmac_sha256, rand

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

HOST = CFG["general"]["host"]
PORT = CFG["general"]["hub_port"]
MASTER_LEN = CFG["models"]["master_len_chars"]

# device_id -> seed0 (seed0 NEVER leaves the hub)
REGISTRY = {}

def enroll(device_id: str) -> bytes:
    seed0 = rand(32)
    REGISTRY[device_id] = seed0
    return seed0

def get_seed2master_vector(target_device_id: str) -> bytes:
    seed0 = REGISTRY.get(target_device_id)
    if seed0 is None:
        return None
    # per-target "weight vector" W derived from seed0_target
    return hmac_sha256(seed0, b"W", target_device_id.encode())

def handle(conn):
    try:
        while True:
            req = recv_msg(conn)
            cmd = req.get("cmd")
            if cmd == "enroll":
                dev = req["device_id"]
                seed0 = enroll(dev)
                # NOTE: returning seed0 is ONLY for demo visibility. Remove in prod.
                send_msg(conn, {"status":"ok","seed0_hex":seed0.hex()})
            elif cmd == "get_seed2master_vec":
                target = req["target_device_id"]
                W = get_seed2master_vector(target)
                if W is None:
                    send_msg(conn, {"status":"error","why":"unknown_target"})
                else:
                    send_msg(conn, {"status":"ok","W_hex": W.hex(), "master_len": MASTER_LEN})
            else:
                send_msg(conn, {"status":"error","why":"unknown_cmd"})
    except ConnectionError:
        pass
    finally:
        conn.close()

def main():
    # TLS server context with client cert required
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_cert_chain(certfile="hub.crt", keyfile="hub.key")
    ctx.load_verify_locations("ca.crt")

    s = socket.socket()
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    print(f"[hub] TLS listening on {HOST}:{PORT}")
    while True:
        c,_ = s.accept()
        try:
            tls_conn = ctx.wrap_socket(c, server_side=True)
            threading.Thread(target=handle, args=(tls_conn,), daemon=True).start()
        except ssl.SSLError as e:
            print("TLS error:", e)
            c.close()

if __name__ == "__main__":
    main()
