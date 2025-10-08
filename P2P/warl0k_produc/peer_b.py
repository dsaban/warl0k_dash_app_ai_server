import socket, ssl, threading, yaml, os
from warlok.net import send_msg, recv_msg
from warlok.crypto import gen_x25519_keypair, pub_bytes_x25519, hexlify, unhex
from warlok.peer_core import PeerCore
from warlok.models.seed2master_model import Seed2MasterModel
from warlok.models.sess2master_drnn import Sess2MasterDRNN

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

HOST=CFG["general"]["host"]; HUB_PORT=CFG["general"]["hub_port"]; PORT=CFG["general"]["peerB_port"]
POLICY=CFG["session"]["policy_id"]; OBF_LEN=CFG["session"]["obf_len"]
MASTER_LEN=CFG["models"]["master_len_chars"]; EPOCHS=CFG["models"]["drnn_epochs_per_session"]

def hub_rpc(req):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls_conn = ctx.wrap_socket(s, server_hostname="hub")
    tls_conn.connect((HOST, HUB_PORT))
    send_msg(tls_conn, req)
    res = recv_msg(tls_conn)
    tls_conn.close()
    return res

def log(*a):
    if CFG["general"]["verbose"]: print("[B]", *a)

def handle(conn):
    try:
        pc = PeerCore("device-B")
        while True:
            msg = recv_msg(conn)
            if msg["type"] == "HELLO":
                A_pub = unhex(msg["A_pub"])
                session_id = msg["session_id"]; counter = int(msg["ctr"])
                b_priv, b_pub = gen_x25519_keypair()
                challengeB = os.urandom(16)
                send_msg(conn, {"type":"HELLO-ACK","B_pub":hexlify(pub_bytes_x25519(b_pub)),"challengeB":hexlify(challengeB)})
                conn._ctx = {"A_pub":A_pub,"counter":counter,"challengeB":challengeB,"b_priv":b_priv,"b_pub":b_pub}
            elif msg["type"] == "ENVELOPE":
                env = msg["body"]; obfA = msg["meta"]["obfA"]
                ctx = conn._ctx; A_pub = ctx["A_pub"]; counter = ctx["counter"]; challengeB=ctx["challengeB"]
                b_priv = ctx["b_priv"]

                # Seed-path master for A
                resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":"device-A"})
                W_A = bytes.fromhex(resp["W_hex"])
                seed_model = Seed2MasterModel("device-A", W_A, MASTER_LEN)
                master_A_seedpath = seed_model.compute_master(); log("master_A seed-path:", master_A_seedpath)

                # Sess-path training & prediction (early-stop)
                # drnn = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
                # train_info = drnn.train_pair(obfA, master_A_seedpath, epochs=EPOCHS)
                # master_A_sesspath = drnn.predict(obfA, out_len=MASTER_LEN); log("DRNN:", train_info, "| master_A sess-path:", master_A_sesspath)
                # drnn = Sess2MasterDRNN()
                # drnn.set_context(peer_id="device-A", W_bytes=W_A, target_len_chars=MASTER_LEN)
                # train_info = drnn.train_pair(obfA, master_A_seedpath, epochs=1)  # no-op
                # master_A_sesspath = drnn.predict(obfA, out_len=MASTER_LEN)
                drnn = Sess2MasterDRNN()
                drnn.set_context(peer_id="device-A", W_bytes=W_A, target_len_chars=MASTER_LEN)
                train_info = drnn.train_pair(obfA, master_A_seedpath, epochs=1)
                master_A_sesspath = drnn.predict(obfA, out_len=MASTER_LEN)
                
                if master_A_sesspath != master_A_seedpath:
                    log("ID verification failed (B’s view). Reject.")
                    send_msg(conn, {"type":"RESULT","crypto_ok": False, "why": "ID_map_mismatch"})
                    return

                # Decrypt AEAD payload under k_session
                ok, plaintext = pc.verify_envelope(env, "device-B","device-A", b_priv, A_pub)
                log("decrypt:", ok, "plaintext:", plaintext)
                send_msg(conn, {"type":"RESULT","crypto_ok": bool(ok)})
                return
            else:
                send_msg(conn, {"status":"error","why":"unknown"})
    except ConnectionError:
        pass
    finally:
        conn.close()

def server():
    # Enroll B (hub stores seed0_B)
    hub_rpc({"cmd":"enroll","device_id":"device-B"})
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT)); s.listen(5)
    log(f"listening on {HOST}:{PORT}")
    while True:
        c,_ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()

if __name__ == "__main__":
    server()
