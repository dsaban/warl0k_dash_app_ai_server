import socket, ssl, threading, yaml, os
from warlok.net import send_msg, recv_msg
from warlok.crypto import gen_x25519_keypair, pub_bytes_x25519, hexlify, unhex, hmac_sha256
from warlok.peer_core import PeerCore
from warlok.storage import TicketedAdapters
from warlok.pretrain import Pretrainer, hexs, obf_ticket, master_seedpath
from warlok.models.sess2master_drnn import Sess2MasterDRNN

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)
# hub monitoring thread (not used in B, but could be)
from warlok.hub_monitor import HubMonitor

MON = CFG.get("monitoring", {})
counter = CFG["session"]["counter_start"]
monitor = HubMonitor(
    my_device_id="device-B",
    hub_host=CFG["general"]["host"],
    hub_port=CFG["general"]["hub_port"],
    adapters_dir=".adapters_B",
    obf_len=CFG["session"]["obf_len"],
    master_len=CFG["models"]["master_len_chars"],
    drnn_hidden=CFG["models"]["drnn_hidden_dim"],
    drnn_lr=CFG["models"]["drnn_lr"],
    poll_interval=MON.get("poll_interval_sec", 5),
    jitter_sec=MON.get("jitter_sec", 2),
    pretrain_window=MON.get("pretrain_window", 8),
    rollout_mode=MON.get("rollout", {}).get("mode", "graceful"),
    overlap_tickets=MON.get("rollout", {}).get("overlap_tickets", 4),
    peers_to_watch=[p for p in MON.get("peers", []) if p != "device-B"],
    get_next_counter=lambda: counter,
    log=lambda *a: print("[B monitor]", *a),
)
monitor.start()

#
HOST=CFG["general"]["host"]; HUB_PORT=CFG["general"]["hub_port"]; PORT=CFG["general"]["peerB_port"]
POLICY=CFG["session"]["policy_id"]; OBF_LEN=CFG["session"]["obf_len"]
MASTER_LEN=CFG["models"]["master_len_chars"]
M_SAMP=CFG["models"]["drnn_meta_samples"]; M_STEPS=CFG["models"]["drnn_meta_steps"]
WIN=CFG["session"]["pretrain_window"]

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

# def handle(conn):
#     try:
#         MY="device-B"; PEER="device-A"
#         store = TicketedAdapters(dirpath=".adapters_B")
#         while True:
#             msg = recv_msg(conn)
#             if msg["type"] == "HELLO":
#                 A_pub = unhex(msg["A_pub"])
#                 session_id = msg["session_id"]; counter = int(msg["ctr"])
#                 b_priv, b_pub = gen_x25519_keypair()
#                 challengeB = os.urandom(16)
#                 send_msg(conn, {"type":"HELLO-ACK","B_pub":hexlify(pub_bytes_x25519(b_pub)),"challengeB":hexlify(challengeB)})
#                 conn._ctx = {"A_pub":A_pub,"counter":counter,"challengeB":challengeB,"b_priv":b_priv,"b_pub":b_pub}
#             elif msg["type"] == "ENVELOPE":
#                 env = msg["body"]; meta = msg["meta"]
#                 n = int(meta["n"]); tagA_hex = meta["tagA"]; obf_len = int(meta["obf_len"])
#                 ctx = conn._ctx; A_pub = ctx["A_pub"]; challengeB=ctx["challengeB"]; b_priv = ctx["b_priv"]
#
#                 # Fetch W for PEER (A) and ensure ticket pretraining
#                 resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":"device-A"})
#                 W_PEER = bytes.fromhex(resp["W_hex"])
#                 pre = Pretrainer(store, obf_len, MASTER_LEN, CFG["models"]["drnn_hidden_dim"], CFG["models"]["drnn_lr"])
#                 pre.schedule_window("device-B", "device-A", W_PEER, start_n=n, window=WIN, meta=(M_SAMP,M_STEPS))
#
#                 # Instant identity via ticket n
#                 drnn = store.load("device-A", n, ctor=lambda: Sess2MasterDRNN(
#                     hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"]))
#                 obf_train = obf_ticket(W_PEER, n, obf_len)
#                 master_seed = master_seedpath(W_PEER, "device-A", MASTER_LEN)
#                 pred = drnn.predict(obf_train, out_len=MASTER_LEN)
#                 if pred != master_seed:
#                     log("Ticketed adapter mismatch (B’s view). Reject.")
#                     send_msg(conn, {"type":"RESULT","crypto_ok": False, "why": "ticket_mismatch"})
#                     return
#
#                 # Verify tag and decrypt
#                 pc = PeerCore("device-B")
#                 k_sess = pc.derive_k_session(b_priv, A_pub, "device-B", "device-A", POLICY, n, challengeB)
#                 obf_real = pc.obf_from_k(k_sess, obf_len)
#                 transcript = f"device-A|device-B|{POLICY}|{n}|{hexlify(challengeB)}".encode()
#                 K_tag = hmac_sha256(W_PEER, b"sessK", n.to_bytes(8,"big"), obf_real.encode())
#                 calc_tag_hex = hexs(hmac_sha256(K_tag, b"TAG", transcript))
#                 if calc_tag_hex != tagA_hex:
#                     log("Tag verify failed. Reject."); send_msg(conn, {"type":"RESULT","crypto_ok": False, "why":"tag_fail"}); return
#
#                 ok, plaintext = pc.verify_envelope(env, "device-B", "device-A", b_priv, A_pub)
#                 log("decrypt:", ok, "plaintext:", plaintext)
#                 send_msg(conn, {"type":"RESULT","crypto_ok": bool(ok)})
#                 return
#             else:
#                 send_msg(conn, {"status":"error","why":"unknown"})
#     except ConnectionError:
#         pass
#     finally:
#         conn.close()
def handle(conn):
    try:
        MY = "device-B"
        PEER = "device-A"
        store = TicketedAdapters(dirpath=".adapters_B")
        pc = PeerCore(MY)

        # Per-connection context lives here (not on the socket object)
        ctx = {}

        while True:
            msg = recv_msg(conn)

            if msg["type"] == "HELLO":
                A_pub = unhex(msg["A_pub"])
                session_id = msg["session_id"]
                counter = int(msg["ctr"])

                b_priv, b_pub = gen_x25519_keypair()
                challengeB = os.urandom(16)

                send_msg(conn, {
                    "type": "HELLO-ACK",
                    "B_pub": hexlify(pub_bytes_x25519(b_pub)),
                    "challengeB": hexlify(challengeB),
                })

                # Save into local context
                ctx.update({
                    "A_pub": A_pub,
                    "counter": counter,
                    "challengeB": challengeB,
                    "b_priv": b_priv,
                    "b_pub": b_pub,
                    "MY": MY,
                    "PEER": PEER,
                })

            elif msg["type"] == "ENVELOPE":
                # Pull back the values from ctx
                if not ctx:
                    send_msg(conn, {"type": "RESULT", "crypto_ok": False, "why": "no_context"})
                    return

                env = msg["body"]
                meta = msg["meta"]
                n = int(meta["n"])
                tagA_hex = meta["tagA"]
                obf_len = int(meta["obf_len"])

                A_pub = ctx["A_pub"]
                challengeB = ctx["challengeB"]
                b_priv = ctx["b_priv"]

                # Ask hub for A's W vector
                resp = hub_rpc({"cmd": "get_seed2master_vec", "target_device_id": "device-A"})
                if resp.get("status") != "ok":
                    send_msg(conn, {"type": "RESULT", "crypto_ok": False, "why": "hub_unknown_target"})
                    return
                W_PEER = bytes.fromhex(resp["W_hex"])

                # Ensure ticket pretraining exists locally for this n
                pre = Pretrainer(store, obf_len, MASTER_LEN,
                                 CFG["models"]["drnn_hidden_dim"],
                                 CFG["models"]["drnn_lr"])
                pre.schedule_window(MY, PEER, W_PEER, start_n=n, window=WIN,
                                    meta=(M_SAMP, M_STEPS))

                # Instant identity via ticket n
                drnn = store.load(PEER, n, ctor=lambda: Sess2MasterDRNN(
                    hidden_dim=CFG["models"]["drnn_hidden_dim"],
                    lr=CFG["models"]["drnn_lr"]))
                obf_train = obf_ticket(W_PEER, n, obf_len)
                master_seed = master_seedpath(W_PEER, PEER, MASTER_LEN)
                pred = drnn.predict(obf_train, out_len=MASTER_LEN)
                if pred != master_seed:
                    log("Ticketed adapter mismatch (B’s view). Reject.")
                    send_msg(conn, {"type": "RESULT", "crypto_ok": False, "why": "ticket_mismatch"})
                    return

                # Verify session tag and decrypt envelope
                k_sess = pc.derive_k_session(b_priv, A_pub, MY, PEER, POLICY, n, challengeB)
                obf_real = pc.obf_from_k(k_sess, obf_len)
                transcript = f"{PEER}|{MY}|{POLICY}|{n}|{hexlify(challengeB)}".encode()
                K_tag = hmac_sha256(W_PEER, b"sessK", n.to_bytes(8, "big"), obf_real.encode())
                calc_tag_hex = hexs(hmac_sha256(K_tag, b"TAG", transcript))
                if calc_tag_hex != tagA_hex:
                    log("Tag verify failed. Reject.")
                    send_msg(conn, {"type": "RESULT", "crypto_ok": False, "why": "tag_fail"})
                    return

                ok, plaintext = pc.verify_envelope(env, MY, PEER, b_priv, A_pub)
                log("decrypt:", ok, "plaintext:", plaintext)
                send_msg(conn, {"type": "RESULT", "crypto_ok": bool(ok)})
                return

            else:
                send_msg(conn, {"status": "error", "why": "unknown"})

    except ConnectionError:
        pass
    finally:
        conn.close()

def server():
    hub_rpc({"cmd":"enroll","device_id":"device-B"})
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT)); s.listen(5)
    log(f"listening on {HOST}:{PORT}")
    while True:
        c,_ = s.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()

if __name__ == "__main__":
    server()
