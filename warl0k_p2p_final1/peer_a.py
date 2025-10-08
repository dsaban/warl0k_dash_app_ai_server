import socket, ssl, yaml, os, binascii
from warlok.net import send_msg, recv_msg
from warlok.crypto import gen_x25519_keypair, pub_bytes_x25519, hexlify, unhex, hmac_sha256
from warlok.peer_core import PeerCore
from warlok.storage import TicketedAdapters
from warlok.pretrain import Pretrainer, hexs, obf_ticket, master_seedpath
from warlok.models.sess2master_drnn import Sess2MasterDRNN

import hashlib, time

# Hub monitoring thread
from warlok.hub_monitor import HubMonitor

# ... load CFG ...
with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)
# Local session counter (captured by the monitor)
counter = CFG["session"]["counter_start"]

MON = CFG.get("monitoring", {})
monitor = HubMonitor(
    my_device_id="device-A",
    hub_host=CFG["general"]["host"],
    hub_port=CFG["general"]["hub_port"],
    adapters_dir=".adapters_A",
    obf_len=CFG["session"]["obf_len"],
    master_len=CFG["models"]["master_len_chars"],
    drnn_hidden=CFG["models"]["drnn_hidden_dim"],
    drnn_lr=CFG["models"]["drnn_lr"],
    poll_interval=MON.get("poll_interval_sec", 5),
    jitter_sec=MON.get("jitter_sec", 2),
    pretrain_window=MON.get("pretrain_window", 8),
    rollout_mode=MON.get("rollout", {}).get("mode", "graceful"),
    overlap_tickets=MON.get("rollout", {}).get("overlap_tickets", 4),
    peers_to_watch=[p for p in MON.get("peers", []) if p != "device-A"],
    get_next_counter=lambda: counter,   # capture the local session counter
    log=lambda *a: print("[A monitor]", *a),
)
monitor.start()

#
def dbg_hex(s, n=16):
    if isinstance(s, bytes): s = s.hex()
    return f"{s[:n]}…{s[-n:]}"

def debug_ticket(W_peer: bytes, peer_id: str, n: int, obf_len: int, master_len: int, drnn=None):
    from warlok.pretrain import obf_ticket, master_seedpath
    obf = obf_ticket(W_peer, n, obf_len)
    master = master_seedpath(W_peer, peer_id, master_len)
    pred = None
    if drnn is not None:
        try:
            pred = drnn.predict(obf, out_len=master_len)
        except Exception as e:
            print("[A] predict error:", e)
    print("[A] --- DEBUG TICKET ---")
    print("[A] peer:", peer_id, "n:", n)
    print("[A] W.sha256:", hashlib.sha256(W_peer).hexdigest())
    print("[A] obf:", obf)
    print("[A] master(target):", master)
    if pred is not None:
        print("[A] master(pred):  ", pred)
    print("[A] ---------------")
    return obf, master, pred

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

HOST=CFG["general"]["host"]; HUB_PORT=CFG["general"]["hub_port"]; PEER_B_PORT=CFG["general"]["peerB_port"]
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
    if CFG["general"]["verbose"]: print("[A]", *a)

def main():
    MY="device-A"; PEER="device-B"
    hub_rpc({"cmd":"enroll","device_id":MY})

    # Connect to B (HELLO)
    s = socket.socket(); s.connect((HOST, PEER_B_PORT))
    a_priv, a_pub = gen_x25519_keypair()
    session_id = binascii.hexlify(os.urandom(8)).decode()
    counter = CFG["session"]["counter_start"]
    send_msg(s, {"type":"HELLO","A_pub":hexlify(pub_bytes_x25519(a_pub)),"session_id":session_id,"ctr":counter})
    ack = recv_msg(s); b_pub = unhex(ack["B_pub"]); challengeB = unhex(ack["challengeB"])

    # Fetch W for PEER and pretrain ticket window
    resp = hub_rpc({"cmd": "get_seed2master_vec", "target_device_id": PEER})
    if resp.get("status") != "ok":
        print("[A] Hub returned:", resp)
        return
    W_PEER = bytes.fromhex(resp["W_hex"])
    
    # NEW: fetch A’s own W (identity vector for A)
    resp_self = hub_rpc({"cmd": "get_seed2master_vec", "target_device_id": MY})
    assert resp_self.get("status") == "ok"
    W_SELF = bytes.fromhex(resp_self["W_hex"])
    store = TicketedAdapters(dirpath=".adapters_A")
    pre = Pretrainer(store, OBF_LEN, MASTER_LEN, CFG["models"]["drnn_hidden_dim"], CFG["models"]["drnn_lr"])
    
    # Always schedule (it will build/rebuild for this ticket)
    pre.schedule_window(MY, PEER, W_PEER, start_n=counter, window=WIN, meta=(M_SAMP, M_STEPS))
    
    # Load adapter for this exact ticket
    def ctor():
        d = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
        d.set_context(peer_id=PEER, W_bytes=W_PEER, target_len_chars=MASTER_LEN)
        return d
    
    drnn = store.load(PEER, counter, ctor=ctor)
    
    # Debug & verify
    from warlok.pretrain import obf_ticket, master_seedpath
    obf_train, master_seed, pred0 = debug_ticket(W_PEER, PEER, counter, OBF_LEN, MASTER_LEN, drnn)
    
    if pred0 != master_seed:
        # Self-heal: hard rebuild adapter for this ticket only
        print("[A] Adapter stale or untrained for this ticket — rebuilding now.")
        d = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
        d.set_context(peer_id=PEER, W_bytes=W_PEER, target_len_chars=MASTER_LEN)
        d.meta_pretrain(m_samples=M_SAMP, steps=M_STEPS, obf_len=OBF_LEN)
        train_info = d.train_pair(obf_train, master_seed, epochs=max(30, CFG["models"]["drnn_epochs_per_session"]),
                                  check_every=5, patience=2)
        store.save(PEER, counter, d)
        pred1 = d.predict(obf_train, out_len=MASTER_LEN)
        debug_ticket(W_PEER, PEER, counter, OBF_LEN, MASTER_LEN, d)
        if pred1 != master_seed:
            print("[A] Ticketed adapter STILL mismatched after rebuild. Abort.")
            return
        drnn = d
    
    print("[A] Ticketed identity OK (A→B), n=", counter)
    
    # start hub minitoring thread
    try:
        # existing run logic...
        pass
    finally:
        try:
            monitor.stop()
        except Exception:
            pass
    
    # resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":PEER})
    # W_PEER = bytes.fromhex(resp["W_hex"])
    # store = TicketedAdapters(dirpath=".adapters_A")
    # pre = Pretrainer(store, OBF_LEN, MASTER_LEN, CFG["models"]["drnn_hidden_dim"], CFG["models"]["drnn_lr"])
    # pre.schedule_window(MY, PEER, W_PEER, start_n=counter, window=WIN, meta=(M_SAMP,M_STEPS))

    # Instant identity via ticket n
    # drnn = store.load(PEER, counter, ctor=lambda: Sess2MasterDRNN(
    #     hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"]))
    # obf_train = obf_ticket(W_PEER, counter, OBF_LEN)
    # master_seed = master_seedpath(W_PEER, PEER, MASTER_LEN)
    drnn = store.load(PEER, counter, ctor=lambda: Sess2MasterDRNN(
        hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"]))
    obf_train = obf_ticket(W_PEER, counter, OBF_LEN)
    master_seed = master_seedpath(W_PEER, PEER, MASTER_LEN)
    
    # Self-heal if stale
    if getattr(drnn, "W", None) != W_PEER or drnn._target_len != MASTER_LEN or \
            drnn.predict(obf_train, out_len=MASTER_LEN) != master_seed:
        d = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
        d.set_context(peer_id=PEER, W_bytes=W_PEER, target_len_chars=MASTER_LEN)
        d.meta_pretrain(m_samples=M_SAMP, steps=M_STEPS, obf_len=OBF_LEN)
        d.train_pair(obf_train, master_seed, epochs=20, check_every=3, patience=2)
        store.save(PEER, counter, d)
        drnn = d
    
    pred = drnn.predict(obf_train, out_len=MASTER_LEN)
    if pred != master_seed:
        log("Ticketed adapter mismatch; abort."); s.close(); return
    log("Ticketed identity OK (A→B), n=", counter)

    # Real session ECDH + session-bound tag
    pc = PeerCore(MY)
    k_preview = pc.derive_k_session(a_priv, b_pub, MY, PEER, POLICY, counter, challengeB)
    obf_real = pc.obf_from_k(k_preview, OBF_LEN)
    transcript = f"{MY}|{PEER}|{POLICY}|{counter}|{hexlify(challengeB)}".encode()
    # K_tag = hmac_sha256(W_PEER, b"sessK", counter.to_bytes(8,"big"), obf_real.encode())
    # Tag_A = hexs(hmac_sha256(K_tag, b"TAG", transcript))
    
    # FIX: use A’s own W to prove A’s identity (B will look up W_A)
    K_tag = hmac_sha256(W_SELF, b"sessK", counter.to_bytes(8, "big"), obf_real.encode())
    Tag_A = hexs(hmac_sha256(K_tag, b"TAG", transcript))

    # Send AEAD envelope + ticket + tag
    env = pc.build_envelope(MY, PEER, a_priv, b_pub, POLICY, counter, challengeB,
                            CFG["general"]["demo_plaintext"].encode())
    send_msg(s, {"type":"ENVELOPE","body":env,"meta":{"n":counter, "tagA":Tag_A, "obf_len": OBF_LEN}})
    res = recv_msg(s); log("RESULT from B:", res)
    s.close()

if __name__ == "__main__":
    main()
