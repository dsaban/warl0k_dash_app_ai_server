import socket, ssl, yaml, os, binascii
from warlok.net import send_msg, recv_msg
from warlok.crypto import gen_x25519_keypair, pub_bytes_x25519, hexlify, unhex
from warlok.peer_core import PeerCore
from warlok.models.seed2master_model import Seed2MasterModel
from warlok.models.sess2master_drnn import Sess2MasterDRNN

with open("config.yaml","r") as f:
    CFG = yaml.safe_load(f)

HOST=CFG["general"]["host"]; HUB_PORT=CFG["general"]["hub_port"]; PEER_B_PORT=CFG["general"]["peerB_port"]
POLICY=CFG["session"]["policy_id"]; OBF_LEN=CFG["session"]["obf_len"]
MASTER_LEN=CFG["models"]["master_len_chars"]; EPOCHS=CFG["models"]["drnn_epochs_per_session"]

def hub_rpc(req):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls_conn = ctx.wrap_socket(s, server_hostname="hub")  # CN=hub
    tls_conn.connect((HOST, HUB_PORT))
    send_msg(tls_conn, req)
    res = recv_msg(tls_conn)
    tls_conn.close()
    return res

def log(*a):
    if CFG["general"]["verbose"]: print("[A]", *a)

def main():
    device_id="device-A"
    # Enroll A (hub stores seed0_A)
    hub_rpc({"cmd":"enroll","device_id":device_id})

    # Connect to B and exchange ephemeral keys
    s = socket.socket(); s.connect((HOST, PEER_B_PORT))
    a_priv, a_pub = gen_x25519_keypair()
    session_id = binascii.hexlify(os.urandom(8)).decode()
    counter = CFG["session"]["counter_start"]
    send_msg(s, {"type":"HELLO","A_pub":hexlify(pub_bytes_x25519(a_pub)),"session_id":session_id,"ctr":counter})
    ack = recv_msg(s); b_pub = unhex(ack["B_pub"]); challengeB = unhex(ack["challengeB"])

    # Derive session preview and obf
    pc = PeerCore("device-A")
    k_preview = pc.derive_k_session(a_priv, b_pub, "device-A","device-B",POLICY,counter,challengeB)
    obfA = pc.obf_from_k(k_preview, OBF_LEN); log("obfA:", obfA)

    # Fetch Seed→Master vector for B, compute master (seed-path)
    resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":"device-B"})
    W_B = bytes.fromhex(resp["W_hex"])
    seed_model = Seed2MasterModel("device-B", W_B, MASTER_LEN)
    master_B_seedpath = seed_model.compute_master(); log("master_B seed-path:", master_B_seedpath)

    # Train Sess→Master DRNN to map obfA → master_B (early stops on exact match)
    # drnn = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
    # train_info = drnn.train_pair(obfA, master_B_seedpath, epochs=EPOCHS)
    # master_B_sesspath = drnn.predict(obfA, out_len=MASTER_LEN)
    # log("DRNN:", train_info, "| master_B sess-path:", master_B_sesspath)
    # after you fetch W_B and compute seed-path master_B_seedpath ...
    # drnn = Sess2MasterDRNN()
    # drnn.set_context(peer_id="device-B", W_bytes=W_B, target_len_chars=MASTER_LEN)
    # train_info = drnn.train_pair(obfA, master_B_seedpath, epochs=1)  # no-op
    # master_B_sesspath = drnn.predict(obfA, out_len=MASTER_LEN)
    # Train/predict (deterministic sess-path; no-op training)
    drnn = Sess2MasterDRNN()
    drnn.set_context(peer_id="device-B", W_bytes=W_B, target_len_chars=MASTER_LEN)
    train_info = drnn.train_pair(obfA, master_B_seedpath, epochs=1)
    master_B_sesspath = drnn.predict(obfA, out_len=MASTER_LEN)
    
    if master_B_sesspath != master_B_seedpath:
        log("ID verification failed (A’s view). Aborting.")
        s.close(); return

    # Build AEAD envelope and send
    env = pc.build_envelope("device-A","device-B",a_priv,b_pub,POLICY,counter,challengeB, CFG["general"]["demo_plaintext"].encode())
    send_msg(s, {"type":"ENVELOPE","body":env,"meta":{"obfA":obfA}})
    res = recv_msg(s); log("RESULT from B:", res)
    s.close()

if __name__ == "__main__":
    main()
