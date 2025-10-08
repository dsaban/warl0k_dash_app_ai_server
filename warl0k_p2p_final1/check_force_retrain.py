#!/usr/bin/env python3
import argparse, os, ssl, socket, yaml, hashlib
from warlok.storage import TicketedAdapters
from warlok.pretrain import Pretrainer, obf_ticket, master_seedpath
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def hub_get_W(peer_id, cfg):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket(); tls = ctx.wrap_socket(s, server_hostname="hub")
    tls.connect((cfg["general"]["host"], cfg["general"]["hub_port"]))
    send_msg(tls, {"cmd":"get_seed2master_vec","target_device_id":peer_id})
    res = recv_msg(tls); tls.close()
    if res.get("status")!="ok": raise SystemExit(f"Hub error: {res}")
    return bytes.fromhex(res["W_hex"])

def main():
    ap=argparse.ArgumentParser(description="Force retrain adapter for (peer_id, n).")
    ap.add_argument("--peer-id", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--adapters", default=".adapters_A")
    args=ap.parse_args()

    cfg = load_cfg()
    OBF = cfg["session"]["obf_len"]; ML = cfg["models"]["master_len_chars"]
    HID = cfg["models"]["drnn_hidden_dim"]; LR = cfg["models"]["drnn_lr"]
    M_S = cfg["models"]["drnn_meta_samples"]; M_ST = cfg["models"]["drnn_meta_steps"]

    W = hub_get_W(args.peer_id, cfg)
    print("W.sha256:", hashlib.sha256(W).hexdigest())
    store = TicketedAdapters(args.adapters)
    obf = obf_ticket(W, args.n, OBF)
    target = master_seedpath(W, args.peer_id, ML)

    d = Sess2MasterDRNN(hidden_dim=HID, lr=LR)
    d.set_context(peer_id=args.peer_id, W_bytes=W, target_len_chars=ML)
    d.meta_pretrain(m_samples=M_S, steps=M_ST, obf_len=OBF)
    info = d.train_pair(obf, target, epochs=30, check_every=5, patience=2)
    d.forced_obf = obf; d.forced_target = target; d.last_training_info = info
    store.save(args.peer_id, args.n, d)

    pred = d.predict(obf, out_len=ML)
    print("obf   :", obf)
    print("target:", target)
    print("pred  :", pred)
    print("epochs:", info["epochs_run"], "meta_used:", info["meta_used"], "early:", info["early_stopped"])
    print("result:", "OK ✅" if pred==target else "MISMATCH ❌")

if __name__=="__main__":
    main()
