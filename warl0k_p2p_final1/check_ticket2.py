#!/usr/bin/env python3
import argparse, os, ssl, socket, yaml, hashlib, sys
from warlok.storage import TicketedAdapters
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.pretrain import Pretrainer, obf_ticket, master_seedpath
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def hub_rpc(req, host, port):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls = ctx.wrap_socket(s, server_hostname="hub")
    tls.connect((host, port))
    send_msg(tls, req)
    res = recv_msg(tls); tls.close()
    return res

def get_W(peer_id, mode, cfg, w_hex_cli=None):
    if mode == "offline":
        if not w_hex_cli: raise SystemExit("--mode offline requires --W-hex")
        return bytes.fromhex(w_hex_cli)
    r = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":peer_id},
                cfg["general"]["host"], cfg["general"]["hub_port"])
    if r.get("status")!="ok":
        print("Hub error:", r, file=sys.stderr); raise SystemExit(2)
    return bytes.fromhex(r["W_hex"])

def main():
    ap = argparse.ArgumentParser(description="Validate ticket obf -> master for (peer_id, n).")
    ap.add_argument("--peer-id", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--adapters", default=".adapters_A")
    ap.add_argument("--mode", choices=["online","offline"], default="online")
    ap.add_argument("--W-hex", default=None)
    ap.add_argument("--window", type=int, default=1)
    args = ap.parse_args()

    cfg = load_cfg()
    OBF = cfg["session"]["obf_len"]; ML = cfg["models"]["master_len_chars"]
    HID = cfg["models"]["drnn_hidden_dim"]; LR = cfg["models"]["drnn_lr"]
    M_S = cfg["models"]["drnn_meta_samples"]; M_ST = cfg["models"]["drnn_meta_steps"]

    W = get_W(args.peer_id, args.mode, cfg, args.W_hex)
    print("[*] peer_id:", args.peer_id)
    print("[*] n:", args.n)
    print("[*] W.sha256:", hashlib.sha256(W).hexdigest())

    store = TicketedAdapters(args.adapters)
    pre = Pretrainer(store, OBF, ML, HID, LR)
    pre.schedule_window("CLI", args.peer_id, W, args.n, max(1,args.window), meta=(M_S,M_ST))

    def ctor():
        from warlok.models.sess2master_drnn import Sess2MasterDRNN
        d = Sess2MasterDRNN(hidden_dim=HID, lr=LR)
        d.set_context(peer_id=args.peer_id, W_bytes=W, target_len_chars=ML); return d
    d = store.load(args.peer_id, args.n, ctor)

    from warlok.pretrain import obf_ticket, master_seedpath
    obf = obf_ticket(W, args.n, OBF)
    target = master_seedpath(W, args.peer_id, ML)
    pred = d.predict(obf, out_len=ML)

    ok = (pred==target)
    print("    obf:", obf)
    print("target :", target)
    print("pred   :", pred)
    print("result :", "OK ✅" if ok else "MISMATCH ❌")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
