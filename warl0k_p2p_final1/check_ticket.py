#!/usr/bin/env python3
"""
Check that a stored adapter for (peer_id, n) maps the ticket obf to the correct master.
Can fetch W from the hub (TLS) or accept --W-hex for offline checks.

Usage examples:
  # Online: ask the hub for W(peer_id) over TLS
  python3 tools/check_ticket.py --peer-id device-B --n 1 --adapters .adapters_A --mode online

  # Offline: provide W-hex string directly (no hub needed)
  python3 tools/check_ticket.py --peer-id device-B --n 1 --adapters .adapters_A --mode offline \
      --W-hex 76228ab7fddc164d204c1062c7c6dc6b1528072a1432ef36e6427f4034de4dfe
"""
import argparse, os, ssl, socket, yaml, hashlib, sys
from warlok.storage import TicketedAdapters
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.pretrain import Pretrainer, obf_ticket, master_seedpath, hexs
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def hub_rpc(req, host, port):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    # client auth (same peer cert is OK for this CLI)
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls_conn = ctx.wrap_socket(s, server_hostname="hub")
    tls_conn.connect((host, port))
    send_msg(tls_conn, req)
    res = recv_msg(tls_conn)
    tls_conn.close()
    return res

def get_W_for_peer(peer_id, mode, cfg, w_hex_cli=None):
    if mode == "offline":
        if not w_hex_cli:
            raise SystemExit("--mode offline requires --W-hex")
        return bytes.fromhex(w_hex_cli)
    elif mode == "online":
        resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":peer_id},
                       cfg["general"]["host"], cfg["general"]["hub_port"])
        if resp.get("status") != "ok":
            print("Hub error:", resp, file=sys.stderr)
            raise SystemExit(2)
        return bytes.fromhex(resp["W_hex"])
    else:
        raise SystemExit("--mode must be 'online' or 'offline'")

def main():
    ap = argparse.ArgumentParser(description="Validate ticket mapping obf -> master for (peer_id, n).")
    ap.add_argument("--peer-id", required=True, help="The target peer identity (whose W you want).")
    ap.add_argument("--n", type=int, required=True, help="Session counter (ticket index).")
    ap.add_argument("--adapters", default=".adapters_A",
                    help="Adapters dir used by the checking side (e.g., .adapters_A or .adapters_B).")
    ap.add_argument("--mode", choices=["online","offline"], default="online",
                    help="Fetch W from hub (online) or pass --W-hex (offline).")
    ap.add_argument("--W-hex", default=None, help="W hex (required in offline mode).")
    ap.add_argument("--window", type=int, default=1, help="How many tickets to (re)prepare from n.")
    args = ap.parse_args()

    cfg = load_cfg()
    OBF_LEN   = cfg["session"]["obf_len"]
    MASTERLEN = cfg["models"]["master_len_chars"]
    HIDDEN    = cfg["models"]["drnn_hidden_dim"]
    LR        = cfg["models"]["drnn_lr"]
    M_SAMP    = cfg["models"]["drnn_meta_samples"]
    M_STEPS   = cfg["models"]["drnn_meta_steps"]

    # 1) Get W for the target peer
    W = get_W_for_peer(args.peer_id, args.mode, cfg, args.W_hex)
    print("[*] peer_id:", args.peer_id)
    print("[*] n:", args.n)
    print("[*] W.sha256:", hashlib.sha256(W).hexdigest())

    # 2) Ensure adapter exists and is aligned (auto-retrain if stale)
    store = TicketedAdapters(dirpath=args.adapters)
    pre   = Pretrainer(store, OBF_LEN, MASTERLEN, HIDDEN, LR)
    pre.schedule_window(owner_id="CLI", peer_id=args.peer_id, W_peer=W,
                        start_n=args.n, window=max(1, args.window), meta=(M_SAMP, M_STEPS))

    # 3) Load adapter for exactly this ticket
    def ctor():
        d = Sess2MasterDRNN(hidden_dim=HIDDEN, lr=LR)
        d.set_context(peer_id=args.peer_id, W_bytes=W, target_len_chars=MASTERLEN)
        return d
    drnn = store.load(args.peer_id, args.n, ctor=ctor)

    # 4) Compute obf and targets, present result
    obf = obf_ticket(W, args.n, OBF_LEN)
    target = master_seedpath(W, args.peer_id, MASTERLEN)
    pred = drnn.predict(obf, out_len=MASTERLEN)

    ok = (pred == target)
    print("    obf:", obf)
    print("target :", target)
    print("pred   :", pred)
    print("result :", "OK ✅" if ok else "MISMATCH ❌")

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
