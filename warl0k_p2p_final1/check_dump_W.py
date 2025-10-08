#!/usr/bin/env python3
import argparse, os, ssl, socket, yaml, sys, hashlib
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def hub_rpc(req, cfg):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls = ctx.wrap_socket(s, server_hostname="hub")
    tls.connect((cfg["general"]["host"], cfg["general"]["hub_port"]))
    send_msg(tls, req); res=recv_msg(tls); tls.close(); return res

def main():
    ap=argparse.ArgumentParser(description="Dump W-hex for device_id from hub.")
    ap.add_argument("--device-id", required=True)
    args=ap.parse_args()
    cfg=load_cfg()
    r = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":args.device_id}, cfg)
    if r.get("status")!="ok":
        print("Hub error:", r); sys.exit(1)
    W_hex = r["W_hex"]; print("W_hex:", W_hex)
    print("W.sha256:", hashlib.sha256(bytes.fromhex(W_hex)).hexdigest())
    sys.exit(0)

if __name__=="__main__":
    main()
