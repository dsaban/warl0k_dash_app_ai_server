#!/usr/bin/env python3
import argparse, os, ssl, socket, yaml, sys
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
    send_msg(tls, req); res = recv_msg(tls); tls.close(); return res

def main():
    ap=argparse.ArgumentParser(description="Enroll device_id at hub.")
    ap.add_argument("--device-id", required=True)
    args=ap.parse_args()
    cfg=load_cfg()
    r = hub_rpc({"cmd":"enroll","device_id":args.device_id}, cfg)
    print(r)
    sys.exit(0 if r.get("status")=="ok" else 1)

if __name__=="__main__":
    main()
