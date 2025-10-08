#!/usr/bin/env python3
import os, ssl, socket, yaml
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def main():
    cfg = load_cfg()
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls = ctx.wrap_socket(s, server_hostname="hub")
    tls.connect((cfg["general"]["host"], cfg["general"]["hub_port"]))
    send_msg(tls, {"cmd":"enroll","device_id":"ping-device"})
    print(recv_msg(tls))
    tls.close()

if __name__=="__main__":
    main()
