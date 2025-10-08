#!/usr/bin/env python3
import argparse, os, ssl, socket, yaml, binascii, hashlib, sys
from warlok.crypto import hmac_sha256
from warlok.peer_core import PeerCore
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def hub_rpc(req, host, port):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls = ctx.wrap_socket(s, server_hostname="hub")
    tls.connect((host, port))
    send_msg(tls, req)
    res = recv_msg(tls); tls.close(); return res

def get_W(device_id, cfg):
    r = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":device_id},
                cfg["general"]["host"], cfg["general"]["hub_port"])
    if r.get("status")!="ok": print("Hub error:", r); sys.exit(2)
    return bytes.fromhex(r["W_hex"])

def main():
    ap = argparse.ArgumentParser(description="Recompute/verify session tag for A->B.")
    ap.add_argument("--A", default="device-A"); ap.add_argument("--B", default="device-B")
    ap.add_argument("--policy", default=None); ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--challengeB-hex", required=True)
    ap.add_argument("--obf-real", help="obf derived from k_session (hex string)", required=True)
    ap.add_argument("--tag-hex", help="tag to verify (hex string)", required=True)
    args=ap.parse_args()

    cfg=load_cfg()
    policy = args.policy or cfg["session"]["policy_id"]
    W_A = get_W(args.A, cfg)

    transcript = f"{args.A}|{args.B}|{policy}|{args.n}|{args.challengeB_hex}".encode()
    K_tag = hmac_sha256(W_A, b"sessK", args.n.to_bytes(8,"big"), args.obf_real.encode())
    calc = binascii.hexlify(hmac_sha256(K_tag, b"TAG", transcript)).decode()

    print("W_A.sha256:", hashlib.sha256(W_A).hexdigest())
    print("tag input  :", transcript)
    print("calc tag   :", calc)
    print("given tag  :", args.tag_hex)
    print("result     :", "OK ✅" if calc==args.tag_hex else "MISMATCH ❌")
    sys.exit(0 if calc==args.tag_hex else 1)

if __name__ == "__main__":
    main()
