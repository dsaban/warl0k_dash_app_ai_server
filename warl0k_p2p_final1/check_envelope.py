#!/usr/bin/env python3
import argparse, yaml, binascii, sys
from warlok.peer_core import PeerCore
from warlok.crypto import unhex

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser(description="Offline verify/decrypt envelope with given keys.")
    ap.add_argument("--my-id", required=True)
    ap.add_argument("--their-id", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ctr", type=int, required=True)
    ap.add_argument("--challenge-hex", required=True)
    ap.add_argument("--nonce-hex", required=True)
    ap.add_argument("--ct-hex", required=True)
    ap.add_argument("--priv-raw-hex", help="Raw 32-byte X25519 private key (hex)", required=True)
    ap.add_argument("--peer-pub-raw-hex", help="Raw 32-byte X25519 public key (hex)", required=True)
    args=ap.parse_args()

    env = {"header":{"from":args.their_id,"to":args.my_id,"policy_id":args.policy,
                     "ctr":args.ctr,"challenge":args.challenge_hex},
           "nonce":args.nonce_hex, "ciphertext":args.ct_hex}

    pc = PeerCore(args.my_id)
    ok, pt = pc.verify_envelope(env, args.my_id, args.their_id,
                                unhex(args.priv_raw_hex), unhex(args.peer_pub_raw_hex))
    print("verify:", ok)
    if ok: print("plaintext:", pt.decode(errors="replace"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
