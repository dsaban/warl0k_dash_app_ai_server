#!/usr/bin/env python3
"""
Verify end-to-end: pretraining -> saved adapter mapping -> live session proof for A <-> B.

Usage:
  python3 tools/verify_peer_flow.py --A device-A --B device-B --n 1 \
      --adapters-A .adapters_A --adapters-B .adapters_B --mode online

Options:
  --mode online|offline (online queries hub for W; offline requires --W-A --W-B hex strings)
  --W-A, --W-B (hex) used in offline mode
"""
import argparse, os, ssl, socket, yaml, hashlib, binascii, sys
from warlok.storage import TicketedAdapters
from warlok.pretrain import obf_ticket, master_seedpath
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.peer_core import PeerCore
from warlok.crypto import gen_x25519_keypair, pub_bytes_x25519, hmac_sha256, hexlify, unhex
from warlok.net import send_msg, recv_msg

def load_cfg():
    with open("config.yaml","r") as f:
        return yaml.safe_load(f)

def hub_rpc(req, cfg):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile="ca.crt")
    if os.path.exists("peer.crt") and os.path.exists("peer.key"):
        ctx.load_cert_chain(certfile="peer.crt", keyfile="peer.key")
    s = socket.socket()
    tls_conn = ctx.wrap_socket(s, server_hostname="hub")
    tls_conn.connect((cfg["general"]["host"], cfg["general"]["hub_port"]))
    send_msg(tls_conn, req)
    res = recv_msg(tls_conn)
    tls_conn.close()
    return res

def get_W(peer_id, mode, cfg, w_hex_cli=None):
    if mode == "offline":
        if not w_hex_cli:
            raise SystemExit("--mode offline requires --W-hex for W_A/W_B")
        return bytes.fromhex(w_hex_cli)
    resp = hub_rpc({"cmd":"get_seed2master_vec","target_device_id":peer_id}, cfg)
    if resp.get("status") != "ok":
        raise RuntimeError(f"Hub response error for {peer_id}: {resp}")
    return bytes.fromhex(resp["W_hex"])

def load_adapter(adapters_dir, peer_id, n, W_bytes, model_cfg):
    store = TicketedAdapters(dirpath=adapters_dir)
    ctor = lambda: Sess2MasterDRNN(hidden_dim=model_cfg["drnn_hidden_dim"], lr=model_cfg["drnn_lr"])
    # Ensure ctor sets context
    def ctor_with_ctx():
        d = ctor()
        d.set_context(peer_id=peer_id, W_bytes=W_bytes, target_len_chars=model_cfg["master_len_chars"])
        return d
    if not store.exists(peer_id, n):
        raise FileNotFoundError(f"Adapter missing: {adapters_dir}/{peer_id}/n_{n:08d}.pkl")
    drnn = store.load(peer_id, n, ctor=ctor_with_ctx)
    return drnn, store

def compute_k_session_and_obf(my_priv, their_pub_bytes, my_id, their_id, policy_id, counter):
    pc = PeerCore(my_id)
    k = pc.derive_k_session(my_priv, their_pub_bytes, my_id, their_id, policy_id, counter, os.urandom(16))
    obf = pc.obf_from_k(k, cfg["session"]["obf_len"])
    return k, obf

def derive_k_session_from_challenge(my_priv, their_pub_bytes, my_id, their_id, policy_id, counter, challenge):
    pc = PeerCore(my_id)
    k = pc.derive_k_session(my_priv, their_pub_bytes, my_id, their_id, policy_id, counter, challenge)
    obf = pc.obf_from_k(k, cfg["session"]["obf_len"])
    return k, obf

def hexstr(b): return binascii.hexlify(b).decode()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--A", required=True)
    ap.add_argument("--B", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--adapters-A", default=".adapters_A")
    ap.add_argument("--adapters-B", default=".adapters_B")
    ap.add_argument("--mode", choices=["online","offline"], default="online")
    ap.add_argument("--W-A", default=None, help="W hex for device A (offline mode)")
    ap.add_argument("--W-B", default=None, help="W hex for device B (offline mode)")
    args = ap.parse_args()

    cfg = load_cfg()
    POLICY = cfg["session"]["policy_id"]
    n = args.n

    try:
        # 1) Obtain W_A, W_B
        W_A = get_W(args.A, args.mode, cfg, args.W_A)
        W_B = get_W(args.B, args.mode, cfg, args.W_B)
        print("[*] W_A sha256:", hashlib.sha256(W_A).hexdigest())
        print("[*] W_B sha256:", hashlib.sha256(W_B).hexdigest())

        # 2) Ensure adapters exist & predict mapping (A has adapter for B, B has adapter for A)
        model_cfg = {
            "drnn_hidden_dim": cfg["models"]["drnn_hidden_dim"],
            "drnn_lr": cfg["models"]["drnn_lr"],
            "master_len_chars": cfg["models"]["master_len_chars"]
        }

        # A must have adapter mapping obf_ticket(W_B, n) -> master_B
        drnn_A_for_B, storeA = load_adapter(args.adapters_A, args.B, n, W_B, model_cfg)
        OBf_A = obf_ticket(W_B, n, cfg["session"]["obf_len"])
        master_B_expected = master_seedpath(W_B, args.B, cfg["models"]["master_len_chars"])
        predA = drnn_A_for_B.predict(OBf_A, out_len=cfg["models"]["master_len_chars"])
        print(f"[*] A's adapter for B: obf={OBf_A} master_expected={master_B_expected} master_pred={predA}")
        if predA != master_B_expected:
            raise RuntimeError("A's adapter prediction mismatch for B")

        # B must have adapter mapping obf_ticket(W_A, n) -> master_A
        drnn_B_for_A, storeB = load_adapter(args.adapters_B, args.A, n, W_A, model_cfg)
        OBf_B = obf_ticket(W_A, n, cfg["session"]["obf_len"])
        master_A_expected = master_seedpath(W_A, args.A, cfg["models"]["master_len_chars"])
        predB = drnn_B_for_A.predict(OBf_B, out_len=cfg["models"]["master_len_chars"])
        print(f"[*] B's adapter for A: obf={OBf_B} master_expected={master_A_expected} master_pred={predB}")
        if predB != master_A_expected:
            raise RuntimeError("B's adapter prediction mismatch for A")

        # 3) Simulate an authenticated session:
        #  - A initiates: we generate ephemeral keys for A and B (simulate real handshake)
        a_priv, a_pub = gen_x25519_keypair()
        b_priv, b_pub = gen_x25519_keypair()
        a_pub_bytes = pub_bytes_x25519(a_pub)
        b_pub_bytes = pub_bytes_x25519(b_pub)

        # B creates a challengeB that will be sent to A in HELLO-ACK (simulate)
        challengeB = os.urandom(16)
        # A will derive k_session using challengeB; B must derive the same
        k_A, obf_real_A = derive_k_session_from_challenge(a_priv, b_pub_bytes, args.A, args.B, POLICY, n, challengeB)
        k_B, obf_real_B = derive_k_session_from_challenge(b_priv, a_pub_bytes, args.B, args.A, POLICY, n, challengeB)

        if k_A != k_B:
            raise RuntimeError("Derived session keys differ (ECDH/HKDF mismatch)")
        if obf_real_A != obf_real_B:
            raise RuntimeError("Session obf mismatch")

        print("[*] Derived same k_session and obf_real OK")

        # 4) A computes Tag_A using W_A, obf_real_A, n and transcript
        transcript = f"{args.A}|{args.B}|{POLICY}|{n}|{hexlify(challengeB)}".encode()
        K_tag = hmac_sha256(W_A, b"sessK", n.to_bytes(8,"big"), obf_real_A.encode())
        tagA = hexlify(hmac_sha256(K_tag, b"TAG", transcript))
        print("[*] A computed tagA:", tagA)

        # 5) A builds envelope (AEAD) with plaintext demo and sends to B (we simulate)
        pcA = PeerCore(args.A)
        plaintext = cfg["general"]["demo_plaintext"].encode()
        env = pcA.build_envelope(args.A, args.B, a_priv, b_pub_bytes, POLICY, n, challengeB, plaintext)
        meta = {"n": n, "tagA": tagA, "obf_len": cfg["session"]["obf_len"]}

        # 6) B side verification (simulate peer_b handler behavior):
        # 6.a) Confirm adapter mapping A->master_A
        # already done above (predB == master_A_expected)

        # 6.b) Verify tag using W_A and transcript and obf_real_B
        K_tag_b = hmac_sha256(W_A, b"sessK", n.to_bytes(8,"big"), obf_real_B.encode())
        calc_tag_b = hexlify(hmac_sha256(K_tag_b, b"TAG", transcript))
        print("[*] B recomputed tag:", calc_tag_b)
        if calc_tag_b != tagA:
            raise RuntimeError("Tag mismatch at B")

        # 6.c) Verify envelope decrypt
        pcB = PeerCore(args.B)
        ok, pt = pcB.verify_envelope(env, args.B, args.A, b_priv, a_pub_bytes)
        print("[*] AEAD decrypt ok:", ok, "plaintext:", pt.decode())
        if not ok:
            raise RuntimeError("AEAD decrypt failed at B")

        # Success
        print("=== VERIFY_PEER_FLOW: SUCCESS: A and B adapters existed and session validated ===")
        sys.exit(0)

    except Exception as e:
        print("=== VERIFY_PEER_FLOW: FAILURE ===")
        import traceback; traceback.print_exc()
        sys.exit(2)
