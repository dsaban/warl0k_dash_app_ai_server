# demo_model.py
import os, binascii, threading, socket, time
from warl0k.hub import Hub
from warl0k.peer import Peer
from warl0k.crypto_utils import dh_generate_keypair
from warl0k.socket_net import send_msg, recv_msg
from model.model_wrapper import SessionModelAgent

HOST = "127.0.0.1"
PORT = 50666

def run_server_peer(hub, ready_evt):
    B = Peer("device-B", hub)
    # create a human-readable device identity (8 hex chars)
    B_identity = binascii.hexlify(os.urandom(4)).decode()
    B.enroll_with_hub(identity_string=B_identity)
    print(f"[B] enrolled with identity: {B.device_identity_string}")

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(1)
    ready_evt.set()
    conn, addr = srv.accept()
    with conn:
        hello = recv_msg(conn)
        assert hello["type"] == "HELLO"
        device_idA = hello["device_id"]
        A_pub = int(hello["A_pub"])
        ctr = int(hello["ctr"])
        print(f"[B] HELLO from {device_idA} ctr={ctr}")

        b_priv, b_pub = dh_generate_keypair()
        challengeB = os.urandom(16)
        # send HELLO-ACK
        send_msg(conn, {"type": "HELLO-ACK", "device_id": B.device_id, "B_pub": str(b_pub), "challengeB": binascii.hexlify(challengeB).decode()})

        env_msg = recv_msg(conn)
        assert env_msg["type"] == "ENVELOPE"
        env = env_msg["body"]

        # Build model agent and attempt prediction-based verification
        # The B peer derives K_session locally and then maps it to obf string; but we need both contributions.
        # For demo, sender (A) will have included its obf string in meta for B to train/predict.
        meta = env_msg.get("meta", {})
        obfA = meta.get("obfA")
        # B computes its local session K and obfB too (to train its model)
        # For verification, B will:
        # - Derive K_session (as in verify_envelope check); if HMAC verifies, accept.
        # - Additionally, train its local model with obfA -> predicted identity and compare to stored identity.
        # For simplicity we do both checks.
        ok_crypto = B.verify_envelope(env, my_priv=b_priv, peer_pub=A_pub, contrib_self=B.contrib(), contrib_peer=binascii.unhexlify(meta.get("contribA_hex")))
        print(f"[B] crypto verify: {'ACCEPT' if ok_crypto else 'REJECT'}")

        # Model-based check: train model for a few epochs and predict A's identity (toy demo)
        modelB = SessionModelAgent()
        if obfA:
            print("[B] training local RNN on observed obf_A -> identity_A (toy demo)...")
            # In a real setting B wouldn't know A's identity string, but for demo we simulate that
            # A provided identity label in meta (only for demo). In production you'd have a different protocol.
            identityA = meta.get("identityA")
            modelB.train_on_pair(obfA, identityA, epochs=800)
            pred = modelB.predict_identity(obfA)
            print(f"[B] predicted identity for A (model): {pred} (expected {identityA})")

        send_msg(conn, {"type": "RESULT", "crypto_ok": ok_crypto})
    srv.close()

def run_client_peer(hub):
    A = Peer("device-A", hub)
    A_identity = binascii.hexlify(os.urandom(4)).decode()
    A.enroll_with_hub(identity_string=A_identity)
    print(f"[A] enrolled with identity: {A.device_identity_string}")

    time.sleep(0.2)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print("[A] connected to B")

    a_priv, a_pub = dh_generate_keypair()
    policy_id = "PAYMENT:limit=100"
    counter = 1
    # send HELLO
    send_msg(s, {"type": "HELLO", "device_id": A.device_id, "A_pub": str(a_pub), "policy_id": policy_id, "ctr": counter})

    ack = recv_msg(s)
    assert ack["type"] == "HELLO-ACK"
    b_pub = int(ack["B_pub"])
    challengeB = binascii.unhexlify(ack["challengeB"])

    # derive K_session locally
    contribA = A.contrib()
    contribB_placeholder = b"\x00" * 32  # in this toy demo we don't need exact contrib from B
    k_sess = A.derive_session_key(peer_device_id="device-B", my_priv=a_priv, peer_pub=b_pub,
                                  policy_id=policy_id, counter=counter, challenge=challengeB,
                                  contrib_self=contribA, contrib_peer=contribB_placeholder)

    # create obf string from k_sess (hex truncated)
    from model.model_wrapper import SessionModelAgent
    agentA = SessionModelAgent()
    obfA = agentA.k_to_obf_string(k_sess, length=8)
    print(f"[A] obf string for this session: {obfA}")

    # For demo only: send the obf and also a "label" for identity (insecure; only demo)
    env = A.build_envelope(peer_pub=b_pub, my_priv=a_priv, peer_device_id="device-B",
                           policy_id=policy_id, counter=counter, challenge=challengeB,
                           plaintext=b"pay 50 tokens to B", contrib_self=contribA, contrib_peer=contribB_placeholder)
    # include contribA and obfA in meta for demo server to use
    send_msg(s, {"type": "ENVELOPE", "body": env, "meta": {"contribA_hex": binascii.hexlify(contribA).decode(), "obfA": obfA, "identityA": A.device_identity_string}})

    res = recv_msg(s)
    print("[A] received RESULT from B:", res)
    print("[A] B crypto result:", res["crypto_ok"])
    s.close()

def main():
    hub = Hub()
    ready = threading.Event()
    th = threading.Thread(target=run_server_peer, args=(hub, ready), daemon=True)
    th.start()
    ready.wait()
    run_client_peer(hub)
    th.join(timeout=1.0)

if __name__ == "__main__":
    main()
