"""
Runnable demo that exercises enroll + peer-to-peer sessions.
Usage:
    python -m warlok.demo
"""

from warl0k_cloud_demo_app_multi_client_server_dash.P2P.warl0k_p2p.hub import Hub
from warl0k_cloud_demo_app_multi_client_server_dash.P2P.warl0k_p2p.peer import Peer
from warl0k_cloud_demo_app_multi_client_server_dash.P2P.warl0k_p2p.crypto_utils import dh_generate_keypair
# from .peer import Peer
# from .crypto_utils import dh_generate_keypair
import os

def run_demo():
    print("\n=== WARL0K P2P Demo (Hub for bootstrap only; peer sessions are hub-free) ===\n")

    hub = Hub()
    print("Hub initialized.\n")

    # Enroll two peers
    A = Peer("device-A", hub); A.enroll_with_hub()
    B = Peer("device-B", hub); B.enroll_with_hub()
    print("Enrollment complete.")
    print("  A.device_id =", A.device_id)
    print("  B.device_id =", B.device_id)
    print("  (master_seed values are kept secret on-device)\n")

    # Session #1: A -> B
    policy_id = "PAYMENT:limit=100"
    counter = 1
    a_priv, a_pub = dh_generate_keypair()
    b_priv, b_pub = dh_generate_keypair()
    contribA = A.contrib()
    contribB = B.contrib()
    challengeB = os.urandom(16)

    print(f"Session #1: A -> B (policy={policy_id}, counter={counter})")
    envelope = A.build_envelope(peer_pub=b_pub, my_priv=a_priv, peer_device_id=B.device_id,
                                policy_id=policy_id, counter=counter, challenge=challengeB,
                                plaintext=b"pay 50 tokens to B", contrib_self=contribA, contrib_peer=contribB)
    print("  A -> B envelope header:", envelope["header"])

    ok = B.verify_envelope(envelope, my_priv=b_priv, peer_pub=a_pub, contrib_self=contribB, contrib_peer=contribA)
    print("  B verification result:", "ACCEPT" if ok else "REJECT")

    # Replay test
    print("\nReplay test: resend the same envelope (expect REJECT)")
    ok2 = B.verify_envelope(envelope, my_priv=b_priv, peer_pub=a_pub, contrib_self=contribB, contrib_peer=contribA)
    print("  B verification result:", "ACCEPT" if ok2 else "REJECT")

    # Session #2
    counter = 2
    print(f"\nSession #2: A -> B (policy={policy_id}, counter={counter})")
    a_priv2, a_pub2 = dh_generate_keypair()
    b_priv2, b_pub2 = dh_generate_keypair()
    challengeB2 = os.urandom(16)
    envelope2 = A.build_envelope(peer_pub=b_pub2, my_priv=a_priv2, peer_device_id=B.device_id,
                                 policy_id=policy_id, counter=counter, challenge=challengeB2,
                                 plaintext=b"pay 25 tokens to B", contrib_self=contribA, contrib_peer=contribB)
    ok3 = B.verify_envelope(envelope2, my_priv=b_priv2, peer_pub=a_pub2, contrib_self=contribB, contrib_peer=contribA)
    print("  B verification result:", "ACCEPT" if ok3 else "REJECT")

    print("\nDemo complete. Notes:")
    print(" - This demo uses HMAC for tagging for simplicity. Replace with AEAD in real deployments.")
    print(" - Consider TPM/TEE sealing for enrollment or split-seed for extra assurance on high-value devices.")

if __name__ == "__main__":
    run_demo()
