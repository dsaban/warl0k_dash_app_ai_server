import uuid
from pprint import pprint

from common.crypto import CryptoBox
from common.protocol import SecureChannel
from peers.peer_near import NearPeer
from peers.peer_far import FarPeer
from attacks.injector import (
    attack_prompt_injection,
    attack_tool_swap_to_unauthorized,
    attack_tamper_args,
    attack_delay,
)

def main():
    sid = f"SID-{uuid.uuid4().hex[:8]}"
    print(f"\n=== WARL0K DEMO SID={sid} ===\n")

    # Shared tunnel key for the demo (in real WARL0K this is reconstructed/ephemeral per session)
    box = CryptoBox.new()
    channel = SecureChannel(crypto=box)

    near = NearPeer(sid=sid, channel=channel)
    far = FarPeer(sid=sid, channel=channel)

    assert near.login_and_auth("demo", "demo"), "auth failed"
    print("[AUTH] granted\n")

    # Baseline (no attack): Task 1
    print("=== BASELINE: Task 1 (no attack) ===")
    t1 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 1: read DB and summarize", inject=None)
    pprint(t1)

    # Baseline (no attack): Task 2
    print("\n=== BASELINE: Task 2 (no attack) ===")
    t2 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 2: write validated result row", inject=None)
    pprint(t2)

    # Attack 1: prompt injection
    print("\n=== ATTACK 1: prompt injection into tool call ===")
    a1 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 2: write validated result row", inject=attack_prompt_injection)
    pprint(a1)

    # Attack 2: tool swap to unauthorized exec
    print("\n=== ATTACK 2: tool swap to unauthorized 'exec' ===")
    a2 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 2: write validated result row", inject=attack_tool_swap_to_unauthorized)
    pprint(a2)

    # Attack 3: tamper args (attempt to write pwn)
    print("\n=== ATTACK 3: tamper write args ===")
    a3 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 2: write validated result row", inject=attack_tamper_args)
    pprint(a3)

    # Attack 4: delay to violate timing window (done by injecting delay then returning same msg)
    def delayed(m):
        return attack_delay(m, seconds=3.5)

    print("\n=== ATTACK 4: delay (timing skew) ===")
    a4 = near.run_task_flow(far.recv_and_execute, task_prompt="Task 1: read DB and summarize", inject=delayed)
    pprint(a4)

    print("\n=== DONE ===\n")

if __name__ == "__main__":
    main()
