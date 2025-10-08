#!/usr/bin/env python3
import argparse, yaml, random
from check_10_sessions import run_session, InMemHub
from warlok.storage import TicketedAdapters

def load_cfg():
    with open("config.yaml","r") as f: return yaml.safe_load(f)

def main():
    ap=argparse.ArgumentParser(description="Simulate N sessions in-process (no sockets).")
    ap.add_argument("--N", type=int, default=10)
    args=ap.parse_args()

    cfg = load_cfg()
    random.seed(99)
    hub = InMemHub(); hub.enroll("device-A"); hub.enroll("device-B")
    storeA = TicketedAdapters(".adapters_A"); storeB = TicketedAdapters(".adapters_B")

    succ=0
    for i in range(args.N):
        ok = run_session(hub, storeA, storeB, n=cfg["session"]["counter_start"]+i)
        succ += 1 if ok else 0
        print(f"session {i+1:02d}: {'OK' if ok else 'FAIL'}")
    print(f"\nSummary: {succ}/{args.N} successful sessions ({(succ/args.N)*100:.1f}% success rate)")

if __name__=="__main__":
    main()
