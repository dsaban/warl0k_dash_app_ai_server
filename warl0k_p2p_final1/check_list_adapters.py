#!/usr/bin/env python3
import argparse, os, pickle, time, pathlib, hashlib

def main():
    ap = argparse.ArgumentParser(description="List stored ticket adapters and training metadata.")
    ap.add_argument("--dir", default=".adapters_A", help="Adapters directory (.adapters_A or .adapters_B)")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if not root.exists():
        print("No adapters dir:", root); return

    rows = []
    for peer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        peer_id = peer_dir.name
        for pkl in sorted(peer_dir.glob("n_*.pkl")):
            with open(pkl, "rb") as f:
                s = pickle.load(f)
            n = int(pkl.stem.split("_")[1])
            W = s.get("W", b"")
            Wsha = hashlib.sha256(W).hexdigest() if W else "-"
            lti = s.get("last_training_info") or {}
            saved = s.get("saved_at_epoch_ms")
            saved_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved)) if saved else "-"
            rows.append([
                peer_id, n, Wsha[:8],
                lti.get("epochs_run"), lti.get("early_stopped"),
                lti.get("meta_used"), saved_ts
            ])

    if not rows:
        print("No adapters found in", root); return

    print(f"{'peer':<16} {'n':>6} {'Wsha8':<10} {'epochs':>6} {'early':>7} {'meta':>6} {'saved_at'}")
    print("-"*70)
    for r in rows:
        print(f"{r[0]:<16} {r[1]:>6} {r[2]:<10} {str(r[3]):>6} {str(r[4]):>7} {str(r[5]):>6} {r[6]}")

if __name__ == "__main__":
    main()
