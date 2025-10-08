#!/usr/bin/env python3
import argparse, json, pathlib, os

def main():
    ap = argparse.ArgumentParser(description="List JSON audits for ticket adapters.")
    ap.add_argument("--dir", default=".adapters_A", help="Adapters dir (.adapters_A or .adapters_B)")
    ap.add_argument("--peer-id", default=None, help="Filter by peer id")
    ap.add_argument("--n", type=int, default=None, help="Filter by ticket n")
    args = ap.parse_args()

    root = pathlib.Path(args.dir)
    if not root.exists():
        print("No adapters dir:", root); return

    audits = []
    for peer_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if args.peer_id and peer_dir.name != args.peer_id:
            continue
        for jf in sorted(peer_dir.glob("n_*.json")):
            if args.n is not None:
                try:
                    jn = int(jf.stem.split("_")[1])
                    if jn != args.n: continue
                except Exception:
                    pass
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                audits.append((peer_dir.name, jf.name, data))
            except Exception as e:
                print("Bad JSON:", jf, e)

    if not audits:
        print("No matching audits.")
        return

    print(f"{'peer':<16} {'file':<16} {'n':>6} {'epochs':>6} {'early':>7} {'meta':>6} {'w_sha256':<16} {'target_sha256':<16}")
    print("-"*100)
    for peer_id, fname, a in audits:
        tr = a.get("training", {})
        print(f"{peer_id:<16} {fname:<16} {str(a.get('n')):>6} "
              f"{str(tr.get('epochs_run')):>6} {str(tr.get('early_stopped')):>7} {str(tr.get('meta_used')):>6} "
              f"{str(a.get('w_sha256','-'))[:16]:<16} {str(a.get('target_sha256','-'))[:16]:<16}")

if __name__ == "__main__":
    main()
