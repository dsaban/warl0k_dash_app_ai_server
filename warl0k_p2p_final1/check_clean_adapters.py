#!/usr/bin/env python3
import argparse, shutil, os, sys

def main():
    ap=argparse.ArgumentParser(description="Remove local ticketed adapters directory.")
    ap.add_argument("--path", default=".adapters_A", help="e.g., .adapters_A or .adapters_B")
    args=ap.parse_args()
    if os.path.exists(args.path):
        shutil.rmtree(args.path)
        print("Removed", args.path)
    else:
        print("Not found:", args.path)
    sys.exit(0)

if __name__=="__main__":
    main()
