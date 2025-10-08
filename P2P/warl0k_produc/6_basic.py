"""
Stage-6 DRNN convergence test
- Generates a synthetic 'seed-path' master from a random W and device_id
- Generates a random obf string
- Trains the Sess→Master DRNN to predict the master from obf
- Reports success, epochs to early-stop, and a few diagnostics

Run:  python test_stage6_basic.py
"""

import os, random, binascii
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.crypto import hmac_sha256

# ---------- Tunables (tighten/loosen as needed) ----------
MASTER_LEN = 16   # 32 hex chars = 128-bit (try 16 first if you want quick wins)
OBF_LEN    = 16   # conditioning size (12–20 is a good range)
EPOCHS     = 2000  # cap (DRNN uses early-stop)
HIDDEN     = 48   # DRNN hidden dim
LR         = 0.005 # learning rate
TRIALS     = 5   # number of independent attempts
# ---------------------------------------------------------

def hexstr(b): return binascii.hexlify(b).decode()

def make_master_from_W(W: bytes, device_id: str, out_len: int) -> str:
    """Same rule as seed-path model: HMAC(W, 'M' || device_id) truncated to hex."""
    m = hmac_sha256(W, b"M", device_id.encode())
    return hexstr(m)[:out_len]

def main():
    random.seed(1337)
    ok = 0
    epochs_sum = 0
    for t in range(1, TRIALS+1):
        # Synthetic W and device_id
        W = os.urandom(32)
        device_id = f"device-{t}"
        target = make_master_from_W(W, device_id, MASTER_LEN)

        # Random obf (what k_session→HKDF would have produced)
        obf = hexstr(os.urandom(32))[:OBF_LEN]

        # Train DRNN
        drnn = Sess2MasterDRNN(hidden_dim=HIDDEN, lr=LR)
        info = drnn.train_pair(obf, target, epochs=EPOCHS)
        pred = drnn.predict(obf, out_len=len(target))
        success = (pred == target)

        print(f"[{t:02d}] success={success}  epochs_run={info['epochs_run']}  early_stop={info['early_stopped']}")
        if not success:
            # Minimal debug on failure: show prefix match length
            prefix = 0
            for a,b in zip(pred, target):
                if a==b: prefix += 1
                else: break
            print(f"     pred={pred}\n     targ={target}\n     prefix_match={prefix}/{len(target)}")
        else:
            ok += 1
            epochs_sum += info["epochs_run"]

    print(f"\nSummary: {ok}/{TRIALS} successful mappings "
          f"({(ok/TRIALS)*100:.1f}%). "
          f"Avg epochs to stop (successful only): {epochs_sum/max(ok,1):.1f}")

if __name__ == "__main__":
    main()
