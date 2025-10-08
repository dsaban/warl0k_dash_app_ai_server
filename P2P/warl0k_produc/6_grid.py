"""
Stage-6 DRNN grid search
- Sweeps over (MASTER_LEN, OBF_LEN, EPOCHS) combos
- Runs a few trials per combo
- Prints a compact scoreboard to pick stable parameters

Run: python test_stage6_grid.py
"""

import os, random, binascii
from statistics import mean
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.crypto import hmac_sha256

def hexstr(b): return binascii.hexlify(b).decode()
def make_master_from_W(W: bytes, device_id: str, out_len: int) -> str:
    m = hmac_sha256(W, b"M", device_id.encode()); return hexstr(m)[:out_len]

# Search space (trim or expand)
MASTER_LEN_CHOICES = [16, 24, 32]  # 64, 96, 128-bit
OBF_LEN_CHOICES    = [12, 16, 20]
EPOCHS_CHOICES     = [200, 300, 500]
TRIALS_PER_COMBO   = 6
HIDDEN             = 48
LR                 = 0.05

def run_trial(master_len, obf_len, epochs):
    W = os.urandom(32); device_id = "device-X"
    target = make_master_from_W(W, device_id, master_len)
    obf = hexstr(os.urandom(32))[:obf_len]
    drnn = Sess2MasterDRNN(hidden_dim=HIDDEN, lr=LR)
    info = drnn.train_pair(obf, target, epochs=epochs)
    pred = drnn.predict(obf, out_len=len(target))
    success = (pred == target)
    return success, info["epochs_run"]

def main():
    random.seed(7)
    rows = []
    for m in MASTER_LEN_CHOICES:
        for o in OBF_LEN_CHOICES:
            for e in EPOCHS_CHOICES:
                succ = []; epc = []
                for _ in range(TRIALS_PER_COMBO):
                    s, er = run_trial(m, o, e)
                    succ.append(1 if s else 0)
                    epc.append(er)
                rate = sum(succ)/len(succ)
                avg_epochs = mean([er for s,er in zip(succ,epc) if s]) if rate>0 else float('inf')
                rows.append((m,o,e,rate,avg_epochs))

    # Print scoreboard sorted by success rate desc, then avg epochs asc
    rows.sort(key=lambda r: (-r[3], r[4]))
    print("MASTER  OBF  EPOCHS | SUCCESS | AVG_EPOCHS (successful runs)")
    print("------- ---- ------- | ------- | --------------------------")
    for m,o,e,rate,avg in rows:
        avg_s = f"{avg:.1f}" if avg != float('inf') else "-"
        print(f"{m:>6} {o:>4} {e:>7} |  {rate*100:5.1f}% | {avg_s}")

if __name__ == "__main__":
    main()
