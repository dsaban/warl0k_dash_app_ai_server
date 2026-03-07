# warlok_gw/_gru_infer_py.py
# Pure Python / NumPy fallback — identical API to Cython gru_infer.pyx
from __future__ import annotations
from collections import namedtuple
import numpy as np

SEQ_LEN   = 48
IN_DIM    = 9
H_DIM     = 64
N_CLS     = 6
MAX_DT    = 200.0
ATK_LABELS = ("none","reorder","drop","replay","timewarp","splice")

Verdict = namedtuple("Verdict",
    ["block","attack_class","confidence","scores","step_count"])

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

class SessionBuffer:
    def __init__(self, session_id=""):
        self.X          = np.zeros((SEQ_LEN, IN_DIM), dtype=np.float32)
        self.n          = 0
        self.session_id = session_id
    def reset(self):
        self.X[:] = 0.0; self.n = 0; self.session_id = ""

class GRUInferEngine:
    def __init__(self, weights_path, threshold=0.80, min_steps=4):
        self.threshold = threshold
        self.min_steps = min_steps
        raw = np.load(weights_path, allow_pickle=False)
        self._p = {k: raw[k] for k in
                   ("Wz","Uz","bz","Wr","Ur","br","Wh","Uh","bh","Wc","bc")}

    def step(self, buf, msg):
        t = buf.n
        if t >= SEQ_LEN:                     # sliding window
            buf.X[:-1] = buf.X[1:]
            t = SEQ_LEN - 1; buf.n = t

        op_id = {"READ":0,"WRITE":1}.get(msg.get("op","READ"), 2)
        row   = buf.X[t]
        row[:] = 0.0
        row[0] = float(np.clip(msg.get("meas",0.0), 0.0, 1.0))
        row[1] = float(np.clip(msg.get("dt_ms",0) / MAX_DT, 0.0, 5.0))
        row[2] = msg.get("step_idx", 0) / 47.0
        row[3] = msg.get("ctr", 0) / float(SEQ_LEN)
        row[4] = 1.0 if msg.get("decision","ACCEPT") == "ACCEPT" else 0.0
        row[5 + op_id] = 1.0
        row[8] = t / 47.0
        buf.n += 1

        n = buf.n
        if n < self.min_steps:
            return Verdict(False,"none",0.0,{l:0.0 for l in ATK_LABELS},n)

        p = self._p
        h = np.zeros(H_DIM, dtype=np.float32)
        s = np.zeros(H_DIM, dtype=np.float32)
        for i in range(n):
            x = buf.X[i]
            z = _sigmoid(x @ p["Wz"].T + h @ p["Uz"].T + p["bz"])
            r = _sigmoid(x @ p["Wr"].T + h @ p["Ur"].T + p["br"])
            g = np.tanh( x @ p["Wh"].T + (r*h) @ p["Uh"].T + p["bh"])
            h = (1-z)*h + z*g
            s += h
        ctx    = s / n
        logits = ctx @ p["Wc"].T + p["bc"]
        logits -= logits.max()
        probs   = np.exp(logits); probs /= probs.sum()

        best   = int(np.argmax(probs))
        label  = ATK_LABELS[best]
        conf   = float(probs[best])
        scores = {ATK_LABELS[i]: float(probs[i]) for i in range(N_CLS)}
        block  = (label != "none") and (conf >= self.threshold)
        return Verdict(block, label, conf, scores, n)

    def score_trace(self, trace):
        buf = SessionBuffer(); v = None
        for msg in trace:
            v = self.step(buf, msg)
        return v
