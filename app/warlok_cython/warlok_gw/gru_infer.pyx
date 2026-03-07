# warlok_gw/gru_infer.pyx
# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False
#
# WARL0K Cython GRU inference engine — gateway/sidecar checkpoint
#
#   [agentic AI agent - far cloud]
#        | outbound TLS session
#        v
#   +------------------------------------------+
#   |  WARL0K sidecar checkpoint               | <- THIS MODULE
#   |  * unwrap TLS record                     |
#   |  * call engine.step(buf, msg)            |
#   |  * Verdict.block -> drop + alert SOC     |
#   |  * Verdict.allow -> forward to asset     |
#   +------------------------------------------+
#        | inbound near-sidecar
#        v
#   [org asset - internal command executor]
#
# BUILD:  pip install cython numpy && python setup.py build_ext --inplace

import  numpy as np
cimport numpy as cnp
from libc.math   cimport expf, tanhf
from libc.string cimport memset
from collections import namedtuple

# Architecture constants — must match training
DEF SEQ_LEN = 48
DEF IN_DIM  = 9
DEF H_DIM   = 64
DEF N_CLS   = 6
DEF MAX_DT  = 200.0

ATK_LABELS = ("none", "reorder", "drop", "replay", "timewarp", "splice")

Verdict = namedtuple("Verdict", [
    "block",
    "attack_class",
    "confidence",
    "scores",
    "step_count",
])

# ---------------------------------------------------------------------------
# Inline scalar helpers
# ---------------------------------------------------------------------------

cdef inline float _sigmoid(float x) nogil:
    if x >  30.0: return 1.0
    if x < -30.0: return 0.0
    return 1.0 / (1.0 + expf(-x))

cdef inline float _clampf(float x, float lo, float hi) nogil:
    if x < lo: return lo
    if x > hi: return hi
    return x

# ---------------------------------------------------------------------------
# SessionBuffer — one per intercepted TLS session (1.7 KB each)
# ---------------------------------------------------------------------------

cdef class SessionBuffer:
    """Per-session feature accumulator. Allocate at session open, drop at close."""

    cdef public cnp.ndarray X
    cdef public int         n
    cdef public str         session_id

    def __cinit__(self, str session_id=""):
        self.X          = np.zeros((SEQ_LEN, IN_DIM), dtype=np.float32)
        self.n          = 0
        self.session_id = session_id

    cpdef void reset(self):
        self.X[:]       = 0.0
        self.n          = 0
        self.session_id = ""

# ---------------------------------------------------------------------------
# _featurise_row — pack one message into a feature row (nogil)
# Feature layout [9]:
#  [0] os_meas           [1] dt_ms/MAX_DT  [2] step/47
#  [3] ctr/48            [4] accept flag
#  [5] op==READ          [6] op==WRITE     [7] op==CONTROL
#  [8] position t/47
# ---------------------------------------------------------------------------

cdef void _featurise_row(
    float[:]  row,
    float     meas,
    int       dt_ms,
    int       step_idx,
    int       ctr,
    bint      is_accept,
    int       op_id,
    int       t,
) nogil:
    row[0] = _clampf(meas, 0.0, 1.0)
    row[1] = _clampf(<float>dt_ms / MAX_DT, 0.0, 5.0)
    row[2] = <float>step_idx / 47.0
    row[3] = <float>ctr / <float>SEQ_LEN
    row[4] = 1.0 if is_accept else 0.0
    row[5] = 1.0 if op_id == 0 else 0.0
    row[6] = 1.0 if op_id == 1 else 0.0
    row[7] = 1.0 if op_id == 2 else 0.0
    row[8] = <float>t / 47.0

# ---------------------------------------------------------------------------
# _gru_cell — single GRU time step, pure C, no Python (nogil)
# ---------------------------------------------------------------------------

cdef void _gru_cell(
    float*        h,
    const float*  x,
    const float*  Wz, const float* Uz, const float* bz,
    const float*  Wr, const float* Ur, const float* br,
    const float*  Wh, const float* Uh, const float* bh,
) nogil:
    # ALL declarations at the very top — Cython requires this inside nogil cdef
    cdef float z[H_DIM]
    cdef float r[H_DIM]
    cdef float g[H_DIM]
    cdef float h_new[H_DIM]
    cdef int   i
    cdef int   j
    cdef float az
    cdef float ar
    cdef float ag

    for i in range(H_DIM):
        az = bz[i]
        ar = br[i]
        ag = bh[i]
        for j in range(IN_DIM):
            az += Wz[i * IN_DIM + j] * x[j]
            ar += Wr[i * IN_DIM + j] * x[j]
            ag += Wh[i * IN_DIM + j] * x[j]
        for j in range(H_DIM):
            az += Uz[i * H_DIM + j] * h[j]
            ar += Ur[i * H_DIM + j] * h[j]
        z[i] = _sigmoid(az)
        r[i] = _sigmoid(ar)
        for j in range(H_DIM):
            ag += Uh[i * H_DIM + j] * (r[i] * h[j])
        g[i]     = tanhf(ag)
        h_new[i] = (1.0 - z[i]) * h[i] + z[i] * g[i]

    for i in range(H_DIM):
        h[i] = h_new[i]

# ---------------------------------------------------------------------------
# _gru_forward — full sequence forward pass, mean-pool context (nogil)
# ---------------------------------------------------------------------------

cdef void _gru_forward(
    const float[:, :] X,
    int               n_steps,
    const float[:]    Wz_v, const float[:] Uz_v, const float[:] bz_v,
    const float[:]    Wr_v, const float[:] Ur_v, const float[:] br_v,
    const float[:]    Wh_v, const float[:] Uh_v, const float[:] bh_v,
    float[:]          ctx,
) nogil:
    # ALL declarations at top — including C pointers (no mid-block cdef)
    cdef float        h[H_DIM]
    cdef float        s[H_DIM]
    cdef int          i
    cdef int          t
    cdef float        inv_t
    cdef const float* x_ptr
    cdef const float* Wz
    cdef const float* Uz
    cdef const float* bz
    cdef const float* Wr
    cdef const float* Ur
    cdef const float* br
    cdef const float* Wh
    cdef const float* Uh
    cdef const float* bh

    memset(h, 0, H_DIM * sizeof(float))
    memset(s, 0, H_DIM * sizeof(float))

    # Assign pointers after declarations (not simultaneous with declaration)
    Wz = &Wz_v[0]
    Uz = &Uz_v[0]
    bz = &bz_v[0]
    Wr = &Wr_v[0]
    Ur = &Ur_v[0]
    br = &br_v[0]
    Wh = &Wh_v[0]
    Uh = &Uh_v[0]
    bh = &bh_v[0]

    for t in range(n_steps):
        x_ptr = &X[t, 0]
        _gru_cell(h, x_ptr, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh)
        for i in range(H_DIM):
            s[i] += h[i]

    inv_t = 1.0 / <float>n_steps
    for i in range(H_DIM):
        ctx[i] = s[i] * inv_t

# ---------------------------------------------------------------------------
# _classify — linear head + softmax, return argmax class index (nogil)
# ---------------------------------------------------------------------------

cdef int _classify(
    const float*   ctx,
    const float[:] Wc_v,
    const float[:] bc_v,
    float*         probs_out,
) nogil:
    # ALL declarations at top
    cdef float        logits[N_CLS]
    cdef float        mx
    cdef float        s
    cdef int          i
    cdef int          j
    cdef int          best
    cdef const float* Wc
    cdef const float* bc

    Wc = &Wc_v[0]
    bc = &bc_v[0]

    for i in range(N_CLS):
        logits[i] = bc[i]
        for j in range(H_DIM):
            logits[i] += Wc[i * H_DIM + j] * ctx[j]

    mx = logits[0]
    for i in range(1, N_CLS):
        if logits[i] > mx:
            mx = logits[i]

    s = 0.0
    for i in range(N_CLS):
        probs_out[i] = expf(logits[i] - mx)
        s += probs_out[i]
    for i in range(N_CLS):
        probs_out[i] /= s

    best = 0
    for i in range(1, N_CLS):
        if probs_out[i] > probs_out[best]:
            best = i
    return best

# ---------------------------------------------------------------------------
# GRUInferEngine — the public class loaded by the gateway
# ---------------------------------------------------------------------------

cdef class GRUInferEngine:
    """
    Compiled GRU inference engine.

    One instance per gateway process, shared read-only across all worker
    threads. Each worker/connection owns its own SessionBuffer.

    Parameters
    ----------
    weights_path : str    path to .npz file from model_io.save_weights()
    threshold    : float  block if confidence >= threshold (default 0.80)
    min_steps    : int    minimum messages before scoring begins (default 4)
    """

    cdef float[:]       Wz, Uz, bz
    cdef float[:]       Wr, Ur, br
    cdef float[:]       Wh, Uh, bh
    cdef float[:]       Wc, bc
    cdef readonly float threshold
    cdef readonly int   min_steps

    def __cinit__(self,
                  str   weights_path,
                  float threshold=0.80,
                  int   min_steps=4):
        self.threshold = threshold
        self.min_steps = min_steps
        self._load(weights_path)

    def _load(self, str path):
        """Load .npz weights into contiguous C-order float32 memoryviews."""
        w = dict(np.load(path, allow_pickle=False))

        def _flat(key):
            return np.ascontiguousarray(w[key].ravel(), dtype=np.float32)

        def _vec(key):
            return np.ascontiguousarray(w[key], dtype=np.float32)

        self.Wz = _flat("Wz")
        self.Uz = _flat("Uz")
        self.bz = _vec("bz")
        self.Wr = _flat("Wr")
        self.Ur = _flat("Ur")
        self.br = _vec("br")
        self.Wh = _flat("Wh")
        self.Uh = _flat("Uh")
        self.bh = _vec("bh")
        self.Wc = _flat("Wc")
        self.bc = _vec("bc")

    cpdef object step(self, SessionBuffer buf, dict msg):
        """
        Featurise msg, append to session buffer, run GRU, return Verdict.

        Call this for every message arriving from the TLS stream,
        BEFORE forwarding to the org asset.

        Parameters
        ----------
        buf : SessionBuffer  per-session accumulator (one per TLS connection)
        msg : dict  keys: meas(float) dt_ms(int) step_idx(int)
                          ctr(int) decision(str) op(str)

        Returns
        -------
        Verdict — .block=True means drop this message and alert the SOC
        """
        # ALL cdef declarations at the top of the cpdef body
        cdef int   t
        cdef int   op_id
        cdef int   best
        cdef int   n
        cdef float probs[N_CLS]
        cdef float[:] ctx_mv
        cdef float[:] row

        t = buf.n
        if t >= SEQ_LEN:
            buf.X[:-1, :] = buf.X[1:, :]
            t = SEQ_LEN - 1
            buf.n = t

        op_str = msg.get("op", "READ")
        if op_str == "WRITE":
            op_id = 1
        elif op_str == "CONTROL":
            op_id = 2
        else:
            op_id = 0

        row = buf.X[t]
        _featurise_row(
            row,
            <float>msg.get("meas",     0.0),
            <int>  msg.get("dt_ms",    0),
            <int>  msg.get("step_idx", 0),
            <int>  msg.get("ctr",      0),
            msg.get("decision", "ACCEPT") == "ACCEPT",
            op_id,
            t,
        )
        buf.n += 1
        n = buf.n

        if n < self.min_steps:
            return Verdict(False, "none", 0.0, {lbl: 0.0 for lbl in ATK_LABELS}, n)

        ctx_mv = np.empty(H_DIM, dtype=np.float32)
        with nogil:
            _gru_forward(
                buf.X[:n, :], n,
                self.Wz, self.Uz, self.bz,
                self.Wr, self.Ur, self.br,
                self.Wh, self.Uh, self.bh,
                ctx_mv,
            )
            best = _classify(&ctx_mv[0], self.Wc, self.bc, probs)

        conf   = float(probs[best])
        label  = ATK_LABELS[best]
        scores = {ATK_LABELS[i]: float(probs[i]) for i in range(N_CLS)}
        block  = (label != "none") and (conf >= self.threshold)

        return Verdict(block, label, conf, scores, n)

    def score_trace(self, list trace_dicts):
        """Score a full trace list. Returns final Verdict. For offline eval."""
        cdef SessionBuffer buf
        buf = SessionBuffer()
        v   = None
        for msg in trace_dicts:
            v = self.step(buf, msg)
        return v
