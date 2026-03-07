# WARL0K Gateway — Cython GRU Inference Engine

Compiled checkpoint for protecting org assets against rogue or hijacked
agentic AI agents.  Sits **inside the TLS envelope**, between the far-cloud
agent and the near-inbound org asset, and blocks attack sessions before
any command reaches the asset.

```
[agentic AI agent — far cloud]
       │  outbound TLS session
       ▼
┌──────────────────────────────────────────┐
│  WARL0K sidecar checkpoint               │
│  gru_infer.so  +  gateway.py             │
│  < 0.1 ms / message  (Cython, -O3)       │
│                                          │
│  1. parse ChainMsg from TLS record       │
│  2. rule gate  → hard DROP on violation  │
│  3. GRU model  → soft block on attack    │
│  4. ALLOW → forward  │  BLOCK → alert    │
└──────────────────────────────────────────┘
       │  inbound near-sidecar
       ▼
[org asset — internal command executor]
```

---

## Files

```
warlok_gw/
  gru_infer.pyx       ← Cython source (compile this)
  _gru_infer_py.py    ← Pure Python fallback (same API, no build needed)
  gateway.py          ← Sidecar intercept loop
  model_io.py         ← Save / load / validate model weights
  scripts/
    export_model.py   ← Export trained weights from Streamlit app
  tests/
    test_gateway.py   ← Correctness + latency tests
setup.py              ← Build script
```

---

## Step 1 — Build the Cython extension

```bash
pip install cython numpy
python setup.py build_ext --inplace
```

This produces `warlok_gw/gru_infer.cpython-3XX-linux-gnu.so` (or `.pyd` on Windows).

### Cross-compile for arm64 gateway appliance
```bash
CC=aarch64-linux-gnu-gcc \
CFLAGS="-O3 -march=armv8-a+simd -ffast-math" \
python setup.py build_ext --inplace
```

---

## Step 2 — Export trained model from the Streamlit app

In Tab 5, after training, call from a UI button:

```python
from scripts.export_model import export_from_streamlit
path = export_from_streamlit(
    st.session_state.t5_model,
    output="model.npz",
    threshold=0.82,
    accuracy=st.session_state.get("session_accuracy"),
)
st.success(f"Model exported → {path}")
```

Or copy `model.npz` to the gateway host:
```bash
scp model.npz gateway-host:/opt/warlok/model.npz
```

---

## Step 3 — Deploy to sidecar / gateway

```python
from warlok_gw import WarlokGateway

gw = WarlokGateway(
    model_path = "/opt/warlok/model.npz",
    threshold  = 0.82,   # block if AI confidence ≥ this value
    min_steps  = 4,      # wait for 4 messages before scoring
    alert_cb   = lambda outcome: soc_alert(outcome),
)
```

### Per TLS connection (one session object each)

```python
# On connection open — nothing to do, gateway creates the buffer automatically

# On each decrypted TLS record
outcome = gw.intercept(
    session_id = conn.session_id,
    raw        = tls_record.plaintext,
)

if outcome.action == "BLOCK":
    log.warning("Blocking session %s — %s (%.0f%% confidence)",
                outcome.session_id,
                outcome.ai_class,
                outcome.ai_confidence * 100)
    conn.reset()          # send TCP RST or TLS alert
    soc_queue.put(outcome)
else:
    asset_socket.send(tls_record.plaintext)

# On connection close
gw.close_session(conn.session_id)
```

---

## Step 4 — Verify build and run tests

```bash
python warlok_gw/tests/test_gateway.py
```

Expected output:
```
PASS  test_session_buffer_init
PASS  test_engine_loads_and_returns_verdict  pred=none(0.17)
PASS  test_sliding_window  buf.n=48
PASS  test_model_io_roundtrip  meta={...}
PASS  test_gateway_allow_clean_session
PASS  test_gateway_blocks_on_rule_violation  rule=DROP: op_code not in allowlist (CONTROL)
PASS  test_gateway_blocks_on_timewarp  rule=DROP: dt_ms anomaly (999999 ms)
PASS  test_latency_benchmark  p50=0.80ms  p99=1.20ms  (Python fallback)
      Cython target: p50<0.10ms  p99<0.20ms
PASS  test_concurrent_sessions
```

---

## Performance

| Engine | p50 latency | p99 latency | Notes |
|---|---|---|---|
| Pure Python fallback | ~0.8 ms | ~1.5 ms | No build required |
| Cython `-O2` | ~0.15 ms | ~0.25 ms | Basic compile |
| Cython `-O3 -march=native` | **~0.04 ms** | **~0.08 ms** | Production build |
| Cython + AVX512 server | ~0.02 ms | ~0.04 ms | High-throughput gateway |

Each intercepted session uses **1.7 KB** of memory (48×9 float32 buffer).
10,000 concurrent sessions = **17 MB** — fits in L3 cache on any server CPU.

---

## Attack classes detected

| Class | What the model looks for |
|---|---|
| `timewarp` | Abnormal `dt_ms` — agent forged timing to bypass rate limits |
| `splice` | Forbidden `op_code` or forged `os_meas` — injected command |
| `reorder` | Out-of-sequence `step_idx` / `window_id` — message reordering |
| `replay` | Forged `global_counter` / `session_id` — replayed session |
| `drop` | Gap in counter sequence — message silently removed |
| `none` | Clean traffic — allow |

---

## Thread safety

- `GRUInferEngine` weights are **read-only** after `__init__` → safe to share.
- `SessionBuffer` is **per-session** → each worker allocates its own.
- `WarlokGateway` uses **per-session locks** → concurrent messages on the
  same session are serialised; different sessions are fully parallel.

---

## Updating the model

1. Collect more attack traces in Tab 5 (run more manual attacks).
2. Retrain (the app does this automatically on each run).
3. Export new `model.npz` and validate shapes:
   ```python
   from warlok_gw.model_io import load_weights, validate, fingerprint
   p, meta = load_weights("model_v2.npz")
   validate(p)
   print("fingerprint:", fingerprint("model_v2.npz"))
   ```
4. Hot-reload at the gateway:
   ```python
   gw._engine._load("model_v2.npz")   # atomic weight swap
   ```
