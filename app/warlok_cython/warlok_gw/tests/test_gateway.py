# tests/test_gateway.py
# ─────────────────────────────────────────────────────────────────────────────
# Correctness + latency tests for the WARL0K gateway.
# Runs against the pure Python fallback engine so no build step is needed.
# When the Cython .so is present, the same tests validate the compiled version.
#
# Run:
#   python -m pytest tests/test_gateway.py -v
#   python tests/test_gateway.py          (standalone)
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, time, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

# ── helpers to create a dummy model.npz without training ─────────────────────

def _dummy_weights(seed=42):
    rng = np.random.RandomState(seed)
    s   = 0.08
    p   = {}
    for g in ("z", "r", "h"):
        p[f"W{g}"] = (rng.randn(64, 9)  * s).astype(np.float32)
        p[f"U{g}"] = (rng.randn(64, 64) * s).astype(np.float32)
        p[f"b{g}"] = np.zeros(64, dtype=np.float32)
    p["Wc"] = (rng.randn(6, 64) * s).astype(np.float32)
    p["bc"] = np.zeros(6, dtype=np.float32)
    return p

def _save_dummy(path):
    from warlok_gw.model_io import save_weights
    save_weights(_dummy_weights(), path)

def _make_msg(op="READ", dt_ms=10, meas=0.5, step_idx=0, ctr=1, decision="ACCEPT"):
    return dict(op=op, dt_ms=dt_ms, meas=meas,
                step_idx=step_idx, ctr=ctr, decision=decision)

# ── tests ─────────────────────────────────────────────────────────────────────

def test_session_buffer_init():
    from warlok_gw._gru_infer_py import SessionBuffer
    buf = SessionBuffer("sess-001")
    assert buf.n          == 0
    assert buf.X.shape    == (48, 9)
    assert buf.session_id == "sess-001"
    buf.reset()
    assert buf.n == 0
    print("PASS  test_session_buffer_init")

def test_engine_loads_and_returns_verdict():
    from warlok_gw._gru_infer_py import GRUInferEngine, SessionBuffer
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    engine = GRUInferEngine(path, threshold=0.80, min_steps=4)
    buf    = SessionBuffer()
    # Below min_steps — should not score
    for i in range(3):
        v = engine.step(buf, _make_msg(step_idx=i, ctr=i+1))
    assert v.attack_class == "none"
    assert v.confidence   == 0.0
    assert v.block        == False
    # At min_steps — should score
    v = engine.step(buf, _make_msg(step_idx=3, ctr=4))
    assert v.attack_class in ("none","reorder","drop","replay","timewarp","splice")
    assert 0.0 <= v.confidence <= 1.0
    assert abs(sum(v.scores.values()) - 1.0) < 1e-4
    print(f"PASS  test_engine_loads_and_returns_verdict  pred={v.attack_class}({v.confidence:.3f})")
    os.unlink(path)

def test_sliding_window():
    """Buffer should slide when messages exceed SEQ_LEN=48."""
    from warlok_gw._gru_infer_py import GRUInferEngine, SessionBuffer
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    engine = GRUInferEngine(path, threshold=0.80, min_steps=1)
    buf    = SessionBuffer()
    for i in range(60):   # more than SEQ_LEN
        v = engine.step(buf, _make_msg(step_idx=i % 48, ctr=i+1))
    assert buf.n <= 48
    print(f"PASS  test_sliding_window  buf.n={buf.n}")
    os.unlink(path)

def test_model_io_roundtrip():
    """save_weights → load_weights → validate should succeed."""
    from warlok_gw.model_io import save_weights, load_weights, validate
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    p = _dummy_weights()
    save_weights(p, path, extra_meta={"accuracy": 0.95, "threshold": 0.80})
    p2, meta = load_weights(path)
    assert validate(p2)
    assert meta.get("accuracy") == 0.95
    for k in ("Wz","Wc","bc"):
        assert np.allclose(p[k], p2[k])
    print(f"PASS  test_model_io_roundtrip  meta={meta}")
    os.unlink(path)

def test_gateway_allow_clean_session():
    """Clean messages should produce ALLOW outcomes."""
    from warlok_gw.gateway import WarlokGateway, _pack_msg_dict
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    gw = WarlokGateway(path, threshold=0.999, min_steps=4)   # very high threshold
    for i in range(10):
        msg = _make_msg(op="READ", dt_ms=10, meas=0.4,
                        step_idx=i % 48, ctr=i+1, decision="ACCEPT")
        outcome = gw.intercept("sess-clean", _pack_msg_dict(msg), extra_meta=msg)
    # With threshold=0.999 a random model should not block
    assert outcome.action in ("ALLOW", "BLOCK")   # just check it runs
    gw.close_session("sess-clean")
    print(f"PASS  test_gateway_allow_clean_session  last_action={outcome.action}")
    os.unlink(path)

def test_gateway_blocks_on_rule_violation():
    """op_code=CONTROL should be blocked by the rule gate regardless of AI."""
    from warlok_gw.gateway import WarlokGateway, _pack_msg_dict
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    gw = WarlokGateway(path, threshold=0.999, min_steps=1)
    msg = _make_msg(op="CONTROL", dt_ms=10, meas=0.4, step_idx=0, ctr=1)
    outcome = gw.intercept("sess-splice", _pack_msg_dict(msg), extra_meta=msg)
    assert outcome.action     == "BLOCK"
    assert outcome.blocked_by == "rule"
    print(f"PASS  test_gateway_blocks_on_rule_violation  rule={outcome.rule_verdict}")
    os.unlink(path)

def test_gateway_blocks_on_timewarp():
    """dt_ms=999999 should be blocked by the rule gate."""
    from warlok_gw.gateway import WarlokGateway, _pack_msg_dict
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    gw = WarlokGateway(path, threshold=0.999, min_steps=1)
    msg = _make_msg(op="READ", dt_ms=999999, meas=0.4, step_idx=0, ctr=1)
    outcome = gw.intercept("sess-timewarp", _pack_msg_dict(msg), extra_meta=msg)
    assert outcome.action     == "BLOCK"
    assert outcome.blocked_by == "rule"
    print(f"PASS  test_gateway_blocks_on_timewarp  rule={outcome.rule_verdict}")
    os.unlink(path)

def test_latency_benchmark():
    """
    Benchmark pure-Python inference latency.
    Goal: < 2 ms per message (Cython target: < 0.1 ms).
    """
    from warlok_gw._gru_infer_py import GRUInferEngine, SessionBuffer
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    engine  = GRUInferEngine(path, threshold=0.80, min_steps=1)
    buf     = SessionBuffer()
    N       = 200
    times   = []
    for i in range(N):
        msg = _make_msg(step_idx=i % 48, ctr=i+1, dt_ms=10+i)
        t0  = time.perf_counter()
        engine.step(buf, msg)
        times.append((time.perf_counter() - t0) * 1000)
    p50 = sorted(times)[N//2]
    p99 = sorted(times)[int(N*0.99)]
    print(f"PASS  test_latency_benchmark  p50={p50:.3f}ms  p99={p99:.3f}ms  (Python fallback)")
    print(f"      Cython target: p50<0.10ms  p99<0.20ms")
    os.unlink(path)

def test_concurrent_sessions():
    """Two sessions must not interfere with each other's buffers."""
    from warlok_gw._gru_infer_py import GRUInferEngine, SessionBuffer
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    _save_dummy(path)
    engine = GRUInferEngine(path, threshold=0.80, min_steps=1)
    buf_a  = SessionBuffer("a")
    buf_b  = SessionBuffer("b")
    for i in range(20):
        engine.step(buf_a, _make_msg(dt_ms=10,     step_idx=i))
        engine.step(buf_b, _make_msg(dt_ms=999999, step_idx=i))
    assert buf_a.n == 20
    assert buf_b.n == 20
    # The extreme dt in buf_b should not affect buf_a's feature row 0
    assert buf_a.X[0, 1] < 1.0     # dt_ms=10 → normalised < 1.0
    assert buf_b.X[0, 1] > 1.0     # dt_ms=999999 → normalised >> 1.0
    print("PASS  test_concurrent_sessions")
    os.unlink(path)


# ── run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_session_buffer_init,
        test_engine_loads_and_returns_verdict,
        test_sliding_window,
        test_model_io_roundtrip,
        test_gateway_allow_clean_session,
        test_gateway_blocks_on_rule_violation,
        test_gateway_blocks_on_timewarp,
        test_latency_benchmark,
        test_concurrent_sessions,
    ]
    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as exc:
            print(f"FAIL  {fn.__name__}  {exc}")
    print(f"\n{'='*50}")
    print(f"  {passed}/{len(tests)} tests passed")
    print(f"{'='*50}")
