# WARL0K PIM Engine — Python Rebuild

Pure Python/NumPy port of the C++ GRU+Attention PIM authentication system,
extended with all-around encryption and tamper-evident chain-proof validation.

---

## Architecture

```
pim_core.py          ← Core engine (all math, no dependencies beyond NumPy)
  ├── XorShift32     ← Deterministic PRNG matching C++ exactly
  ├── OS chain gen   ← generate_os_chain() — PN watermark + delta projection
  ├── GRU + BPTT    ← gru_forward() / gru_backward()
  ├── Attention      ← attention_forward() / attention_backward()
  ├── MS head        ← ms_head_forward() / ms_head_backward()
  ├── Adam           ← per-parameter Adam with freeze mask
  ├── train_phase1   ← MS recon + token scaffold (90 epochs)
  ├── train_phase2   ← ID/Window/Validity heads (100 epochs, frozen backbone)
  ├── verify_chain   ← single-sample inference + 6 gates
  ├── save/load      ← NumPy .npz (compressed)
  │
  ├── encrypt_chain  ← AES-256-GCM (cryptography) or HMAC-XOR fallback
  ├── decrypt_chain  ← symmetric decryption + tag verification
  ├── derive_key     ← HKDF-SHA256 key derivation
  │
  └── ChainProof     ← HMAC chain: state_n = SHA3-256(state_{n-1} || event_n)
      PIMSession     ← high-level: verify + encrypt + chain-log

server.py            ← Flask REST API + SSE streaming
  GET  /             → UI (static/index.html)
  GET  /api/status   → model/crypto status
  POST /api/train    → start background training
  GET  /api/train/stream → SSE loss stream
  POST /api/load     → load saved model.npz
  POST /api/verify   → single verify (tamper modes: none/shuffle/truncate/wrong_win/wrong_id/oob)
  POST /api/encrypt  → encrypt token+meas payload, verify roundtrip
  GET  /api/chain    → chain state + last 10 events
  POST /api/benchmark → N inference calls, return timing stats

static/index.html    ← Dark terminal UI (no build step)
```

---

## Run

```bash
pip install flask numpy
pip install cryptography      # optional — enables AES-256-GCM

python server.py
# → http://localhost:5050
```

Or headless:
```bash
python pim_core.py            # runs full pipeline, prints results
```

---

## Encryption Layer

Two modes (auto-selected):

| Mode | Cipher | KDF |
|------|--------|-----|
| `cryptography` installed | AES-256-GCM | HKDF-SHA256 |
| fallback | XOR stream + HMAC-SHA256 tag | SHA-256 |

Each payload: `{ salt, nonce, ciphertext, tag, chain_hash (SHA3-256) }`

---

## Chain Proof (HMAC chain)

Each verify event is linked:

```
state_0 = SHA3-256(b'WARL0K_GENESIS')
state_n = SHA3-256(state_{n-1} || event_n_json)
proof_n = HMAC-SHA256(state_n, event_n_json)
```

- Any event modification breaks all subsequent proofs
- `ChainProof.verify_chain()` returns (ok, first_bad_index)

---

## Verification Gates (6 total)

| Gate | Threshold |
|------|-----------|
| PN pilot correlation | ≥ 0.02 |
| p_valid (behavior head) | ≥ 0.80 |
| id_pred == claimed_id | exact |
| w_pred == expected_w | exact |
| pid (claimed ID prob) | ≥ 0.70 |
| pw (expected window prob) | ≥ 0.40 |

All 6 must pass for `ok=True`.

---

## Fidelity vs C++

The Python engine is a faithful port:
- XorShift32 PRNG byte-for-byte identical → same OS chains
- PN pilot watermark identical
- GRU equations identical (z/r/htilde, mask logic)
- Attention: W_att·h → tanh → v_att dot-product → masked softmax
- Adam with bias/weight separation, global grad clipping
- Phase1/2 freeze masks identical
- Save format: NumPy .npz (different from C++ binary, but portable)
