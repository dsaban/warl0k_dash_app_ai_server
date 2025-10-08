# WarL0k P2P — Final (TLS + 128-bit master + early-stop DRNN)

**Core model**
- **Hub(s)** hold **seed0** per device (seed0 never leaves a hub).
- For each counterparty X, a peer requests a **Seed→Master vector** `W_X` from the hub (derived from X’s seed0). Peers store `W_X` (not seed0).
- Per session, peers derive `k_session` (X25519→HKDF), compute `obf`, and train a **Sess→Master DRNN** so that `DRNN(obf) == Master` (the same value the Seed→Master path yields).
- A transaction proceeds only if **both paths yield identical `Master`** on both sides.

**Security**
- Forward secrecy: X25519 + HKDF → ChaCha20-Poly1305 (AEAD).
- **128-bit Master** (32 hex chars).
- Hub RPC over **TLS with mutual authentication** (mTLS).
- DRNN uses **early-stop** as soon as equality holds (fast in practice).

---

## 0) Dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
