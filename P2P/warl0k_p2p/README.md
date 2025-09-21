# WARL0K P2P Demo Codebase

This repository contains a minimal, didactic Python implementation of the WARL0K peer-to-peer session protocol:

- Hub does **enrollment** only (issues `seed0`).
- Devices derive `master_seed = HMAC(seed0, device_nonce)`.
- Peers perform ephemeral DH (MODP demo) and derive a per-session `K_session` by mixing:
  - shared ECDH,
  - both peers' contributions (derived from master_seed),
  - policy/counter/challenge info.
- Messages are protected with HMAC (demo); replace with AEAD (ChaCha20-Poly1305 / AES-GCM) in production.

**Run demo:**
```bash
python -m warlok.demo
