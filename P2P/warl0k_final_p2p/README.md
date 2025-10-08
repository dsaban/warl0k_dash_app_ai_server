# WarL0k P2P — Vanishing Secrets with Per-Transaction Learning

WarL0k provides **hub-introduced, hub-less P2P** sessions where each device:
- stores only **model weights** (`W_peer` per counterparty, plus a tiny **adapter**),
- **re-generates** the master identity per session,
- validates sessions by **two-path equality**:
  1) **Seed-path**: `Master = HMAC(W_peer, "M" || peer_id)`
  2) **Session-path**: tiny **DRNN** learns `obf → Master` each session (few-shot, early-stop)

## Quick start

### 0) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
