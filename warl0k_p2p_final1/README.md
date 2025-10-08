# WarL0k P2P — Vanishing Secrets with Pretrained (Ticketed) Session Learning

This version implements:
- **Hub as introducer** (enrollment + W vector; seed0 never leaves hub).
- **Per-counterparty adapters** (DRNN) that are meta-pretrained once and then
  **pre-learn** a rolling window of **ticketed obfuscations** for instant validation.
- **Per-session ECDH** for payload crypto + **session-bound tag** using W and the live obf.
- **No keys at rest**: peers store only W and tiny adapters (weights). Masters vanish.

## Quick Start

### 0) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
