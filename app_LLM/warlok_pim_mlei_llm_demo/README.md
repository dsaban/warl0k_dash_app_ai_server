# WARL0K PIM + MLEI LLM-Agent Session Demo (2 Peers, Encrypted, CSV DB)

This demo simulates:
- Near peer (verifier + policy gate)
- Far peer (proofer + executor + DB writer)
- Encrypted tunnel (AES-GCM)
- PIM: per-message chain integrity (counter, prev-hash, timestamp window)
- MLEI: nano-gates (tiny numpy model) at both peers to detect malicious intent + anomalies
- Mock cloud LLM "agent" that returns tool calls to complete tasks
- Attack injector that tampers / reorders / delays / prompt-injects / unauthorized DB writes

## Run
```bash
pip install -r requirements.txt
python run_demo.py
```

## What to look for
- `ALLOW` vs `BLOCK` logs from near/far nano-gates
- PIM chain verification failures (hash mismatch / counter mismatch / timing window)
- Attacks are detected and *foiled* (message dropped, DB write blocked)

## Tasks
- Task 1: read DB and summarize rows
- Task 2: write a new row after LLM tool-call approval

The "LLM" is mocked but behaves like an agent: it emits tool calls (read_db, write_db)
that must pass MLEI policy and PIM chain validation.
