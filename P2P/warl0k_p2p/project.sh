# shellcheck disable=SC2287
warl0k_p2p/
├─ hub.py                   # FastAPI monitor & logger (no payloads)
├─ warlok_common.py         # Shared crypto + helpers
├─ peer_a.py                # Sender peer (CLI)
├─ peer_b.py                # Receiver peer (CLI loop; polls wire/)
├─ streamlit_log_dash.py    # Read-only dashboard over hub logs
├─ wire/                    # “network” drop folder (created on first run)
└─ logs/
   └─ events.jsonl          # hub log (auto-created)
