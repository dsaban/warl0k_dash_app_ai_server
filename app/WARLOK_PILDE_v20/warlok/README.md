# WARLOK — Pilot Console v2.0

Upgraded from Streamlit → FastAPI + single-page dark Web UI.

## Stack
- **Backend**: FastAPI (same core logic — block, dag, hsm, hub, simulator, attack)
- **Frontend**: Vanilla HTML/CSS/JS, dark pilot console UI
- **No Streamlit dependency**

## Run

```bash
pip install -r requirements.txt
uvicorn api.server:app --reload --port 8000
```

Then open: http://localhost:8000

## Pages
| Page | Description |
|------|-------------|
| Dashboard | Live stats, quick create/validate, event stream |
| DAG Graph | Canvas-rendered topology, click nodes to inspect |
| Execution | Full chronological event log |
| Pipeline | Animated multi-agent runner |
| Inspector | Hash lookup + deep validation |
| Blocks | Full block registry table |
| Attack Sim | Tamper payload / break signature |
