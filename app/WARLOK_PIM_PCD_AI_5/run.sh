# This environment (Flask shim — works now):
python3 pim_pcd_fastapi.py

# Your environment (real FastAPI — install once):
#pip install fastapi uvicorn numpy
#uvicorn pim_pcd_fastapi:app --host 0.0.0.0 --port 5053 --reload
#
## Interactive API docs (FastAPI only):
#http://localhost:5053/docs

# SSE real-time stream (FastAPI only — no polling needed):
#http://localhost:5053/api/events
