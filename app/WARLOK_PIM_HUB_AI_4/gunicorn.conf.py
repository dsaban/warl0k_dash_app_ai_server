# gunicorn.conf.py — WARL0K PIM Engine v2
import multiprocessing
import os

# Workers: 1 for dev (shared in-process state), 2+ for prod (use Redis for shared state)
workers      = int(os.environ.get("WORKERS", 1)) if __import__("os").environ.get("WORKERS") else 1
worker_class = "uvicorn.workers.UvicornWorker"
bind         = f"0.0.0.0:{__import__('os').environ.get('PORT', '5051')}"
timeout      = 300          # long for training runs
graceful_timeout = 30
keepalive    = 5
loglevel     = "info"
accesslog    = "-"
errorlog     = "-"

# For multi-worker with shared peer state, set WORKERS=1 (default)
# or move _peers dict to Redis:
#   pip install redis
#   import redis; r = redis.Redis(); r.set('peers', json.dumps({}))
