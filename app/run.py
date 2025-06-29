import os
import subprocess
import time

print("[BOOT] Starting WARL0K TCP demo...")

# --- Create required directories ---
os.makedirs("logs", exist_ok=True)
os.makedirs("_session_keys", exist_ok=True)

# --- Start the TCP Server as a daemon ---
print("[SERVER] Launching TCP server (daemon)...")
subprocess.Popen(
    ["python3", "server_ai.py"],
    stdout=open("logs/server.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True  # Detach from terminal
)

# --- Give the server time to start ---
time.sleep(2)

# --- Start the Streamlit Client Dashboard ---
print("[CLIENT DASHBOARD] Launching Streamlit dashboard...")
subprocess.Popen(
    ["streamlit", "run", "dash6.py"],
    stdout=open("logs/client.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)

print("[✓] WARL0K system is running as background daemons. Logs are in ./logs/")
