import os
import subprocess
import time

print("[BOOT] Starting WARL0K TCP demo...")

# --- Create required directories ---
os.makedirs("logs", exist_ok=True)
os.makedirs("_session_keys", exist_ok=True)

# --- Start the TCP Server ---
print("[SERVER] Launching TCP server...")
subprocess.Popen(
    ["python3", "server_ai.py"],
    stdout=open("logs/server.log", "w"),
    stderr=subprocess.STDOUT
)

# --- Wait for server to initialize ---
# time.sleep(2)

# --- Start the Streamlit Client Dashboard ---
print("[CLIENT DASHBOARD] Launching Streamlit dashboard...")
subprocess.Popen(
    ["streamlit", "run", "client_ai_dash2.py"],
    stdout=open("logs/client.log", "w"),
    stderr=subprocess.STDOUT
)

print("[✓] WARL0K system is running. Logs are in ./logs/")
