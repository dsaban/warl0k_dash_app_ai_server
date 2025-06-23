#!/bin/bash

echo "[BOOT] Starting WARL0K TCP demo..."

# --- Create required directories ---
mkdir -p logs
mkdir -p _session_keys

# --- Start the TCP Server ---
echo "[SERVER] Launching TCP server..."
#nohup python3 server.py > logs/server.log 2>&1 &
nohup python3 server_ai.py > logs/server.log 2>&1 &
#SLEEP 2  # Give the server time to start
sleep 2
# --- Start the client ---
echo "[CLIENT] Launching Streamlit client..."
# Ensure the client script is executable
#chmod +x client_ai.py
# Run the client script in the background
#nohup python3 client.py > logs/client.log 2>&1 &
#nohup python3 client_ai_dash2.py > logs/client.log 2>&1 &

# --- Create session keys directory if it doesn't exist ---

# --- Give the server time to start ---
sleep 1

# --- Start the Streamlit Client Dashboard ---
echo "[CLIENT DASHBOARD] Launching Streamlit dashboard..."
#nohup streamlit run client.py > logs/dashboard.log 2>&1 &
#nohup streamlit run client_ai.py > logs/dashboard.log 2>&1 &
nohup streamlit run client_ai_dash2.py > logs/client.log 2>&1 &

echo "[✓] WARL0K system is running. Logs are in ./logs/"
