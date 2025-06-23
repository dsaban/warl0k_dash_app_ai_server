import socket
import threading
import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import datetime

HOST = '0.0.0.0'
PORT = 9999  # Port to listen on (non-privileged ports are > 1023)
KEY_DIR = "_session_keys"
LOG_FILE = "logs/server.log"

os.makedirs(KEY_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Logging helper
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    print(f"[{timestamp}] {msg}")

# Key management
def load_key(session_id):
    key_path = os.path.join(KEY_DIR, f"{session_id}.key")
    if not os.path.exists(key_path):
        log(f"[KEY] ❌ Key file not found: {session_id}")
        raise FileNotFoundError(f"Key for session {session_id} not found.")
    with open(key_path, "r") as f:
        key = bytes.fromhex(f.read())
    log(f"[KEY] ✅ Key loaded for session: {session_id}")
    return key

# Handle each client
def handle_client(conn, addr):
    log(f"[CONNECTION] Accepted from {addr}")
    try:
        data = conn.recv(4096)
        if not data or len(data) < 48:
            log("[ERROR] Insufficient data received")
            return

        session_id = data[:36].decode()
        nonce = data[36:48]
        ciphertext = data[48:]

        key = load_key(session_id)
        aesgcm = AESGCM(key)
        try:
            decrypted = aesgcm.decrypt(nonce, ciphertext, None).decode()
            log(f"[SESSION:{session_id}] ✅ Decrypted: {decrypted}")
        except Exception as e:
            log(f"[SESSION:{session_id}] ❌ Decryption error: {type(e).__name__}: {e}")
            return

        if decrypted == "KILL_SERVER":
            log("[CONTROL] Shutdown command received.")
            os._exit(0)

        # Prepare response
        response = f"ACK:{decrypted}"
        nonce_out = os.urandom(12)
        ct_out = aesgcm.encrypt(nonce_out, response.encode(), None)
        payload_out = session_id.encode() + nonce_out + ct_out
        conn.sendall(payload_out)
        log(f"[SESSION:{session_id}] → Sent encrypted ACK response")

    except Exception as e:
        log(f"[ERROR] Unexpected error: {e}")
    finally:
        conn.close()
        log(f"[CONNECTION] Closed for {addr}")

# TCP Server Loop
def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        log(f"[SERVER] TCP server started on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            thread.start()

if __name__ == "__main__":
    start_server()
