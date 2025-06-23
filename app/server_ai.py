import socket, threading, pickle
import torch
from model import train_secret_regenerator, evaluate_secret_regenerator
from utils import (
    load_key,create_key,decrypt, encrypt, generate_secret, aead_decrypt, log)
import random

HOST = '0.0.0.0'
PORT = 9996
SESSIONS = {}

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def handle_client(conn):
    session_id = conn.recv(36).decode()
    print(f"[SESSION] New request for: {session_id}")
    log(f"New session started with ID: {session_id}")

    if session_id not in SESSIONS:
        master = generate_secret()
        obfs = generate_secret()
        model_master = train_secret_regenerator(master, vocab)
        model_obfs = train_secret_regenerator(master, vocab, input_override=obfs)

        SESSIONS[session_id] = {
            "master": master,
            "obfs": obfs,
            "model_master": model_master,
            "model_obfs": model_obfs
        }

        conn.send(obfs.encode())
        print(f"[SEND] Obfuscated secret sent.")
        # Example after sending obfuscated secret:
        log(f"Obfuscated secret: {obfs}")
        
        return

    # Handle fingerprint + encrypted payload
    raw = conn.recv(4096)
    if not raw:
        return

    try:
        fingerprint, encrypted_payload = pickle.loads(raw)
        obfs_model = SESSIONS[session_id]["model_obfs"]
        master_expected = SESSIONS[session_id]["master"]
        recovered = evaluate_secret_regenerator(obfs_model, fingerprint, vocab)

        print(f"[RECONSTRUCT] Master: {recovered}")
        log(f"Recovered master: {recovered}")
        if recovered != master_expected:
            conn.send(b"[FAIL] Authentication failed.")
            return

        decrypted = aead_decrypt(fingerprint.numpy().tobytes()[:16], encrypted_payload.encode())
        print(f"[✓] Payload Decrypted: {decrypted.decode()}")
        log(f"Decrypted payload: {decrypted.decode()}")
        conn.send(b"[OK] Authenticated & Decrypted.")
    except Exception as e:
        print(f"[ERROR] {e}")
        conn.send(b"[ERR] Processing failed.")

def start_server():
    print("[BOOT] WARL0K TCP Server Starting...")
    log("Server is starting...")
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

start_server()
