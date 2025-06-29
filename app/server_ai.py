import socket, threading, pickle
import torch
from model import (
    train_secret_regenerator,
    evaluate_secret_regenerator,
    add_noise_to_tensor,
    inject_patterned_noise)
from utils import (
    load_key,
    create_key,
    decrypt,
    encrypt,
    generate_secret,
    aead_decrypt,
    log)
import random

HOST = '0.0.0.0'
PORT = 9995
SESSIONS = {}

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def text_to_tensor(text):
    return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

def handle_client(conn):
    global SESSIONS
    session_id = conn.recv(36).decode()
    # print(f"[SESSION] New request for: {session_id}")
    log(f"[Server] New session started with ID: {session_id}")

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
        # print(f"[SEND] Obfuscated secret sent.")
        # Example after sending obfuscated secret:
        log(f"[Server] Obfuscated secret: {obfs}")
        
        return

    # Handle fingerprint + encrypted payload
    raw = conn.recv(4096)
    if not raw:
        return

    try:
        _fingerprint, encrypted_payload = pickle.loads(raw)
        obfs_model = SESSIONS[session_id]["model_obfs"]
        master_expected = SESSIONS[session_id]["master"]
        recovered = evaluate_secret_regenerator(obfs_model, _fingerprint, vocab)
        # print(f"[RECONSTRUCT] Master: {recovered}")
        
        #  get obfs from SESSIONS
        obfs = SESSIONS[session_id]["obfs"]
        torch.manual_seed(int(session_id[:8], 16))
        # fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
        
        noisy_tensor = inject_patterned_noise(
            seq_tensor=text_to_tensor(obfs).squeeze(1),
            vocab_size=len(vocab),
            error_rate=0.25,
            pattern_ratio=0.6,  # Example ratio, can be adjusted
            seed=session_id
        ).unsqueeze(1)
        
        noisy_obf_secret = ''.join([vocab[i] for i in noisy_tensor.squeeze(1).tolist()])
        # print(f"[RECEIVED] Fingerprint: {fingerprint}")
        # log(f"Received fingerprint: {fingerprint}")
        log(f"[Server] Received fingerprint: {noisy_obf_secret}")
        obfs_model = SESSIONS[session_id]["model_obfs"]
        master_expected = SESSIONS[session_id]["master"]
        recovered = evaluate_secret_regenerator(obfs_model, _fingerprint, vocab)
        
        log(f"[Server][RECONSTRUCT] Master: {recovered}")
        log(f"[Server] Recovered master: {recovered}")
        if recovered != master_expected:
            conn.send(b"[FAIL] Authentication failed.")
            return

        decrypted = aead_decrypt(_fingerprint.numpy().tobytes()[:16], encrypted_payload.encode())
        log(f"[Server] [✓] Payload Decrypted: {decrypted.decode()}")
        log(f"[Server] Decrypted payload: {decrypted.decode()}")
        conn.send(b"[OK] Authenticated & Decrypted.")
    except Exception as e:
        print(f"[Server] [ERROR] {e}")
        conn.send(b"[ERR] Processing failed.")

def start_server():
    print("[BOOT] WARL0K TCP Server Starting...")
    log("[Server] Server is starting...")
    s = socket.socket()
    s.bind((HOST, PORT))
    s.listen(5)
    while True:
        conn, _ = s.accept()
        threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

start_server()
