import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Secure Data Transfer System using WARL0K Secret Obfuscation Protocol

# --- utils.py ---
# Common utilities: secret generation, encryption helpers

import os
import base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import datetime
LOG_FILE = "./logs/server.log"
#  log for client
LOG_FILE_CLIENT = "./logs/client.log"
# --- Configuration ---
def generate_secret(length=16):
    charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join([charset[b % len(charset)] for b in os.urandom(length)])

def aead_encrypt(key: bytes, plaintext: bytes, associated_data: bytes = b""):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, plaintext, associated_data)
    return base64.b64encode(nonce + ct).decode()

def aead_decrypt(key: bytes, token: str, associated_data: bytes = b""):
    data = base64.b64decode(token)
    nonce, ct = data[:12], data[12:]
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, associated_data)

def create_key():
    return AESGCM.generate_key(bit_length=128)

def save_key(session_id, key, key_dir="session_keys"):
    os.makedirs(key_dir, exist_ok=True)
    with open(os.path.join(key_dir, f"{session_id}.key"), "w") as f:
        f.write(key.hex())

def load_key(session_id, key_dir="session_keys"):
    path = os.path.join(key_dir, f"{session_id}.key")
    with open(path, "r") as f:
        return bytes.fromhex(f.read())

def encrypt(key, message):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, message.encode(), None)
    return nonce, ct

def decrypt(key, nonce, ct):
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ct, None).decode()

# Logging helper
def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[Server][{timestamp}] {msg}\n")
    # print(f"[Server][{timestamp}] {msg}")

#  log for client
def log_client(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE_CLIENT, "a") as f:
        f.write(f"[CLIENT] [{timestamp}] {msg}\n")
    # print(f"[CLIENT] [{timestamp}] {msg}")
