from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import streamlit as st
import socket
import uuid
import os
import pandas as pd
import time

# --- Configuration ---
SERVER_HOST = "localhost"
SERVER_PORT = 9999
KEY_DIR = "./_session_keys"
os.makedirs(KEY_DIR, exist_ok=True)

# --- Session Management ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "ephemeral_key" not in st.session_state:
    st.session_state.ephemeral_key = AESGCM.generate_key(bit_length=128)
    with open(os.path.join(KEY_DIR, f"{st.session_state.session_id}.key"), "w") as f:
        f.write(st.session_state.ephemeral_key.hex())
if "response" not in st.session_state:
    st.session_state.response = None
if "session_log" not in st.session_state:
    st.session_state.session_log = []

# --- AES-GCM Utility ---
def encrypt_message(key, message):
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    ct = aesgcm.encrypt(nonce, message.encode(), None)
    return nonce, ct

def decrypt_message(key, nonce, ciphertext):
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, None).decode()

# --- Streamlit Layout ---
st.set_page_config("WARL0K TCP Client Dashboard", layout="wide")
st.title("🔐 WARL0K TCP Client Dashboard")

st.sidebar.header("Session Info")
st.sidebar.code(f"Session ID: {st.session_state.session_id}")
st.sidebar.code(f"Ephemeral Key: {st.session_state.ephemeral_key.hex()}")

message = st.text_input("Message to Send", value="AUTH_REQUEST")
if st.button("Send Message"):
    try:
        # --- Encrypt Message ---
        nonce, ct = encrypt_message(st.session_state.ephemeral_key, message)
        payload = st.session_state.session_id.encode() + nonce + ct

        # --- Send to Server ---
        with socket.create_connection((SERVER_HOST, SERVER_PORT), timeout=5) as sock:
            sock.sendall(payload)
            data = sock.recv(2048)

        session_id_resp = data[:36].decode()
        nonce_recv = data[36:48]
        ct_recv = data[48:]

        if session_id_resp == st.session_state.session_id:
            response = decrypt_message(st.session_state.ephemeral_key, nonce_recv, ct_recv)
            st.success("Response received and decrypted.")
            st.session_state.response = response
            st.session_state.session_log.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "message": message,
                "response": response
            })
        else:
            st.warning("⚠️ Session ID mismatch")
    except Exception as e:
        st.error(f"[ERROR] {type(e).__name__}: {e}")

if st.session_state.response:
    st.subheader("🔐 Server Response")
    st.code(st.session_state.response)

# --- History Table ---
if st.session_state.session_log:
    st.subheader("📊 Session Log")
    df = pd.DataFrame(st.session_state.session_log)
    st.dataframe(df, use_container_width=True)
