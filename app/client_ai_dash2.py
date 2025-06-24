import streamlit as st
import socket, uuid, random, torch, pickle, os
import pandas as pd
from model import add_noise_to_tensor, inject_patterned_noise
from utils import aead_encrypt, log_client

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def text_to_tensor(text):
    return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

st.set_page_config("WARL0K Client Dashboard", layout="wide")
st.title("🔐 WARL0K Client – Secure Session & Authentication")

# Sidebar reload button
if st.sidebar.button("🔄 Reload Authentication Process"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Session init
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
st.sidebar.title("📌 Session Info")
st.sidebar.subheader("Session ID")
st.sidebar.code(session_id)

# Step 1: Connect and get obfuscated secret
try:
    s = socket.socket()
    s.connect(('localhost', 9996))
    s.send(session_id.encode())
    obfs = s.recv(64).decode()
    log_client(f"[RECEIVED] Obfuscated secret: {obfs}")
    st.success("✅ Obfuscated Secret Received")
    st.code(obfs, language="text")
except Exception as e:
    st.error(f"Failed to receive obfuscated secret: {e}")
    st.stop()

# Step 2: Generate fingerprint
torch.manual_seed(int(session_id[:8], 16))
fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
#  convert to ascii characters into 1 dimensional tensor
noisy_tensor = inject_patterned_noise(
            seq_tensor=text_to_tensor(obfs).squeeze(1),
            vocab_size=len(vocab),
            error_rate=0.25,
            pattern_ratio=0.6,  # Example ratio, can be adjusted
            seed=session_id
        ).unsqueeze(1)

noisy_obf_secret = ''.join([vocab[i] for i in noisy_tensor.squeeze(1).tolist()])
print(f"[CLIENT] Injected noisy obf_secret: {noisy_obf_secret}")
log_client(f"[GENERATED] Fingerprint: {noisy_obf_secret}")

# Step 3: AEAD encryption
key = fingerprint.numpy().tobytes()[:16]
secure_payload = aead_encrypt(key, b"This is my secure message")
log_client(f"[ENCRYPTED] Payload: {secure_payload}")
st.code(secure_payload, "Encrypted Payload")

# Step 4: Send fingerprint + payload
try:
    s2 = socket.socket()
    s2.connect(('localhost', 9996))
    s2.send(session_id.encode())
    s2.send(pickle.dumps((fingerprint, secure_payload)))
    log_client(f"[SENT] Fingerprint and encrypted payload sent.")
except Exception as e:
    st.error(f"Failed to send payload: {e}")
    st.stop()

# Step 5: Receive server response
try:
    response = s2.recv(1024).decode()
    log_client(f"[RESPONSE] Server: {response}")
    if "[OK]" in response:
        st.success("✅ Authentication successful")
    else:
        st.error("❌ Authentication failed")
except Exception as e:
    st.error(f"Failed to receive verification response: {e}")
    st.stop()

# Logs display
st.subheader("📜 Server Log")
if os.path.exists("./logs/server.log"):
    with open("./logs/server.log") as f:
        lines = f.readlines()
    st.text_area("Server Log Output", "".join(lines[-20:]), height=300)
else:
    st.info("No server log file found.")

st.subheader("📜 Client Log")
if os.path.exists("logs/client.log"):
    with open("logs/client.log", "r") as f:
        lines = f.readlines()
    st.code("".join(lines[-20:]), language="bash")
    st.sidebar.subheader("📊 Metrics")
    st.sidebar.metric("Messages Sent", sum("Payload" in l for l in lines))
    st.sidebar.metric("Auth Success", sum("[OK]" in l for l in lines))
else:
    st.warning("Client log not found.")

