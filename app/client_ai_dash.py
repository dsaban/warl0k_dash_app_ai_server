import streamlit as st
import socket, uuid, random, torch, pickle, os
import pandas as pd
from model import add_noise_to_tensor
from utils import aead_encrypt, log_client

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def text_to_tensor(text):
    return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

# Set up Streamlit page
st.set_page_config("WARL0K Client Dashboard", layout="wide")
st.title("🔐 WARL0K Client – Secure Session & Authentication")

# Initialize
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
st.sidebar.title("📌 Session Info")
# st.sidebar.code(session_id, label="Session ID")
st.sidebar.subheader("Session ID")
st.sidebar.code(session_id)

# Step 1: Connect and Receive Obfuscated Secret
s = socket.socket()
s.connect(('localhost', 9996))
s.send(session_id.encode())

obfs = s.recv(64).decode()
log_client(f"[RECEIVED] Obfuscated secret: {obfs}")

st.success("✅ Obfuscated Secret Received")
st.code(obfs, language="text")

# Step 2: Generate fingerprint
torch.manual_seed(int(session_id[:8], 16))
fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
log_client(f"[GENERATED] Fingerprint: {fingerprint.numpy().tobytes().hex()}")

# Step 3: Encrypt data with fingerprint-derived key
key = fingerprint.numpy().tobytes()[:16]
secure_payload = aead_encrypt(key, b"This is my secure message")
log_client(f"[ENCRYPTED] Payload: {secure_payload}")

st.code(secure_payload, "Encrypted Payload")

# Step 4: Send fingerprint + payload
s2 = socket.socket()
s2.connect(('localhost', 9996))
s2.send(session_id.encode())
s2.send(pickle.dumps((fingerprint, secure_payload)))
log_client(f"[SENT] Fingerprint and encrypted payload sent.")

# Step 5: Receive Verification
response = s2.recv(1024).decode()
log_client(f"[RESPONSE] Server: {response}")

if "[OK]" in response:
    st.success("✅ Authentication successful")
else:
    st.error("❌ Authentication failed")

# Display Logs
st.subheader("📜 Server Log")
if os.path.exists("./logs/server.log"):
    with open("./logs/server.log") as f:
        lines = f.readlines()
    st.text_area("Log Output", "".join(lines), height=300)
else:
    st.info("No log file found.")

# # Basic stats (if log has fingerprints)
# with open("logs/client.log", "r") as f:
#     lines = f.readlines()
#
# if any("Fingerprint" in line for line in lines):
#     st.sidebar.subheader("📊 Stats")
#     st.sidebar.metric("Messages Sent", sum("Sent" in l for l in lines))
#     st.sidebar.metric("Auth Success", sum("✅" in l for l in lines))
if os.path.exists("logs/client.log"):
    with open("logs/client.log", "r") as f:
        lines = f.readlines()

    st.subheader("📜 Client Log Preview")
    st.code("".join(lines[-20:]), language="bash")

    if any("Fingerprint" in line for line in lines):
        st.success("🔍 Fingerprint-related process detected.")
    else:
        st.info("ℹ️ No fingerprint process detected yet.")
else:
    st.warning("Client log file not found.")

st.sidebar.title("📌 Session Info")
st.sidebar.subheader("Session ID")
st.sidebar.code(session_id)
