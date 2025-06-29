import streamlit as st
import socket, uuid, torch, pickle, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import add_noise_to_tensor, inject_patterned_noise
from utils import aead_encrypt, log_client
from plotly import express as px
import random

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def text_to_tensor(text):
	return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

st.set_page_config("WARL0K Control Room", layout="wide")
st.title("WARL0K Client Control Room")

if st.sidebar.button("🔄 Reload Authentication Process"):
	for key in list(st.session_state.keys()):
		del st.session_state[key]
	st.rerun()

if "session_id" not in st.session_state:
	st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

st.sidebar.title("📌 Session Info")
st.sidebar.subheader("Session ID")
st.sidebar.code(session_id)

# --- Receive obfuscated secret
try:
	s = socket.socket()
	s.connect(('localhost', 9995))
	s.send(session_id.encode())
	obfs = s.recv(64).decode()
	log_client(f"[RECEIVED] Obfuscated secret: {obfs}")
except Exception as e:
	st.error(f"Failed to receive obfuscated secret: {e}")
	st.stop()

# --- Fingerprint + Noisy variant
torch.manual_seed(int(session_id[:8], 16))
fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
noisy_tensor = inject_patterned_noise(
	seq_tensor=text_to_tensor(obfs).squeeze(1),
	vocab_size=len(vocab),
	error_rate=0.25,
	pattern_ratio=0.6,
	seed=session_id
).unsqueeze(1)
noisy_obf_secret = ''.join([vocab[i] for i in noisy_tensor.squeeze(1).tolist()])
log_client(f"[FINGERPRINT] {fingerprint.squeeze(1).tolist()}")
log_client(f"[NOISY] {noisy_obf_secret}")

# --- Encrypt
key = fingerprint.numpy().tobytes()[:16]
secure_payload = aead_encrypt(key, b"This is my secure message")
log_client(f"[ENCRYPTED] {secure_payload}")

# --- Send to server
try:
	s2 = socket.socket()
	s2.connect(('localhost', 9995))
	s2.send(session_id.encode())
	s2.send(pickle.dumps((fingerprint, secure_payload)))
	response = s2.recv(1024).decode()
	log_client(f"[RESPONSE] Server: {response}")
	auth_status = "✅ Authentication successful" if "[OK]" in response else "❌ Authentication failed"
except Exception as e:
	auth_status = f"Transmission error: {e}"

# --- Three-column layout ---
col1, col2, col3 = st.columns(3)

# ----- 📊 Column 1: Graphs -----
with col1:
	st.subheader("📡 Fingerprint Drift Analysis")
	fp_df = pd.DataFrame({
		"Index": list(range(len(fingerprint))),
		"Original": fingerprint.squeeze(1).tolist(),
		"Noisy": noisy_tensor.squeeze(1).tolist()
	})
	st.line_chart(fp_df.set_index("Index"))

	st.subheader("🧬 Secret Comparison Lattice")
	fig, ax = plt.subplots(figsize=(10, 3))
	df_lattice = pd.DataFrame({
		"Obfuscated": list(obfs),
		"Noisy": list(noisy_obf_secret),
		"Fingerprint": [vocab[i] for i in fingerprint.squeeze(1).tolist()]
	})
	annot_data = df_lattice.T.values.tolist()
	sns.heatmap(
		data=[[ord(c) for c in row] for row in annot_data],
		cmap="Blues", cbar=False,
		annot=annot_data, fmt='s', ax=ax
	)
	ax.set_yticklabels(df_lattice.columns)
	ax.set_xticks([])
	ax.set_title("Alignment Matrix")
	st.pyplot(fig)

# ----- 📜 Column 2: Logs -----
with col2:
	st.subheader("📜 Server Log")
	if os.path.exists("logs/server.log"):
		with open("logs/server.log", "r") as f:
			lines = f.readlines()
		st.text_area("Server Log", "".join(lines[-20:]), height=200)
	else:
		st.info("No server log found.")

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

# ----- 🔐 Column 3: Secrets & Status -----
with col3:
	st.subheader("🔐 Obfuscated Secret")
	st.code(obfs, language="text")

	st.subheader("🧬 Noisy Fingerprint")
	st.code(noisy_obf_secret, language="text")

	st.subheader("🔍 Fingerprint Vector")
	st.code(fingerprint.squeeze(1).tolist(), language="python")

	st.subheader("🔐 Encrypted Payload")
	st.code(secure_payload, language="bash")

	st.subheader("✅ Status")
	if "Authentication successful" in auth_status:
		st.success(auth_status)
	elif "Authentication failed" in auth_status:
		st.error(auth_status)
	else:
		st.warning(auth_status)

# 4th panel: Telemetry + IP map
st.divider()
col_a, col_b = st.columns(2)

with col_a:
	st.subheader("📊 Sensor Telemetry Simulation")
	telemetry_df = pd.DataFrame({
		"Temp": torch.randn(30).numpy() * 5 + 30,
		"Voltage": torch.randn(30).numpy() * 0.1 + 3.3,
		"Noise": torch.rand(30).numpy()
	})
	st.line_chart(telemetry_df)

with col_b:
	st.subheader("🌐 IP/Network Map")
	network_df = pd.DataFrame({
		"Client": [f"Client-{i}" for i in range(1, 6)],
		"IP": [f"192.168.0.{i + 10}" for i in range(5)],
		"Status": random.choices(["Active", "Idle", "Disconnected"], k=5),
		"Latency": [random.randint(10, 90) for _ in range(5)]
	})
	fig = px.scatter(network_df, x="Latency", y="Client", color="Status", size="Latency", hover_data=["IP"])
	st.plotly_chart(fig, use_container_width=True)
