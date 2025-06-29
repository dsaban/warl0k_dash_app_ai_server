import streamlit as st
import socket, uuid, torch, pickle, os, threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from model import (
	add_noise_to_tensor, inject_patterned_noise,
	train_secret_regenerator, evaluate_secret_regenerator
)
from utils import (
	aead_encrypt, aead_decrypt, generate_secret, log, log_client
)
# --- Ensure logs directory and log files exist ---
os.makedirs("logs", exist_ok=True)
open("logs/client.log", "a").close()
open("logs/server.log", "a").close()

# --- Shared Config ---
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
HOST = '0.0.0.0'
PORT = 9992
SESSIONS = {}

def text_to_tensor(text):
	return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

# --- TCP Server Code ---
def handle_client(conn):
	session_id = conn.recv(36).decode()
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
		log(f"[Server] Obfuscated secret: {obfs}")
		return

	raw = conn.recv(4096)
	if not raw:
		return

	try:
		_fingerprint, encrypted_payload = pickle.loads(raw)
		obfs = SESSIONS[session_id]["obfs"]
		torch.manual_seed(int(session_id[:8], 16))
		noisy_tensor = inject_patterned_noise(
			text_to_tensor(obfs).squeeze(1), len(vocab), 0.25, 0.6, session_id
		).unsqueeze(1)
		noisy_obf_secret = ''.join([vocab[i] for i in noisy_tensor.squeeze(1).tolist()])
		log(f"[Server] Received fingerprint: {noisy_obf_secret}")

		recovered = evaluate_secret_regenerator(SESSIONS[session_id]["model_obfs"], _fingerprint, vocab)
		log(f"[Server][RECONSTRUCT] Master: {recovered}")

		if recovered != SESSIONS[session_id]["master"]:
			conn.send(b"[FAIL] Authentication failed.")
			return

		decrypted = aead_decrypt(_fingerprint.numpy().tobytes()[:16], encrypted_payload)
		log(f"[Server] [✓] Payload Decrypted: {decrypted.decode()}")
		conn.send(b"[OK] Authenticated & Decrypted.")
	except Exception as e:
		log(f"[Server] [ERROR] {e}")
		conn.send(b"[ERR] Processing failed.")

def start_server():
	# s = socket.socket()
	# s.bind((HOST, PORT))
	# s.listen(5)
	s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # 🔥 Important fix
	s.bind((HOST, PORT))
	s.listen(5)
	
	while True:
		conn, _ = s.accept()
		threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

# Start server in background
threading.Thread(target=start_server, daemon=True).start()

# --- Streamlit UI ---
st.set_page_config("WARL0K Control Room", layout="wide")
st.title("WARL0K Client Control Room")

# Sidebar
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

# Connect to server and receive obfuscated secret
try:
	s = socket.socket()
	s.connect(('localhost', PORT))
	s.send(session_id.encode())
	obfs = s.recv(64).decode()
	log_client(f"[RECEIVED] Obfuscated secret: {obfs}")
except Exception as e:
	st.error(f"Failed to receive obfuscated secret: {e}")
	st.stop()

torch.manual_seed(int(session_id[:8], 16))
fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
noisy_tensor = inject_patterned_noise(
	text_to_tensor(obfs).squeeze(1), len(vocab), 0.25, 0.6, session_id
).unsqueeze(1)
noisy_obf_secret = ''.join([vocab[i] for i in noisy_tensor.squeeze(1).tolist()])
log_client(f"[FINGERPRINT] {fingerprint.squeeze(1).tolist()}")
log_client(f"[NOISY] {noisy_obf_secret}")

key = fingerprint.numpy().tobytes()[:16]
secure_payload = aead_encrypt(key, b"This is my secure message")
log_client(f"[ENCRYPTED] {secure_payload}")

# Send payload
try:
	s2 = socket.socket()
	s2.connect(('localhost', PORT))
	s2.send(session_id.encode())
	s2.send(pickle.dumps((fingerprint, secure_payload)))
	response = s2.recv(1024).decode()
	log_client(f"[RESPONSE] Server: {response}")
	auth_status = "✅ Authentication successful" if "[OK]" in response else "❌ Authentication failed"
except Exception as e:
	auth_status = f"Transmission error: {e}"

# Display: Three Columns
col1, col2, col3 = st.columns(3)

with col1:
	st.subheader("📡 Fingerprint Drift")
	df = pd.DataFrame({
		"Index": range(len(fingerprint)),
		"Original": fingerprint.squeeze(1).tolist(),
		"Noisy": noisy_tensor.squeeze(1).tolist()
	})
	st.line_chart(df.set_index("Index"))
	st.subheader("🧬 Secret Alignment")
	fig, ax = plt.subplots(figsize=(10, 3))
	df_lattice = pd.DataFrame({
		"Obfuscated": list(obfs),
		"Noisy": list(noisy_obf_secret),
		"Fingerprint": [vocab[i] for i in fingerprint.squeeze(1).tolist()]
	})
	sns.heatmap([[ord(c) for c in row] for row in df_lattice.T.values.tolist()],
				cmap="Blues", cbar=False, annot=df_lattice.T.values.tolist(), fmt='s', ax=ax)
	ax.set_yticklabels(df_lattice.columns)
	ax.set_xticks([])
	st.pyplot(fig)

with col2:
	st.subheader("📜 Server Log")
	if os.path.exists("logs/server.log"):
		with open("logs/server.log") as f:
			st.text_area("Server", "".join(f.readlines()[-20:]), height=200)
	st.subheader("📜 Client Log")
	if os.path.exists("logs/client.log"):
		with open("logs/client.log") as f:
			lines = f.readlines()
			st.code("".join(lines[-20:]), language="bash")
			st.sidebar.metric("Messages Sent", sum("Payload" in l for l in lines))
			st.sidebar.metric("Auth OK", sum("[OK]" in l for l in lines))

with col3:
	#  add obsfucated secret
	st.subheader("🧬 Obfuscated Secret")
	st.code(obfs, language="text")
	st.subheader("🧬 Noisy Fingerprint")
	st.code(noisy_obf_secret, language="text")
	st.subheader("🔍 Fingerprint Vector")
	st.code(fingerprint.squeeze(1).tolist(), language="python")
	st.subheader("🔐 Encrypted Payload")
	st.code(secure_payload, language="bash")
	st.subheader("✅ Status")
	if "✅" in auth_status:
		st.success(auth_status)
	elif "❌" in auth_status:
		st.error(auth_status)
	else:
		st.warning(auth_status)
