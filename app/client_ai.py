import socket, uuid, random, torch, pickle
from model import add_noise_to_tensor
from utils import aead_encrypt, log

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

def text_to_tensor(text):
    return torch.tensor([vocab.index(c) for c in text], dtype=torch.long).unsqueeze(1)

session_id = str(uuid.uuid4())
s = socket.socket()
s.connect(('localhost', 9996))
s.send(session_id.encode())

# Step 1: Receive Obfuscated Secret
obfs = s.recv(64).decode()
print("[RECEIVED] Obfuscated Secret:", obfs)
# Example after recving obfuscated secret:
log(f"Obfuscated secret: {obfs}")

# Step 2: Generate fingerprint
torch.manual_seed(int(session_id[:8], 16))
fingerprint = add_noise_to_tensor(text_to_tensor(obfs).squeeze(1), len(vocab)).unsqueeze(1)
log(f"Fingerprint: {fingerprint.numpy().tobytes().hex()}")

# Step 3: Encrypt data with fingerprint-derived key
key = fingerprint.numpy().tobytes()[:16]
encrypted_payload = aead_encrypt(key, b"This is my secure message")
print("[ENCRYPTED] Payload with Fingerprint-derived Key:", encrypted_payload)
log(f"Encrypted payload: {encrypted_payload}")

# Step 4: Send fingerprint + payload
s2 = socket.socket()
s2.connect(('localhost', 9996))
s2.send(session_id.encode())
s2.send(pickle.dumps((fingerprint, encrypted_payload)))
print("[SENT] Fingerprint + Encrypted Payload")
# Example after sending fingerprint + encrypted payload:
log(f"Sent fingerprint and encrypted payload to server: {session_id}")


# Step 5: Receive Verification
response = s2.recv(1024)
print("[RESPONSE]", response.decode())
log(f"response_received: {response.decode()}")

if response.decode() == "[OK] Authenticated & Decrypted.":
    print("[CLIENT] ✅ Authentication successful.")
else:
    print("[CLIENT] ❌ Authentication failed.")
# Close connections
