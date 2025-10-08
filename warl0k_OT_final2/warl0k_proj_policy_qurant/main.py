
import json, time, hashlib, re
import numpy as np
from warlok.tiny_mcu import TinyModelMCUWithNoise
from warlok.counter_mcu import TinyModelCounterMCU
from warlok.broker_rnn import GatedRNNBrokerHot
from warlok.crypto_aead import encrypt_aead, decrypt_aead
from warlok.mqtt_utils import MqttApp

BROKER_HOST = "localhost"
BROKER_PORT = 1883

TOP_SECRETS    = "warlok/secrets"
TOP_VALIDATE   = "warlok/validation/{device}"
TOP_INGEST     = "warlok/ingest"
TOP_QUAR       = "warlok/quarantine"

VOCAB = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!~@#$%^&*_+")  # using full vocab
SECRET_LEN = 24

def sha8(s):
    return hashlib.sha256(s.encode()).hexdigest()[:8]

def policy_for(device_id: str, role: str):
    return {
        "policy_id": "POL-VALVE-OPEN@v1",
        "role": role,
        "template": "VALVE_OPEN",
        "bounds": {"min": 40, "max": 50},
        "device_id": device_id,
        "window": "T+30s"
    }

def canon_cmd(template: str, device_id: str, valve_id: int, value: int):
    return f"{template}:{device_id}:valve={valve_id},value={value}"

def parse_value(cmd: str):
    m = re.search(r"value=(\d+)", cmd)
    return int(m.group(1)) if m else None

def in_policy(cmd: str, policy: dict) -> bool:
    v = parse_value(cmd)
    if v is None: 
        return False
    b = policy["bounds"]
    return b["min"] <= v <= b["max"]

# ------------------ Device ------------------
def device_setup(client):
    client.device_id = "edge-PLC-001"
    client.role = "VendorTech"
    client.subscribe(TOP_VALIDATE.format(device=client.device_id), qos=1)
    print(f"[DEVICE] Subscribed: {TOP_VALIDATE.format(device=client.device_id)}")

def device_on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"[DEVICE] Validation result: {data}")

def device_loop(client):
    if getattr(client, "_sent", False):
        return
    client._sent = True

    session_seed = time.time_ns() % 2**32  # Use a session seed based on current time
    print(f"[DEVICE] Using session_seed={session_seed}")
    # Generate a secret string using the TinyModelMCUWithNoise
    # This simulates the device generating a secret string for the session.
    mcu = TinyModelMCUWithNoise(vocab=VOCAB, hidden_dim=8, seed=42)
    secret_string = mcu.generate_secret_string(SECRET_LEN, session_seed)
    print(f"[DEVICE] Generated secret_string={secret_string} session_seed={session_seed}")

    rng = np.random.RandomState(session_seed)
    seed_input = ''.join(rng.choice(VOCAB, size=SECRET_LEN))
    teacher = TinyModelCounterMCU(VOCAB, 8, 42).regenerate_secret_string(SECRET_LEN, session_seed)
    rnn_local = GatedRNNBrokerHot(vocab=VOCAB, hidden_dim=64, lr=0.02)
    rnn_local.train(seed_input, teacher, epochs=500)
    regenerated_local = rnn_local.regenerate(seed_input)
    local_ok = (regenerated_local == secret_string)
    if not local_ok:
        print(f"[DEVICE] ❌ Local RNN attestation failed: secret_digest={sha8(secret_string)} recon={regenerated_local}")
        return

    policy_ctx = policy_for(client.device_id, client.role)
    aad_bytes = json.dumps(policy_ctx, sort_keys=True).encode()

    cmds = [
        canon_cmd("VALVE_OPEN", client.device_id, valve_id=12, value=47),  # in policy
        canon_cmd("VALVE_OPEN", client.device_id, valve_id=12, value=60),  # out of policy
    ]

    for idx, payload_cmd in enumerate(cmds, 1):
        enc = encrypt_aead(secret_string, payload_cmd, aad_bytes)
        msg = {
            "vocab": "".join(VOCAB),
            "secret_len": SECRET_LEN,
            "session_seed": session_seed,
            "nonce_b64": enc["nonce_b64"],
            "cipher_b64": enc["cipher_b64"],
            "meta": {
                "device_id": client.device_id,
                "role": client.role,
                "ts": int(time.time()),
                "seq": idx
            }
        }
        client.on_message = device_on_message
        client.publish(TOP_SECRETS, json.dumps(msg), qos=1)
        print(f"[DEVICE] ▶ Published {TOP_SECRETS}: seq={idx} device={client.device_id} secret_digest={sha8(secret_string)} cmd={payload_cmd}")

# ------------------ Interceptor ------------------
def interceptor_setup(client):
    client.subscribe(TOP_SECRETS, qos=1)
    print(f"[INTERCEPTOR] Subscribed: {TOP_SECRETS}")

def interceptor_on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    vocab = list(data["vocab"])
    secret_len = int(data["secret_len"])
    session_seed = int(data["session_seed"])
    nonce_b64 = data["nonce_b64"]
    cipher_b64 = data["cipher_b64"]
    meta = data.get("meta", {})
    device_id = meta.get("device_id", "unknown")
    role = meta.get("role", "unknown")

    rng = np.random.RandomState(session_seed)
    seed_input = ''.join(rng.choice(vocab, size=secret_len))
    teacher = TinyModelCounterMCU(vocab, 8, 42).regenerate_secret_string(secret_len, session_seed)
    rnn = GatedRNNBrokerHot(vocab=vocab, hidden_dim=64, lr=0.02)
    rnn.train(seed_input, teacher, epochs=500)
    regenerated_secret = rnn.regenerate(seed_input)
    local_ok = (regenerated_secret == teacher)
    if not local_ok:
        print(f"[INTERCEPTOR] ❌ Local RNN attestation failed: secret_digest={sha8(regenerated_secret)} recon={regenerated_secret}")
        return
    print(f"[INTERCEPTOR] Regenerated secret: {regenerated_secret} "
          f"session_seed={session_seed} "
          f"seed_input={seed_input} "
            f"device_id={device_id} "
            f"role={role}")
    # Validate against policy
    if not device_id or not role:
        print(f"[INTERCEPTOR] ❌ Missing device_id or role in metadata: {meta}")
        return
    if not re.match(r"^[a-zA-Z0-9_-]+$", device_id):
        print(f"[INTERCEPTOR] ❌ Invalid device_id format: {device_id}")
        return

    policy_ctx = policy_for(device_id, role)
    aad_bytes = json.dumps(policy_ctx, sort_keys=True).encode()

    try:
        plaintext = decrypt_aead(regenerated_secret, nonce_b64, cipher_b64, aad_bytes)
        ok = in_policy(plaintext, policy_ctx)
        out_topic = TOP_INGEST if ok else TOP_QUAR

        out = {
            "meta": meta,
            "policy_id": policy_ctx["policy_id"],
            "plaintext": plaintext,
            "validated": ok,
            "reason": None if ok else "policy_violation",
            "secret_digest": sha8(regenerated_secret)
        }
        client.publish(out_topic, json.dumps(out), qos=1)
        client.publish(TOP_VALIDATE.format(device=device_id), json.dumps(out), qos=1)

        tag = "✅ AUTH" if ok else "🚫 QUAR"
        print(f"[INTERCEPTOR] {tag} seq={meta.get('seq')} device={device_id} cmd={plaintext} secret_digest={sha8(regenerated_secret)}")

    except Exception as e:
        out = {
            "meta": meta,
            "policy_id": policy_ctx["policy_id"],
            "reason": f"auth_failed:{repr(e)}",
            "validated": False
        }
        client.publish(TOP_QUAR, json.dumps(out), qos=1)
        client.publish(TOP_VALIDATE.format(device=device_id), json.dumps(out), qos=1)
        print(f"[INTERCEPTOR] ❌ FAIL (cryptographic) seq={meta.get('seq')} device={device_id} reason={out['reason']}")

# ------------------ Gateway ------------------
def gateway_setup(client):
    client.subscribe(TOP_INGEST, qos=1)
    client.subscribe(TOP_QUAR, qos=1)
    print(f"[GATEWAY] Subscribed: {TOP_INGEST}, {TOP_QUAR}")

def gateway_on_message(client, userdata, msg):
    data = json.loads(msg.payload.decode())
    print(f"[GATEWAY] {msg.topic}: {data}")

def main():
    gateway = MqttApp("gateway", on_setup=gateway_setup, host=BROKER_HOST, port=BROKER_PORT)
    interceptor = MqttApp("interceptor", on_setup=interceptor_setup, host=BROKER_HOST, port=BROKER_PORT)
    device = MqttApp("device", on_setup=device_setup, on_loop=device_loop, host=BROKER_HOST, port=BROKER_PORT)

    gateway.client.on_message = gateway_on_message
    interceptor.client.on_message = interceptor_on_message

    gateway.start()
    interceptor.start()
    device.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
