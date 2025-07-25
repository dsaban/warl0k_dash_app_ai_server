# streamlit_rnn_app.py
import streamlit as st
import numpy as np
from tiny_rnn_combine_secret_proof import TinyModelMCU, GatedRNNBroker

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

st.set_page_config("WARL0K MCU ↔ Broker RNN", layout="wide")
st.title("🔐 WARL0K Fingerprint Verification using Gated GATEWAY")

# --- Sidebar Parameters ---
with st.sidebar:
    seed = np.random.randint(0, 1000)
    st.header("🔧 Parameters")
    secret_len = st.slider("Secret Length per Segment", 4, 16, 8)
    mcu_dim = st.slider("MCU Hidden Dim", 4, 64, 8)
    rnn_dim = st.slider("Broker RNN Hidden Dim", 8, 128, 32)
    epochs = st.slider("Training Epochs", 5000, 20000, 16000, step=100)
    seed = st.number_input("Random Seed", value=seed, min_value=0, max_value=1000, step=1)

# --- Instantiate MCU + Generate Secret ---
tiny_mcu = TinyModelMCU(vocab=vocab, hidden_dim=mcu_dim, seed=seed)
secrets, secrets_str = tiny_mcu.generate_3_secrets(length=secret_len)

# --- Instantiate Broker RNN ---
broker = GatedRNNBroker(input_dim=mcu_dim, hidden_dim=rnn_dim, vocab=vocab)

# --- Train Broker ---
with st.spinner("Training Broker RNN on MCU secret..."):
    broker.train(secrets, secrets_str, secrets, epochs=epochs)
    st.success("✅ Broker training complete")

# --- Reconstruct Secrets ---
reconstructed_logits, _ = broker.forward(secrets)
reconstructed = []
for seq in reconstructed_logits:
    decoded = ''.join([vocab[np.argmax(logit)] for logit in seq])
    reconstructed.append(decoded)

# --- Output Results ---
st.subheader("📊 Verification Results")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔐 MCU Generated Secret")
    for i, s in enumerate(secrets_str):
        st.code(f"Secret {i+1}: {s}")
    st.metric("TinyModel RAM (KB)", f"{tiny_mcu.ram_usage_kb():.2f}")

with col2:
    st.subheader("🧠 GATEWAY validating Secrets")
    for i, s in enumerate(reconstructed):
        st.code(f"Reconstructed {i+1}: {s}")
        proof = "✅" if s == secrets_str[i] else "❌"
        st.metric(f"Proof {i+1} Match", proof)
    st.metric("Broker RNN RAM (KB)", f"{broker.ram_usage_kb():.2f}")

if st.button("🔄 Restart Simulation"):
    st.rerun()
