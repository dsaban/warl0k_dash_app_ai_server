import streamlit as st
# from gated_rnn_broker import GatedRNNBroker
# from tiny_model_mcu import TinyModelMCU
from new_RNN_Tiny_models import GatedRNNBroker, TinyModelMCU
import numpy as np

st.set_page_config(page_title="WARL0K Fingerprint Verification", layout="wide")

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*=+[]{}|;:<>?~")

# Sidebar Parameters
with st.sidebar:
    st.title("⚙️ Parameters")
    seed = st.number_input("Seed", value=42, step=1)
    secret_len = st.slider("Secret Length", 4, 32, 16)
    train_epochs = st.slider("Training Epochs", 100, 5000, 2000, step=100)
    hidden_dim = st.slider("Hidden Dim (Broker)", 8, 128, 64)
    mcu_hidden = st.slider("Hidden Dim (MCU)", 4, 32, 8)
    train_button = st.button("Generate + Train")

# MCU Secret Generation
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔐 PLC/MCU: Secret Generation")
    mcu = TinyModelMCU(vocab, hidden_dim=mcu_hidden, seed=seed)
    vec, secret = mcu.generate_secret(secret_len)
    st.code(secret, language='text')
    st.text(f"RAM: {mcu.ram_usage_kb():.2f} KB")

# Broker RNN
with col2:
    st.subheader("🧠 Gateway Broker RNN")
    broker = GatedRNNBroker(vocab, hidden_dim=hidden_dim, lr=0.01)
    if train_button:
        broker.train(secret, epochs=train_epochs)
    recon = broker.fingerprint_verify(secret)
    st.code(recon, language='text')
    st.text(f"RAM: {broker.ram_usage_kb():.2f} KB")

# Verification
with col3:
    st.subheader("✅ Verification Result")
    if train_button:
        valid = recon == secret
        st.markdown(f"### {'✅ Match' if valid else '❌ Mismatch'}")
        st.text(f"Target:      {secret}")
        st.text(f"Reconstructed: {recon}")
