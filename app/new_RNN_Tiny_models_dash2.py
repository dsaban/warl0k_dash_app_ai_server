import streamlit as st
from new_RNN_Tiny_models import GatedRNNBroker, TinyModelMCU
import numpy as np

st.set_page_config(page_title="WARL0K Fingerprint Verification", layout="wide")

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*=+[]{}|;:<>?~")

# Sidebar Parameters
with st.sidebar:
    st.title("⚙️ Parameters")
    seed = st.number_input("Seed", value=99, step=1)
    secret_len = st.slider("Secret Length", 2, 16, 8)
    train_epochs = st.slider("Training Epochs", 100, 2000, 1000, step=100)
    hidden_dim = st.slider("Hidden Dim (Broker)", 8, 128, 64)
    mcu_hidden = st.slider("Hidden Dim (MCU)", 4, 32, 8)
    train_button = st.button("Generate + Train")

# MCU Secret Generation
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🔐 PLC/MCU: Tiny Secret Fragments")
    mcu = TinyModelMCU(vocab, hidden_dim=mcu_hidden, seed=seed)
    fragments, fragment_strs = mcu.generate_3_secrets(length=secret_len)
    concat_secret = "".join(fragment_strs)
    for i, frag in enumerate(fragment_strs):
        st.code(f"Fragment {i+1}: {frag}", language='text')
    st.markdown(f"**🧬 Concatenated Secret:** `{concat_secret}`")
    st.text(f"RAM: {mcu.ram_usage_kb():.2f} KB")

# Broker RNN
with col2:
    st.subheader("🧠 Gateway Broker RNN")
    broker = GatedRNNBroker(vocab, hidden_dim=hidden_dim, lr=0.01)
    if train_button:
        broker.train(concat_secret, epochs=train_epochs)
    recon = broker.fingerprint_verify(concat_secret)
    st.code(recon, language='text')
    st.text(f"RAM: {broker.ram_usage_kb():.2f} KB")

# Verification
with col3:
    st.subheader("✅ Verification Result")
    if train_button:
        valid = recon == concat_secret
        st.markdown(f"### {'✅ Match' if valid else '❌ Mismatch'}")
        st.text(f"Target:        {concat_secret}")
        st.text(f"Reconstructed: {recon}")
