# streamlit_mcu_rnn_proof.py
import streamlit as st
import numpy as np
import random
from warlok_nano_micro_models import TinyModelMCU, GatedRNNBroker

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

st.set_page_config("WARL0K MCU vs RNN Proof", layout="wide")
st.title("🔐 WARL0K MCU ↔ Gateway RNN Fingerprint Verification")

with st.sidebar:
    st.header("🔧 Parameters")
    secret_len = st.slider("Secret Length per TinyModel", 8, 6, 4)
    mcu_dim = st.slider("TinyModel Hidden Dim", 4, 64, 8)
    rnn_dim = st.slider("RNN Hidden Dim", 8, 64, 8)
    heads = st.slider("Attention Heads", 1, 4, 2)
    epochs = st.slider("Training Epochs", 10000, 12000, 15000, step=100)
    seed = st.number_input("Random Seed", value=random.randint(0, 1000), min_value=0, max_value=1000, step=1)

# --- Instantiate Models ---
mcu = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed)
rnn = GatedRNNBroker(input_dim=mcu_dim,
                     hidden_dim=rnn_dim,
                     vocab=vocab,
                     num_heads=heads,
                     seed=seed)

# --- Generate and Train ---
sequences, secrets_str = mcu.generate_3_secrets(length=secret_len)
rnn.train(sequences, secrets_str, sequences, epochs=epochs)
reconstructed_logits, _ = rnn.forward(sequences)

# --- Evaluate ---
reconstructed = ''.join([vocab[np.argmax(logit)] for logit in reconstructed_logits[0]])
proof_valid = secrets_str[0] == reconstructed

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔐 MCU Concatenated Secret")
    st.code(secrets_str[0], language="text")
    st.metric("TinyModel RAM (KB)", f"{mcu.ram_usage_kb():.2f}")

with col2:
    st.subheader("🧠 Gateway RNN Reconstruction")
    st.code(reconstructed, language="text")
    st.metric("RNN RAM (KB)", f"{rnn.ram_usage_kb():.2f}")
    st.metric("Proof Verified", "✅" if proof_valid else "❌")
