# Streamlit WARL0K App: MCU TinyModel + Broker RNN Auth Flow
import streamlit as st
import numpy as np
from warlok_nano_micro_models import TinyModelMCU, GatedRNNBroker

# --- Vocab ---
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

# --- Streamlit UI ---
st.set_page_config("WARL0K Auth Demo", layout="wide")
st.title("🔐 WARL0K OT Authentication Simulator")

with st.sidebar:
    st.header("🛠️ Model Parameters")
    secret_len = st.slider("Secret Length", 4, 6, 8)
    mcu_dim = st.slider("MCU Hidden Dim", 4, 32, 8)
    rnn_dim = st.slider("RNN Hidden Dim", 32, 16, 8)
    epochs = st.slider("Training Epochs", 10000, 18000, 20000, step=100)
    seed = st.number_input("Random Seed", value=18, step=1)
    
#      add a button to reset the session state
    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.rerun()
#  add models size on RAM
    st.sidebar.subheader("Model RAM Usage")
    mcu_ram = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed).ram_usage_kb()
    rnn_ram = GatedRNNBroker(input_dim=mcu_dim, hidden_dim=rnn_dim, vocab=vocab).ram_usage_kb()
    st.sidebar.write(f"MCU Model: {mcu_ram:.2f} KB")
    st.sidebar.write(f"RNN Broker: {rnn_ram:.2f} KB")


# Run Models
mcu = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed)
sequences, secrets_str = mcu.generate_3_secrets(length=secret_len)
rnn = GatedRNNBroker(input_dim=mcu_dim, hidden_dim=rnn_dim, vocab=vocab)
rnn.train(sequences, secrets_str, sequences, epochs=epochs)
reconstructed_logits, _ = rnn.forward(sequences)

# Layout
col1, col2 = st.columns(2)
with col1:
    st.subheader("🔑 MCU Secrets")
    for i, s in enumerate(secrets_str):
        st.code(s, language="text")
with col2:
    st.subheader("🧠 RNN Reconstructed")
    for i, logits_seq in enumerate(reconstructed_logits):
        reconstructed = ''.join([vocab[np.argmax(l)] for l in logits_seq])
        st.code(reconstructed, language="text")
