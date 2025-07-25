# streamlit_icvae_app.py
import streamlit as st
import numpy as np
from warlok_nano_micro_models import TinyModelMCU, IncrementalVAE

vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

st.set_page_config("WARL0K IC-VAE Demo", layout="wide")
st.title("🔐 WARL0K MCU ↔ Gateway IC-VAE Fingerprint Verification")

with st.sidebar:
    st.header("🧪 IC-VAE Parameters")
    secret_len = st.slider("Secret Length per Segment", 4, 16, 8)
    mcu_dim = st.slider("TinyModel Hidden Dim", 4, 64, 8)
    latent_dim = st.slider("Latent Dim (z)", 4, 32, 8)
    hidden_dim = st.slider("Encoder/Decoder Hidden Dim", 8, 64, 32)
    epochs = st.slider("Training Epochs", 100, 3000, 1000, step=100)
    beta = st.slider("KL Beta", 0.01, 2.0, 0.5, step=0.01)
    seed = st.number_input("Random Seed", value=224)

# --- MCU: Generate Concatenated Secret ---
mcu = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed)
x_data, secrets_str = mcu.generate_3_secrets(length=secret_len)

# Convert secret string to one-hot
vocab_size = len(vocab)
seq_len = len(secrets_str[0])
onehot = np.zeros((seq_len, vocab_size))
for i, c in enumerate(secrets_str[0]):
    onehot[i, vocab.index(c)] = 1
x_data = [onehot.reshape(1, -1)]

# --- IC-VAE Model ---
beta_schedule = lambda epoch: beta
icvae = IncrementalVAE(input_dim=seq_len * vocab_size, latent_dim=latent_dim, hidden_dim=hidden_dim, beta_schedule=beta_schedule)

with st.spinner("Training IC-VAE model..."):
    icvae.train(x_data, epochs=epochs, lr=0.01)
    st.success("✅ Training complete!")

# --- Inference ---
recon, mu, logvar = icvae.forward(np.array(x_data[0]))
kl = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
mse = np.mean((np.array(x_data[0]) - recon)**2)
recon_str = ''.join([vocab[np.argmax(logit)] for logit in recon[0]]) if recon.ndim > 1 else ''.join([vocab[np.argmax(logit)] for logit in recon])
proof_valid = recon_str == secrets_str[0]

# --- Output Summary ---
st.subheader("📊 IC-VAE Final Output Summary")
st.text(f"Latent Mean (mu):\n{mu.flatten()[:min(10, len(mu.flatten()))]}")
st.text(f"Latent LogVar:\n{logvar.flatten()[:min(10, len(logvar.flatten()))]}")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🔐 MCU Combined Secret")
    st.code(secrets_str[0], language="text")
    st.metric("TinyModel RAM (KB)", f"{mcu.ram_usage_kb():.2f}")

with col2:
    st.subheader("🧠 IC-VAE Reconstruction & KL")
    st.code(recon_str, language="text")
    st.metric("Reconstruction Error", f"{mse:.4f}")
    st.metric("KL Divergence", f"{kl:.4f}")
    st.metric("Proof Verified", "✅" if proof_valid else "❌")
    st.metric("VAE RAM (KB)", f"{icvae.ram_usage_kb():.2f}")

if st.button("🔄 Restart Training"):
    st.rerun()
