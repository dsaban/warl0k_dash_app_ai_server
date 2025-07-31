# WARL0K Unified Streamlit Dashboard: Edge Auth + SCADA Anomaly Detection

import streamlit as st
import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, plotly.express as px
import uuid, random
from sklearn.preprocessing import MinMaxScaler
from torch import optim
from torch.utils.data import DataLoader
from warlok_models import ICVAE, HICVAE
from new_RNN_Tiny_models import GatedRNNBroker, TinyModelMCU
from model import train_secret_regenerator, evaluate_secret_regenerator, inject_patterned_noise, add_noise_to_tensor
from utils import generate_secret, aead_encrypt, aead_decrypt, log_client

# Config
st.set_page_config("WARL0K Unified Dashboard", layout="wide")
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*=+[]{}|;:<>?~")
session_id = str(uuid.uuid4())[:8]

# Sidebar
st.sidebar.header("🔧 MCU & Broker Settings")
seed = st.sidebar.number_input("Seed", value=42, step=1)
secret_len = st.sidebar.slider("Secret Length", 2, 16, 8)
epochs = st.sidebar.slider("Training Epochs", 100, 2000, 1000, 100)
broker_dim = st.sidebar.slider("Broker Hidden", 8, 128, 64)
mcu_dim = st.sidebar.slider("MCU Hidden", 4, 32, 16)

st.sidebar.header("🧠 Anomaly Detection")
latent_dim = st.sidebar.slider("Latent Dim", 1, 8, 2)
learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
vae_epochs = st.sidebar.slider("VAE Epochs", 10, 200, 100, 100)
batch_size = st.sidebar.slider("Batch Size", 8, 128, 16, 16)
patience = st.sidebar.slider("Early-Stop Patience", 3, 30, 15)
spike_thresh = st.sidebar.number_input("Spike Threshold", value=3.0)
time_dev_percent = st.sidebar.slider("Timing Deviation (%)", 5, 100, 20, 5)

if st.sidebar.button("▶ Run Unified Process"):
    st.session_state.run = True
if st.sidebar.button("🔁 Reset"):
    st.session_state.clear()
    st.rerun()

if "run" in st.session_state:

    # Layer 1: MCU + Gateway + SCADA
    col1, col2 = st.columns(2)

    with col1:
        st.header("🔐 Edge: MCU & Gateway")
        mcu = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed)
        fragments, frag_strs = mcu.generate_3_secrets(length=secret_len)
        full_secret = ''.join(frag_strs)
        broker = GatedRNNBroker(vocab, hidden_dim=broker_dim, lr=0.01)
        broker.train(full_secret, epochs=epochs)
        recon = broker.fingerprint_verify(full_secret)
        valid = recon == full_secret

        st.code(f"Fragments: {frag_strs}")
        st.markdown(f"🔗 Concatenated: `{full_secret}`")
        st.markdown(f"📡 Reconstructed: `{recon}` — {'✅ Match' if valid else '❌ Mismatch'}")
        st.caption(f"MCU RAM: {mcu.ram_usage_kb():.2f} KB | Broker RAM: {broker.ram_usage_kb():.2f} KB")

    with col2:
        st.header("🧬 SCADA Session")
        master = generate_secret()
        obfs = generate_secret()
        model_master = train_secret_regenerator(master, vocab)
        model_obfs = train_secret_regenerator(master, vocab, input_override=obfs)

        torch.manual_seed(int(session_id, 16))
        noisy = inject_patterned_noise(torch.tensor([vocab.index(c) for c in obfs]), len(vocab), 0.25, 0.6, session_id)
        fingerprint = add_noise_to_tensor(torch.tensor([vocab.index(c) for c in obfs]), len(vocab)).unsqueeze(1)
        key = fingerprint.numpy().tobytes()[:16]
        encrypted = aead_encrypt(key, b"Payload: Hello WARL0K")
        decrypted = aead_decrypt(key, encrypted.encode())
        recovered = evaluate_secret_regenerator(model_obfs, fingerprint, vocab)
        status = "✅ AUTH OK" if recovered == master else "❌ AUTH FAIL"

        st.code(f"Injected Obfuscated: {obfs}")
        st.code(f"Injected Noisy: {''.join([vocab[i] for i in noisy.tolist()])}")
        st.line_chart(pd.DataFrame({
            "Index": list(range(len(obfs))),
            "Obfuscated": [vocab.index(c) for c in obfs],
            "Noisy": noisy.tolist(),
            "Fingerprint": fingerprint.squeeze(1).tolist()
        }).set_index("Index"))
        st.markdown(f"🔐 Recovered: `{recovered}` — {status}")
        st.code(decrypted, language="bash")

    # Layer 2: Anomaly Detection
    st.header("📈 IC‑VAE & HIC‑VAE Anomaly Detection")
    x = np.linspace(-10, 10, 1000)
    base = np.sin(x) + np.sin(2*x) + np.sin(3*x)
    noise = 0.5 * np.random.normal(size=1000)
    spikes = np.random.choice([0,1], size=1000, p=[0.95,0.05]) * np.random.uniform(3,6,size=1000)
    y = base + noise + spikes
    scaler = MinMaxScaler()
    scaled_y = scaler.fit_transform(y.reshape(-1,1))
    tensor_y = torch.tensor(scaled_y, dtype=torch.float32)
    train_data, test_data = torch.utils.data.random_split(tensor_y, [800, 200])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=False)

    ic, hic = ICVAE(1, latent_dim), HICVAE(1, latent_dim)
    opt_ic = optim.Adam(ic.parameters(), lr=learning_rate)
    opt_hic = optim.Adam(hic.parameters(), lr=learning_rate)
    loss_ic = loss_hic = []
    best_ic = best_hic = np.inf
    stall_ic = stall_hic = 0

    for ep in range(vae_epochs):
        ic.train(); hic.train()
        bl_ic = bl_hic = 0
        for b in train_loader:
            opt_ic.zero_grad(); opt_hic.zero_grad()
            oic, mic, lic, kic = ic(b)
            loss = ic.loss_function(oic, b, mic, lic, kic)
            loss.backward(); opt_ic.step(); bl_ic += loss.item()
            ohic, m1, l1, m2, l2, kh = hic(b)
            loss = hic.loss_function(ohic, b, m1, l1, m2, l2, kh)
            loss.backward(); opt_hic.step(); bl_hic += loss.item()
        loss_ic.append(bl_ic/len(train_loader))
        loss_hic.append(bl_hic/len(train_loader))
        if loss_ic[-1] < best_ic: best_ic, stall_ic = loss_ic[-1], 0
        else: stall_ic += 1
        if loss_hic[-1] < best_hic: best_hic, stall_hic = loss_hic[-1], 0
        else: stall_hic += 1
        if stall_ic >= patience and stall_hic >= patience: break

    def reconstruct(model):
        model.eval()
        outs = [model(b)[0].detach().numpy() for b in test_loader]
        return np.vstack(outs)

    def detect_spikes(sig):
        return np.where(np.abs(sig) >= spike_thresh)[0]

    def detect_timing_anoms(idx):
        if len(idx)<2: return np.array([], int)
        intervals = np.diff(idx)
        med = np.median(intervals)
        return idx[np.where(np.abs(intervals - med) > med * (time_dev_percent/100))[0]+1]

    rec_ic = scaler.inverse_transform(reconstruct(ic))
    rec_hic = scaler.inverse_transform(reconstruct(hic))
    orig = scaler.inverse_transform(test_data.dataset[:][test_data.indices].numpy())

    spikes_ic = detect_spikes(rec_ic.flatten())
    spikes_hic = detect_spikes(rec_hic.flatten())
    time_ic = detect_timing_anoms(spikes_ic)
    time_hic = detect_timing_anoms(spikes_hic)

    c1, c2 = st.columns(2)
    for col, name, rec, spk, tim in [
        (c1,"IC‑VAE",  rec_ic,  spikes_ic,  time_ic),
        (c2,"HIC‑VAE", rec_hic, spikes_hic, time_hic)]:

        col.subheader(f"{name} Reconstruction & Anomalies")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(orig, label="Original", alpha=0.4)
        ax.plot(rec,  label="Reconstruction", alpha=0.8)
        ax.scatter(spk, rec[spk], color="orange", s=20, label="Spike")
        ax.scatter(tim, rec[tim], color="red", s=25, label="Timing∆")
        ax.set_xlabel("Index"); ax.set_ylabel("Value"); ax.legend()
        col.pyplot(fig)

        col.metric("Spikes", len(spk))
        col.metric("Timing anomalies", len(tim))
