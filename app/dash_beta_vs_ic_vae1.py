"""warlok_vae_dashboard.py – Streamlit dashboard for Beta‑VAE vs IC‑VAE
---------------------------------------------------------------------
Run with:
    streamlit run warlok_vae_dashboard.py

This interactive dashboard lets you:
• **Generate** a synthetic sinusoid + spikes dataset.
• **Calibrate** hyper‑parameters from the sidebar.
• **Train** Beta‑VAE and IC‑VAE (imported from `warl0k_models.py`).
• **Compare** reconstructions, losses, and success/error metrics in a
  two‑column view.

NOTE: Ensure that `warl0k_models.py` is discoverable in PYTHONPATH and
exports `BetaVAE` and `ICVAE` classes matching the expected signatures.
"""

# ----------------------------- Imports ------------------------------
import streamlit as st
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings

# Import the models from the user‑supplied module
try:
    from warlok_models import BetaVAE, IC_VAE  # noqa: F401 (import unused)
except ModuleNotFoundError as e:
    st.error("❌ Could not import `warl0k_models.py`. Make sure it is in PYTHONPATH.")
    raise e

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ------------------------- Page Config ------------------------------
st.set_page_config(page_title="WARL0K VAE Dashboard", layout="wide")
st.title("WARL0K – Beta‑VAE vs IC‑VAE Dashboard")

# -------------------------- Sidebar UI ------------------------------
st.sidebar.header("Data & Hyper‑parameters")

# Data generation controls
points        = st.sidebar.slider("Data Points", 500, 5000, 2000, 500)
noise_std     = st.sidebar.number_input("Gaussian Noise σ", 0.0, 1.0, 0.33, 0.01, format="%0.2f")
spike_every_n = st.sidebar.number_input("Spike Every N Points", 10, 200, 50, step=10)
spike_mu      = st.sidebar.number_input("Spike μ", 0.5, 5.0, 2.0, 0.1)
spike_sigma   = st.sidebar.number_input("Spike σ", 0.1, 2.0, 0.5, 0.1)

st.sidebar.markdown("---")

# Model hyper‑parameters
latent_dim    = st.sidebar.slider("Latent Dim", 1, 16, 4)
beta_value    = st.sidebar.number_input("β (KL weight)", 0.0001, 1.0, 0.004, 0.0001, format="%0.4f")
learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs        = st.sidebar.slider("Epochs", 5, 200, 20, 5)
batch_size    = st.sidebar.slider("Batch Size", 2, 256, 32, 2)

st.sidebar.markdown("---")

success_thresh = st.sidebar.number_input("Success Threshold (abs error)", 0.01, 1.0, 0.1, 0.01)

# --------------------- Synthetic Data Gen ---------------------------

def generate_data(num_points:int, noise_std:float, spike_every:int, mu:float, sigma:float):
    x = np.linspace(-10, 10, num_points)
    y = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
    y += np.random.normal(0, noise_std, y.shape)
    spike_idx = np.arange(0, num_points, spike_every)
    y[spike_idx] += np.random.normal(mu, sigma, spike_idx.shape)
    return x, y

x, y = generate_data(points, noise_std, spike_every_n, spike_mu, spike_sigma)

# Normalize & split
scaler = MinMaxScaler()
scaled = scaler.fit_transform(y.reshape(-1, 1))

data_tensor = torch.tensor(scaled, dtype=torch.float32)
train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

# ------------------------ Data Preview ------------------------------
with st.expander("🔍 Preview Generated Data", expanded=False):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(y, label="Complex Signal")
    ax.set_title("Synthetic Data Preview")
    ax.legend()
    st.pyplot(fig)

# --------------------- Data Loaders ---------------------------------

def make_loaders(bs:int):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=bs, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = make_loaders(batch_size)

# ------------------------ Training ----------------------------------

def train_epoch(model: nn.Module, loader, opt):
    model.train(); losses = []
    for batch in loader:
        opt.zero_grad()
        if isinstance(model, IC_VAE):
            recon, mu, logvar, kl_w = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar, kl_w)
        else:  # BetaVAE
            recon, mu, logvar = model(batch)
            loss = model.loss_function(recon, batch, mu, logvar)
        loss.backward(); opt.step(); losses.append(loss.item())
    return np.mean(losses)

@st.cache_resource(show_spinner=True)
def train_models(latent:int, beta:float, lr:float, ep:int, bs:int):
    train_loader, _ = make_loaders(bs)
    beta_model = BetaVAE(1, latent, beta)
    ic_model   = IC_VAE(1, latent, beta)
    opt_beta = optim.Adam(beta_model.parameters(), lr=lr)
    opt_ic   = optim.Adam(ic_model.parameters(),   lr=lr)
    loss_beta, loss_ic = [], []
    for _ in range(ep):
        loss_beta.append(train_epoch(beta_model, train_loader, opt_beta))
        loss_ic.append(train_epoch(ic_model,   train_loader, opt_ic))
    return beta_model, ic_model, loss_beta, loss_ic

# ------------------ Train button & caching ---------------------------

if st.sidebar.button("🚀 Train / Retrain Models"):
    st.session_state["train_stamp"] = str(np.random.rand())  # force cache miss

cache_stamp = st.session_state.get("train_stamp", "init")

with st.spinner("Training models – this may take a moment..."):
    beta_model, ic_model, beta_losses, ic_losses = train_models(latent_dim, beta_value, learning_rate, epochs, batch_size)

# ---------------------- Evaluation ----------------------------------

def evaluate(model, loader):
    model.eval(); outs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(model, IC_VAE):
                recon, *_ = model(batch)
            else:
                recon, *_ = model(batch)
            outs.append(recon.cpu().numpy())
    recon = np.vstack(outs)
    return scaler.inverse_transform(recon)

_, test_loader_eval = make_loaders(batch_size)
recon_beta = evaluate(beta_model, test_loader_eval)
recon_ic   = evaluate(ic_model,   test_loader_eval)
orig_test  = scaler.inverse_transform(test_data.numpy())

# -------------------- Success/Error ---------------------------------

def success_error(orig, rec, thresh):
    err = np.abs(orig.flatten() - rec.flatten())
    success = (err <= thresh).mean() * 100
    return success, 100 - success

s_beta, e_beta = success_error(orig_test, recon_beta, success_thresh)
s_ic,   e_ic   = success_error(orig_test, recon_ic,   success_thresh)

# ------------------------ Visuals -----------------------------------

col_beta, col_ic = st.columns(2)

with col_beta:
    st.subheader("Beta‑VAE Reconstruction")
    df_beta = pd.DataFrame({"Original": orig_test.flatten(), "Beta‑VAE": recon_beta.flatten()})
    st.line_chart(df_beta)
    st.metric("Success %", f"{s_beta:.2f}%")
    st.metric("Error %",   f"{e_beta:.2f}%")

with col_ic:
    st.subheader("IC‑VAE Reconstruction")
    df_ic = pd.DataFrame({"Original": orig_test.flatten(), "IC‑VAE": recon_ic.flatten()})
    st.line_chart(df_ic)
    st.metric("Success %", f"{s_ic:.2f}%")
    st.metric("Error %",   f"{e_ic:.2f}%")

# ----------- Optional: show training loss curves --------------------
with st.expander("📉 Training Loss Curves", expanded=False):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(beta_losses, label="Beta‑VAE")
    ax.plot(ic_losses,   label="IC‑VAE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss History")
    ax.legend()
    st.pyplot(fig)

# -------------------------- Footer ----------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Warl0k Innovations – Powered by Streamlit & PyTorch")
