"""icvae_streamlit_app.py – WARL0K anomaly‑aware dashboard
-----------------------------------------------------------
Run with:
    streamlit run icvae_streamlit_app.py

NOTE  •  The actual neural‑net classes **ICVAE** and **HICVAE** must already
exist in your PYTHONPATH, e.g.
    from warlok_models import ICVAE, HICVAE
This file shows only their *names* (per request) and focuses on the anomaly
pipeline & UI.
"""

# ------------------------- Imports -------------------------------
import streamlit as st
import torch, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch import optim
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---- Import model classes (implementations live elsewhere) -----
from warlok_models import ICVAE, HICVAE  # <‑‑ ensure these exist

# --------------------- Page & Sidebar ---------------------------
st.set_page_config(page_title="WARL0K VAE Dashboard", layout="wide")
st.title("WARL0K IC‑VAE & HIC‑VAE Dashboard")

sb = st.sidebar
sb.header("Hyper‑parameters & Thresholds")
latent_dim    = sb.slider("Latent Dim", 1, 8, 2)
learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs        = sb.slider("Epochs", 10, 200, 50, 10)
batch_size    = sb.slider("Batch Size", 8, 128, 32, 8)
patience      = sb.slider("Early‑Stop Patience", 3, 30, 10)

sb.markdown("---")
spike_thresh     = sb.number_input("Spike Amplitude Threshold", value=3.0)
time_dev_percent = sb.slider("Timing Deviation (%)", 5, 100, 20, 5)

# ---------------- Synthetic data --------------------------------
np.random.seed(42)
N = 1000
x = np.linspace(-10, 10, N)
base = np.sin(x) + np.sin(2*x) + np.sin(3*x)
noise = 0.5 * np.random.normal(size=N)
spikes = np.random.choice([0,1], size=N, p=[0.95,0.05]) * np.random.uniform(3,6,size=N)
y = base + noise + spikes

scaler = MinMaxScaler()
scaled_y = scaler.fit_transform(y.reshape(-1,1))

tensor_y = torch.tensor(scaled_y, dtype=torch.float32)
train_data, test_data = train_test_split(tensor_y, test_size=0.2, random_state=42)

# -------------- Helper: dataloaders -----------------------------
from torch.utils.data import DataLoader

def make_loaders(bs):
    return (DataLoader(train_data, bs, shuffle=True),
            DataLoader(test_data,  bs, shuffle=False))

train_loader, test_loader = make_loaders(batch_size)

# ----------------- Train / cache logic --------------------------
@st.cache_resource(show_spinner=False)
def train_models(latent_dim, lr, epochs, patience, bs):
    ic, hic = ICVAE(1, latent_dim), HICVAE(1, latent_dim)
    opt_ic, opt_hic = optim.Adam(ic.parameters(), lr=lr), optim.Adam(hic.parameters(), lr=lr)
    best_ic = best_hic = np.inf
    stall_ic = stall_hic = 0
    losses_ic, losses_hic = [], []

    train_loader, _ = make_loaders(bs)
    for ep in range(epochs):
        # one epoch per model
        ic.train(); hic.train()
        batch_loss_ic = batch_loss_hic = 0
        for b in train_loader:
            opt_ic.zero_grad(); opt_hic.zero_grad()
            out_ic, mu_ic, lv_ic, kw_ic = ic(b)
            loss_ic = ic.loss_function(out_ic, b, mu_ic, lv_ic, kw_ic)
            loss_ic.backward(); opt_ic.step()
            batch_loss_ic += loss_ic.item()

            out_hic, mu1, lv1, mu2, lv2, kw_hic = hic(b)
            loss_hic = hic.loss_function(out_hic, b, mu1, lv1, mu2, lv2, kw_hic)
            loss_hic.backward(); opt_hic.step()
            batch_loss_hic += loss_hic.item()
        losses_ic.append(batch_loss_ic/len(train_loader))
        losses_hic.append(batch_loss_hic/len(train_loader))
        # patience check
        if losses_ic[-1] < best_ic: best_ic, stall_ic = losses_ic[-1], 0
        else: stall_ic += 1
        if losses_hic[-1] < best_hic: best_hic, stall_hic = losses_hic[-1], 0
        else: stall_hic += 1
        if stall_ic>=patience and stall_hic>=patience:
            break
    return ic.eval(), hic.eval(), losses_ic, losses_hic

# ----------------- Evaluation utils -----------------------------
@torch.no_grad()
def reconstruct(model, loader):
    outs = []
    for b in loader:
        if isinstance(model, HICVAE):
            o, *_ = model(b)
        else:
            o, *_ = model(b)
        outs.append(o.cpu().numpy())
    return np.vstack(outs)  # shape (N_test,1) scaled

def detect_spikes(sig, thresh):
    return np.where(np.abs(sig) >= thresh)[0]

def detect_timing_anoms(idx, percent):
    if len(idx)<2:
        return np.array([], int)
    intervals = np.diff(idx)
    med = np.median(intervals)
    bad = np.where(np.abs(intervals - med) > med * (percent/100))[0]
    return idx[bad+1]

def success_err(orig, recon, thr=0.01):
    err = np.abs(orig.flatten()-recon.flatten())
    suc = (err<=thr).mean()*100
    return suc, 100-suc

# --------------------- UI Trigger -------------------------------
if "trained" not in st.session_state:
    st.session_state.trained=False

if sb.button("🚀 Train / Refresh"):
    with st.spinner("Training models…"):
        ic_model, hic_model, loss_ic, loss_hic = train_models(latent_dim, learning_rate, epochs, patience, batch_size)
        st.session_state.update({
            "trained": True,
            "ic": ic_model,
            "hic": hic_model,
            "loss_ic": loss_ic,
            "loss_hic": loss_hic
        })
        st.success("Training finished")

# ------------------- When trained, show -------------------------
if st.session_state.get("trained", False):
    # plot losses in sidebar
    sb.subheader("Training Losses")
    fig_l, ax_l = plt.subplots(figsize=(4,2))
    ax_l.plot(st.session_state.loss_ic, label="IC‑VAE")
    ax_l.plot(st.session_state.loss_hic, label="HIC‑VAE")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
    ax_l.legend(); sb.pyplot(fig_l)

    # reconstructions (scaled‑back)
    test_loader = make_loaders(batch_size)[1]
    ic_rec_scaled  = reconstruct(st.session_state.ic,  test_loader)
    hic_rec_scaled = reconstruct(st.session_state.hic, test_loader)

    ic_rec = scaler.inverse_transform(ic_rec_scaled)
    print(f"IC‑VAE Reconstruction shape: {ic_rec.shape}")
    hic_rec = scaler.inverse_transform(hic_rec_scaled)
    print(f"HIC‑VAE Reconstruction shape: {hic_rec.shape}")
    orig_test = scaler.inverse_transform(test_data.numpy())
    print(f"Original Test shape: {orig_test.shape}")

    # anomaly detection
    ic_spk  = detect_spikes(ic_rec.flatten(),  spike_thresh)
    print(f"IC‑VAE Spikes: {len(ic_spk)}")
    hic_spk = detect_spikes(hic_rec.flatten(), spike_thresh)
    print(f"HIC‑VAE Spikes: {len(hic_spk)}")
    ic_tim  = detect_timing_anoms(ic_spk,  time_dev_percent)
    print(f"IC‑VAE Timing anomalies: {len(ic_tim)}")
    hic_tim = detect_timing_anoms(hic_spk, time_dev_percent)
    print(f"HIC‑VAE Timing anomalies: {len(hic_tim)}")

    ic_succ, ic_err   = success_err(orig_test, ic_rec)
    print(f" IC: {ic_succ, ic_err}")
    hic_succ, hic_err = success_err(orig_test, hic_rec)
    print(f"HIC: {hic_succ, hic_err}")

    # two‑column display
    c1, c2 = st.columns(2)
    for col, name, rec, spk, tim, succ, err in [
        (c1,"IC‑VAE",  ic_rec,  ic_spk,  ic_tim,  ic_succ,  ic_err),
        (c2,"HIC‑VAE", hic_rec, hic_spk, hic_tim, hic_succ, hic_err)]:

        col.subheader(f"{name} Reconstruction & Anomalies")
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(orig_test, label="Original", alpha=0.4)
        ax.plot(rec,        label="Reconstruction", alpha=0.8)
        # markers
        ax.scatter(spk,  rec[spk],  color="orange", s=20, label="Spike")
        ax.scatter(tim,  rec[tim],  color="red",    s=25, label="Timing∆")
        ax.set_xlabel("Index"); ax.set_ylabel("Value"); ax.legend()
        col.pyplot(fig)

        col.metric("Spikes",            len(spk))
        col.metric("Timing anomalies", len(tim))
        col.metric("Success %",        f"{100 - succ:.2f}")
        col.metric("Error %",          f"{100 - err:.2f}")
else:
    st.info("Use the sidebar to configure parameters, then click ▶️ *Train / Refresh*.")
