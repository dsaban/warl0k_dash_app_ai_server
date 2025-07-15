"""icvae_streamlit_app.py – Streamlit dashboard for IC‑VAE & HIC‑VAE
---------------------------------------------------------------------
Run with:
    streamlit run icvae_streamlit_app.py

July 2025 update – Explicit anomaly marking
==========================================
After model reconstruction we now *re‑detect* spikes and timing‑delay anomalies
**on the reconstructed signal itself**, then highlight them directly on the
reconstruction plots. You get:

* Orange markers → amplitude‑based spikes in the reconstruction.
* Red markers    → spikes whose timing deviates from the median interval.
* Metrics for number of spikes & timing anomalies captured by each model.
"""

# ------------------------- Imports -----------------------------------
import streamlit as st
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IC‑VAE Dashboard", layout="wide")

st.title("WARL0K Incremental Control VAE Dashboard")
st.write(
    "Train and compare **IC‑VAE** and **Hierarchical IC‑VAE** on a synthetic "
    "spike‑laden signal. Adjust hyper‑parameters and anomaly thresholds in the "
    "sidebar, then click **Train Models** to watch the training and see how each "
    "model marks detected spikes and timing‑delay anomalies on its own "
    "reconstruction."
)

# -------------------- Sidebar Hyper‑parameters -----------------------
st.sidebar.header("Hyper‑parameters")
latent_dim = st.sidebar.slider("Latent Dimension", 1, 8, 2)
beta_value = st.sidebar.number_input("Beta (KL‑initial)", value=0.0004, format="%.5f")
learning_rate = st.sidebar.number_input("Learning Rate", value=1e-3, format="%.5f")
epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
patience = st.sidebar.slider("Early‑Stop Patience", 5, 30, 10)
threshold = st.sidebar.number_input("Success Threshold (abs error)", value=0.1, format="%.2f")

st.sidebar.header("Spike‑Anomaly Thresholds")
amp_thr     = st.sidebar.slider("Spike Amplitude Threshold", 0.5, 10.0, 3.0, 0.1)
time_dev    = st.sidebar.slider("Timing Deviation (%)",      0.05, 1.0, 0.20, 0.05)

train_button = st.sidebar.button("Train Models")

# ------------------ Synthetic Data Generator ------------------------
@st.cache_data(show_spinner=False)
def generate_data():
    np.random.seed(42)
    x = np.linspace(-10, 10, 2000)
    y = np.sin(x) + np.sin(2 * x) + np.sin(3 * x) + 0.5 * np.random.normal(size=x.shape)
    spikes = (
        np.random.choice([0, 1], size=x.shape, p=[0.95, 0.05])
        * np.random.uniform(3, 6, size=x.shape)
    )
    y += spikes
    return x, y

x_full, y_full = generate_data()

# ---------------- Spike / Timing‑Anomaly Detection -------------------
@st.cache_data(show_spinner=False)
def detect_spikes(y_arr: np.ndarray, x_arr: np.ndarray, amplitude_thr: float, timing_dev: float):
    """Return indices of amplitude spikes and subset with timing anomalies."""
    spike_idx = np.where(y_arr >= amplitude_thr)[0]
    if spike_idx.size < 2:
        return spike_idx, np.array([])
    intervals   = np.diff(x_arr[spike_idx])
    med_int     = np.median(intervals)
    anomaly_msk = np.abs(intervals - med_int) > med_int * timing_dev
    anomaly_idx = spike_idx[1:][anomaly_msk]  # anomalous current spike
    return spike_idx, anomaly_idx

# Raw‑signal anomaly preview
spike_idx_raw, anom_idx_raw = detect_spikes(y_full, x_full, amp_thr, time_dev)
fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
ax_raw.plot(x_full, y_full, label="Signal", linewidth=1)
if spike_idx_raw.size:
    ax_raw.scatter(x_full[spike_idx_raw], y_full[spike_idx_raw], color="orange", s=10, label="Spikes (amp)")
if anom_idx_raw.size:
    ax_raw.scatter(x_full[anom_idx_raw], y_full[anom_idx_raw], color="red", s=15, label="Timing Anom.")
ax_raw.set_title("Synthetic Signal & Spike‑Timing Anomalies (Raw)")
ax_raw.set_xlabel("X")
ax_raw.set_ylabel("Y")
ax_raw.legend(loc="upper right")
st.pyplot(fig_raw)

col_a, col_b = st.columns(2)
col_a.metric("Raw Spikes Detected",   f"{spike_idx_raw.size}")
col_b.metric("Raw Timing Anomalies",  f"{anom_idx_raw.size}")

# ------------------ VAE Model Definitions ---------------------------
class ICVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, beta_initial: float):
        super().__init__()
        self.beta = beta_initial
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu = nn.Linear(64, latent_dim)
        self.fc3_logvar = nn.Linear(64, latent_dim)
        self.aux_network = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))
        self.fc4 = nn.Linear(latent_dim, 64)
        self.fc5 = nn.Linear(64, 128)
        self.fc6 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3_mu(h), self.fc3_logvar(h)

    def reparameterize(self, mu, logvar):
        std, eps = torch.exp(0.5 * logvar), torch.randn_like(logvar)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc4(z))
        h = torch.relu(self.fc5(h))
        return torch.sigmoid(self.fc6(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        kl_weight = torch.sigmoid(self.aux_network(mu)).mean()
        return recon_x, mu, logvar, kl_weight

    @staticmethod
    def loss_function(recon_x, x, mu, logvar, kl_weight):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight * KLD


class HierarchicalICVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, beta_initial: float):
        super().__init__()
        self.beta = beta_initial
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        # Level‑1 latent
        self.fc3_mu1 = nn.Linear(64, latent_dim)
        self.fc3_logvar1 = nn.Linear(64, latent_dim)
        # Level‑2 latent
        self.fc4_mu2 = nn.Linear(64, latent_dim)
        self.fc4_logvar2 = nn.Linear(64, latent_dim)
        # Attention and aux
        self.attn = nn.Sequential(
            nn.Linear(latent_dim * 2, 64), nn.ReLU(), nn.Linear(64, latent_dim * 2), nn.Sigmoid()
        )
        self.aux_network = nn.Sequential(nn.Linear(latent_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))
        # Decoder
        self.fc5 = nn.Linear(latent_dim * 2, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, input_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu1, logvar1 = self.fc3_mu1(h), self.fc3_logvar1(h)
        z1 = self.reparameterize(mu1, logvar1)
        mu2, logvar2 = self.fc4_mu2(h), self.fc4_logvar2(h)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat([z1, z2], 1)
        return z * self.attn(z), mu1, logvar1, mu2, logvar2

    def decode(self, z):
        h = torch.relu(self.fc5(z))
        h = torch.relu(self.fc6(h))
        return torch.sigmoid(self.fc7(h))

    def forward(self, x):
        z, mu1, logvar1, mu2, logvar2 = self.encode(x)
        recon_x = self.decode(z)
        kl_weight = torch.sigmoid(self.aux_network(z)).mean()
        return recon_x, mu1, logvar1, mu2, logvar2, kl_weight

    @staticmethod
    def loss_function(recon_x, x, mu1, logvar1, mu2, logvar2, kl_weight):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
        KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        return BCE + kl_weight * (KLD1 + KLD2)

# ------------------ Data Preparation Helpers ------------------------

def prepare_loaders(batch: int):
    data = y_full.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data)
    tensor = torch.tensor(data_norm, dtype=torch.float32)
    train, test = train_test_split(tensor, test_size=0.2, random_state=42)
    return (scaler,
            torch.utils.data.DataLoader(train, batch_size=batch, shuffle=True),
            torch.utils.data.DataLoader(test, batch_size=batch, shuffle=False),
            test)

# ---------------- Training / Evaluation ----------------------------

def train(model, opt, loader, n_epoch, patience, pbar):
    best, no_imp, losses = float("inf"), 0, []
    for ep in range(n_epoch):
        model.train()
        batch_losses = []
        for batch in loader:
            opt.zero_grad()
            if isinstance(model, HierarchicalICVAE):
                r, m1, l1, m2, l2, kw = model(batch)
                loss = model.loss_function(r, batch, m1, l1, m2, l2, kw)
            else:
                r, m, l, kw = model(batch)
                loss = model.loss_function(r, batch, m, l, kw)
            loss.backward(); opt.step(); batch_losses.append(loss.item())
        ep_loss = np.mean(batch_losses); losses.append(ep_loss)
        pbar.progress((ep + 1) / n_epoch)
        if ep_loss < best:
            best, no_imp = ep_loss, 0
        else:
            no_imp += 1
        if no_imp >= patience:
            break
    return losses


def evaluate(model, loader, scaler):
    model.eval(); recon = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(model, HierarchicalICVAE):
                r, _, _, _, _, _ = model(batch)
            else:
                r, _, _, _ = model(batch)
            recon.append(r.numpy())
    return scaler.inverse_transform(np.vstack(recon))


def success_error(orig, rec, th):
    err = np.abs(orig.flatten() - rec.flatten())
    suc = (err <= th).mean() * 100
    return suc, 100 - suc

# ------------------ Train Button Action -----------------------------
if train_button:
    scaler, train_loader, test_loader, test_tensor = prepare_loaders(batch_size)
    test_orig = scaler.inverse_transform(test_tensor.numpy())
    test_x = np.arange(len(test_orig))  # pseudo‑time axis for test subset

    icvae, hicvae = ICVAE(1, latent_dim, beta_value), HierarchicalICVAE(1, latent_dim, beta_value)
    opt_ic, opt_hic = optim.Adam(icvae.parameters(), lr=learning_rate), optim.Adam(hicvae.parameters(), lr=learning_rate)

    st.subheader("Training …")
    losses_ic = train(icvae, opt_ic, train_loader, epochs, patience, st.progress(0.))
    losses_hic = train(hicvae, opt_hic, train_loader, epochs, patience, st.progress(0.))

    # Loss curve
    st.subheader("Loss Curves")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(losses_ic, label="IC‑VAE"); ax_loss.plot(losses_hic, label="HIC‑VAE")
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()
    st.pyplot(fig_loss)

    # Reconstructions
    recon_ic  = evaluate(icvae, test_loader, scaler)
    recon_hic = evaluate(hicvae, test_loader, scaler)

    # --------- Detect anomalies on reconstructions ------------------
    spk_ic, anm_ic   = detect_spikes(recon_ic.flatten(), test_x, amp_thr, time_dev)
    spk_hic, anm_hic = detect_spikes(recon_hic.flatten(), test_x, amp_thr, time_dev)

    # --------- Plot Reconstructions with Markers --------------------
    st.subheader("Reconstructions with Anomaly Markers")
    fig_rec, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.plot(test_orig, label="Original", alpha=0.4)
    ax1.plot(recon_ic,  label="IC‑VAE",  alpha=0.8)
    if spk_ic.size:
        ax1.scatter(spk_ic, recon_ic[spk_ic], color="orange", s=12, label="Spikes")
    if anm_ic.size:
        ax1.scatter(anm_ic, recon_ic[anm_ic], color="red", s=16, label="Timing Anom.")
    ax1.set_title("IC‑VAE Reconstruction")
    ax1.legend(loc="upper right")

    ax2.plot(test_orig, label="Original", alpha=0.4)
    ax2.plot(recon_hic, label="HIC‑VAE", alpha=0.8)
    if spk_hic.size:
        ax2.scatter(spk_hic, recon_hic[spk_hic], color="orange", s=12, label="Spikes")
    if anm_hic.size:
        ax2.scatter(anm_hic, recon_hic[anm_hic], color="red", s=16, label="Timing Anom.")
    ax2.set_title("Hierarchical IC‑VAE Reconstruction")
    ax2.legend(loc="upper right")

    st.pyplot(fig_rec)

    # ------------------ Metrics -------------------------------------
    suc_ic, err_ic   = success_error(test_orig, recon_ic, threshold)
    suc_hic, err_hic = success_error(test_orig, recon_hic, threshold)

    st.subheader("Performance & Anomaly Capture")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("IC‑VAE Success %", f"{suc_ic:.2f}")
    c2.metric("IC‑VAE Spikes",    f"{len(spk_ic)}")
    c3.metric("HIC‑VAE Success %", f"{suc_hic:.2f}")
    c4.metric("HIC‑VAE Spikes",    f"{len(spk_hic)}")

    c5, c6 = st.columns(2)
    c5.metric("IC‑VAE Timing Anom.", f"{len(anm_ic)}")
    c6.metric("HIC‑VAE Timing Anom.", f"{len(anm_hic)}")

    st.success("Training, reconstruction & anomaly marking complete!")
