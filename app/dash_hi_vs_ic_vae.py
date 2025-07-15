import streamlit as st
import torch
from torch import nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="IC‑VAE Dashboard", layout="wide")

st.title("Incremental Control VAE Dashboard")
st.write("Train and compare IC‑VAE and Hierarchical IC‑VAE on synthetic spike‑laden signals. Adjust hyper‑parameters in the sidebar, then click **Train Models** to see results.")

# Sidebar hyper‑parameters
st.sidebar.header("Hyper‑parameters")
latent_dim = st.sidebar.slider("Latent Dimension", 1, 8, 2)
beta_value = st.sidebar.number_input("Beta (KL‑initial)", value=0.0004, format="%.5f")
learning_rate = st.sidebar.number_input("Learning Rate", value=1e-3, format="%.5f")
epochs = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size = st.sidebar.selectbox("Batch Size", [8, 16, 32, 64], index=1)
patience = st.sidebar.slider("Early‑Stop Patience", 5, 30, 10)
threshold = st.sidebar.number_input("Success Threshold (abs error)", value=0.1, format="%.2f")

train_button = st.sidebar.button("Train Models")

st.sidebar.header("Spike‑Anomaly Thresholds")
amp_thr     = st.sidebar.slider("Spike Amplitude Threshold", 0.5, 10.0, 3.0, 0.1)
time_dev    = st.sidebar.slider("Timing Deviation (%)",      0.05, 1.0, 0.20, 0.05)

# Synthetic data generator
@st.cache_data(show_spinner=False)
def generate_data():
    np.random.seed(42)
    x = np.linspace(-10, 10, 2000)
    y = np.sin(x) + np.sin(2 * x) + np.sin(3 * x) + 0.5 * np.random.normal(size=x.shape)
    spikes = np.random.choice([0, 1], size=x.shape, p=[0.95, 0.05]) * np.random.uniform(3, 6, size=x.shape)
    y += spikes
    return x, y

x, y = generate_data()

# with st.expander("Show raw data"):
#     fig_raw, ax_raw = plt.subplots(figsize=(10, 3))
#     ax_raw.plot(x, y, label="Raw signal")
#     ax_raw.set_title("Synthetic data with random spikes")
#     ax_raw.legend()
#     st.pyplot(fig_raw)

# ------------------ Spike / Timing‑Anomaly Detection -----------------
@st.cache_data(show_spinner=False)
def detect_spikes(y_arr, x_arr, amplitude_thr: float, timing_dev: float):
    # Indices where amplitude exceeds threshold
    spike_idx = np.where(y_arr >= amplitude_thr)[0]
    if spike_idx.size < 2:
        return spike_idx, np.array([])
    # Inter‑spike intervals in x‑space
    intervals = np.diff(x_arr[spike_idx])
    med_int   = np.median(intervals)
    # Any interval deviating more than timing_dev proportion is anomalous
    anomaly_mask = np.abs(intervals - med_int) > med_int * timing_dev
    anomaly_idx  = spike_idx[1:][anomaly_mask]  # the *current* spike is anomalous
    return spike_idx, anomaly_idx

spike_idx, anomaly_idx = detect_spikes(y, x, amp_thr, time_dev)

# ---------------- Initial Visualisation of Raw Data ------------------
fig_raw, ax_raw = plt.subplots(figsize=(10, 4))
ax_raw.plot(x, y, label="Signal", linewidth=1)
if spike_idx.size:
    ax_raw.scatter(x[spike_idx], y[spike_idx], color="orange", s=10, label="Detected Spikes")
if anomaly_idx.size:
    ax_raw.scatter(x[anomaly_idx], y[anomaly_idx], color="red", s=15, label="Timing Anomalies")
ax_raw.set_title("Synthetic Signal & Spike‑Timing Anomalies")
ax_raw.set_xlabel("X")
ax_raw.set_ylabel("Y")
ax_raw.legend(loc="upper right")
st.pyplot(fig_raw)

# Anomaly metrics
col_a, col_b = st.columns(2)
col_a.metric("Total Spikes Detected", f"{spike_idx.size}")
col_b.metric("Timing Anomalies",       f"{anomaly_idx.size}")


# ----- Model Definitions -----
class ICVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta_initial):
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
        mu = self.fc3_mu(h)
        logvar = self.fc3_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
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

    def loss_function(self, recon_x, x, mu, logvar, kl_weight):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + kl_weight * KLD

class HierarchicalICVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, beta_initial):
        super().__init__()
        self.beta = beta_initial
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3_mu1 = nn.Linear(64, latent_dim)
        self.fc3_logvar1 = nn.Linear(64, latent_dim)
        self.fc4_mu2 = nn.Linear(64, latent_dim)
        self.fc4_logvar2 = nn.Linear(64, latent_dim)
        self.attn = nn.Sequential(nn.Linear(latent_dim * 2, 64), nn.ReLU(), nn.Linear(64, latent_dim * 2), nn.Sigmoid())
        self.aux_network = nn.Sequential(nn.Linear(latent_dim * 2, 32), nn.ReLU(), nn.Linear(32, 1))
        self.fc5 = nn.Linear(latent_dim * 2, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu1 = self.fc3_mu1(h)
        logvar1 = self.fc3_logvar1(h)
        z1 = self.reparameterize(mu1, logvar1)
        mu2 = self.fc4_mu2(h)
        logvar2 = self.fc4_logvar2(h)
        z2 = self.reparameterize(mu2, logvar2)
        z = torch.cat([z1, z2], dim=1)
        z = z * self.attn(z)
        return z, mu1, logvar1, mu2, logvar2

    def decode(self, z):
        h = torch.relu(self.fc5(z))
        h = torch.relu(self.fc6(h))
        return torch.sigmoid(self.fc7(h))

    def forward(self, x):
        z, mu1, logvar1, mu2, logvar2 = self.encode(x)
        recon_x = self.decode(z)
        kl_weight = torch.sigmoid(self.aux_network(z)).mean()
        return recon_x, mu1, logvar1, mu2, logvar2, kl_weight

    def loss_function(self, recon_x, x, mu1, logvar1, mu2, logvar2, kl_weight):
        BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
        KLD2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
        return BCE + kl_weight * (KLD1 + KLD2)

# ----- Training Helpers -----

def prepare_loaders(batch_size):
    data = y.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
    train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return scaler, train_loader, test_loader, test_data


def train(model, optimizer, train_loader, epochs, patience, progress_bar):
    losses, best, no_impr = [], float('inf'), 0
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            if isinstance(model, HierarchicalICVAE):
                recon, mu1, logvar1, mu2, logvar2, kl_w = model(batch)
                loss = model.loss_function(recon, batch, mu1, logvar1, mu2, logvar2, kl_w)
            else:
                recon, mu, logvar, kl_w = model(batch)
                loss = model.loss_function(recon, batch, mu, logvar, kl_w)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)
        progress_bar.progress((epoch + 1) / epochs)
        if mean_loss < best:
            best, no_impr = mean_loss, 0
        else:
            no_impr += 1
        if no_impr >= patience:
            break
    return losses


def evaluate(model, test_loader, scaler):
    model.eval()
    recon_batches = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(model, HierarchicalICVAE):
                r, _, _, _, _, _ = model(batch)
            else:
                r, _, _, _ = model(batch)
            recon_batches.append(r.numpy())
    recon = np.vstack(recon_batches)
    return scaler.inverse_transform(recon)


def success_error(orig, rec, th):
    errors = np.abs(orig.flatten() - rec.flatten())
    success = (errors <= th).mean() * 100
    return success, 100 - success

# ----- Main Training Trigger -----
if train_button:
    scaler, train_loader, test_loader, test_data_tensor = prepare_loaders(batch_size)
    test_original = scaler.inverse_transform(test_data_tensor.numpy())

    icvae = ICVAE(1, latent_dim, beta_value)
    hicvae = HierarchicalICVAE(1, latent_dim, beta_value)
    opt_ic = optim.Adam(icvae.parameters(), lr=learning_rate)
    opt_hic = optim.Adam(hicvae.parameters(), lr=learning_rate)

    st.subheader("Training …")
    pbar1 = st.progress(0.)
    losses_ic = train(icvae, opt_ic, train_loader, epochs, patience, pbar1)
    pbar2 = st.progress(0.)
    losses_hic = train(hicvae, opt_hic, train_loader, epochs, patience, pbar2)

    # Loss curves
    st.subheader("Loss Curves")
    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
    ax_loss.plot(losses_ic, label="IC‑VAE")
    ax_loss.plot(losses_hic, label="HIC‑VAE")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    st.pyplot(fig_loss)

    # Reconstructions
    recon_ic = evaluate(icvae, test_loader, scaler)
    recon_hic = evaluate(hicvae, test_loader, scaler)

    st.subheader("Reconstructions vs Original")
    fig_rec, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.plot(test_original, label="Original", alpha=0.5)
    ax1.plot(recon_ic, label="IC‑VAE", alpha=0.7)
    ax1.set_title("IC‑VAE")
    ax1.legend()
    ax2.plot(test_original, label="Original", alpha=0.5)
    ax2.plot(recon_hic, label="HIC‑VAE", alpha=0.7)
    ax2.set_title("Hierarchical IC‑VAE")
    ax2.legend()
    st.pyplot(fig_rec)

    # Metrics
    suc_ic, err_ic = success_error(test_original, recon_ic, threshold)
    suc_hic, err_hic = success_error(test_original, recon_hic, threshold)

    st.subheader("Success / Error Rates (%)")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("IC‑VAE Success", f"{suc_ic:.2f}")
        st.metric("IC‑VAE Error", f"{err_ic:.2f}")
    with col2:
        st.metric("HIC‑VAE Success", f"{suc_hic:.2f}")
        st.metric("HIC‑VAE Error", f"{err_hic:.2f}")

    st.success("Training & evaluation complete!")
