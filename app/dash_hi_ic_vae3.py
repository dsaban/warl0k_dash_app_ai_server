"""icvae_streamlit_app.py – Streamlit dashboard for IC‑VAE & HIC‑VAE
---------------------------------------------------------------------
Run with:
	streamlit run icvae_streamlit_app.py

July 2025 – Compact 2‑Column Layout & Loss‑in‑Sidebar
====================================================
* Sidebar hosts hyper‑params **and** a mini training‑loss chart.
* Main panel: 2 responsive columns (IC‑VAE ↔ HIC‑VAE) showing
  reconstructions with amplitude‑ & timing‑anomaly markers plus metrics.
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ----------------------- Page Config ---------------------------------
st.set_page_config(page_title="IC‑VAE Dashboard", layout="wide")

# --------------------- Sidebar Controls ------------------------------
st.sidebar.header("Hyperparameters & Thresholds")
latent_dim     = st.sidebar.slider("Latent Dim", 1, 8, 2)
beta_value     = st.sidebar.number_input("β (KL weight)", value=0.0004, format="%f")
learning_rate  = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs         = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size     = st.sidebar.slider("Batch Size", 8, 128, 16, 8)
patience       = st.sidebar.slider("Early‑Stop Patience", 3, 30, 10)

st.sidebar.markdown("---")
spike_thresh       = st.sidebar.number_input("Spike Amplitude Threshold", value=3.0)
time_dev_percent   = st.sidebar.slider("Timing Deviation (%)", 5, 100, 20, 5)

# ----------------- Synthetic Data Generation -------------------------
np.random.seed(42)
NUM_POINTS = 2000
x = np.linspace(-10, 10, NUM_POINTS)
base_signal = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
noise = 0.5 * np.random.normal(size=x.shape)
spikes = np.random.choice([0, 1], size=x.shape, p=[0.95, 0.05]) * np.random.uniform(3, 6, size=x.shape)
y = base_signal + noise + spikes

data = y.reshape(-1, 1)
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

# ---------------- Utils: Dataloaders ---------------------------------

def make_dataloaders(bs):
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
	test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=bs, shuffle=False)
	return train_loader, test_loader

train_loader, test_loader = make_dataloaders(batch_size)

# -------------------- Model Definitions ------------------------------
class ICVAE(nn.Module):
	def __init__(self, input_dim, latent_dim):
		super().__init__()
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

	@staticmethod
	def reparameterize(mu, logvar):
		std = torch.exp(0.5 * logvar)
		return mu + torch.randn_like(std) * std

	def decode(self, z):
		h = torch.relu(self.fc4(z))
		h = torch.relu(self.fc5(h))
		return torch.sigmoid(self.fc6(h))

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		kl_w = torch.sigmoid(self.aux_network(mu)).mean()
		return recon, mu, logvar, kl_w

	def loss_function(self, recon, x, mu, logvar, kl_w):
		bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
		kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return bce + kl_w * kld

class HICVAE(nn.Module):
	def __init__(self, input_dim, latent_dim):
		super().__init__()
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

	@staticmethod
	def reparameterize(mu, logvar):
		std = torch.exp(0.5 * logvar)
		return mu + torch.randn_like(std) * std

	def encode(self, x):
		h = torch.relu(self.fc1(x))
		h = torch.relu(self.fc2(h))
		mu1, logvar1 = self.fc3_mu1(h), self.fc3_logvar1(h)
		mu2, logvar2 = self.fc4_mu2(h), self.fc4_logvar2(h)
		z1 = self.reparameterize(mu1, logvar1)
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
		recon = self.decode(z)
		kl_w = torch.sigmoid(self.aux_network(z)).mean()
		return recon, mu1, logvar1, mu2, logvar2, kl_w

	def loss_function(self, recon, x, mu1, logvar1, mu2, logvar2, kl_w):
		bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
		kld1 = -0.5 * torch.sum(1 + logvar1 - mu1.pow(2) - logvar1.exp())
		kld2 = -0.5 * torch.sum(1 + logvar2 - mu2.pow(2) - logvar2.exp())
		return bce + kl_w * (kld1 + kld2)

# ---------------------- Training Helpers -----------------------------
@st.cache_resource(show_spinner=True)
def train_models(latent_dim, lr, epochs, patience, batch_size):
	train_loader, _ = make_dataloaders(batch_size)
	ic = ICVAE(1, latent_dim)
	hic = HICVAE(1, latent_dim)
	opt_ic = optim.Adam(ic.parameters(), lr=lr)
	opt_hic = optim.Adam(hic.parameters(), lr=lr)

	def run(model, opt):
		best = np.inf
		no_imp = 0
		losses = []
		for ep in range(epochs):
			model.train(); batch_loss = []
			for b in train_loader:
				opt.zero_grad()
				if isinstance(model, HICVAE):
					out, mu1, lv1, mu2, lv2, kw = model(b)
					loss = model.loss_function(out, b, mu1, lv1, mu2, lv2, kw)
				else:
					out, mu, lv, kw = model(b)
					loss = model.loss_function(out, b, mu, lv, kw)
				loss.backward(); opt.step(); batch_loss.append(loss.item())
			ep_loss = np.mean(batch_loss); losses.append(ep_loss)
			if ep_loss < best:
				best, no_imp = ep_loss, 0
			else:
				no_imp += 1
			if no_imp >= patience:
				break
		return model, losses

	ic, ic_loss = run(ic, opt_ic)
	hic, hic_loss = run(hic, opt_hic)
	return ic, hic, ic_loss, hic_loss

# ---------------------- Anomaly Detection ----------------------------

def detect_spikes(signal, threshold):
	return np.where(np.abs(signal) >= threshold)[0]

def detect_timing_anomalies(indices, deviation_percent):
	if len(indices) < 2:
		return np.array([])
	intervals = np.diff(indices)
	median = np.median(intervals)
	abnormal = np.where(np.abs(intervals - median) >= (deviation_percent / 100.0) * median)[0] + 1
	return indices[abnormal]

# ------------------------- Evaluation --------------------------------

def evaluate_model(model, loader):
	model.eval(); preds = []
	with torch.no_grad():
		for b in loader:
			if isinstance(model, HICVAE):
				out, *_ = model(b)
			else:
				out, *_ = model(b)
			preds.append(out.numpy())
	recon = np.vstack(preds)
	return scaler.inverse_transform(recon)

# ------------------ Training Trigger & Display -----------------------

if st.button("🚀 Train Models", type="primary"):
	with st.spinner("Training IC‑VAE & HIC‑VAE …"):
		ic_vae, hic_vae, ic_loss, hic_loss = train_models(latent_dim, learning_rate, epochs, patience, batch_size)
		st.success("Training complete! 🎉")
		st.session_state.ic_vae = ic_vae
		st.session_state.hic_vae = hic_vae
		st.session_state.ic_loss = ic_loss
		st.session_state.hic_loss = hic_loss
# ------------------ Display Training Losses --------------------------
if "ic_loss" in st.session_state and "hic_loss" in st.session_state:
	st.sidebar.subheader("Training Losses")
	st.sidebar.line_chart({
		"IC‑VAE": st.session_state.ic_loss,
		"HIC‑VAE": st.session_state.hic_loss
	})
# ------------------ Display Model Reconstructions --------------------
#  show the reconstructions only if both models are trained
if "ic_vae" in st.session_state and "hic_vae" in st.session_state:
	st.header("Model Reconstructions")
	ic_recon = evaluate_model(st.session_state.ic_vae, test_loader)
	hic_recon = evaluate_model(st.session_state.hic_vae, test_loader)
	print(f"IC‑VAE Reconstructions Shape: {ic_recon.shape}")
	print(f"HIC‑VAE Reconstructions Shape: {hic_recon.shape}")
# 	 show data in two responsive columns
	print("Displaying reconstructions in two columns...")
	# print("IC‑VAE Reconstructions:", ic_recon)
	# print("HIC‑VAE Reconstructions:", hic_recon)
	
# 	 show default training data
	st.subheader("Default Training Data")
	st.line_chart(data_tensor.numpy().flatten(), use_container_width=True, height=300)
# 	 show the default data as the training scaler
	st.sidebar.subheader("Default Data Scaler")
	st.sidebar.write("Min:", scaler.data_min_[0])
	st.sidebar.write("Max:", scaler.data_max_[0])
# 	 show the IC‑VAE reconstructions
	st.subheader("IC‑VAE Reconstructions")
	st.line_chart(ic_recon.flatten(), use_container_width=True, height=300)
# 	 show the HIC‑VAE reconstructions
	st.subheader("HIC‑VAE Reconstructions")
	st.line_chart(hic_recon.flatten(), use_container_width=True, height=300)
# 	# ------------------ Display Reconstructions (Commented Out) ---------



# ------------------ Display Reconstructions (Commented Out) ---------
# ...
# ------------------ Anomaly Detection & Metrics ---------------------
#  show the anomaly detection only if both models are trained
#  and reconstructions are available
if "ic_vae" in st.session_state and "hic_vae" in st.session_state:
	st.header("Anomaly Detection & Metrics")
	ic_anomalies = detect_spikes(ic_recon.flatten(), spike_thresh)
	hic_anomalies = detect_spikes(hic_recon.flatten(), spike_thresh)
	ic_timing_anomalies = detect_timing_anomalies(ic_anomalies, time_dev_percent)
	hic_timing_anomalies = detect_timing_anomalies(hic_anomalies, time_dev_percent)
	st.subheader("IC‑VAE Anomalies")
	st.write(f"Spike Indices: {ic_anomalies}")
	st.write(f"Timing Anomalies: {ic_timing_anomalies}")
	st.subheader("HIC‑VAE Anomalies")
	st.write(f"Spike Indices: {hic_anomalies}")
	st.write(f"Timing Anomalies: {hic_timing_anomalies}")
	# Plotting anomalies: arrays used as indices must be of integer (or boolean) type
	ic_anomalies = ic_anomalies.astype(int)
	hic_anomalies = hic_anomalies.astype(int)
	ic_timing_anomalies = ic_timing_anomalies.astype(int)
	hic_timing_anomalies = hic_timing_anomalies.astype(int)
	# Plotting the anomalies
	fig, ax = plt.subplots(2, 1, figsize=(10, 8))
	ax[0].plot(ic_recon.flatten(), label='IC‑VAE Reconstruction')
	ax[0].scatter(ic_anomalies, ic_recon.flatten()[ic_anomalies], color='red', label='Spikes')
	ax[0].scatter(ic_timing_anomalies, ic_recon.flatten()[ic_timing_anomalies], color='orange', label='Timing Anomalies')
	ax[0].set_title("IC‑VAE Anomalies")
	ax[0].legend()
	ax[1].plot(hic_recon.flatten(), label='HIC‑VAE Reconstruction')
	ax[1].scatter(hic_anomalies, hic_recon.flatten()[hic_anomalies], color='red', label='Spikes')
	ax[1].scatter(hic_timing_anomalies, hic_recon.flatten()[hic_timing_anomalies], color='orange', label='Timing Anomalies')
	ax[1].set_title("HIC‑VAE Anomalies")
	ax[1].legend()
	st.pyplot(fig)
# ------------------ Metrics & Thresholds -----------------------------
	# Metrics
	ic_spike_count = len(ic_anomalies)
	hic_spike_count = len(hic_anomalies)
	ic_timing_count = len(ic_timing_anomalies)
	hic_timing_count = len(hic_timing_anomalies)
	st.sidebar.subheader("Anomaly Metrics")
	st.sidebar.metric("IC‑VAE Spike Count", ic_spike_count)
	st.sidebar.metric("HIC‑VAE Spike Count", hic_spike_count)
	st.sidebar.metric("IC‑VAE Timing Anomalies", ic_timing_count)
	st.sidebar.metric("HIC‑VAE Timing Anomalies", hic_timing_count)
	st.sidebar.markdown("---")
	st.sidebar.subheader("Thresholds")
	st.sidebar.write(f"Spike Amplitude Threshold: {spike_thresh}")
	st.sidebar.write(f"Timing Deviation: {time_dev_percent}%")
# ------------------ Footer & Info -----------------------------------
st.sidebar.markdown("---")
st.sidebar.info("IC‑VAE & HIC‑VAE Dashboard")
st.sidebar.markdown("This dashboard allows you to train and evaluate IC‑VAE and HIC‑VAE models on synthetic time series data. Adjust hyperparameters in the sidebar and visualize model performance, reconstructions, and anomaly detection results.")
st.sidebar.markdown("© 2025 Warl0k Innovations")
