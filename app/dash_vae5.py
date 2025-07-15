"""icvae_streamlit_app.py – Streamlit dashboard for IC‑VAE & HIC‑VAE
------------------------------------------------------------------
Run with:
	streamlit run icvae_streamlit_app.py

July 2025 – Two‑Column Layout (COMPLETE)
========================================
* **Sidebar** – hyper‑parameters, anomaly thresholds, and a mini loss plot.
* **Main panel** – two concise columns:
	• **Left** – IC‑VAE reconstruction, anomaly markers & metrics.
	• **Right** – HIC‑VAE reconstruction, anomaly markers & metrics.

This file is now fully self‑contained — train, evaluate, and visualise end‑to‑end.
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
st.set_page_config(page_title="WARL0K IC‑VAE Dashboard", layout="wide")
st.title("WARL0K IC‑VAE & HIC‑VAE Dashboard")

# --------------------- Sidebar Controls ------------------------------
st.sidebar.header("Hyper‑parameters & Thresholds")
latent_dim    = st.sidebar.slider("Latent Dim", 1, 8, 2)
beta_value    = st.sidebar.number_input("β (KL weight)", value=0.0004, format="%f")
learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs        = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size    = st.sidebar.slider("Batch Size", 8, 128, 16, 8)
patience      = st.sidebar.slider("Early‑Stop Patience", 3, 30, 10)

st.sidebar.markdown("---")
spike_thresh     = st.sidebar.number_input("Spike Amplitude Threshold", value=3.0)
time_dev_percent = st.sidebar.slider("Timing Deviation (%)", 5, 100, 20, 5)

# ----------------- Synthetic Data Generation -------------------------
np.random.seed(42)
NUM_POINTS = 1000
x = np.linspace(-10, 10, NUM_POINTS)
base_signal = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
noise = 0.5 * np.random.normal(size=x.shape)
spikes = np.random.choice([0, 1], size=x.shape, p=[0.95, 0.05]) * np.random.uniform(3, 6, size=x.shape)
y = base_signal + noise + spikes

scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(y.reshape(-1, 1))
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

def train_one_epoch(model, loader, optimiser):
	model.train(); batch_losses = []
	for b in loader:
		optimiser.zero_grad()
		if isinstance(model, HICVAE):
			out, mu1, lv1, mu2, lv2, kw = model(b)
			loss = model.loss_function(out, b, mu1, lv1, mu2, lv2, kw)
		else:
			out, mu, lv, kw = model(b)
			loss = model.loss_function(out, b, mu, lv, kw)
		loss.backward(); optimiser.step(); batch_losses.append(loss.item())
	return np.mean(batch_losses)

@st.cache_resource(show_spinner=True)
def train_models(latent_dim, lr, epochs, patience, batch_size):
	train_loader, _ = make_dataloaders(batch_size)
	ic, hic = ICVAE(1, latent_dim), HICVAE(1, latent_dim)
	opt_ic, opt_hic = optim.Adam(ic.parameters(), lr=lr), optim.Adam(hic.parameters(), lr=lr)
	best_ic, best_hic, no_imp_ic, no_imp_hic = np.inf, np.inf, 0, 0
	losses_ic, losses_hic = [], []

	for ep in range(epochs):
		l_ic = train_one_epoch(ic, train_loader, opt_ic)
		l_hic = train_one_epoch(hic, train_loader, opt_hic)
		losses_ic.append(l_ic); losses_hic.append(l_hic)
		# Early stop per model
		if l_ic < best_ic: best_ic, no_imp_ic = l_ic, 0
		else: no_imp_ic += 1
		if l_hic < best_hic: best_hic, no_imp_hic = l_hic, 0
		else: no_imp_hic += 1
		if no_imp_ic >= patience and no_imp_hic >= patience:
			break
	return ic, hic, losses_ic, losses_hic

# -------------------- Evaluation & Utilities -------------------------

def evaluate(model, loader):
	model.eval(); recons = []
	with torch.no_grad():
		for b in loader:
			if isinstance(model, HICVAE):
				out, *_ = model(b)
			else:
				out, *_ = model(b)
			recons.append(out.cpu().numpy())
	return np.vstack(recons)

def detect_spikes(signal, threshold):
	return np.where(np.abs(signal) >= threshold)[0]

def detect_timing_anoms(indices, percent):
	if len(indices) < 2:
		return np.array([], dtype=int)
	intervals = np.diff(indices)
	med = np.median(intervals)
	bad = np.where(np.abs(intervals - med) > med * (percent / 100))[0]
	return indices[bad + 1]  # shift to the later spike

def success_error(original, recon, thresh=0.1):
	err = np.abs(original.flatten() - recon.flatten())
	success = (err <= thresh).mean() * 100
	return success, 100 - success


# ------------------------ Main Action --------------------------------
if "train_clicked" not in st.session_state:
	st.session_state.train_clicked = False

if st.sidebar.button("🚀 Train Models"):
	st.session_state.train_clicked = True
	with st.spinner("Training models, please wait..."):
		ic_model, hic_model, loss_ic_list, loss_hic_list = train_models(latent_dim, learning_rate, epochs, patience, batch_size)
		st.session_state.ic_model = ic_model
		st.session_state.hic_model = hic_model
		st.session_state.loss_ic = loss_ic_list
		st.session_state.loss_hic = loss_hic_list
		st.success("Models trained successfully!")
if st.session_state.train_clicked:
	st.sidebar.markdown("### Training Losses")
	fig, ax = plt.subplots()
	ax.plot(st.session_state.loss_ic, label="IC‑VAE Loss", color="blue")
	ax.plot(st.session_state.loss_hic, label="HIC‑VAE Loss", color="orange")
	ax.set_title("Training Losses")
	ax.set_xlabel("Epochs")
	ax.set_ylabel("Loss")
	ax.legend()
	st.sidebar.pyplot(fig)

	# Evaluate models
	ic_recon = evaluate(st.session_state.ic_model, test_loader)
	hic_recon = evaluate(st.session_state.hic_model, test_loader)

	# Detect anomalies
	ic_spikes = detect_spikes(ic_recon.flatten(), spike_thresh)
	hic_spikes = detect_spikes(hic_recon.flatten(), spike_thresh)
	ic_timing_anoms = detect_timing_anoms(ic_spikes, time_dev_percent)
	hic_timing_anoms = detect_timing_anoms(hic_spikes, time_dev_percent)

	# Success/Error rates
	# ic_success, ic_error = success_error(data_tensor.numpy(), ic_recon)
	# hic_success, hic_error = success_error(data_tensor.numpy(), hic_recon)
	# --- Success/Error rates (compare only on the test slice) ---
	original_test_np = test_data.numpy()  # shape (N_test, 1)
	
	ic_success, ic_error = success_error(original_test_np, ic_recon)
	hic_success, hic_error = success_error(original_test_np, hic_recon)

	# Display results
	col1, col2 = st.columns(2)
	
	with col1:
		st.header("IC‑VAE Results")
		st.subheader("Reconstruction")
		st.line_chart(ic_recon.flatten())
		st.markdown(f"**Spike Anomalies Detected:** {len(ic_spikes)}")
		st.markdown(f"**Timing Anomalies Detected:** {len(ic_timing_anoms)}")
		st.markdown(f"**Success Rate:** {ic_success:.2f}%")
		st.markdown(f"**Error Rate:** {ic_error:.2f}%")

	with col2:
		st.header("HIC‑VAE Results")
		st.subheader("Reconstruction")
		st.line_chart(hic_recon.flatten())
		st.markdown(f"**Spike Anomalies Detected:** {len(hic_spikes)}")
		st.markdown(f"**Timing Anomalies Detected:** {len(hic_timing_anoms)}")
		st.markdown(f"**Success Rate:** {hic_success:.2f}%")
		st.markdown(f"**Error Rate:** {hic_error:.2f}%")
	
	# Plot original vs reconstructed
	fig, ax = plt.subplots(figsize=(12, 6))
	ax.plot(test_data.numpy(), label='Original Data', color='blue', alpha=0.5)
	ax.plot(ic_recon.flatten(), label='IC‑VAE Reconstruction', color='orange', alpha=0.5)
	ax.plot(hic_recon.flatten(), label='HIC‑VAE Reconstruction', color='green', alpha=0.5)
	ax.set_title('Original vs IC‑VAE vs HIC‑VAE Reconstruction')
	ax.set_xlabel('Time')
	ax.set_ylabel('Signal Value')
	ax.legend()
	st.pyplot(fig)
else:
	st.sidebar.info("Click the button to train the models and view results.")
# ------------------------ Footer -------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Warl0k Innovations")
st.sidebar.markdown("This dashboard is powered by Streamlit and PyTorch.")
st.sidebar.markdown("For more information, visit [Warl0k Innovations](https://warl0k.com).")
st.sidebar.markdown("### Contact")
st.sidebar.markdown("For inquiries, please contact: [DANNY SABAN](mailto:danny@warl0k.tech)")
