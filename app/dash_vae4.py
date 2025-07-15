"""icvae_streamlit_app.py – Streamlit dashboard for IC‑VAE & HIC‑VAE
------------------------------------------------------------------
Run:
	streamlit run icvae_streamlit_app.py

Highlights (July 2025)
======================
• **Sidebar** – Hyper‑parameters, anomaly thresholds, and a mini loss chart.
• **Main panel** – **Three concise columns**:
	1. Original test signal with detected spikes & timing anomalies.
	2. IC‑VAE reconstruction with anomalies.
	3. HIC‑VAE reconstruction with anomalies.
  Each column shows metrics: success / error %, spikes reproduced & timing capture‑rate.

The script is self‑contained and trains on synthetic data.
"""

# ------------------------- Imports ----------------------------------
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

# ----------------------- Page Config --------------------------------
st.set_page_config(page_title="WARL0K IC‑VAE Dashboard", layout="wide")
st.title("WARL0K — AI‑Powered Anomaly Reconstruction Dashboard 🛰️")

# --------------------- Sidebar Controls -----------------------------
st.sidebar.header("Hyper‑parameters & Thresholds")
latent_dim     = st.sidebar.slider("Latent Dim", 1, 8, 2)
beta_value     = st.sidebar.number_input("β (KL weight)", value=0.0004, format="%f")
learning_rate  = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs         = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size     = st.sidebar.slider("Batch Size", 8, 128, 16, 8)
patience       = st.sidebar.slider("Early‑Stop Patience", 3, 30, 10)

st.sidebar.markdown("---")
spike_thresh     = st.sidebar.number_input("Spike Amplitude Threshold", value=3.0)
time_dev_percent = st.sidebar.slider("Timing Deviation (%)", 5, 100, 20, 5)

# ---------------- Synthetic Data Generation -------------------------
np.random.seed(42)
NUM_POINTS = 1000
x = np.linspace(-10, 10, NUM_POINTS)
base_signal = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
noise = 0.5 * np.random.normal(size=x.shape)
spikes = np.random.choice([0, 1], size=x.shape, p=[0.95, 0.05]) * np.random.uniform(3, 6, size=x.shape)
y = base_signal + noise + spikes

# Scale data 0‑1------------------------------------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(y.reshape(-1, 1))

data_tensor = torch.tensor(scaled, dtype=torch.float32)
train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

# ---------------- Utils: Dataloaders --------------------------------

def make_dataloaders(bs):
	train_ldr = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
	test_ldr  = torch.utils.data.DataLoader(test_data,  batch_size=bs, shuffle=False)
	return train_ldr, test_ldr

train_loader, test_loader = make_dataloaders(batch_size)

# -------------------- Model Definitions -----------------------------
class ICVAE(nn.Module):
	def __init__(self, input_dim: int, latent_dim: int):
		super().__init__()
		self.fc1, self.fc2 = nn.Linear(input_dim, 128), nn.Linear(128, 64)
		self.fc_mu, self.fc_logvar = nn.Linear(64, latent_dim), nn.Linear(64, latent_dim)
		self.aux = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1))
		self.fc4, self.fc5, self.fc6 = nn.Linear(latent_dim, 64), nn.Linear(64, 128), nn.Linear(128, input_dim)

	def encode(self, x):
		h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
		return self.fc_mu(h), self.fc_logvar(h)

	@staticmethod
	def reparameterize(mu, logvar):
		std = torch.exp(0.5 * logvar)
		return mu + torch.randn_like(std) * std

	def decode(self, z):
		h = torch.relu(self.fc5(torch.relu(self.fc4(z))))
		return torch.sigmoid(self.fc6(h))

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon = self.decode(z)
		kl_w = torch.sigmoid(self.aux(mu)).mean()
		return recon, mu, logvar, kl_w

	def loss_function(self, recon, x, mu, logvar, kl_w):
		bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
		kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return bce + kl_w * kld

class HICVAE(nn.Module):
	def __init__(self, input_dim: int, latent_dim: int):
		super().__init__()
		self.fc1, self.fc2 = nn.Linear(input_dim, 128), nn.Linear(128, 64)
		self.mu1, self.lv1, self.mu2, self.lv2 = (nn.Linear(64, latent_dim) for _ in range(4))
		self.attn = nn.Sequential(nn.Linear(latent_dim*2, 64), nn.ReLU(), nn.Linear(64, latent_dim*2), nn.Sigmoid())
		self.aux = nn.Sequential(nn.Linear(latent_dim*2, 32), nn.ReLU(), nn.Linear(32, 1))
		self.fc5, self.fc6, self.fc7 = nn.Linear(latent_dim*2, 64), nn.Linear(64, 128), nn.Linear(128, input_dim)

	@staticmethod
	def reparameterize(mu, logvar):
		std = torch.exp(0.5 * logvar)
		return mu + torch.randn_like(std) * std

	def encode(self, x):
		h = torch.relu(self.fc2(torch.relu(self.fc1(x))))
		mu1, lv1 = self.mu1(h), self.lv1(h)
		mu2, lv2 = self.mu2(h), self.lv2(h)
		z = torch.cat([self.reparameterize(mu1, lv1), self.reparameterize(mu2, lv2)], dim=1)
		z = z * self.attn(z)
		return z, mu1, lv1, mu2, lv2

	def decode(self, z):
		h = torch.relu(self.fc6(torch.relu(self.fc5(z))))
		return torch.sigmoid(self.fc7(h))

	def forward(self, x):
		z, mu1, lv1, mu2, lv2 = self.encode(x)
		recon = self.decode(z)
		kl_w = torch.sigmoid(self.aux(z)).mean()
		return recon, mu1, lv1, mu2, lv2, kl_w

	def loss_function(self, recon, x, mu1, lv1, mu2, lv2, kl_w):
		bce = nn.functional.binary_cross_entropy(recon, x, reduction="sum")
		kld  = -0.5 * (torch.sum(1 + lv1 - mu1.pow(2) - lv1.exp()) + torch.sum(1 + lv2 - mu2.pow(2) - lv2.exp()))
		return bce + kl_w * kld

# ---------------------- Training Helpers ----------------------------
@st.cache_resource(show_spinner=False)
def train_models(latent_dim, lr, epochs, patience, batch_size):
	train_loader, _ = make_dataloaders(batch_size)
	ic, hic = ICVAE(1, latent_dim), HICVAE(1, latent_dim)
	opt_ic, opt_hic = optim.Adam(ic.parameters(), lr=lr), optim.Adam(hic.parameters(), lr=lr)

	def run(model, optimizer):
		best, no_imp, losses = np.inf, 0, []
		for ep in range(epochs):
			model.train(); batch_l = []
			for b in train_loader:
				optimizer.zero_grad()
				if isinstance(model, HICVAE):
					out, mu1, lv1, mu2, lv2, kw = model(b)
					loss = model.loss_function(out, b, mu1, lv1, mu2, lv2, kw)
				else:
					out, mu, lv, kw = model(b)
					loss = model.loss_function(out, b, mu, lv, kw)
				loss.backward(); optimizer.step(); batch_l.append(loss.item())
			ep_loss = np.mean(batch_l); losses.append(ep_loss)
			if ep_loss < best:
				best, no_imp = ep_loss, 0
			else:
				no_imp += 1
			if no_imp >= patience:
				break
		return model, losses

	ic, ic_loss   = run(ic, opt_ic)
	hic, hic_loss = run(hic, opt_hic)
	return ic, hic, ic_loss, hic_loss

# ---------------------- Anomaly Detection ---------------------------

def detect_spikes(signal, threshold):
	return np.where(np.abs(signal) >= threshold)[0]


def detect_timing_anomalies(indices, deviation_percent):
	if len(indices) < 2:
		return np.array([])
	intervals = np.diff(indices)
	med = np.median(intervals)
	abnormal = np.where(np.abs(intervals - med) >= (deviation_percent/100) * med)[0] + 1
	return indices[abnormal]

# ------------------------- Evaluation -------------------------------

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

# ------------------ Training Trigger & Session State ---------------
if st.button("🚀 Train Models", type="primary"):
	with st.spinner("Training IC‑VAE & HIC‑VAE …"):
		ic_vae, hic_vae, ic_loss, hic_loss = train_models(latent_dim, learning_rate, epochs, patience, batch_size)
		st.session_state.update({
			"ic_vae": ic_vae,
			"hic_vae": hic_vae,
			"ic_loss": ic_loss,
			"hic_loss": hic_loss
		})
		st.success("Training complete! 🎉")

# ---------------- Sidebar: Training Loss Plot -----------------------
if "ic_loss" in st.session_state:
	st.sidebar.subheader("Training Losses")
	st.sidebar.line_chart({"IC‑VAE": st.session_state.ic_loss, "HIC‑VAE": st.session_state.hic_loss})

# ---------------- Display Reconstructions & Anomalies --------------
if "ic_vae" in st.session_state:
	# Evaluate models
	ic_recon  = evaluate_model(st.session_state.ic_vae,  test_loader)
	hic_recon = evaluate_model(st.session_state.hic_vae, test_loader)
	original  = scaler.inverse_transform(test_data.numpy())

	# Detect anomalies
	def process(signal):
		spike_idx = detect_spikes(signal, spike_thresh)
		timing_idx = detect_timing_anomalies(spike_idx, time_dev_percent)
		return spike_idx, timing_idx

	orig_sp, orig_ti = process(original.squeeze())
	ic_sp,   ic_ti   = process(ic_recon.squeeze())
	hic_sp,  hic_ti  = process(hic_recon.squeeze())

	# Success / Error (reconstruction accuracy)
	def success(original, recon, thr=0.1):
		err = np.abs(original.flatten() - recon.flatten())
		suc = (err <= thr).mean()*100
		return suc, 100 - suc

	suc_ic, err_ic   = success(original, ic_recon)
	suc_hic, err_hic = success(original, hic_recon)

	# ---------------- Three‑Column Layout -------------------------
	col_orig, col_ic, col_hic = st.columns(3)

	# Helper to plot signal
	# IndexError: arrays used as indices must be of integer (or boolean) type
	# convert to integer type
	def plot_signal(ax, y_vals, spike_idx, timing_idx, title):
		spike_idx = spike_idx.astype(int)
		timing_idx = timing_idx.astype(int)
		ax.plot(y_vals, linewidth=0.8, label="Signal")
		ax.scatter(spike_idx, y_vals[spike_idx], color="orange", s=10, label="Spikes")
		ax.scatter(timing_idx, y_vals[timing_idx], color="red", s=12, label="Timing Anoms")
		ax.set_title(title)
		ax.set_xlabel("Time")
		ax.set_ylabel("Amplitude")
		ax.legend(fontsize=6)
	
	# Original -----------------------------------------------------
	with col_orig:
		fig, ax = plt.subplots(figsize=(5,3))
		plot_signal(ax, original.squeeze(), orig_sp, orig_ti, "Original Test Signal")
		st.pyplot(fig, use_container_width=True)
		st.metric("Total Spikes", len(orig_sp))
		st.metric("Timing Anoms", len(orig_ti))

	# IC‑VAE --------------------------------------------------------
	with col_ic:
		fig, ax = plt.subplots(figsize=(5,3))
		plot_signal(ax, ic_recon.squeeze(), ic_sp, ic_ti, "IC‑VAE Reconstruction")
		st.pyplot(fig, use_container_width=True)
		st.metric("Success %", f"{suc_ic:.2f}")
		st.metric("Error %", f"{err_ic:.2f}")
		st.metric("Spike Capture", f"{len(set(ic_sp) & set(orig_sp))}/{len(orig_sp)}")
		st.metric("Timing Capture", f"{len(set(ic_ti) & set(orig_ti))}/{len(orig_ti)}")

	# HIC‑VAE -------------------------------------------------------
	with col_hic:
		fig, ax = plt.subplots(figsize=(5,3))
		plot_signal(ax, hic_recon.squeeze(), hic_sp, hic_ti, "HIC‑VAE Reconstruction")
		st.pyplot(fig, use_container_width=True)
		st.metric("Success %", f"{suc_hic:.2f}")
		st.metric("Error %", f"{err_hic:.2f}")
		st.metric("Spike Capture", f"{len(set(hic_sp) & set(orig_sp))}/{len(orig_sp)}")
		st.metric("Timing Capture", f"{len(set(hic_ti) & set(orig_ti))}/{len(orig_ti)}")

	st.caption("Orange = amplitude spikes • Red = timing‑delay anomalies")

else:
	st.info("Click **🚀 Train Models** to generate reconstructions & anomaly analysis.")
