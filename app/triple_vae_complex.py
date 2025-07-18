"""warlok_triple_vae_dashboard.py – Streamlit dashboard for Beta‑VAE, IC‑VAE, Hi‑IC‑VAE
------------------------------------------------------------------------------------------------
Run with:
	streamlit run warlok_triple_vae_dashboard.py

This dashboard lets you:
• Generate a complex synthetic signal (stacked sinusoids + Gaussian noise + random spikes).
• Tune hyper‑parameters from the sidebar.
• Train three VAE variants – BetaVAE, ICVAE, HICVAE – imported from `warl0k_models.py`.
• Compare loss curves and reconstructions side‑by‑side in three columns.

NOTE ▸ The classes **BetaVAE, ICVAE, HICVAE** must be defined in `warl0k_models.py` and importable.
"""

# ------------------------- Imports -----------------------------------
import streamlit as st
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from warlok_models import BetaVAE, ICVAE, HICVAE  # <-- provide these in your PYTHONPATH
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ----------------------- Page Config ---------------------------------
st.set_page_config(page_title="WARL0K Triple‑VAE Dashboard", layout="wide")
st.title("WARL0K Beta‑VAE vs IC‑VAE vs Hi‑IC‑VAE")

# --------------------- Sidebar Controls ------------------------------
st.sidebar.header("Hyper‑parameters & Data Gen")
latent_dim    = st.sidebar.slider("Latent Dim", 1, 8, 2)
beta_value    = st.sidebar.number_input("β (for Beta‑VAE)", value=4.0, step=0.5)
learning_rate = st.sidebar.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)
epochs        = st.sidebar.slider("Epochs", 10, 200, 50, 10)
batch_size    = st.sidebar.slider("Batch Size", 8, 128, 32, 8)
patience      = st.sidebar.slider("Early‑Stop Patience", 3, 30, 10)

st.sidebar.markdown("---")
noise_amp   = st.sidebar.slider("Gaussian Noise Amplitude", 0.0, 1.0, 0.5, 0.1)
spike_prob  = st.sidebar.slider("Spike Probability", 0.0, 0.2, 0.05, 0.01)
spike_min   = st.sidebar.number_input("Spike Min Amp", value=3.0)
spike_max   = st.sidebar.number_input("Spike Max Amp", value=6.0)

# ----------------- Synthetic Data Generation -------------------------
@st.cache_data(show_spinner=False)
def generate_data(noise_amp, spike_prob, spike_min, spike_max, num_points=1000):
	x = np.linspace(-10, 10, num_points)
	base_signal = np.sin(x) + np.sin(2 * x) + np.sin(3 * x)
	noise = noise_amp * np.random.normal(size=x.shape)
	spikes = np.random.choice([0, 1], size=x.shape, p=[1 - spike_prob, spike_prob]) * np.random.uniform(spike_min, spike_max, size=x.shape)
	y = base_signal + noise + spikes
	scaler = MinMaxScaler()
	y_norm = scaler.fit_transform(y.reshape(-1, 1)).astype(np.float32)
	return x, y, y_norm, scaler

x, y_raw, y_norm, scaler = generate_data(noise_amp, spike_prob, spike_min, spike_max)

data_tensor = torch.tensor(y_norm)
train_data, test_data = train_test_split(data_tensor, test_size=0.2, random_state=42)

# ---------------- Utils: Dataloaders ---------------------------------

def make_dataloaders(bs):
	train_loader = torch.utils.data.DataLoader(train_data, batch_size=bs, shuffle=True)
	test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=bs, shuffle=False)
	return train_loader, test_loader

train_loader, test_loader = make_dataloaders(batch_size)

# ---------------------- Training Helpers -----------------------------

def train_one_epoch(model, loader, optimiser):
	model.train(); losses = []
	for batch in loader:
		optimiser.zero_grad()
		out, mu, logvar, *rest = model(batch)
		if isinstance(model, BetaVAE):
			loss = model.loss_function(out, batch, mu, logvar)
		elif isinstance(model, HICVAE):
			mu2, logvar2, kl_w = rest  # HICVAE returns extra parts
			loss = model.loss_function(out, batch, mu, logvar, mu2, logvar2, kl_w)
		else:  # ICVAE
			kl_w, = rest
			loss = model.loss_function(out, batch, mu, logvar, kl_w)
		loss.backward(); optimiser.step(); losses.append(loss.item())
	return np.mean(losses)

@st.cache_resource(show_spinner=True)
def train_models(latent, lr, epochs, patience, bs):
	tl, _ = make_dataloaders(bs)
	beta_model = BetaVAE(1, latent, beta=beta_value)
	ic_model   = ICVAE(1, latent)
	hi_model   = HICVAE(1, latent)
	opt_b = optim.Adam(beta_model.parameters(), lr=lr)
	opt_i = optim.Adam(ic_model.parameters(),   lr=lr)
	opt_h = optim.Adam(hi_model.parameters(),   lr=lr)
	losses_b, losses_i, losses_h = [], [], []
	best_b = best_i = best_h = np.inf
	no_imp_b = no_imp_i = no_imp_h = 0

	for ep in range(epochs):
		lb = train_one_epoch(beta_model, tl, opt_b)
		li = train_one_epoch(ic_model,   tl, opt_i)
		lh = train_one_epoch(hi_model,   tl, opt_h)
		losses_b.append(lb); losses_i.append(li); losses_h.append(lh)
		if lb < best_b: best_b, no_imp_b = lb, 0
		else: no_imp_b += 1
		if li < best_i: best_i, no_imp_i = li, 0
		else: no_imp_i += 1
		if lh < best_h: best_h, no_imp_h = lh, 0
		else: no_imp_h += 1
		if max(no_imp_b, no_imp_i, no_imp_h) >= patience:
			break
	return beta_model, ic_model, hi_model, losses_b, losses_i, losses_h

# -------------------- Evaluation & Utilities -------------------------

def evaluate(model, loader):
	model.eval(); recons = []
	with torch.no_grad():
		for b in loader:
			out, *_ = model(b)
			recons.append(out.cpu().numpy())
	return np.vstack(recons)

def success_error(orig, recon, thresh=0.1):
	err = np.abs(orig.flatten() - recon.flatten())
	success = (err <= thresh).mean() * 100
	return success, 100 - success

# ------------------------ Streamlit action ---------------------------

if "train_click" not in st.session_state:
	st.session_state.train_click = False

if st.sidebar.button("🚀 Train / Retrain Models"):
	st.session_state.train_click = True
	with st.spinner("Training models…"):
		(beta_m, ic_m, hi_m,
		 loss_b, loss_i, loss_h) = train_models(latent_dim, learning_rate, epochs, patience, batch_size)
		st.session_state.beta_m = beta_m
		st.session_state.ic_m   = ic_m
		st.session_state.hi_m   = hi_m
		st.session_state.loss_b = loss_b
		st.session_state.loss_i = loss_i
		st.session_state.loss_h = loss_h
		st.success("Training complete!")

# ------------------------ Display -----------------------------------

if st.session_state.get("train_click"):
	# Sidebar loss mini‑plot
	st.sidebar.markdown("### Training Losses")
	fig_loss, ax_loss = plt.subplots()
	ax_loss.plot(st.session_state.loss_b, label="Beta‑VAE")
	ax_loss.plot(st.session_state.loss_i, label="IC‑VAE")
	ax_loss.plot(st.session_state.loss_h, label="Hi‑IC‑VAE")
	ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss"); ax_loss.legend()
	st.sidebar.pyplot(fig_loss)

	# Evaluate on test set
	_, test_loader = make_dataloaders(batch_size)
	beta_rec = evaluate(st.session_state.beta_m, test_loader)
	ic_rec   = evaluate(st.session_state.ic_m,   test_loader)
	hi_rec   = evaluate(st.session_state.hi_m,   test_loader)

	# Inverse scale
	beta_rec_inv = scaler.inverse_transform(beta_rec)
	ic_rec_inv   = scaler.inverse_transform(ic_rec)
	hi_rec_inv   = scaler.inverse_transform(hi_rec)
	orig_test_inv = scaler.inverse_transform(test_data.numpy())

	# Success/Error metrics
	b_succ, b_err = success_error(orig_test_inv, beta_rec_inv)
	i_succ, i_err = success_error(orig_test_inv, ic_rec_inv)
	h_succ, h_err = success_error(orig_test_inv, hi_rec_inv)

	# 3‑column layout
	col_b, col_i, col_h = st.columns(3)
	for col, name, rec, succ, err in [
		(col_b, "Beta‑VAE", beta_rec_inv, b_succ, b_err),
		(col_i, "IC‑VAE",   ic_rec_inv,   i_succ, i_err),
		(col_h, "Hi‑IC‑VAE", hi_rec_inv,   h_succ, h_err)]:
		with col:
			st.subheader(name)
			st.line_chart(rec.flatten())
			st.markdown(f"**Success:** {succ:.2f}%  |  **Error:** {err:.2f}%")

	# Overlay plot of all reconstructions vs original
	fig_all, ax_all = plt.subplots(figsize=(12, 5))
	ax_all.plot(orig_test_inv, label="Original", color="black", alpha=0.5)
	ax_all.plot(beta_rec_inv, label="Beta‑VAE", alpha=0.7)
	ax_all.plot(ic_rec_inv,   label="IC‑VAE",   alpha=0.7)
	ax_all.plot(hi_rec_inv,   label="Hi‑IC‑VAE", alpha=0.7)
	ax_all.set_title("Original vs Reconstructions (test slice)")
	ax_all.legend(); st.pyplot(fig_all)
else:
	st.info("Use the sidebar to configure hyper‑parameters, then click the button to train and compare the three VAE models.")

#  display success/error rates
st.sidebar.markdown("### Success/Error Rates")
if "train_click" in st.session_state and st.session_state.train_click:
	st.sidebar.metric("Beta‑VAE Success %", f"{b_succ:.2f}%")
	st.sidebar.metric("IC‑VAE Success %",   f"{i_succ:.2f}%")
	st.sidebar.metric("Hi‑IC‑VAE Success %", f"{h_succ:.2f}%")
	st.sidebar.metric("Beta‑VAE Error %",   f"{b_err:.2f}%")
	st.sidebar.metric("IC‑VAE Error %",     f"{i_err:.2f}%")
	st.sidebar.metric("Hi‑IC‑VAE Error %",   f"{h_err:.2f}%")
# ------------------------ Optional: Training Loss Curves ------------
# ------------------------ Footer ------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 WARL0K Innovations")
st.sidebar.markdown("Dashboard powered by Streamlit & PyTorch")
