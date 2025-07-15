from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
