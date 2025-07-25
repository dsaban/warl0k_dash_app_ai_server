import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class TinySecretRegenerator(nn.Module):
	def __init__(self, vocab_size, hidden_size=32):
		super().__init__()
		self.fc1 = nn.Linear(vocab_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, vocab_size)
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		return F.softmax(self.fc2(x), dim=-1)


vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
vocab_size = len(vocab)


def text_to_onehot(text):
	onehot = torch.zeros(len(text), vocab_size)
	for i, char in enumerate(text):
		onehot[i][vocab.index(char)] = 1.0
	return onehot


def inject_simple_noise(onehot_tensor, noise_level=0.1):
	noise = torch.rand_like(onehot_tensor) * noise_level
	return torch.clamp(onehot_tensor + noise, 0, 1)


def train_tiny_model(secret, epochs=100):
	model = TinySecretRegenerator(vocab_size)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	criterion = nn.MSELoss()
	x = inject_simple_noise(text_to_onehot(secret))
	y = text_to_onehot(secret)
	for _ in range(epochs):
		optimizer.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		loss.backward()
		optimizer.step()
	return model


def save_model_and_check_size(model, filename="tiny_model.pth"):
	torch.save(model.state_dict(), filename)
	size = os.path.getsize(filename) / 1024  # size in KB
	print(f"✅ Model saved as '{filename}' — size: {size:.2f} KB")
	return size


def evaluate(model, noisy_input):
	with torch.no_grad():
		prediction = model(noisy_input)
	pred_indices = prediction.argmax(dim=-1)
	return ''.join([vocab[i] for i in pred_indices])


if __name__ == "__main__":
	# secret = "TestSecret"
	#  generate a 16-character secret from vocab
	import random
	secret = ''.join(random.choices(vocab, k=16))
	print("Original Secret:", secret)
	model = train_tiny_model(secret)
	noisy_input = inject_simple_noise(text_to_onehot(secret))
	reconstructed = evaluate(model, noisy_input)
	print("Reconstructed:", reconstructed)
	
	# Save and check size
	model_size_kb = save_model_and_check_size(model)
	
	# Optional: Show compression size with zip
	import zipfile
	
	zip_filename = "tiny_model.zip"
	with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
		zipf.write("tiny_model.pth")
	zip_size = os.path.getsize(zip_filename) / 1024
	print(f"📦 Compressed model size: {zip_size:.2f} KB")
