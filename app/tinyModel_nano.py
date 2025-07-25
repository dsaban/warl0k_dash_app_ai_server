# Step 1: Tiny Inference + Training On-Device (tiny_self_train.py)
import numpy as np
import time

class TinySecretRegenerator:
	def __init__(self, vocab_size, hidden_dim=8):
		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.init_weights()

	def init_weights(self):
		# Initialize weights and biases internally (MCU friendly)
		np.random.seed(42)
		self.w1 = np.random.randn(self.vocab_size, self.hidden_dim) * 0.1
		self.b1 = np.zeros((1, self.hidden_dim))
		self.w2 = np.random.randn(self.hidden_dim, self.vocab_size) * 0.1
		self.b2 = np.zeros((1, self.vocab_size))

	def forward(self, x):
		x = np.dot(x, self.w1) + self.b1
		x = np.tanh(x)
		x = np.dot(x, self.w2) + self.b2
		return self._softmax(x)

	def _softmax(self, x):
		e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
		return e_x / e_x.sum(axis=-1, keepdims=True)

	def train(self, x, y, epochs=500, lr=0.05):
		for epoch in range(epochs):
			z1 = np.dot(x, self.w1) + self.b1
			a1 = np.tanh(z1)
			z2 = np.dot(a1, self.w2) + self.b2
			a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
			a2 /= np.sum(a2, axis=1, keepdims=True)

			loss = np.mean((a2 - y) ** 2)
			dz2 = 2 * (a2 - y) / y.shape[0]
			dw2 = np.dot(a1.T, dz2)
			db2 = np.sum(dz2, axis=0, keepdims=True)
			da1 = np.dot(dz2, self.w2.T) * (1 - a1 ** 2)
			dw1 = np.dot(x.T, da1)
			db1 = np.sum(da1, axis=0, keepdims=True)

			self.w1 -= lr * dw1
			self.b1 -= lr * db1
			self.w2 -= lr * dw2
			self.b2 -= lr * db2

			if epoch % 100 == 0:
				print(f"Epoch {epoch}, Loss: {loss:.6f}")

# Encoding utility
def text_to_onehot(text, vocab):
	onehot = np.zeros((len(text), len(vocab)))
	for i, c in enumerate(text):
		onehot[i, vocab.index(c)] = 1
	return onehot

# Reconstruction
def reconstruct(model, input_tensor, vocab):
	output = model.forward(input_tensor)
	indices = np.argmax(output, axis=-1)
	return ''.join([vocab[i] for i in indices])

def est_size_time(model=TinySecretRegenerator):
	est_ram_usage = (model.w1.nbytes + model.b1.nbytes + model.w2.nbytes + model.b2.nbytes + x.nbytes) / 1024
	print(f"✅ Estimated RAM usage (weights + input): {est_ram_usage:.2f} KB")
	print(f"⚡ Inference Latency: {latency_ms:.3f} ms")
	#      add all together time to run the entire script
	end_time0 = time.perf_counter()
	total_time = (end_time0 - start_time0) * 1000  # Convert to milliseconds
	print(f"Total execution time: {total_time:.3f} ms")


# Full on-device training + inference pipeline
if __name__ == "__main__":
	start_time0 = time.perf_counter()
	vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
	secret = ''.join(np.random.choice(vocab, 16))
	print(f"Generated Secret (on-device): {secret}")

	x = text_to_onehot(secret, vocab)
	y = x.copy()

	model = TinySecretRegenerator(len(vocab))
	model.train(x, y, epochs=500)

	# Latency measurement for inference
	start_time = time.perf_counter()
	reconstructed = reconstruct(model, x, vocab)
	end_time = time.perf_counter()
	latency_ms = (end_time - start_time) * 1000

	print(f"Reconstructed Secret (on-device): {reconstructed}")
	
	est_size_time(model=model)

