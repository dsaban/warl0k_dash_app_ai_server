import numpy as np

# Define the vocabulary
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*=+[]{}|;:<>?~")

# Define the TinyModelMCU class
class TinyModelMCU:
    def __init__(self, vocab, hidden_dim=8, seed=42):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        np.random.seed(seed)
        self.weights = np.random.randn(self.vocab_size, hidden_dim) * 0.1

    def generate_secret(self, length=8):
        secret = ''.join(np.random.choice(self.vocab, length))
        print(f"Generated Secret (on-device): {secret}")
        one_hot = np.eye(self.vocab_size)[[self.vocab.index(c) for c in secret]]
        secret_vector = one_hot @ self.weights
        return secret_vector, secret

    def generate_3_secrets(self, length=8):
        secrets = [self.generate_secret(length) for _ in range(3)]
        vectors, strings = zip(*secrets)
        return list(vectors), list(strings)

    def ram_usage_kb(self):
        return self.weights.nbytes / 1024

# Function to inject Gaussian noise into each secret vector
def inject_noise(vectors, noise_level=0.05):
    noisy_vectors = [vec + np.random.normal(0, noise_level, vec.shape) for vec in vectors]
    return noisy_vectors

# Initialize model
model = TinyModelMCU(vocab=vocab, hidden_dim=8, seed=99)

# Generate 3 secrets
secret_vectors, secret_strings = model.generate_3_secrets(length=32)

# Inject noise
noisy_secret_vectors = inject_noise(secret_vectors, noise_level=0.05)

# Display results
for i, (secret, vec, noisy_vec) in enumerate(zip(secret_strings, secret_vectors, noisy_secret_vectors), 1):
    print(f"\n🔐 Secret {i}: {secret}")
    # print("Original Vector:\n", vec)
    # print("Noisy Vector:\n", noisy_vec)

# Optional: show RAM usage
print(f"\n💾 Model RAM usage: {model.ram_usage_kb():.2f} KB")
