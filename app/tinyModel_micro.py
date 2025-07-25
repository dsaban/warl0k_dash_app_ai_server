# Step 1: Tiny Inference Library (tiny_inference.py)
import numpy as np

class TinySecretRegenerator:
    def __init__(self, vocab_size, hidden_dim=8):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.weights = None

    def load_weights(self, weight_path):
        data = np.load(weight_path)
        self.w1 = data['w1']
        self.b1 = data['b1']
        self.w2 = data['w2']
        self.b2 = data['b2']

    def forward(self, x):
        x = np.dot(x, self.w1) + self.b1
        x = np.tanh(x)
        x = np.dot(x, self.w2) + self.b2
        return self._softmax(x)

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

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

# Step 2: Tiny Trainer and Model Saver (train_and_save.py)
def train_and_save_model(secret, vocab, epochs=500, hidden_dim=8):
    np.random.seed(42)
    vocab_size = len(vocab)
    x = text_to_onehot(secret, vocab)
    y = x.copy()

    # Initialize weights for two layers
    w1 = np.random.randn(vocab_size, hidden_dim) * 0.1
    b1 = np.zeros((1, hidden_dim))
    w2 = np.random.randn(hidden_dim, vocab_size) * 0.1
    b2 = np.zeros((1, vocab_size))

    learning_rate = 0.05

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        z1 = np.dot(x, w1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = np.exp(z2 - np.max(z2, axis=1, keepdims=True))
        a2 /= np.sum(a2, axis=1, keepdims=True)

        # Loss (mean squared error)
        loss = np.mean((a2 - y) ** 2)

        # Backpropagation
        dz2 = 2 * (a2 - y) / y.shape[0]
        dw2 = np.dot(a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = np.dot(dz2, w2.T) * (1 - a1 ** 2)
        dw1 = np.dot(x.T, da1)
        db1 = np.sum(da1, axis=0, keepdims=True)

        # Gradient descent update
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.6f}")

    # Save weights
    np.savez_compressed("tiny_model.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    print("Model saved as tiny_model.npz")

    model_size = os.path.getsize("tiny_model.npz") / 1024
    print(f"✅ Model size: {model_size:.2f} KB")

# Step 3: Execute Full Pipeline
if __name__ == "__main__":
    import os
    import time
    start = time.perf_counter()
    vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    secret = ''.join(np.random.choice(vocab, 16))
    print(f"Generated Secret: {secret}")

    train_and_save_model(secret, vocab)
    start2 = time.perf_counter()
    
    model = TinySecretRegenerator(len(vocab))
    model.load_weights("tiny_model.npz")
    input_tensor = text_to_onehot(secret, vocab)
    reconstructed = reconstruct(model, input_tensor, vocab)
    end = time.perf_counter()
    print(f"Reconstructed Secret: {reconstructed}")
    print(f"Total time taken: {(end - start) * 1000:.2f} ms")
    print(f"Latency for inference: {(end - start2) * 1000:.2f} ms")
    
    lib_size = os.path.getsize(__file__) / 1024
    model_size = os.path.getsize("tiny_model.npz") / 1024
    total_size = lib_size + model_size
    print(f"📦 Total library + model size: {total_size:.2f} KB")
