import numpy as np
import random
eps = 13000  # Number of training epochs
seed = random.randint(0, 1000)  # Random seed for reproducibility
print(f"[Using Random Seed: {seed}]")
# ---------------- Tiny Model (MCU Side) ---------------- #
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
        return (self.weights.nbytes) / 1024


# ---------------- Broker RNN with Vocab Decoding + Hidden State Supervision ---------------- #

# class GatedRNNBroker:
#     def __init__(self, input_dim, hidden_dim=16, vocab=None):
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.vocab = vocab
#         self.vocab_size = len(vocab)
#         np.random.seed(seed)
#         self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.1
#         self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
#         self.Wgate = np.random.randn(hidden_dim, hidden_dim) * 0.1
#         self.W_vocab = np.random.randn(self.vocab_size, hidden_dim) * 0.1
#         self.W_target = np.random.randn(hidden_dim, input_dim) * 0.1  # Projection layer
#
#     def gate(self, h):
#         gate_values = 1 / (1 + np.exp(-np.dot(self.Wgate, h)))
#         return gate_values
#
#     def rnn_step(self, x, h_prev):
#         h_raw = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev))
#         gate = self.gate(h_prev)
#         h = gate * h_raw + (1 - gate) * h_prev
#         return h
#
#     def forward(self, sequence):
#         vocab_logits, hidden_states = [], []
#         for secret in sequence:
#             h = np.zeros(self.hidden_dim)
#             seq_logits, seq_hidden = [], []
#             for x in secret:
#                 h = self.rnn_step(x, h)
#                 logits = np.dot(self.W_vocab, h)
#                 seq_logits.append(logits)
#                 seq_hidden.append(h.copy())
#             vocab_logits.append(np.array(seq_logits))
#             hidden_states.append(np.array(seq_hidden))
#         return vocab_logits, hidden_states
#
#     def train(self, sequence, target_strings, target_vectors, epochs=eps, lr=0.01):
#         targets_idx = [[self.vocab.index(c) for c in s] for s in target_strings]
#         for _ in range(epochs):
#             for secret, idx_seq, target_vec_seq in zip(sequence, targets_idx, target_vectors):
#                 vocab_logits, hidden_states = self.forward([secret])
#                 seq_logits, h_seq = vocab_logits[0], hidden_states[0]
#                 for logits, target_idx, h, target_x in zip(seq_logits, idx_seq, h_seq, target_vec_seq):
#                     probs = np.exp(logits - np.max(logits))
#                     probs /= np.sum(probs)
#                     grad = probs.copy()
#                     grad[target_idx] -= 1
#                     self.W_vocab -= lr * np.outer(grad, h)
#                     target_x_proj = np.dot(self.W_target, target_x)
#                     h_grad = 2 * (h - target_x_proj)
#                     self.Wxh -= lr * np.outer(h_grad, target_x)
#         #     print after 500 epochs
#         #     if _ % 500 == 0:
#         #         print(f"Epoch {_}: Loss = {np.mean([np.sum((v - target_vec) ** 2) for v, target_vec in zip(vocab_logits, target_vectors)])}")
#         #
#         # targets_idx = [[self.vocab.index(c) for c in s] for s in target_strings]
#         # for _ in range(epochs):
#         #     vocab_logits, hidden_states = self.forward(sequence)
#         #     for seq_logits, idx_seq, h_seq, target_vec_seq in zip(vocab_logits, targets_idx, hidden_states, target_vectors):
#         #         for logits, target_idx, h, target_x in zip(seq_logits, idx_seq, h_seq, target_vec_seq):
#         #             probs = np.exp(logits - np.max(logits))
#         #             probs /= np.sum(probs)
#         #             grad = probs.copy()
#         #             grad[target_idx] -= 1
#         #             self.W_vocab -= lr * np.outer(grad, h)
#         #             # hidden state supervision
#         #             target_x_proj = np.dot(self.W_target, target_x)
#         #             h_grad = 2 * (h - target_x_proj)
#         #             self.Wxh -= lr * np.outer(h_grad, target_x)
#
#     def ram_usage_kb(self):
#         return (self.Wxh.nbytes + self.Whh.nbytes + self.Wgate.nbytes + self.W_vocab.nbytes) / 1024

class GatedRNNBroker:
    def __init__(self, vocab=None, hidden_dim=64, lr=0.01):
        self.vocab = vocab
        self.input_dim = len(vocab)
        self.hidden_dim = hidden_dim
        self.lr = lr

        self.Wxh = np.random.randn(hidden_dim, self.input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(self.input_dim, hidden_dim) * 0.01
        self.bh = np.zeros(hidden_dim)
        self.by = np.zeros(self.input_dim)

    def one_hot_seq(self, string):
        return [np.eye(self.input_dim)[self.vocab.index(c)] for c in string]

    def softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / np.sum(e)

    def predict(self, x_seq):
        h = np.zeros(self.hidden_dim)
        out = ""
        for x in x_seq:
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            out += self.vocab[np.argmax(self.softmax(y))]
        return out

    def train(self, string, epochs=1000):
        x_seq = self.one_hot_seq(string)
        t_seq = [self.vocab.index(c) for c in string]

        for epoch in range(epochs):
            hs = [np.zeros(self.hidden_dim)]
            ps = []
            loss = 0

            for t in range(len(x_seq)):
                h = np.tanh(self.Wxh @ x_seq[t] + self.Whh @ hs[-1] + self.bh)
                y = self.Why @ h + self.by
                p = self.softmax(y)
                loss -= np.log(p[t_seq[t]] + 1e-9)
                hs.append(h)
                ps.append(p)

            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dWhy = np.zeros_like(self.Why)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])

            for t in reversed(range(len(x_seq))):
                dy = ps[t]
                dy[t_seq[t]] -= 1
                dby += dy
                dWhy += np.outer(dy, hs[t+1])
                dh = self.Why.T @ dy + dh_next
                dt = (1 - hs[t+1]**2) * dh
                dbh += dt
                dWxh += np.outer(dt, x_seq[t])
                dWhh += np.outer(dt, hs[t])
                dh_next = self.Whh.T @ dt

            for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(dparam, -5, 5, out=dparam)

            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.Why -= self.lr * dWhy
            self.bh  -= self.lr * dbh
            self.by  -= self.lr * dby

            if epoch % 100 == 0:
                recon = self.predict(x_seq)
                print(f"[Epoch {epoch}] Loss: {loss:.4f} | Recon: {recon}")

    def fingerprint_verify(self, string):
        x_seq = self.one_hot_seq(string)
        return self.predict(x_seq)

    def ram_usage_kb(self):
        return sum(w.nbytes for w in [self.Wxh, self.Whh, self.Why, self.bh, self.by]) / 1024.0


# ---------------- Workflow ---------------- #
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")

# Tiny Model
tiny_mcu = TinyModelMCU(vocab=vocab)
secrets, string_secrets = tiny_mcu.generate_3_secrets(length=8)

# Broker Gated RNN
# broker_rnn = GatedRNNBroker(input_dim=tiny_mcu.hidden_dim, hidden_dim=32, vocab=vocab)
broker_rnn = GatedRNNBroker(vocab=vocab, hidden_dim=32, lr=0.01)
print(f"\n[Training Gated RNN Broker with Hybrid Loss on 8-char secrets...]")
broker_rnn.train(secrets, string_secrets, secrets)

# Forward pass after training
reconstructed_sequences, _ = broker_rnn.forward(secrets)

print("\n=== Tiny Model Secret Strings (MCU) ===")
for idx, s in enumerate(string_secrets, 1):
    print(f"Secret {idx}: {s}")

print("\n=== Reconstructed Secrets (Decoded to Vocab) ===")
for idx, recon_seq in enumerate(reconstructed_sequences, 1):
    decoded = ""
    for logits in recon_seq:
        idx_max = np.argmax(logits)
        decoded += vocab[idx_max]
    print(f"Reconstructed Secret {idx}: {decoded}")

print(f"\n[Tiny Model RAM Usage]: {tiny_mcu.ram_usage_kb():.2f} KB")
print(f"[Broker Gated RNN RAM Usage]: {broker_rnn.ram_usage_kb():.2f} KB")
