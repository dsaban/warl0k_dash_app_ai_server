# model/rnn_model.py
# (Copied from the user-provided file)
import numpy as np

class GatedRNNBroker:
    def __init__(self, vocab, hidden_dim=64, lr=0.01):
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
