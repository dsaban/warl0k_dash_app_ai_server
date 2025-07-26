import numpy as np

LR = 0.01
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
target = ""
#  generate a secret of 16 characters from vocab by random seed
import random
import time
random.seed(time.time_ns())  # Fixed seed for reproducibility
target += ''.join(random.choices(vocab, k=32))

seq_len = len(target)
input_dim = len(vocab)
hidden_dim = 64
lr = 0.01

def one_hot(c):
    vec = np.zeros(input_dim)
    vec[vocab.index(c)] = 1.0
    return vec

x_seq = [one_hot(c) for c in target]
t_seq = [vocab.index(c) for c in target]

# Initialize weights small to prevent saturation
Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
Why = np.random.randn(input_dim, hidden_dim) * 0.01
bh = np.zeros(hidden_dim)
by = np.zeros(input_dim)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def predict(x_seq):
    h = np.zeros(hidden_dim)
    out = ""
    for x in x_seq:
        h = np.tanh(Wxh @ x + Whh @ h + bh)
        y = Why @ h + by
        out += vocab[np.argmax(softmax(y))]
    return out

def clip(grad, limit=5.0):
    return np.clip(grad, -limit, limit)

# Training
for epoch in range(501):
    hs = [np.zeros(hidden_dim)]
    ps = []
    loss = 0

    # Forward
    for t in range(seq_len):
        h = np.tanh(Wxh @ x_seq[t] + Whh @ hs[-1] + bh)
        y = Why @ h + by
        p = softmax(y)
        loss -= np.log(p[t_seq[t]] + 1e-8)
        hs.append(h)
        ps.append(p)

    # Backward
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)
    dh_next = np.zeros_like(hs[0])

    for t in reversed(range(seq_len)):
        dy = ps[t].copy()
        dy[t_seq[t]] -= 1
        dby += dy
        dWhy += np.outer(dy, hs[t+1])

        dh = Why.T @ dy + dh_next
        dt = (1 - hs[t+1] ** 2) * dh
        dbh += dt
        dWxh += np.outer(dt, x_seq[t])
        dWhh += np.outer(dt, hs[t])
        dh_next = Whh.T @ dt

    # Clip gradients for stability
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)

    # Update weights
    Wxh -= lr * dWxh
    Whh -= lr * dWhh
    Why -= lr * dWhy
    bh  -= lr * dbh
    by  -= lr * dby

    if epoch % 100 == 0:
        print(f"[Epoch {epoch}] Loss: {loss:.4f} | Recon: {predict(x_seq)}")

# Final report
print(f"\n[Final]: {predict(x_seq)} | Target: {target}")
ram = sum(w.nbytes for w in [Wxh, Whh, Why, bh, by])
print(f"[RAM usage]: {ram / 1024:.2f} KB")
