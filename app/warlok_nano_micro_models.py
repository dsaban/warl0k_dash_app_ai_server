# tiny_rnn_model.py
import numpy as np

# icvae_model.py
class IncrementalVAE:
    def __init__(self, input_dim, latent_dim=8, hidden_dim=32, beta_schedule=None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta_schedule = beta_schedule or (lambda epoch: 1.0)
        np.random.seed(42)
        
        self.W_enc1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b_enc1 = np.zeros((1, hidden_dim))
        self.W_mu = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_mu = np.zeros((1, latent_dim))
        self.W_logvar = np.random.randn(hidden_dim, latent_dim) * 0.1
        self.b_logvar = np.zeros((1, latent_dim))
        
        self.W_dec1 = np.random.randn(latent_dim, hidden_dim) * 0.1
        self.b_dec1 = np.zeros((1, hidden_dim))
        self.W_out = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b_out = np.zeros((1, input_dim))
    
    def encode(self, x):
        h = np.tanh(np.dot(x, self.W_enc1) + self.b_enc1)
        mu = np.dot(h, self.W_mu) + self.b_mu
        logvar = np.dot(h, self.W_logvar) + self.b_logvar
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = np.exp(0.5 * logvar)
        eps = np.random.randn(*mu.shape)
        return mu + eps * std
    
    def decode(self, z):
        h = np.tanh(np.dot(z, self.W_dec1) + self.b_dec1)
        return np.dot(h, self.W_out) + self.b_out
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def train(self, data, epochs=1000, lr=0.01):
        for epoch in range(epochs):
            beta = self.beta_schedule(epoch)
            for x in data:
                x = x.reshape(1, -1)
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                
                # Reconstruction + KL Loss
                recon_loss = np.mean((x - recon) ** 2)
                kl_loss = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
                loss = recon_loss + beta * kl_loss
                
                # Gradients (simple SGD steps, pseudo-backprop)
                grad_out = 2 * (recon - x) / x.shape[0]
                dW_out = np.dot(np.tanh(np.dot(z, self.W_dec1) + self.b_dec1).T, grad_out)
                self.W_out -= lr * dW_out
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}, Beta: {beta:.2f}")
    
    def ram_usage_kb(self):
        weights = [self.W_enc1, self.W_mu, self.W_logvar, self.W_dec1, self.W_out]
        return sum(w.nbytes for w in weights) / 1024


class TinyModelMCU:
    def __init__(self, vocab, hidden_dim=8, seed=42):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        np.random.seed(seed)
        self.weights = np.random.randn(self.vocab_size, hidden_dim) * 0.1

    def generate_secret(self, length=8):
        secret = ''.join(np.random.choice(self.vocab, length))
        one_hot = np.eye(self.vocab_size)[[self.vocab.index(c) for c in secret]]
        secret_vector = one_hot @ self.weights
        return secret_vector, secret

    def generate_3_secrets(self, length=8):
        secrets = [self.generate_secret(length) for _ in range(3)]
        vectors, strings = zip(*secrets)
        concatenated_vector = np.concatenate(vectors, axis=0)
        concatenated_string = ''.join(strings)
        return [concatenated_vector], [concatenated_string]

    def ram_usage_kb(self):
        return self.weights.nbytes / 1024

class GatedRNNBroker:
    def __init__(self, input_dim, hidden_dim=16, vocab=None, num_heads=2, seed=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vocab = vocab
        self.vocab_size = len(vocab)
        np.random.seed(seed)
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.1
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.Wgate = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_vocab = np.random.randn(self.vocab_size, hidden_dim) * 0.1
        self.W_target = np.random.randn(hidden_dim, input_dim) * 0.1
        self.attn_heads = [np.random.randn(hidden_dim, hidden_dim) * 0.1 for _ in range(num_heads)]
        self.W_attn_proj = np.random.randn(hidden_dim * num_heads + hidden_dim, hidden_dim) * 0.1

    def gate(self, h):
        return 1 / (1 + np.exp(-np.dot(self.Wgate, h)))

    def rnn_step(self, x, h_prev):
        h_raw = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_prev))
        gate = self.gate(h_prev)
        h = gate * h_raw + (1 - gate) * h_prev
        return h / (np.linalg.norm(h) + 1e-8)

    def multi_head_attention(self, h_seq, h_t):
        if len(h_seq) == 0:
            return np.zeros(self.hidden_dim * self.num_heads)
        heads = []
        for W in self.attn_heads:
            scores = [np.dot(W @ h_t, h_i) / np.sqrt(self.hidden_dim) for h_i in h_seq]
            weights = np.exp(scores - np.max(scores))
            weights /= np.sum(weights)
            context = np.sum([w * h for w, h in zip(weights, h_seq)], axis=0)
            heads.append(context)
        return np.concatenate(heads)

    def forward(self, sequence):
        vocab_logits, hidden_states = [], []
        for secret in sequence:
            h = np.zeros(self.hidden_dim)
            seq_logits, seq_hidden = [], []
            for x in secret:
                h = self.rnn_step(x, h)
                attn_ctx = self.multi_head_attention(seq_hidden, h) if seq_hidden else np.zeros(self.hidden_dim * self.num_heads)
                fusion_input = np.concatenate([h, attn_ctx])
                fused = np.tanh(np.dot(fusion_input, self.W_attn_proj))
                logits = np.dot(self.W_vocab, fused)
                seq_logits.append(logits)
                seq_hidden.append(h.copy())
            vocab_logits.append(np.array(seq_logits))
            hidden_states.append(np.array(seq_hidden))
        return vocab_logits, hidden_states

    def train(self, sequence, target_strings, target_vectors, epochs=1000, lr=0.01):
        targets_idx = [[self.vocab.index(c) for c in s] for s in target_strings]
        for _ in range(epochs):
            vocab_logits, hidden_states = self.forward(sequence)
            for seq_logits, idx_seq, h_seq, target_vec_seq in zip(vocab_logits, targets_idx, hidden_states, target_vectors):
                for logits, target_idx, h, target_x in zip(seq_logits, idx_seq, h_seq, target_vec_seq):
                    probs = np.exp(logits - np.max(logits))
                    probs /= np.sum(probs)
                    grad = probs.copy()
                    grad[target_idx] -= 1
                    attn_ctx = self.multi_head_attention(h_seq, h)
                    fusion_input = np.concatenate([h, attn_ctx])
                    fused = np.tanh(np.dot(fusion_input, self.W_attn_proj))
                    self.W_vocab -= lr * np.outer(grad, fused)
                    h_target = np.dot(self.W_target, target_x)
                    h_grad = 2 * (h - h_target)
                    self.Wxh -= lr * np.outer(h_grad, target_x)

    def ram_usage_kb(self):
        total = self.Wxh.nbytes + self.Whh.nbytes + self.Wgate.nbytes + self.W_vocab.nbytes + self.W_target.nbytes + self.W_attn_proj.nbytes
        total += sum(w.nbytes for w in self.attn_heads)
        return total / 1024
