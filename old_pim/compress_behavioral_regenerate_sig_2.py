import numpy as np

# =======================================================
# CONFIG
# =======================================================

MS_DIM     = 8          # master secret dimension (8 bytes -> 16 ASCII tokens)
SEQ_LEN    = 16         # obfuscated secret length in tokens
HIDDEN_DIM = 32         # RNN hidden size
SIG_DIM    = 8          # behavioral signature size
LR         = 0.01
EPOCHS     = 2000       # train on a single mapping; small net so it's fine

np.random.seed(2)

# Nibble vocab: 16 ASCII tokens 'A'..'P'
VOCAB = [chr(65 + i) for i in range(60)]
print(f"Vocab: {VOCAB}")
char_to_idx = {c: i for i, c in enumerate(VOCAB)}


# =======================================================
# ENCODING / DECODING: MS <-> bytes <-> ASCII tokens
# =======================================================

def ms_bytes_to_ascii_tokens(ms_bytes):
    """
    ms_bytes: array of 8 integers [0..255]
    Returns a 16-char string (2 nibble tokens per byte).
    """
    tokens = []
    for v in ms_bytes:
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        tokens.append(VOCAB[hi])
        tokens.append(VOCAB[lo])
    return ''.join(tokens)  # length 16


def ms_continuous_from_bytes(ms_bytes):
    """
    Map 0..255 bytes to continuous range [-0.9, 0.9] for training.
    """
    ms_norm = ms_bytes.astype(np.float32) / 255.0   # [0,1]
    ms_cont = ms_norm * 1.8 - 0.9                   # [-0.9,0.9]
    return ms_cont


def ms_bytes_from_continuous(ms_vec):
    """
    Inverse mapping: continuous [-0.9,0.9] back to [0,255] ints.
    """
    ms_norm = (ms_vec + 0.9) / 1.8                  # ~[0,1]
    ms_int  = np.clip(ms_norm * 255.0, 0, 255).astype(np.int32)
    return ms_int


def ms_vector_to_ascii_tokens(ms_vec):
    """
    Full reconstruction: continuous vector -> bytes -> ASCII tokens.
    """
    ms_int = ms_bytes_from_continuous(ms_vec)
    return ms_bytes_to_ascii_tokens(ms_int), ms_int


# =======================================================
# TinyModelMCU: generate a NEW (MS, OS) each run
# =======================================================

class TinyModelMCU:
    def __init__(self):
        pass

    def generate_ms_and_secret(self):
        """
        Generate:
          - master secret as 8 bytes in [0,255]
          - continuous MS vector in [-0.9,0.9]
          - OS as 16 ASCII tokens encoding MS
        """
        ms_bytes = np.random.randint(0, 256, size=MS_DIM)
        ms_cont  = ms_continuous_from_bytes(ms_bytes)
        secret   = ms_bytes_to_ascii_tokens(ms_bytes)  # 16-char OS
        return ms_bytes, ms_cont, secret


# =======================================================
# GatedRNNBrokerPIM:
#  - RNN over one-hot OS
#  - final hidden -> reconstruct MS (continuous)
#  - same hidden + (seed,counter,window) -> behavioral signature
# =======================================================

class GatedRNNBrokerPIM:
    def __init__(self, vocab, hidden_dim=HIDDEN_DIM, ms_dim=MS_DIM, sig_dim=SIG_DIM, lr=LR):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        self.ms_dim = ms_dim
        self.sig_dim = sig_dim
        self.lr = lr

        # RNN weights
        self.Wxh = np.random.randn(hidden_dim, self.vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.bh  = np.zeros(hidden_dim, dtype=np.float32)

        # Master secret mapping
        self.W_ms = np.random.randn(ms_dim, hidden_dim) * 0.01
        self.b_ms = np.zeros(ms_dim, dtype=np.float32)

        # Behavioral signature mapping
        self.W_sig = np.random.randn(sig_dim, hidden_dim + 3) * 0.01
        self.b_sig = np.zeros(sig_dim, dtype=np.float32)

    # -------------------------
    # Utilities
    # -------------------------
    def one_hot_seq(self, s):
        T = len(s)
        X = np.zeros((T, self.vocab_size), dtype=np.float32)
        for t, ch in enumerate(s):
            X[t, char_to_idx[ch]] = 1.0
        return X

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def dtanh(x):
        t = np.tanh(x)
        return 1.0 - t * t

    # -------------------------
    # Forward RNN
    # -------------------------
    def forward_rnn(self, X):
        """
        X: (T, vocab_size)
        Returns:
          h_last: (hidden_dim,)
          hs: list of h_t
          pre_acts: list of pre-activation vectors
        """
        T = X.shape[0]
        hs = []
        pre_acts = []
        h = np.zeros(self.hidden_dim, dtype=np.float32)

        for t in range(T):
            pre = self.Wxh @ X[t] + self.Whh @ h + self.bh
            h = self.tanh(pre)
            pre_acts.append(pre)
            hs.append(h)

        return h, hs, pre_acts

    # -------------------------
    # Reconstruct MS
    # -------------------------
    def reconstruct_ms(self, secret_str):
        X = self.one_hot_seq(secret_str)
        h_last, _, _ = self.forward_rnn(X)
        ms_hat = self.W_ms @ h_last + self.b_ms
        return ms_hat, h_last

    # -------------------------
    # Train on SINGLE (OS, MS) mapping
    # -------------------------
    def train_single_mapping(self, secret_str, ms_target_cont, epochs=EPOCHS):
        X = self.one_hot_seq(secret_str)
        T = X.shape[0]
        ms_target_cont = ms_target_cont.astype(np.float32)

        for epoch in range(epochs):
            # Forward
            h_last, hs, pre_acts = self.forward_rnn(X)
            ms_hat = self.W_ms @ h_last + self.b_ms
            diff   = ms_hat - ms_target_cont
            loss   = 0.5 * np.mean(diff ** 2)

            # Backward
            d_ms_hat = diff / self.ms_dim
            dW_ms = np.outer(d_ms_hat, h_last)
            db_ms = d_ms_hat
            dh_last = self.W_ms.T @ d_ms_hat

            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dbh  = np.zeros_like(self.bh)
            dh_next = dh_last

            for t in reversed(range(T)):
                pre_t = pre_acts[t]
                h_prev = np.zeros(self.hidden_dim, dtype=np.float32) if t == 0 else hs[t-1]

                dpre = self.dtanh(pre_t) * dh_next
                dWxh += np.outer(dpre, X[t])
                dWhh += np.outer(dpre, h_prev)
                dbh  += dpre

                dh_next = self.Whh.T @ dpre

            # Gradient clipping
            for g in [dWxh, dWhh, dW_ms]:
                np.clip(g, -5, 5, out=g)

            # Update
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.bh  -= self.lr * dbh
            self.W_ms -= self.lr * dW_ms
            self.b_ms -= self.lr * db_ms

            if (epoch + 1) % (epochs // 10) == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss:.6f}")

    # -------------------------
    # Behavioral signature
    # -------------------------
    def behavioral_signature(self, secret_str, seed, counter, window):
        """
        Use final hidden state from OS + normalized (seed,counter,window)
        to produce a behavioral signature.
        """
        X = self.one_hot_seq(secret_str)
        h_last, _, _ = self.forward_rnn(X)

        p_seed    = (seed % 1000) / 1000.0
        p_counter = (counter % 100) / 100.0
        p_window  = (window % 100) / 100.0
        params    = np.array([p_seed, p_counter, p_window], dtype=np.float32)

        combined = np.concatenate([h_last, params], axis=0)  # (hidden_dim + 3,)
        sig = self.tanh(self.W_sig @ combined + self.b_sig)  # (SIG_DIM,)
        return sig


# =======================================================
# DEMO: single (MS, OS) + ASCII reconstruct + behavior
# =======================================================

if __name__ == "__main__":
    # 1) NEW master + OS each run
    mcu = TinyModelMCU()
    ms_bytes_true, ms_cont_true, secret = mcu.generate_ms_and_secret()

    print("=== Generated Master Secret (bytes) ===")
    print(ms_bytes_true)
    print("=== Obfuscated Secret (16 ASCII tokens) ===")
    print(secret)

    # 2) Train GatedRNNBrokerPIM on this single mapping
    broker = GatedRNNBrokerPIM(VOCAB)
    broker.train_single_mapping(secret, ms_cont_true, epochs=EPOCHS)

    # 3) Honest reconstruction
    ms_hat_cont, h_last = broker.reconstruct_ms(secret)
    recon_err = float(np.linalg.norm(ms_hat_cont - ms_cont_true))

    ascii_recon, ms_bytes_recon = ms_vector_to_ascii_tokens(ms_hat_cont)

    print("\n=== Honest Reconstruction ===")
    print("Reconstructed MS (continuous):")
    print(np.round(ms_hat_cont, 3))
    print("L2 error (continuous):", recon_err)

    print("\nTrue MS bytes:       ", ms_bytes_true)
    print("Reconstructed bytes: ", ms_bytes_recon)
    print("\nTrue MS ASCII tokens:      ", ms_bytes_to_ascii_tokens(ms_bytes_true))
    print("Reconstructed MS ASCII tok:", ascii_recon)

    # 4) Behavioral signature baseline
    seed    = 123
    counter = 7
    window  = 3
    sig_ref = broker.behavioral_signature(secret, seed, counter, window)

    print("\n=== Behavioral Signature (honest) ===")
    print("sig_ref:", np.round(sig_ref, 4))

    def check_behavior(desc, sec, s, c, w):
        sig = broker.behavioral_signature(sec, s, c, w)
        err = float(np.linalg.norm(sig - sig_ref))
        print(f"\n--- {desc} ---")
        print("sig:", np.round(sig, 4))
        print("L2 vs honest:", err)

    # Wrong seed / counter / window
    check_behavior("Wrong SEED", secret, seed + 1, counter, window)
    check_behavior("Wrong COUNTER", secret, seed, counter + 5, window)
    check_behavior("Wrong WINDOW", secret, seed, counter, window + 2)

    # 5) Tampered OS
    tampered_list = list(secret)
    idx0 = char_to_idx[tampered_list[0]]
    tampered_list[0] = VOCAB[(idx0 + 1) % len(VOCAB)]
    tampered_secret = ''.join(tampered_list)

    ms_hat_t_cont, _ = broker.reconstruct_ms(tampered_secret)
    ascii_recon_t, ms_bytes_recon_t = ms_vector_to_ascii_tokens(ms_hat_t_cont)
    recon_err_t = float(np.linalg.norm(ms_hat_t_cont - ms_cont_true))

    print("\n=== Tampered OS ===")
    print("Original OS: ", secret)
    print("Tampered OS: ", tampered_secret)
    print("Reconstructed MS (cont.) from tampered:")
    print(np.round(ms_hat_t_cont, 3))
    print("L2 error (cont.) vs true:", recon_err_t)
    print("Reconstructed bytes (tampered):", ms_bytes_recon_t)
    print("Reconstructed ASCII (tampered):", ascii_recon_t)

    # Behavioral signature on tampered OS (same params)
    check_behavior("Tampered OS (behavior)", tampered_secret, seed, counter, window)
