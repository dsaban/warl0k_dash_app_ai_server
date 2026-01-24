import numpy as np
import shutil
import os
# =======================================================
# CONFIG
# =======================================================

SEQ_LEN    = 16         # length of obfuscated secret (OS) in ASCII tokens
HIDDEN_DIM = 32         # RNN hidden size
MS_DIM     = 16          # master secret dimension
SIG_DIM    = 16         # behavioral signature dimension
LR         = 0.01
EPOCHS     = 10000

np.random.seed(2)

# Simple ASCII vocab (16 chars)
VOCAB = [chr(65 + i) for i in range(60)]  # 'A'..'P'
# shuffle the vocab to avoid any ordering bias
np.random.shuffle(VOCAB)
# print is the same why?
#  shuffle changes each run otherwise


print(f"Vocab: {VOCAB}")
char_to_idx = {c: i for i, c in enumerate(VOCAB)}


# =======================================================
# TinyModelMCU: Generates a single random OS string
# =======================================================

class TinyModelMCU:
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)

    def generate_secret(self, length=SEQ_LEN):
        secret = ''.join(np.random.choice(self.vocab, length))
        return secret


# =======================================================
# GatedRNNBrokerPIM:
#  - RNN over one-hot OS sequence
#  - Learns to map final hidden state -> Master Secret
#  - Uses same hidden state + (seed,counter,window)
#    to produce behavioral signature
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

        # Master secret mapping (from final hidden state)
        self.W_ms = np.random.randn(ms_dim, hidden_dim) * 0.01
        self.b_ms = np.zeros(ms_dim, dtype=np.float32)

        # Behavioral signature mapping (from hidden + params)
        self.W_sig = np.random.randn(sig_dim, hidden_dim + 3) * 0.01
        self.b_sig = np.zeros(sig_dim, dtype=np.float32)

    # -------------------------
    # Utilities
    # -------------------------
    def one_hot_seq(self, s):
        X = np.zeros((len(s), self.vocab_size), dtype=np.float32)
        for t, ch in enumerate(s):
            X[t, char_to_idx[ch]] = 1.0
        return X

    def tanh(self, x):
        return np.tanh(x)

    def dtanh(self, x):
        t = np.tanh(x)
        return 1.0 - t * t

    # -------------------------
    # Forward RNN on secret
    # -------------------------
    def forward_rnn(self, X):
        """
        X: (T, vocab_size)
        Returns:
          h_last: (hidden_dim,)
          hs: list of hidden states
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
    # Reconstruct Master Secret
    # -------------------------
    def reconstruct_ms(self, secret_str):
        X = self.one_hot_seq(secret_str)
        h_last, _, _ = self.forward_rnn(X)
        ms_hat = self.W_ms @ h_last + self.b_ms
        return ms_hat, h_last

    # -------------------------
    # Train on ONE (secret, MS) pair
    # -------------------------
    def train_single_mapping(self, secret_str, master_secret, epochs=EPOCHS):
        X = self.one_hot_seq(secret_str)
        T = X.shape[0]
        ms_target = master_secret.astype(np.float32)

        for epoch in range(epochs):
            # Forward
            h_last, hs, pre_acts = self.forward_rnn(X)
            ms_hat = self.W_ms @ h_last + self.b_ms
            diff = ms_hat - ms_target
            loss = 0.5 * np.mean(diff ** 2)

            # Backward: dL/d(ms_hat)
            d_ms_hat = diff / self.ms_dim

            # Gradients for MS mapping
            dW_ms = np.outer(d_ms_hat, h_last)
            db_ms = d_ms_hat
            dh_last = self.W_ms.T @ d_ms_hat

            # Backprop through RNN
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dbh  = np.zeros_like(self.bh)

            dh_next = dh_last

            for t in reversed(range(T)):
                h_t = hs[t]
                pre_t = pre_acts[t]
                h_prev = np.zeros(self.hidden_dim, dtype=np.float32) if t == 0 else hs[t-1]

                dpre = self.dtanh(pre_t) * dh_next
                dWxh += np.outer(dpre, X[t])
                dWhh += np.outer(dpre, h_prev)
                dbh  += dpre

                dh_next = self.Whh.T @ dpre

            # Update
            for grad in [dWxh, dWhh, dW_ms]:
                np.clip(grad, -5, 5, out=grad)

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
        Use final hidden state from secret + encoded (seed,counter,window)
        to produce a behavioral signature vector.
        """
        X = self.one_hot_seq(secret_str)
        h_last, _, _ = self.forward_rnn(X)

        # Normalize params into [0,1]-ish
        p_seed    = (seed % 1000) / 1000.0
        p_counter = (counter % 100) / 100.0
        p_window  = (window % 100) / 100.0

        params_vec = np.array([p_seed, p_counter, p_window], dtype=np.float32)

        combined = np.concatenate([h_last, params_vec], axis=0)  # (hidden_dim + 3,)
        sig = self.tanh(self.W_sig @ combined + self.b_sig)      # (sig_dim,)
        return sig


# =======================================================
# DEMO: Single OS→MS + Behavioral PIM checks
# =======================================================

if __name__ == "__main__":
    # 1) Generate NEW secret each run
    mcu = TinyModelMCU(VOCAB)
    secret = mcu.generate_secret()
    print("Generated OS (16 ASCII tokens):", secret)

    # 2) Define master secret for this session (random)
    master_secret = np.random.randn(MS_DIM).astype(np.float32)
    print("Master secret (target):", np.round(master_secret, 3))

    # 3) Train GatedRNNBrokerPIM on this SINGLE mapping
    broker = GatedRNNBrokerPIM(VOCAB)
    broker.train_single_mapping(secret, master_secret, epochs=EPOCHS)

    # 4) Honest reconstruction
    ms_hat, h_last = broker.reconstruct_ms(secret)
    recon_err = float(np.linalg.norm(ms_hat - master_secret))
    print("\n=== Honest Reconstruction ===")
    print("Reconstructed MS:", np.round(ms_hat, 3))
    print("L2 error:", recon_err)

    # 5) Honest behavioral signature
    seed    = 123
    counter = 7
    window  = 3
    sig_ref = broker.behavioral_signature(secret, seed, counter, window)
    print("\nBehavioral signature (honest):", np.round(sig_ref, 4))

    # ---------------------------------------------------
    # Deviation tests
    # ---------------------------------------------------
    def check_behavioral_deviation(desc, secret_str, seed_, counter_, window_):
        sig = broker.behavioral_signature(secret_str, seed_, counter_, window_)
        sig_err = float(np.linalg.norm(sig - sig_ref))
        print(f"\n=== {desc} ===")
        print("Signature:", np.round(sig, 4))
        print("L2 vs honest:", sig_err)

    # Same secret, wrong SEED
    check_behavioral_deviation("Wrong SEED", secret, seed + 1, counter, window)

    # Same secret, wrong COUNTER
    check_behavioral_deviation("Wrong COUNTER", secret, seed, counter + 5, window)

    # Same secret, wrong WINDOW
    check_behavioral_deviation("Wrong WINDOW", secret, seed, counter, window + 2)

    # Tampered OS (single char changed)
    tampered_list = list(secret)
    tampered_list[0] = VOCAB[(char_to_idx[tampered_list[0]] + 1) % len(VOCAB)]
    tampered_secret = ''.join(tampered_list)

    ms_hat_tampered, _ = broker.reconstruct_ms(tampered_secret)
    recon_err_tampered = float(np.linalg.norm(ms_hat_tampered - master_secret))

    print("\n=== Tampered OS ===")
    print("Tampered secret:", tampered_secret)
    print("Reconstructed MS:", np.round(ms_hat_tampered, 3))
    print("L2 error vs true MS:", recon_err_tampered)

    # Behavioral signature on tampered OS
    check_behavioral_deviation("Tampered OS (behavior)", tampered_secret, seed, counter, window)
