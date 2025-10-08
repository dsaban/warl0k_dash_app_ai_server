
import numpy as np

class TinyModelMCUWithNoise:
    """Tiny model that generates a secret string deterministically per session seed
    and (optionally) adds reversible noise in vector space. This demo uses strings.
    """
    def __init__(self, vocab, hidden_dim=8, seed=42):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        rng = np.random.RandomState(seed)
        self.weights = rng.randn(self.vocab_size, hidden_dim) * 0.1

    def generate_secret_string(self, length, session_seed):
        rng = np.random.RandomState(session_seed)
        chars = rng.choice(self.vocab, size=length)
        return ''.join(chars)
