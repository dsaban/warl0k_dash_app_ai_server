from dataclasses import dataclass


@dataclass
class PIMHashConfig:
    """
    Configuration for the Proof-in-Motion inspired hash/HMAC-like model.

    Parameters
    ----------
    digest_size : int
        Size (in bytes) of the low-level digest used to build the obfuscated secret.
    secret_dim : int
        Dimensionality of the internal representation of the master secret.
    hidden_dim : int
        Size of the hidden layer of the tiny MLP.
    learning_rate : float
        Learning rate used in gradient descent.
    epochs : int
        Number of training epochs.
    window_size : int
        Size of the valid sliding window of counters. If counter is outside
        the current [start, start+window_size-1], verification fails.
    noise_std : float
        Standard deviation of noise injected during training into the
        obfuscated vectors (simulating "model jitter" / PQ robustness stress).
    """
    digest_size: int = 32
    secret_dim: int = 32
    hidden_dim: int = 64
    learning_rate: float = 0.01
    epochs: int = 1000
    window_size: int = 32
    noise_std: float = 0.0
