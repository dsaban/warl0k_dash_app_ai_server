from dataclasses import dataclass


@dataclass
class DualPIMHashConfig:
    """
    Configuration for the dual-model Proof-in-Motion inspired primitive.

    One model obfuscates the master secret, another reconstructs it.

    Parameters
    ----------
    secret_dim : int
        Dimensionality of internal master-secret vector.
    obf_dim : int
        Dimensionality of the obfuscated (temporary) secret vector.
    obf_hidden_dim : int
        Hidden size of the obfuscation MLP.
    recon_hidden_dim : int
        Hidden size of the reconstruction MLP.
    learning_rate : float
        Gradient descent learning rate.
    epochs : int
        Number of training epochs.
    window_size : int
        Sliding window size for valid counters.
    noise_std : float
        Std-dev of noise injected into obfuscated vectors during training
        (simulating jitter / PQ stress).
    """
    secret_dim: int = 32
    obf_dim: int = 32
    obf_hidden_dim: int = 64
    recon_hidden_dim: int = 64
    learning_rate: float = 0.01
    epochs: int = 500
    window_size: int = 32
    noise_std: float = 0.0
