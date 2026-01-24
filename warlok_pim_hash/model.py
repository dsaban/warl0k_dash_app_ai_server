import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np

from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash.config import PIMHashConfig
from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash.exceptions import VerificationError


@dataclass
class VerificationResult:
    ok: bool
    error: float
    counter: int
    seed: int
    window_start: int
    window_end: int


class PIMHashModel:
    """
    Proof-in-Motion inspired "learned hash/HMAC" model.

    Conceptual behavior:
    --------------------
    1. A fixed master_secret (bytes) is the root identity of a device/session.
    2. For each (seed, counter), we derive an obfuscated_secret using a
       deterministic digest (HMAC-like).
    3. A tiny neural network is trained to reconstruct a fixed internal
       representation of the master_secret from (obfuscated_secret, seed, counter).
    4. At verification time, if the model reconstructs the expected master secret
       within a tolerance, the secret is considered valid (like a hash/HMAC check).
    5. The model can be probed across a sliding window of counters to monitor
       behavioral changes and detect tampering or out-of-window use.

    ⚠ NOT cryptographically secure. For research and WARL0K proto experiments only.
    """

    def __init__(self, config: Optional[PIMHashConfig] = None):
        self.config = config or PIMHashConfig()
        # Input: obfuscated digest bytes + normalized seed + normalized counter
        self.input_dim = self.config.digest_size + 2
        self.hidden_dim = self.config.hidden_dim
        self.output_dim = self.config.secret_dim

        # Initialize tiny MLP parameters
        rng = np.random.default_rng()
        self.W1 = rng.normal(0, 0.1, (self.input_dim, self.hidden_dim)).astype(np.float32)
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (self.hidden_dim, self.output_dim)).astype(np.float32)
        self.b2 = np.zeros((self.output_dim,), dtype=np.float32)

        # Placeholders for learned master representation and normalization
        self._master_vec: Optional[np.ndarray] = None
        self._max_seed: int = 1
        self._max_counter: int = 1
        self._trained: bool = False

    # ---------------------------------------------------------------------
    # Low-level helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_bytes(value: int, length: int = 8) -> bytes:
        return int(value).to_bytes(length, byteorder="big", signed=False)

    def _obfuscate_secret_digest(self, master_secret: bytes, seed: int, counter: int) -> np.ndarray:
        """
        Create a deterministic obfuscated digest of (master_secret, seed, counter).

        This is analogous to an HMAC-style construction but simplified.
        """
        h = hashlib.sha512()
        h.update(master_secret)
        h.update(self._to_bytes(seed))
        h.update(self._to_bytes(counter))
        digest = h.digest()[: self.config.digest_size]
        # Map bytes to float32 [0, 1]
        return (np.frombuffer(digest, dtype=np.uint8).astype(np.float32) / 255.0)

    def _encode_master_secret(self, master_secret: bytes) -> np.ndarray:
        """
        Encode a master_secret into a fixed internal vector representation.

        We just hash it and slice to secret_dim.
        """
        h = hashlib.sha512(master_secret).digest()
        vec = np.frombuffer(h[: self.config.secret_dim], dtype=np.uint8).astype(np.float32) / 255.0
        return vec

    def _normalize_seed_counter(self, seed: int, counter: int) -> np.ndarray:
        seed_norm = seed / max(self._max_seed, 1)
        counter_norm = counter / max(self._max_counter, 1)
        return np.array([seed_norm, counter_norm], dtype=np.float32)

    def _forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through tiny MLP.

        x: (batch, input_dim)
        Returns: (batch, output_dim)
        """
        z1 = x @ self.W1 + self.b1  # (batch, hidden_dim)
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2 + self.b2  # (batch, output_dim)
        # We keep output linear; you can also apply tanh if you want strict [-1,1]
        return z2

    def _train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        One gradient descent step with MSE loss.
        """
        lr = self.config.learning_rate
        batch_size = X.shape[0]

        # Forward
        z1 = X @ self.W1 + self.b1
        h1 = np.tanh(z1)
        y_pred = h1 @ self.W2 + self.b2

        # Loss and gradient
        diff = (y_pred - y)
        loss = float((diff ** 2).mean())

        # Backprop
        dL_dy = (2.0 / batch_size) * diff  # (batch, output_dim)
        dL_dW2 = h1.T @ dL_dy  # (hidden_dim, output_dim)
        dL_db2 = dL_dy.sum(axis=0)

        dL_dh1 = dL_dy @ self.W2.T  # (batch, hidden_dim)
        dL_dz1 = dL_dh1 * (1 - np.tanh(z1) ** 2)  # derivative tanh

        dL_dW1 = X.T @ dL_dz1  # (input_dim, hidden_dim)
        dL_db1 = dL_dz1.sum(axis=0)

        # Gradient descent
        self.W2 -= lr * dL_dW2
        self.b2 -= lr * dL_db2
        self.W1 -= lr * dL_dW1
        self.b1 -= lr * dL_db1

        return loss

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(
        self,
        master_secret: bytes,
        seed_range: Tuple[int, int] = (1, 100),
        counter_range: Tuple[int, int] = (1, 100),
    ) -> None:
        """
        Train the model to reconstruct a fixed master_secret from obfuscated secrets
        derived from (seed, counter).

        The model learns:
        f(obfuscated_digest, seed, counter) -> master_vec

        Parameters
        ----------
        master_secret : bytes
            The master secret representing device/identity.
        seed_range : (int, int)
            Inclusive range of seeds used for training.
        counter_range : (int, int)
            Inclusive range of counters used for training.
        """
        self._master_vec = self._encode_master_secret(master_secret)
        self._max_seed = max(seed_range[1], 1)
        self._max_counter = max(counter_range[1], 1)

        seeds = np.arange(seed_range[0], seed_range[1] + 1, dtype=np.int64)
        counters = np.arange(counter_range[0], counter_range[1] + 1, dtype=np.int64)

        pairs = [(int(s), int(c)) for s in seeds for c in counters]
        rng = np.random.default_rng()
        rng.shuffle(pairs)

        X_list = []
        y_list = []

        for seed, counter in pairs:
            digest_vec = self._obfuscate_secret_digest(master_secret, seed, counter)
            if self.config.noise_std > 0.0:
                digest_vec = digest_vec + rng.normal(0.0, self.config.noise_std, digest_vec.shape).astype(
                    np.float32
                )
            meta = self._normalize_seed_counter(seed, counter)
            x = np.concatenate([digest_vec, meta], axis=0)
            X_list.append(x)
            y_list.append(self._master_vec)

        X = np.stack(X_list, axis=0).astype(np.float32)
        y = np.stack(y_list, axis=0).astype(np.float32)

        # Training loop
        for epoch in range(self.config.epochs):
            loss = self._train_step(X, y)
            # You can add logging here if needed

        self._trained = True

    def reconstruct_master(
        self,
        master_secret: bytes,
        seed: int,
        counter: int,
    ) -> np.ndarray:
        """
        Reconstruct an internal master-secret vector based on a derived obfuscated
        secret (digest of master_secret, seed, counter).

        Returns
        -------
        master_vec_hat : np.ndarray
            Reconstructed internal master secret representation (secret_dim,).
        """
        if not self._trained:
            raise RuntimeError("Model must be trained with `fit` before reconstructing.")

        digest_vec = self._obfuscate_secret_digest(master_secret, seed, counter)
        meta = self._normalize_seed_counter(seed, counter)
        x = np.concatenate([digest_vec, meta], axis=0)[None, :].astype(np.float32)
        y_hat = self._forward(x)[0]
        return y_hat

    def verify(
        self,
        master_secret: bytes,
        seed: int,
        counter: int,
        window_start: int,
        tolerance: float = 1e-3,
    ) -> VerificationResult:
        """
        Verify that the current (seed, counter) produces an obfuscated secret that
        maps back to the same master-secret representation within tolerance, and
        that counter is inside a sliding window [window_start, window_start + window_size - 1].

        Acts conceptually like a learned HMAC verification.

        Raises
        ------
        VerificationError
            If verification fails or counter is out of window.
        """
        if not self._trained or self._master_vec is None:
            raise RuntimeError("Model must be trained with `fit` before verification.")

        window_end = window_start + self.config.window_size - 1
        if not (window_start <= counter <= window_end):
            raise VerificationError(
                f"Counter {counter} outside valid window [{window_start}, {window_end}]"
            )

        master_vec_hat = self.reconstruct_master(master_secret, seed, counter)
        err = float(((master_vec_hat - self._master_vec) ** 2).mean())

        if err > tolerance:
            raise VerificationError(
                f"Verification failed: reconstruction error {err:.6f} > tolerance {tolerance:.6f}"
            )

        return VerificationResult(
            ok=True,
            error=err,
            counter=counter,
            seed=seed,
            window_start=window_start,
            window_end=window_end,
        )

    def monitor_chain(
        self,
        master_secret: bytes,
        seed: int,
        start_counter: int,
        length: int,
        window_start: int,
        tolerance: float = 1e-3,
    ) -> List[VerificationResult]:
        """
        Monitor how the reconstruction behaves across a chain of counters
        in a sliding window.

        This lets you:
        - observe how the model reacts to sequential counters
        - detect when counters leave a valid window
        - see where reconstruction error starts to drift

        Returns
        -------
        results : List[VerificationResult]
        """
        results: List[VerificationResult] = []
        for i in range(length):
            counter = start_counter + i
            window_end = window_start + self.config.window_size - 1
            if not (window_start <= counter <= window_end):
                results.append(
                    VerificationResult(
                        ok=False,
                        error=float("inf"),
                        counter=counter,
                        seed=seed,
                        window_start=window_start,
                        window_end=window_end,
                    )
                )
                continue

            master_vec_hat = self.reconstruct_master(master_secret, seed, counter)
            print("master_vec_hat:", master_vec_hat)
            err = float(((master_vec_hat - self._master_vec) ** 2).mean())
            ok = err <= tolerance
            results.append(
                VerificationResult(
                    ok=ok,
                    error=err,
                    counter=counter,
                    seed=seed,
                    window_start=window_start,
                    window_end=window_end,
                )
            )
        return results


# -------------------------------------------------------------------------
# Optional: simple demo when run as a script
# -------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = PIMHashConfig(epochs=1000, window_size=8, noise_std=0.01)
    model = PIMHashModel(cfg)

    master = b"this is some master secret representing device identity"
    model.fit(master, seed_range=(1, 4), counter_range=(1, 16))

    seed = 2
    window_start = 5

    # Single verification
    try:
        result = model.verify(master, seed=seed, counter=7, window_start=window_start, tolerance=1e-3)
        print("Verification OK:", result)
    except VerificationError as e:
        print("Verification FAILED:", e)

    # Monitor chain
    chain_results = model.monitor_chain(
        master_secret=master,
        seed=seed,
        start_counter=4,
        length=12,
        window_start=window_start,
        tolerance=1e-3,
    )
    for r in chain_results:
        print(
            f"ctr={r.counter} ok={r.ok} err={r.error:.6f} "
            f"window=[{r.window_start},{r.window_end}]"
        )
