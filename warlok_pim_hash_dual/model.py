import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash_dual.config import DualPIMHashConfig
from warl0k_cloud_demo_app_multi_client_server_dash.warlok_pim_hash_dual.exceptions import VerificationError


@dataclass
class VerificationResult:
    ok: bool
    error: float
    counter: int
    seed: int
    window_start: int
    window_end: int


class PIMHashDualModel:
    """
    Dual-model Proof-in-Motion inspired primitive.

    - Obfuscation model:
        [master_vec, seed, counter] -> obf_vec (temporary secret)
    - Reconstruction model:
        [obf_vec, seed, counter] -> master_vec_hat

    The model learns to map any obfuscated secret generated from
    (master_secret, seed, counter) back to the same internal master vector.

    This behaves like a *learned HMAC*:
    - You can generate a temp secret (obfuscated) based on seed & counter.
    - You can reconstruct and verify the master identity from the temp secret.
    - You enforce sliding windows on counters.
    - You can monitor how the chain behaves across counters.

    ⚠ NOT cryptographically secure. For research and WARL0K proto experiments only.
    """

    def __init__(self, config: Optional[DualPIMHashConfig] = None):
        self.config = config or DualPIMHashConfig()

        self.secret_dim = self.config.secret_dim
        self.obf_dim = self.config.obf_dim

        # Obfuscation net: [master_vec + seed + counter] -> obf_vec
        self.obf_input_dim = self.secret_dim + 2
        self.obf_hidden_dim = self.config.obf_hidden_dim

        # Reconstruction net: [obf_vec + seed + counter] -> master_vec
        self.rec_input_dim = self.obf_dim + 2
        self.rec_hidden_dim = self.config.recon_hidden_dim

        rng = np.random.default_rng()

        # Obfuscation parameters
        self.W1_o = rng.normal(0, 0.1, (self.obf_input_dim, self.obf_hidden_dim)).astype(np.float32)
        self.b1_o = np.zeros((self.obf_hidden_dim,), dtype=np.float32)
        self.W2_o = rng.normal(0, 0.1, (self.obf_hidden_dim, self.obf_dim)).astype(np.float32)
        self.b2_o = np.zeros((self.obf_dim,), dtype=np.float32)

        # Reconstruction parameters
        self.W1_r = rng.normal(0, 0.1, (self.rec_input_dim, self.rec_hidden_dim)).astype(np.float32)
        self.b1_r = np.zeros((self.rec_hidden_dim,), dtype=np.float32)
        self.W2_r = rng.normal(0, 0.1, (self.rec_hidden_dim, self.secret_dim)).astype(np.float32)
        self.b2_r = np.zeros((self.secret_dim,), dtype=np.float32)

        # State
        self._master_vec: Optional[np.ndarray] = None
        self._max_seed: int = 1
        self._max_counter: int = 1
        self._trained: bool = False

    # ------------------------------------------------------------------
    # Encoding & normalization helpers
    # ------------------------------------------------------------------
    def _encode_master_secret(self, master_secret: bytes) -> np.ndarray:
        """
        Encode a master_secret into an internal vector representation.
        """
        h = hashlib.sha512(master_secret).digest()
        vec = np.frombuffer(h[: self.secret_dim], dtype=np.uint8).astype(np.float32) / 255.0
        return vec

    def _normalize_seed_counter(self, seed: int, counter: int) -> np.ndarray:
        seed_norm = seed / max(self._max_seed, 1)
        counter_norm = counter / max(self._max_counter, 1)
        return np.array([seed_norm, counter_norm], dtype=np.float32)

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------
    def _forward_obf(self, master_vec_batch: np.ndarray, meta_batch: np.ndarray):
        """
        Obfuscation net forward pass.

        master_vec_batch: (B, secret_dim)
        meta_batch:       (B, 2)  [seed_norm, counter_norm]
        Returns:
            obf_vec: (B, obf_dim)   in [-1, 1] via tanh
            cache:   intermediates for backprop
        """
        X = np.concatenate([master_vec_batch, meta_batch], axis=1)  # (B, secret_dim+2)
        z1 = X @ self.W1_o + self.b1_o                              # (B, H_o)
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2_o + self.b2_o                             # (B, obf_dim)
        obf_vec = np.tanh(z2)                                       # keep in [-1, 1]
        return obf_vec, (X, z1, h1, z2)

    def _forward_rec(self, obf_batch: np.ndarray, meta_batch: np.ndarray):
        """
        Reconstruction net forward pass.

        obf_batch: (B, obf_dim)
        meta_batch: (B, 2)

        Returns:
            y_pred: (B, secret_dim)
            cache:  intermediates for backprop
        """
        X = np.concatenate([obf_batch, meta_batch], axis=1)  # (B, obf_dim+2)
        z1 = X @ self.W1_r + self.b1_r                       # (B, H_r)
        h1 = np.tanh(z1)
        z2 = h1 @ self.W2_r + self.b2_r                      # (B, secret_dim)
        y_pred = z2
        return y_pred, (X, z1, h1, z2)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(
        self,
        master_secret: bytes,
        # seed_range: tuple[int, int] = (1, 100),
        # counter_range: tuple[int, int] = (1, 100),
        seed_range: Tuple[int, int] = (0, 15),
        counter_range: Tuple[int, int] = (0, 15),
    ) -> None:
        """
        Train both models jointly to reconstruct a fixed master_secret
        from obfuscated temp secrets based on (seed, counter).

        After training, this instance is "bound" to that master_secret.
        """
        self._master_vec = self._encode_master_secret(master_secret)
        self._max_seed = max(seed_range[1], 1)
        self._max_counter = max(counter_range[1], 1)

        # Build training pairs
        seeds = np.arange(seed_range[0], seed_range[1] + 1, dtype=np.int64)
        counters = np.arange(counter_range[0], counter_range[1] + 1, dtype=np.int64)
        pairs = [(int(s), int(c)) for s in seeds for c in counters]

        rng = np.random.default_rng()
        rng.shuffle(pairs)

        # Build batch tensors
        master_vec = self._master_vec[None, :].astype(np.float32)
        master_batch = np.repeat(master_vec, len(pairs), axis=0)  # (N, secret_dim)

        meta_list = []
        for seed, counter in pairs:
            meta_list.append(self._normalize_seed_counter(seed, counter))
        meta_batch = np.stack(meta_list, axis=0).astype(np.float32)  # (N, 2)

        lr = self.config.learning_rate

        for epoch in range(self.config.epochs):
            # Forward obfuscation
            obf, obf_cache = self._forward_obf(master_batch, meta_batch)

            # Optional training noise on obfuscated vector
            if self.config.noise_std > 0.0:
                obf = obf + rng.normal(0.0, self.config.noise_std, obf.shape).astype(np.float32)

            # Forward reconstruction
            y_pred, rec_cache = self._forward_rec(obf, meta_batch)
            y_true = master_batch

            # Loss = MSE
            diff = (y_pred - y_true)
            loss = float((diff ** 2).mean())

            # ---------------------------
            # Backprop: reconstruction net
            # ---------------------------
            batch_size = master_batch.shape[0]
            dL_dy = (2.0 / batch_size) * diff  # (N, secret_dim)

            X_r, z1_r, h1_r, _z2_r = rec_cache

            dL_dW2_r = h1_r.T @ dL_dy                     # (H_r, secret_dim)
            dL_db2_r = dL_dy.sum(axis=0)                  # (secret_dim,)
            dL_dh1_r = dL_dy @ self.W2_r.T                # (N, H_r)
            dL_dz1_r = dL_dh1_r * (1 - np.tanh(z1_r) ** 2)
            dL_dW1_r = X_r.T @ dL_dz1_r                   # (obf_dim+2, H_r)
            dL_db1_r = dL_dz1_r.sum(axis=0)               # (H_r,)

            # Gradient wrt X_r (contains obf + meta)
            dL_dX_r = dL_dz1_r @ self.W1_r.T              # (N, obf_dim+2)
            dL_dobf = dL_dX_r[:, : self.obf_dim]          # gradient wrt obf output

            # ---------------------------
            # Backprop: obfuscation net
            # ---------------------------
            X_o, z1_o, h1_o, z2_o = obf_cache
            d_tanh_z2 = (1 - np.tanh(z2_o) ** 2)
            dL_dz2_o = dL_dobf * d_tanh_z2                # (N, obf_dim)

            dL_dW2_o = h1_o.T @ dL_dz2_o                  # (H_o, obf_dim)
            dL_db2_o = dL_dz2_o.sum(axis=0)               # (obf_dim,)
            dL_dh1_o = dL_dz2_o @ self.W2_o.T             # (N, H_o)
            dL_dz1_o = dL_dh1_o * (1 - np.tanh(z1_o) ** 2)
            dL_dW1_o = X_o.T @ dL_dz1_o                   # (secret_dim+2, H_o)
            dL_db1_o = dL_dz1_o.sum(axis=0)               # (H_o,)

            # ---------------------------
            # Gradient descent update
            # ---------------------------
            self.W2_r -= lr * dL_dW2_r
            self.b2_r -= lr * dL_db2_r
            self.W1_r -= lr * dL_dW1_r
            self.b1_r -= lr * dL_db1_r

            self.W2_o -= lr * dL_dW2_o
            self.b2_o -= lr * dL_db2_o
            self.W1_o -= lr * dL_dW1_o
            self.b1_o -= lr * dL_db1_o

            # (Optional) you can print loss if you want to monitor training:
            # if epoch % 50 == 0:
            #     print(f"[epoch {epoch}] loss={loss:.6f}")

        self._trained = True

    # ------------------------------------------------------------------
    # Encoding & decoding obfuscated secrets (for transport)
    # ------------------------------------------------------------------
    def _encode_obf_to_bytes(self, obf_vec: np.ndarray) -> bytes:
        """
        Map obf_vec in [-1, 1] to uint8 bytes [0, 255].
        """
        v = np.clip(obf_vec, -1.0, 1.0)
        u8 = np.round((v + 1.0) * 127.5).astype(np.uint8)
        return u8.tobytes()

    def _decode_obf_from_bytes(self, data: bytes) -> np.ndarray:
        """
        Map uint8 bytes [0, 255] back to float32 in [-1, 1].
        """
        u8 = np.frombuffer(data, dtype=np.uint8).astype(np.float32)
        v = (u8 / 127.5) - 1.0
        return v

    # ------------------------------------------------------------------
    # Public API: generate, reconstruct, verify, monitor
    # ------------------------------------------------------------------
    def generate_obfuscated_secret(self, seed: int, counter: int) -> bytes:
        """
        Generate an obfuscated (temporary) secret for this master identity
        and the given (seed, counter). This simulates the *device side*.
        """
        if not self._trained or self._master_vec is None:
            raise RuntimeError("Model must be trained with `fit` before use.")

        master_batch = self._master_vec[None, :].astype(np.float32)
        meta = self._normalize_seed_counter(seed, counter)[None, :]
        obf, _ = self._forward_obf(master_batch, meta)  # (1, obf_dim)
        return self._encode_obf_to_bytes(obf[0])

    def reconstruct_master(self, obfuscated_secret: bytes, seed: int, counter: int) -> np.ndarray:
        """
        Reconstruct the internal master-secret vector from an obfuscated
        secret and (seed, counter). This simulates the *verifier side*.
        """
        if not self._trained or self._master_vec is None:
            raise RuntimeError("Model must be trained with `fit` before use.")

        obf_vec = self._decode_obf_from_bytes(obfuscated_secret)[None, :].astype(np.float32)
        meta = self._normalize_seed_counter(seed, counter)[None, :]
        y_pred, _ = self._forward_rec(obf_vec, meta)
        return y_pred[0]
    
    def roundtrip(self, seed: int, counter: int):
        """
        Full roundtrip:
        1. generate obfuscated secret
        2. reconstruct master vector
        3. convert vector --> raw-like bytes for printing
        4. compute MSE error
        """
        if not self._trained:
            raise RuntimeError("Model must be trained first with fit().")

        obf_bytes = self.generate_obfuscated_secret(seed=seed, counter=counter)
        master_hat_vec = self.reconstruct_master(obf_bytes, seed=seed, counter=counter)
        mse_error = float(((master_hat_vec - self._master_vec) ** 2).mean())

        # NEW: convert reconstructed vector into printable bytes
        master_hat_raw = self.vector_to_bytes(master_hat_vec)

        return obf_bytes, master_hat_vec, master_hat_raw, mse_error


    def vector_to_bytes(self, vec: np.ndarray) -> bytes:
        """
        Convert reconstructed master vector back into a stable byte-string.

        This is NOT encryption — it's only for visualization and debugging.
        Each float is mapped into a single byte by scaling into [0,255].
        """
        # Normalize vec to [0,1]
        v = vec - vec.min()
        if v.max() > 0:
            v = v / v.max()

        # Scale to bytes
        b = (v * 255).astype(np.uint8)

        return bytes(b)

    def verify(
        self,
        obfuscated_secret: bytes,
        seed: int,
        counter: int,
        window_start: int,
        tolerance: float = 1e-3,
    ) -> VerificationResult:
        """
        Verify a temp secret like a learned HMAC:

        - Checks that counter is within [window_start, window_start + window_size - 1]
        - Reconstructs master-secret vector from obfuscated_secret
        - Compares to the stored master identity (self._master_vec)

        Raises VerificationError if invalid.
        """
        if not self._trained or self._master_vec is None:
            raise RuntimeError("Model must be trained with `fit` before use.")

        window_end = window_start + self.config.window_size - 1
        if not (window_start <= counter <= window_end):
            raise VerificationError(
                f"Counter {counter} outside valid window [{window_start}, {window_end}]"
            )

        master_hat = self.reconstruct_master(obfuscated_secret, seed, counter)
        err = float(((master_hat - self._master_vec) ** 2).mean())

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
        seed: int,
        start_counter: int,
        length: int,
        window_start: int,
        tolerance: float = 1e-3,
    ) -> List[VerificationResult]:
        """
        Monitor how verification behaves across a sequence of counters.

        This simulates:
        - generating and sending obfuscated secrets for counters in a loop
        - reconstructing and verifying them
        - observing when they go out of window or exceed tolerance
        """
        if not self._trained or self._master_vec is None:
            raise RuntimeError("Model must be trained with `fit` before use.")

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

            # Simulate full loop: device obfuscates, verifier reconstructs & compares
            obf = self.generate_obfuscated_secret(seed, counter)
            master_hat = self.reconstruct_master(obf, seed, counter)
            err = float(((master_hat - self._master_vec) ** 2).mean())
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
