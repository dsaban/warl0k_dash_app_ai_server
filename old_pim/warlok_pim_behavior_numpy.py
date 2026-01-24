import numpy as np


# -----------------------
# Basic utilities
# -----------------------

def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def cross_entropy_loss(probs, target_idx):
    eps = 1e-12
    p = probs[target_idx] + eps
    loss = -np.log(p)
    grad_logits = probs.copy()
    grad_logits[target_idx] -= 1.0  # dL/dz = p - y
    return loss, grad_logits


# -----------------------
# RNN + Attention PIM model (NumPy only)
# -----------------------

class RNNAttentionPIM:
    """
    WARL0K-style RNN + Attention core with:

      - seed, counter -> normalized input
      - encoder RNN over bytes + (seed,counter)
      - dot-product attention -> context
      - decoder RNN with teacher forcing during training
      - softmax over 256 bytes per step
      - training across multiple (seed, counter) pairs
      - chain monitor with counter windows & anomaly flags
    """

    def __init__(self, vocab_size=256, embed_dim=16, hidden_dim=32, seed=42,
                 max_seed=8, max_counter=64):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_seed = max_seed
        self.max_counter = max_counter

        rng = np.random.default_rng(seed)

        # Shared byte embedding
        self.W_emb = rng.normal(0, 0.1, size=(vocab_size, embed_dim))

        # Encoder RNN: input = [embed_dim + 2]  (seed_norm, counter_norm)
        self.enc_input_dim = embed_dim + 2
        self.Wxh_enc = rng.normal(0, 0.1, size=(self.enc_input_dim, hidden_dim))
        self.Whh_enc = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.bh_enc = np.zeros(hidden_dim)

        # Attention: q = [seed_norm, counter_norm] @ W_q  -> (hidden_dim,)
        self.W_q = rng.normal(0, 0.1, size=(2, hidden_dim))

        # Decoder RNN: input = [embed_dim + hidden_dim] (prev byte emb + context)
        self.dec_input_dim = embed_dim + hidden_dim
        self.Wxh_dec = rng.normal(0, 0.1, size=(self.dec_input_dim, hidden_dim))
        self.Whh_dec = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.bh_dec = np.zeros(hidden_dim)

        # Map context -> initial decoder hidden
        self.W_ctx = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.b_ctx = np.zeros(hidden_dim)

        # Output projection: hidden -> logits over bytes
        self.W_ho = rng.normal(0, 0.1, size=(hidden_dim, vocab_size))
        self.b_o = np.zeros(vocab_size)

    # -----------------------
    # One (seq, seed, counter) forward
    # -----------------------
    def _forward_single(self, seq_bytes, seed_value, counter_value, noise_std=0.0):
        """
        seq_bytes: (L,) int64 in [0..255]
        """
        L = seq_bytes.shape[0]

        # Normalize seed / counter into [0,1]
        s_norm = seed_value / float(max(self.max_seed, 1))
        c_norm = counter_value / float(max(self.max_counter, 1))
        sc_vec = np.array([s_norm, c_norm])

        # ===== Encoder =====
        enc_x_cat = []
        enc_a = []
        enc_h = []
        enc_h_prev = []

        h_prev = np.zeros(self.hidden_dim)

        for t in range(L):
            idx = int(seq_bytes[t])
            emb = self.W_emb[idx].copy()              # (E,)
            if noise_std > 0.0:
                emb += np.random.normal(0.0, noise_std, size=emb.shape)

            x_cat = np.concatenate([emb, sc_vec])     # (E+2,)
            a_t = x_cat @ self.Wxh_enc + h_prev @ self.Whh_enc + self.bh_enc
            h_t = np.tanh(a_t)

            enc_x_cat.append(x_cat)
            enc_a.append(a_t)
            enc_h.append(h_t)
            enc_h_prev.append(h_prev)

            h_prev = h_t

        enc_h = np.array(enc_h)  # (L, H)

        # ===== Attention over encoder states (dot with q(seed,counter)) =====
        q = sc_vec @ self.W_q                # (H,)
        e = enc_h @ q                        # (L,)

        e_shift = e - np.max(e)
        alpha = np.exp(e_shift)
        alpha = alpha / np.sum(alpha)        # (L,)

        context = np.sum(alpha[:, None] * enc_h, axis=0)  # (H,)

        # ===== Decoder RNN with teacher forcing =====
        a0_dec = context @ self.W_ctx + self.b_ctx
        h_dec_prev = np.tanh(a0_dec)
        dec_a0 = a0_dec.copy()
        dec_h0 = h_dec_prev.copy()

        dec_x_cat = []
        dec_a = []
        dec_h = []
        dec_h_prev = []
        grad_logits_list = []

        total_loss = 0.0
        BOS = 0

        for t in range(L):
            in_idx = BOS if t == 0 else int(seq_bytes[t - 1])

            emb_dec = self.W_emb[in_idx].copy()
            if noise_std > 0.0:
                emb_dec += np.random.normal(0.0, noise_std, size=emb_dec.shape)

            x_cat = np.concatenate([emb_dec, context])   # (E+H,)
            a_t = x_cat @ self.Wxh_dec + h_dec_prev @ self.Whh_dec + self.bh_dec
            h_t = np.tanh(a_t)
            logits = h_t @ self.W_ho + self.b_o
            probs = softmax(logits)

            loss_t, grad_logits_t = cross_entropy_loss(probs, int(seq_bytes[t]))
            total_loss += loss_t

            dec_x_cat.append(x_cat)
            dec_a.append(a_t)
            dec_h.append(h_t)
            dec_h_prev.append(h_dec_prev)
            grad_logits_list.append(grad_logits_t)

            h_dec_prev = h_t

        avg_loss = total_loss / L

        cache = {
            "L": L,
            "seq_bytes": seq_bytes,
            "sc_vec": sc_vec,
            "enc_x_cat": enc_x_cat,
            "enc_a": enc_a,
            "enc_h": enc_h,
            "enc_h_prev": enc_h_prev,
            "q": q,
            "e": e,
            "alpha": alpha,
            "context": context,
            "a0_dec": dec_a0,
            "h0_dec": dec_h0,
            "dec_x_cat": dec_x_cat,
            "dec_a": dec_a,
            "dec_h": dec_h,
            "dec_h_prev": dec_h_prev,
            "grad_logits_list": grad_logits_list,
        }

        return avg_loss, cache

    # -----------------------
    # Backprop for one (seq, seed, counter)
    # -----------------------
    def _backward_single(self, cache):
        L = cache["L"]
        seq_bytes = cache["seq_bytes"]
        sc_vec = cache["sc_vec"]
        enc_x_cat = cache["enc_x_cat"]
        enc_h = cache["enc_h"]
        enc_h_prev = cache["enc_h_prev"]
        q = cache["q"]
        alpha = cache["alpha"]
        context = cache["context"]
        h0_dec = cache["h0_dec"]
        dec_x_cat = cache["dec_x_cat"]
        dec_h = cache["dec_h"]
        dec_h_prev = cache["dec_h_prev"]
        grad_logits_list = cache["grad_logits_list"]

        # Init grads
        grad_W_emb = np.zeros_like(self.W_emb)
        grad_Wxh_enc = np.zeros_like(self.Wxh_enc)
        grad_Whh_enc = np.zeros_like(self.Whh_enc)
        grad_bh_enc = np.zeros_like(self.bh_enc)

        grad_W_q = np.zeros_like(self.W_q)

        grad_Wxh_dec = np.zeros_like(self.Wxh_dec)
        grad_Whh_dec = np.zeros_like(self.Whh_dec)
        grad_bh_dec = np.zeros_like(self.bh_dec)

        grad_W_ctx = np.zeros_like(self.W_ctx)
        grad_b_ctx = np.zeros_like(self.b_ctx)

        grad_W_ho = np.zeros_like(self.W_ho)
        grad_b_o = np.zeros_like(self.b_o)

        # Decoder
        dcontext = np.zeros_like(context)
        dh_next_dec = np.zeros(self.hidden_dim)

        for t in reversed(range(L)):
            grad_logits = grad_logits_list[t] / L
            h_t = dec_h[t]
            h_prev = dec_h_prev[t]
            x_cat = dec_x_cat[t]

            # Output layer
            grad_W_ho += np.outer(h_t, grad_logits)
            grad_b_o += grad_logits

            dh = grad_logits @ self.W_ho.T + dh_next_dec
            da = dh * (1.0 - h_t ** 2)

            grad_Wxh_dec += np.outer(x_cat, da)
            grad_Whh_dec += np.outer(h_prev, da)
            grad_bh_dec += da

            dx_cat = da @ self.Wxh_dec.T
            dh_prev = da @ self.Whh_dec.T

            d_emb = dx_cat[:self.embed_dim]
            d_ctx_t = dx_cat[self.embed_dim:]

            BOS = 0
            in_idx = BOS if t == 0 else int(seq_bytes[t - 1])
            grad_W_emb[in_idx] += d_emb
            dcontext += d_ctx_t

            dh_next_dec = dh_prev

        # Initial decoder hidden gradient
        dh0 = dh_next_dec
        da0 = dh0 * (1.0 - h0_dec ** 2)
        grad_W_ctx += np.outer(context, da0)
        grad_b_ctx += da0
        dcontext += da0 @ self.W_ctx.T

        # Attention + encoder
        d_enc_h = np.zeros_like(enc_h)
        d_alpha = np.zeros(L)

        # from context
        for t in range(L):
            d_enc_h[t] += dcontext * alpha[t]
            d_alpha[t] = np.dot(dcontext, enc_h[t])

        dot = np.sum(alpha * d_alpha)
        d_e = alpha * (d_alpha - dot)

        dq = np.zeros_like(q)
        for t in range(L):
            d_enc_h[t] += d_e[t] * q
            dq += d_e[t] * enc_h[t]

        grad_W_q += np.outer(sc_vec, dq)

        dh_next_enc = np.zeros(self.hidden_dim)
        for t in reversed(range(L)):
            h_t = enc_h[t]
            h_prev = enc_h_prev[t]
            x_cat = enc_x_cat[t]

            dh_total = d_enc_h[t] + dh_next_enc
            da = dh_total * (1.0 - h_t ** 2)

            grad_Wxh_enc += np.outer(x_cat, da)
            grad_Whh_enc += np.outer(h_prev, da)
            grad_bh_enc += da

            dx_cat_enc = da @ self.Wxh_enc.T
            dh_prev_enc = da @ self.Whh_enc.T

            d_emb_enc = dx_cat_enc[:self.embed_dim]
            in_idx = int(seq_bytes[t])
            grad_W_emb[in_idx] += d_emb_enc

            dh_next_enc = dh_prev_enc

        grads = {
            "W_emb": grad_W_emb,
            "Wxh_enc": grad_Wxh_enc,
            "Whh_enc": grad_Whh_enc,
            "bh_enc": grad_bh_enc,
            "W_q": grad_W_q,
            "Wxh_dec": grad_Wxh_dec,
            "Whh_dec": grad_Whh_dec,
            "bh_dec": grad_bh_dec,
            "W_ctx": grad_W_ctx,
            "b_ctx": grad_b_ctx,
            "W_ho": grad_W_ho,
            "b_o": grad_b_o,
        }
        return grads

    def _init_zero_grads(self):
        return {
            "W_emb": np.zeros_like(self.W_emb),
            "Wxh_enc": np.zeros_like(self.Wxh_enc),
            "Whh_enc": np.zeros_like(self.Whh_enc),
            "bh_enc": np.zeros_like(self.bh_enc),
            "W_q": np.zeros_like(self.W_q),
            "Wxh_dec": np.zeros_like(self.Wxh_dec),
            "Whh_dec": np.zeros_like(self.Whh_dec),
            "bh_dec": np.zeros_like(self.bh_dec),
            "W_ctx": np.zeros_like(self.W_ctx),
            "b_ctx": np.zeros_like(self.b_ctx),
            "W_ho": np.zeros_like(self.W_ho),
            "b_o": np.zeros_like(self.b_o),
        }

    def _accumulate_grads(self, acc, g):
        for k in acc.keys():
            acc[k] += g[k]

    def apply_grads(self, grads, lr):
        self.W_emb -= lr * grads["W_emb"]
        self.Wxh_enc -= lr * grads["Wxh_enc"]
        self.Whh_enc -= lr * grads["Whh_enc"]
        self.bh_enc -= lr * grads["bh_enc"]
        self.W_q -= lr * grads["W_q"]
        self.Wxh_dec -= lr * grads["Wxh_dec"]
        self.Whh_dec -= lr * grads["Whh_dec"]
        self.bh_dec -= lr * grads["bh_dec"]
        self.W_ctx -= lr * grads["W_ctx"]
        self.b_ctx -= lr * grads["b_ctx"]
        self.W_ho -= lr * grads["W_ho"]
        self.b_o -= lr * grads["b_o"]

    # -----------------------
    # Training across (seed, counter) grid
    # -----------------------
    def train(self, master_secret: bytes, seed_range=(1, 4), counter_range=(1, 16),
              epochs=200, lr=1e-2, noise_std=0.0):
        seq = np.frombuffer(master_secret, dtype=np.uint8).astype(np.int64)
        L = seq.shape[0]
        seeds = list(range(seed_range[0], seed_range[1] + 1))
        counters = list(range(counter_range[0], counter_range[1] + 1))
        num_pairs = len(seeds) * len(counters)
        print(f"[train] seq_len={L}, pairs={num_pairs}, epochs={epochs}")

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            acc_grads = self._init_zero_grads()

            for s in seeds:
                for c in counters:
                    loss, cache = self._forward_single(seq, s, c, noise_std=noise_std)
                    grads = self._backward_single(cache)
                    self._accumulate_grads(acc_grads, grads)
                    total_loss += loss

            # Average grads over all (seed,counter) pairs
            for k in acc_grads.keys():
                acc_grads[k] /= num_pairs
            avg_loss = total_loss / num_pairs

            self.apply_grads(acc_grads, lr)

            if epoch == 1 or epoch % max(1, epochs // 10) == 0:
                print(f"[epoch {epoch:4d}] avg_loss={avg_loss:.4f}")

    # -----------------------
    # Roundtrip for a single (seed, counter)
    # -----------------------
    def roundtrip(self, master_secret: bytes, seed_value, counter_value,
                  noise_std=0.0):
        seq = np.frombuffer(master_secret, dtype=np.uint8).astype(np.int64)
        L = seq.shape[0]

        loss, cache = self._forward_single(seq, seed_value, counter_value,
                                           noise_std=noise_std)

        # Greedy decoding using the attention context
        context = cache["context"]
        a0_dec = context @ self.W_ctx + self.b_ctx
        h_dec_prev = np.tanh(a0_dec)

        BOS = 0
        prev_idx = BOS
        out_indices = []

        for _ in range(L):
            emb_dec = self.W_emb[prev_idx]
            x_cat = np.concatenate([emb_dec, context])
            a_t = x_cat @ self.Wxh_dec + h_dec_prev @ self.Whh_dec + self.bh_dec
            h_t = np.tanh(a_t)
            logits = h_t @ self.W_ho + self.b_o
            probs = softmax(logits)
            idx = int(np.argmax(probs))
            out_indices.append(idx)
            prev_idx = idx
            h_dec_prev = h_t

        recon = np.array(out_indices, dtype=np.int64)
        recon_bytes = recon.astype(np.uint8).tobytes()
        mismatches = np.sum(recon != seq)
        err_rate = mismatches / L

        return loss, recon_bytes, err_rate

    # -----------------------
    # Chain monitor with counter windows & behavior curve
    # -----------------------
    def monitor_chain(self, master_secret: bytes, seed_value,
                      start_counter: int, length: int,
                      window_size: int, noise_std=0.0,
                      err_threshold=0.1):
        """
        Sliding chain behavioral monitor.

        - Only counters in [start_counter, start_counter+window_size-1] are 'legal'.
        - For each counter:
            * run roundtrip
            * compute err_rate
            * flag:
                - OUT_OF_WINDOW if counter not in legal window
                - ANOMALY if inside window but err_rate > threshold
                - OK otherwise
        """
        seq = np.frombuffer(master_secret, dtype=np.uint8).astype(np.int64)
        window_start = start_counter
        window_end = start_counter + window_size - 1

        print(f"[chain] seed={seed_value}, window=[{window_start},{window_end}], "
              f"length={length}, err_threshold={err_threshold}")
        records = []

        for i in range(length):
            counter = start_counter + i
            loss, recon_bytes, err_rate = self.roundtrip(
                master_secret, seed_value, counter, noise_std=noise_std
            )

            inside_window = (window_start <= counter <= window_end)
            ok = inside_window and (err_rate <= err_threshold)

            if not inside_window:
                status = "OUT_OF_WINDOW"
            elif not ok:
                status = "ANOMALY"
            else:
                status = "OK"

            records.append({
                "counter": counter,
                "loss": loss,
                "err_rate": err_rate,
                "inside_window": inside_window,
                "ok": ok,
                "status": status,
            })

            print(f" ctr={counter:3d}  loss={loss:.4f}  err_rate={err_rate:.3f}  "
                  f"inside_window={inside_window}  status={status}")

        return records


# -----------------------
# Demo: training + chain behavior
# -----------------------

def demo():
    master_secret = b"this is the device master secret"
    print("Master (raw):", master_secret)
    print("Master (hex):", master_secret.hex())

    model = RNNAttentionPIM(hidden_dim=32, embed_dim=16,
                            max_seed=8, max_counter=64)

    # Train across a grid of (seed, counter)
    model.train(
        master_secret,
        seed_range=(1, 4),
        counter_range=(1, 8),
        epochs=500,
        lr=5e-1,
        noise_std=0.001,
    )

    # Single roundtrip check
    seed = 3
    counter = 5
    loss, recon_bytes, err_rate = model.roundtrip(master_secret, seed, counter)
    print("\n[roundtrip]")
    print(" original:", master_secret)
    print(" recon   :", recon_bytes)
    print(" equal?  :", master_secret == recon_bytes, " err_rate=", err_rate)

    # Chain behavior with a window
    print("\n[chain monitor]")
    model.monitor_chain(
        master_secret,
        seed_value=seed,
        start_counter=1,
        length=16,
        window_size=8,     # legal window: counters 1..8
        noise_std=0.0,
        err_threshold=0.2,
    )


if __name__ == "__main__":
    demo()
