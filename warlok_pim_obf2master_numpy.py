import numpy as np
import time

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
    grad_logits[target_idx] -= 1.0
    return loss, grad_logits

# -----------------------
# Obfuscation generator: master XOR PRNG(seed,counter)
# -----------------------

class ObfuscatorXOR:
    """
    Simple deterministic obfuscator:
    obf = master XOR PRNG(seed, counter)

    This simulates an obfuscated secret generated on-device per (seed, counter).
    """

    def __init__(self):
        pass

    @staticmethod
    def _prng_stream(length, seed, counter):
        # derive combined seed (simple mix)
        s = (seed * 1000003 + counter * 9176) & 0xFFFFFFFF
        rng = np.random.default_rng(s)
        return rng.integers(0, 256, size=length, dtype=np.uint8)

    def generate(self, master_bytes: bytes, seed: int, counter: int) -> bytes:
        arr = np.frombuffer(master_bytes, dtype=np.uint8)
        mask = self._prng_stream(arr.shape[0], seed, counter)
        obf = np.bitwise_xor(arr, mask)
        return obf.tobytes()

# -----------------------
# One-hot RNN + Attention: obf -> master
# -----------------------

class OneHotRNNAttentionObf2Master:
    """
    One-hot RNN + Attention model that learns:

        (obfuscated_bytes, seed, counter)  --->  master_bytes

    - Input to encoder: [one_hot(obf_byte) || seed_norm || counter_norm]
    - Attention query from (seed,counter)
    - Decoder: [one_hot(previous_master_byte) || context]
    - Output: softmax over 256 bytes (deterministic via argmax)
    """

    def __init__(self, vocab_size=256, hidden_dim=64,
                 max_seed=8, max_counter=64, rng_seed=42):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_seed = max_seed
        self.max_counter = max_counter

        rng = np.random.default_rng(rng_seed)

        # Encoder RNN: input = [vocab_size + 2] (one-hot obf + seed_norm + counter_norm)
        self.enc_input_dim = vocab_size + 2
        self.Wxh_enc = rng.normal(0, 0.1, size=(self.enc_input_dim, hidden_dim))
        self.Whh_enc = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.bh_enc = np.zeros(hidden_dim)

        # Attention: q = [seed_norm, counter_norm] @ W_q -> (hidden_dim,)
        self.W_q = rng.normal(0, 0.1, size=(2, hidden_dim))

        # Decoder RNN: input = [vocab_size + hidden_dim] (prev master one-hot + context)
        self.dec_input_dim = vocab_size + hidden_dim
        self.Wxh_dec = rng.normal(0, 0.1, size=(self.dec_input_dim, hidden_dim))
        self.Whh_dec = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.bh_dec = np.zeros(hidden_dim)

        # Context -> initial decoder hidden
        self.W_ctx = rng.normal(0, 0.1, size=(hidden_dim, hidden_dim))
        self.b_ctx = np.zeros(hidden_dim)

        # Output: hidden -> logits over bytes
        self.W_ho = rng.normal(0, 0.1, size=(hidden_dim, vocab_size))
        self.b_o = np.zeros(vocab_size)

    # -------- helpers --------

    def _norm_sc(self, seed, counter):
        s_norm = seed / float(max(self.max_seed, 1))
        c_norm = counter / float(max(self.max_counter, 1))
        return np.array([s_norm, c_norm], dtype=np.float32)

    # -------- encoder + attention --------

    def _encode(self, obf_seq, sc_vec):
        """
        obf_seq: (L,) int
        sc_vec: (2,)
        returns: enc_h (L,H), enc_x_cat list, enc_h_prev list
        """
        L = obf_seq.shape[0]
        enc_x_cat = []
        enc_h = []
        enc_h_prev = []
        h_prev = np.zeros(self.hidden_dim)

        for t in range(L):
            idx = int(obf_seq[t])
            one_hot = np.zeros(self.vocab_size)
            one_hot[idx] = 1.0

            x_cat = np.concatenate([one_hot, sc_vec])
            a_t = x_cat @ self.Wxh_enc + h_prev @ self.Whh_enc + self.bh_enc
            h_t = np.tanh(a_t)

            enc_x_cat.append(x_cat)
            enc_h.append(h_t)
            enc_h_prev.append(h_prev)

            h_prev = h_t

        return np.array(enc_h), enc_x_cat, enc_h_prev

    def _attention(self, enc_h, sc_vec):
        q = sc_vec @ self.W_q            # (H,)
        e = enc_h @ q                    # (L,)
        e_shift = e - np.max(e)
        alpha = np.exp(e_shift)
        alpha = alpha / np.sum(alpha)    # (L,)
        context = np.sum(alpha[:, None] * enc_h, axis=0)  # (H,)
        return q, alpha, context

    # -------- decoder (teacher forcing) --------

    def _decode_teacher(self, master_seq, context):
        """
        master_seq: (L,) int target master bytes
        context: (H,)
        returns: avg_loss, cache
        """
        L = master_seq.shape[0]
        a0 = context @ self.W_ctx + self.b_ctx
        h_prev = np.tanh(a0)
        h0 = h_prev.copy()

        dec_x_cat = []
        dec_h = []
        dec_h_prev = []
        grad_logits_list = []

        total_loss = 0.0
        BOS = 0

        for t in range(L):
            in_idx = BOS if t == 0 else int(master_seq[t - 1])

            one_hot_prev = np.zeros(self.vocab_size)
            one_hot_prev[in_idx] = 1.0

            x_cat = np.concatenate([one_hot_prev, context])
            a_t = x_cat @ self.Wxh_dec + h_prev @ self.Whh_dec + self.bh_dec
            h_t = np.tanh(a_t)
            logits = h_t @ self.W_ho + self.b_o
            probs = softmax(logits)

            loss_t, grad_logits_t = cross_entropy_loss(probs, int(master_seq[t]))
            total_loss += loss_t

            dec_x_cat.append(x_cat)
            dec_h.append(h_t)
            dec_h_prev.append(h_prev)
            grad_logits_list.append(grad_logits_t)

            h_prev = h_t

        avg_loss = total_loss / L

        cache = {
            "L": L,
            "master_seq": master_seq,
            "context": context,
            "h0_dec": h0,
            "dec_x_cat": dec_x_cat,
            "dec_h": dec_h,
            "dec_h_prev": dec_h_prev,
            "grad_logits_list": grad_logits_list
        }
        return avg_loss, cache

    # -------- full forward (one sample) --------

    def forward_single(self, master_seq, obf_seq, seed, counter):
        sc_vec = self._norm_sc(seed, counter)
        enc_h, enc_x_cat, enc_h_prev = self._encode(obf_seq, sc_vec)
        q, alpha, context = self._attention(enc_h, sc_vec)
        loss, dec_cache = self._decode_teacher(master_seq, context)

        cache = {
            "sc_vec": sc_vec,
            "enc_h": enc_h,
            "enc_x_cat": enc_x_cat,
            "enc_h_prev": enc_h_prev,
            "q": q,
            "alpha": alpha,
            "context": context
        }
        cache.update(dec_cache)
        return loss, cache

    # -------- backward (one sample) --------

    def backward_single(self, cache):
        L = cache["L"]
        master_seq = cache["master_seq"]
        sc_vec = cache["sc_vec"]
        enc_h = cache["enc_h"]
        enc_x_cat = cache["enc_x_cat"]
        enc_h_prev = cache["enc_h_prev"]
        q = cache["q"]
        alpha = cache["alpha"]
        context = cache["context"]
        h0_dec = cache["h0_dec"]
        dec_x_cat = cache["dec_x_cat"]
        dec_h = cache["dec_h"]
        dec_h_prev = cache["dec_h_prev"]
        grad_logits_list = cache["grad_logits_list"]

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

        # --- Decoder ---
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

            # split dx_cat: [vocab_size | hidden_dim]
            d_ctx_t = dx_cat[self.vocab_size:]
            dcontext += d_ctx_t

            dh_next_dec = dh_prev

        # initial decoder hidden
        dh0 = dh_next_dec
        da0 = dh0 * (1.0 - h0_dec ** 2)

        grad_W_ctx += np.outer(context, da0)
        grad_b_ctx += da0
        dcontext += da0 @ self.W_ctx.T

        # --- Attention + Encoder ---
        d_enc_h = np.zeros_like(enc_h)
        d_alpha = np.zeros(L)

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
            # dx_cat_enc[:vocab_size] is gradient wrt obf one-hot, ignored

            dh_next_enc = dh_prev_enc

        grads = {
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
            "b_o": grad_b_o
        }
        return grads

    # -------- grad helpers --------

    def _init_zero_grads(self):
        return {
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
            "b_o": np.zeros_like(self.b_o)
        }

    def _accumulate_grads(self, acc, g):
        for k in acc:
            acc[k] += g[k]

    def apply_grads(self, grads, lr):
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

    # -------- training over (seed, counter) grid --------

    def train(self, master_secret: bytes, obfuscator: ObfuscatorXOR,
              seed_range=(1, 4), counter_range=(1, 16),
              epochs=100, lr=1e-2):
        master_seq = np.frombuffer(master_secret, dtype=np.uint8).astype(np.int64)
        seeds = list(range(seed_range[0], seed_range[1] + 1))
        counters = list(range(counter_range[0], counter_range[1] + 1))
        num_pairs = len(seeds) * len(counters)

        print(f"[train] seq_len={len(master_seq)}, pairs={num_pairs}, epochs={epochs}")

        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            acc_grads = self._init_zero_grads()

            for s in seeds:
                for c in counters:
                    obf_bytes = obfuscator.generate(master_secret, s, c)
                    obf_seq = np.frombuffer(obf_bytes, dtype=np.uint8).astype(np.int64)

                    loss, cache = self.forward_single(master_seq, obf_seq, s, c)
                    grads = self.backward_single(cache)

                    self._accumulate_grads(acc_grads, grads)
                    total_loss += loss

            for k in acc_grads:
                acc_grads[k] /= num_pairs
            avg_loss = total_loss / num_pairs

            self.apply_grads(acc_grads, lr)

            if epoch == 1 or epoch % max(1, epochs // 10) == 0:
                print(f"[epoch {epoch:4d}] avg_loss={avg_loss:.4f}")

    # -------- inference: obf -> master --------

    def reconstruct_from_obf(self, obf_bytes: bytes, seed: int, counter: int, seq_len: int):
        obf_seq = np.frombuffer(obf_bytes, dtype=np.uint8).astype(np.int64)
        assert obf_seq.shape[0] == seq_len

        sc_vec = self._norm_sc(seed, counter)
        enc_h, enc_x_cat, enc_h_prev = self._encode(obf_seq, sc_vec)
        q, alpha, context = self._attention(enc_h, sc_vec)

        # Greedy decode (no teacher forcing)
        a0 = context @ self.W_ctx + self.b_ctx
        h_prev = np.tanh(a0)

        BOS = 0
        prev_idx = BOS
        out_indices = []

        for _ in range(seq_len):
            one_hot_prev = np.zeros(self.vocab_size)
            one_hot_prev[prev_idx] = 1.0

            x_cat = np.concatenate([one_hot_prev, context])
            a_t = x_cat @ self.Wxh_dec + h_prev @ self.Whh_dec + self.bh_dec
            h_t = np.tanh(a_t)
            logits = h_t @ self.W_ho + self.b_o
            probs = softmax(logits)
            idx = int(np.argmax(probs))

            out_indices.append(idx)
            prev_idx = idx
            h_prev = h_t

        recon = np.array(out_indices, dtype=np.int64)
        return recon.astype(np.uint8).tobytes()

    # -------- save / load --------

    def save(self, path: str):
        np.savez(path,
                 Wxh_enc=self.Wxh_enc,
                 Whh_enc=self.Whh_enc,
                 bh_enc=self.bh_enc,
                 W_q=self.W_q,
                 Wxh_dec=self.Wxh_dec,
                 Whh_dec=self.Whh_dec,
                 bh_dec=self.bh_dec,
                 W_ctx=self.W_ctx,
                 b_ctx=self.b_ctx,
                 W_ho=self.W_ho,
                 b_o=self.b_o,
                 vocab_size=self.vocab_size,
                 hidden_dim=self.hidden_dim,
                 max_seed=self.max_seed,
                 max_counter=self.max_counter)

    @classmethod
    def load(cls, path: str):
        data = np.load(path, allow_pickle=True)
        model = cls(vocab_size=int(data["vocab_size"]),
                    hidden_dim=int(data["hidden_dim"]),
                    max_seed=int(data["max_seed"]),
                    max_counter=int(data["max_counter"]))
        model.Wxh_enc = data["Wxh_enc"]
        model.Whh_enc = data["Whh_enc"]
        model.bh_enc = data["bh_enc"]
        model.W_q = data["W_q"]
        model.Wxh_dec = data["Wxh_dec"]
        model.Whh_dec = data["Whh_dec"]
        model.bh_dec = data["bh_dec"]
        model.W_ctx = data["W_ctx"]
        model.b_ctx = data["b_ctx"]
        model.W_ho = data["W_ho"]
        model.b_o = data["b_o"]
        return model

    # -------- chain monitoring: seed, counter, window behavior --------

    def monitor_chain(self, master_secret: bytes, obfuscator: ObfuscatorXOR,
                      seed: int, start_counter: int, length: int,
                      window_size: int, err_threshold: float = 0.1):
        """
        Trace behavioral pattern across counters:

        - legal counter window: [start_counter, start_counter+window_size-1]
        - for each counter:
            * generate obf
            * reconstruct master
            * compute err_rate
            * classify status: OK / ANOMALY / OUT_OF_WINDOW
        """
        master_seq = np.frombuffer(master_secret, dtype=np.uint8).astype(np.int64)
        seq_len = master_seq.shape[0]
        window_start = start_counter
        window_end = start_counter + window_size - 1

        print(f"[chain] seed={seed}, window=[{window_start},{window_end}], length={length}")
        records = []

        for i in range(length):
            counter = start_counter + i
            obf_bytes = obfuscator.generate(master_secret, seed, counter)
            recon_bytes = self.reconstruct_from_obf(obf_bytes, seed, counter, seq_len)
            recon_seq = np.frombuffer(recon_bytes, dtype=np.uint8).astype(np.int64)

            mismatches = np.sum(recon_seq != master_seq)
            err_rate = mismatches / seq_len

            inside_window = window_start <= counter <= window_end
            ok = inside_window and (err_rate <= err_threshold)

            if not inside_window:
                status = "OUT_OF_WINDOW"
            elif not ok:
                status = "ANOMALY"
            else:
                status = "OK"

            records.append({
                "counter": counter,
                "err_rate": err_rate,
                "inside_window": inside_window,
                "status": status
            })

            print(f" ctr={counter:3d}  err_rate={err_rate:.3f}  "
                  f"inside_window={inside_window}  status={status}")

        return records

# -----------------------
# Demo: training, save/load timing, chain behavior
# -----------------------

def demo():
    master_secret = b"this is the device master secret"
    print("Master (raw):", master_secret)
    print("Master (hex):", master_secret.hex())

    obf = ObfuscatorXOR()
    model = OneHotRNNAttentionObf2Master(hidden_dim=64, max_seed=8, max_counter=64)

    # 1) Train mapping obf -> master
    model.train(
        master_secret,
        obfuscator=obf,
        seed_range=(1, 4),
        counter_range=(1, 5),
        epochs=1000,      # increase for better reconstruction
        lr=5e-1,
    )

    # 2) Single roundtrip test
    seed = 3
    counter = 5
    obf_bytes = obf.generate(master_secret, seed, counter)
    recon_bytes = model.reconstruct_from_obf(obf_bytes, seed, counter, len(master_secret))

    print("\n[roundtrip]")
    print("master:", master_secret)
    print("recon :", recon_bytes)
    print("equal?:", master_secret == recon_bytes)

    # 3) Save model, load it, and time load+inference from obf -> master
    model.save("pim_model.npz")

    t0 = time.perf_counter()
    loaded = OneHotRNNAttentionObf2Master.load("pim_model.npz")
    t1 = time.perf_counter()
    t_load_ms = (t1 - t0) * 1000.0

    t2 = time.perf_counter()
    recon2 = loaded.reconstruct_from_obf(obf_bytes, seed, counter, len(master_secret))
    t3 = time.perf_counter()
    t_inf_ms = (t3 - t2) * 1000.0

    print(f"\n[perf] load_ms={t_load_ms:.3f}  infer_ms={t_inf_ms:.3f}")
    print("loaded recon equal?:", recon2 == master_secret)

    # 4) Chain monitoring: behavior across counters, within a window
    print("\n[chain monitor]")
    model.monitor_chain(
        master_secret,
        obfuscator=obf,
        seed=seed,
        start_counter=1,
        length=16,
        window_size=8,   # legal counters: 1..8
        err_threshold=0.0,
    )

if __name__ == "__main__":
    demo()
