# # import numpy as np
# #
# # # ---------------------------
# # # 1. Text + vocabulary setup
# # # ---------------------------
# #
# # text = """
# # Warl0k nano-models can generate deterministic signatures over long sequences of text.
# # This example uses a tiny RNN + attention written in pure NumPy, with one-hot inputs.
# # """
# #
# # # Build character-level vocabulary
# # chars = sorted(list(set(text)))
# # vocab_size = len(chars)
# # char_to_idx = {ch: i for i, ch in enumerate(chars)}
# # idx_to_char = {i: ch for ch, i in char_to_idx.items()}
# #
# # # Encode text as indices
# # indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int32)
# # seq_len = len(indices)
# #
# # def one_hot(indices, vocab_size):
# #     """Convert sequence of indices to one-hot vectors (seq_len, vocab_size)."""
# #     oh = np.zeros((len(indices), vocab_size), dtype=np.float32)
# #     oh[np.arange(len(indices)), indices] = 1.0
# #     return oh
# #
# # X = one_hot(indices, vocab_size)  # (seq_len, vocab_size)
# #
# # # ---------------------------
# # # 2. Tiny RNN + Attention
# # # ---------------------------
# #
# # np.random.seed(42)  # deterministic
# #
# # hidden_size = 32
# # attn_size = 16
# #
# # # RNN parameters
# # W_xh = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.1
# # W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.1
# # b_h  = np.zeros((hidden_size,), dtype=np.float32)
# #
# # # Attention parameters
# # W_att = np.random.randn(hidden_size, attn_size).astype(np.float32) * 0.1
# # v_att = np.random.randn(attn_size).astype(np.float32)
# #
# # # Output layer (hidden -> vocab logits, optional for reconstruction)
# # W_hy = np.random.randn(hidden_size, vocab_size).astype(np.float32) * 0.1
# # b_y  = np.zeros((vocab_size,), dtype=np.float32)
# #
# #
# # def rnn_forward(X):
# #     """
# #     Simple Elman RNN:
# #     X: (seq_len, vocab_size) one-hot
# #     Returns:
# #       H: (seq_len, hidden_size) all hidden states
# #     """
# #     seq_len, _ = X.shape
# #     H = np.zeros((seq_len, hidden_size), dtype=np.float32)
# #     h_t = np.zeros((hidden_size,), dtype=np.float32)
# #
# #     for t in range(seq_len):
# #         x_t = X[t]  # (vocab_size,)
# #         h_t = np.tanh(x_t @ W_xh + h_t @ W_hh + b_h)
# #         H[t] = h_t
# #     return H
# #
# #
# # def attention(H):
# #     """
# #     Simple additive attention over time:
# #     H: (seq_len, hidden_size)
# #     Returns:
# #       context: (hidden_size,) - weighted sum of hidden states
# #       alphas:  (seq_len,) - attention weights
# #     """
# #     # Score for each timestep: e_t = v^T tanh(W_att h_t)
# #     # H @ W_att -> (seq_len, attn_size)
# #     scores = np.tanh(H @ W_att) @ v_att  # (seq_len,)
# #
# #     # Softmax over time
# #     max_score = np.max(scores)  # for numerical stability
# #     exp_scores = np.exp(scores - max_score)
# #     alphas = exp_scores / np.sum(exp_scores)
# #
# #     # Weighted sum of hidden states
# #     context = (alphas[:, None] * H).sum(axis=0)  # (hidden_size,)
# #     return context, alphas
# #
# #
# # def forward_signature(X):
# #     """
# #     Full forward: RNN -> attention -> signature vector.
# #     Deterministic given fixed weights + seed.
# #     """
# #     H = rnn_forward(X)
# #     context, alphas = attention(H)
# #     return context, alphas, H
# #
# #
# # signature, attn_weights, H = forward_signature(X)
# #
# # # ---------------------------
# # # 3. (Optional) Greedy reconstruction demo
# # #    (NOT trained, just shows how you'd decode)
# # # ---------------------------
# #
# # def softmax(logits):
# #     logits = logits - np.max(logits)
# #     exp = np.exp(logits)
# #     return exp / np.sum(exp)
# #
# # def greedy_decode(initial_h, max_len):
# #     """
# #     Toy greedy decoder:
# #       - start from an initial hidden state
# #       - at each step, compute logits and pick argmax char
# #       - feed that char back in
# #     NOTE: with random untrained weights this is nonsense, but it
# #     shows the reconstruction path.
# #     """
# #     h_t = initial_h.copy()
# #     decoded_indices = []
# #     prev_char_idx = indices[0]  # start from first char of text
# #     for _ in range(max_len):
# #         # Build one-hot for previous char
# #         x_t = np.zeros((vocab_size,), dtype=np.float32)
# #         x_t[prev_char_idx] = 1.0
# #
# #         # One RNN step
# #         h_t = np.tanh(x_t @ W_xh + h_t @ W_hh + b_h)
# #
# #         # Predict next char
# #         logits = h_t @ W_hy + b_y
# #         probs = softmax(logits)
# #         next_idx = int(np.argmax(probs))
# #
# #         decoded_indices.append(next_idx)
# #         prev_char_idx = next_idx
# #
# #     return ''.join(idx_to_char[i] for i in decoded_indices)
# #
# # reconstructed_demo = greedy_decode(signature, max_len=min(80, seq_len))
# #
# # # ---------------------------
# # # 4. Size / footprint metrics
# # # ---------------------------
# #
# # # Raw text size (UTF-8)
# # raw_bytes = len(text.encode("utf-8"))
# #
# # # One-hot representation size:
# # # seq_len * vocab_size floats (assuming float32 -> 4 bytes)
# # one_hot_floats = seq_len * vocab_size
# # one_hot_bytes = one_hot_floats * 4  # float32
# #
# # # Model parameter counting
# # def count_params():
# #     params = 0
# #     params += W_xh.size
# #     params += W_hh.size
# #     params += b_h.size
# #     params += W_att.size
# #     params += v_att.size
# #     params += W_hy.size
# #     params += b_y.size
# #     return params
# #
# # total_params = count_params()
# # model_bytes = total_params * 4  # float32
# #
# # # Vocab size & approximate mapping size:
# # # simplistic: each char entry ~1 byte char + overhead; we just show count.
# # vocab_bytes_min = vocab_size  # lower bound (1 byte / char in ASCII range)
# #
# # print("=== BASIC STATS ===")
# # print(f"Raw text length (chars):        {len(text)}")
# # print(f"Raw text size (UTF-8 bytes):    {raw_bytes}")
# # print()
# # print("=== VOCAB / ONE-HOT ===")
# # print(f"Vocab size (unique chars):      {vocab_size}")
# # print(f"Sequence length (tokens):       {seq_len}")
# # print(f"One-hot floats:                 {one_hot_floats}")
# # print(f"One-hot size (float32 bytes):   {one_hot_bytes}")
# # print(f"Approx. vocab table min bytes:  {vocab_bytes_min}")
# # print()
# # print("=== MODEL WEIGHTS ===")
# # print(f"Total parameters:               {total_params}")
# # print(f"Model weights size (float32):   {model_bytes} bytes")
# # print()
# # print("=== SIGNATURE ===")
# # print(f"Signature vector shape:         {signature.shape}")
# # print(f"First 8 values of signature:    {np.round(signature[:8], 4)}")
# # print()
# # print("=== RECONSTRUCTION DEMO (UNTRAINED) ===")
# # print(reconstructed_demo)
# # print(chars)
# # print(indices)
# # ---------------------------------------------------------------------------
#
# import numpy as np
#
# # ---------------------------
# # 1. Text and vocabulary
# # ---------------------------
#
# text = """
# Warl0k nano-models can generate deterministic signatures over long sequences of text.
# This example uses a tiny RNN with an attention layer written in pure NumPy, with one-hot inputs.
# """
#
# # Build character-level vocabulary
# chars = sorted(list(set(text)))
# vocab_size = len(chars)
# char_to_idx = {ch: i for i, ch in enumerate(chars)}
# idx_to_char = {i: ch for ch, i in char_to_idx.items()}
#
# # Encode text as indices
# indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int32)
# seq_len = len(indices)
#
# # One-hot encoding (seq_len, vocab_size)
# X = np.zeros((seq_len, vocab_size), dtype=np.float32)
# X[np.arange(seq_len), indices] = 1.0
#
# # ---------------------------
# # 2. RNN + Attention weights
# # ---------------------------
#
# np.random.seed(42)  # deterministic
#
# hidden_size = vocab_size  # for exact copy we set hidden_size = vocab_size
#
# # RNN weights
# # We design a *linear* RNN (no nonlinearity) that simply copies the input one-hot:
# #   h_t = x_t @ W_xh + h_{t-1} @ W_hh + b_h
# # With W_xh = I, W_hh = 0, b_h = 0 -> h_t = x_t
# W_xh = np.eye(vocab_size, dtype=np.float32)                # (vocab_size, hidden_size)
# W_hh = np.zeros((hidden_size, hidden_size), dtype=np.float32)
# b_h  = np.zeros((hidden_size,), dtype=np.float32)
#
# # Attention weights – used only to build a signature over the entire sequence
# # Simple vector attention: score_t = v_att^T h_t
# v_att = np.random.randn(hidden_size).astype(np.float32)    # (hidden_size,)
#
# # Output layer: hidden -> vocab
# # For exact reconstruction, we again use identity: y_t = h_t @ I = x_t
# W_hy = np.eye(hidden_size, vocab_size, dtype=np.float32)   # (hidden_size, vocab_size)
# b_y  = np.zeros((vocab_size,), dtype=np.float32)
#
# # ---------------------------
# # 3. Forward pass: RNN + Attention
# # ---------------------------
#
# def rnn_forward(X):
#     """
#     Run the linear RNN over the sequence.
#     X: (seq_len, vocab_size) one-hot
#     Returns:
#       H: (seq_len, hidden_size) hidden states
#     """
#     seq_len, vocab = X.shape
#     H = np.zeros((seq_len, hidden_size), dtype=np.float32)
#     h_t = np.zeros((hidden_size,), dtype=np.float32)
#
#     for t in range(seq_len):
#         x_t = X[t]  # (vocab_size,)
#         # Linear RNN step (no activation):
#         h_t = x_t @ W_xh + h_t @ W_hh + b_h
#         H[t] = h_t
#     return H
#
#
# def attention_signature(H):
#     """
#     Compute a global attention-based signature over time.
#     H: (seq_len, hidden_size)
#     Returns:
#       context: (hidden_size,) global signature
#       alphas:  (seq_len,) attention weights
#     """
#     # scores_t = v_att^T h_t
#     scores = H @ v_att  # (seq_len,)
#     # softmax over time
#     scores = scores - np.max(scores)
#     exp_scores = np.exp(scores)
#     alphas = exp_scores / np.sum(exp_scores)
#     # weighted sum of hidden states
#     context = (alphas[:, None] * H).sum(axis=0)  # (hidden_size,)
#     return context, alphas
#
#
# def reconstruct_text(H):
#     """
#     Use the RNN hidden states and output layer to reconstruct the text exactly.
#     H: (seq_len, hidden_size)
#     Returns:
#       reconstructed string
#     """
#     seq_len, hidden_size = H.shape
#     out_indices = []
#
#     for t in range(seq_len):
#         h_t = H[t]
#         logits = h_t @ W_hy + b_y  # (vocab_size,)
#         idx = int(np.argmax(logits))
#         out_indices.append(idx)
#
#     return ''.join(idx_to_char[i] for i in out_indices)
#
#
# # Run the model
# H = rnn_forward(X)
# signature, alphas = attention_signature(H)
# reconstructed = reconstruct_text(H)
#
# # ---------------------------
# # 4. Size / footprint metrics
# # ---------------------------
#
# # Raw text size (UTF-8)
# raw_bytes = len(text.encode("utf-8"))
#
# # One-hot storage size (if you stored X explicitly)
# one_hot_bytes = X.size * 4  # float32 => 4 bytes
#
# # Model parameter counting
# def count_params():
#     params = 0
#     params += W_xh.size
#     params += W_hh.size
#     params += b_h.size
#     params += v_att.size
#     params += W_hy.size
#     params += b_y.size
#     return params
#
# total_params = count_params()
# model_bytes = total_params * 4  # float32
#
# # ---------------------------
# # 5. Print results
# # ---------------------------
#
# print("=== ORIGINAL TEXT ===")
# print(text)
#
# print("\n=== RECONSTRUCTED TEXT ===")
# print(reconstructed)
#
# print("\nExact match:", text == reconstructed)
#
# print("\n=== VOCAB / SEQUENCE ===")
# print(f"Vocab size (unique chars):        {vocab_size}")
# print(f"Sequence length (chars):          {seq_len}")
# print(f"Sequence length of chars in (bytes UTF-8): {raw_bytes}")
#
# print("\n=== SIZE METRICS ===")
# print(f"Raw text size (UTF-8 bytes):      {raw_bytes}")
# print(f"One-hot matrix size (bytes):      {one_hot_bytes}")
# print(f"Total model parameters:           {total_params}")
# print(f"Model weights size (float32):     {model_bytes} bytes")
#
# print("\n=== SIGNATURE (ATTENTION CONTEXT) ===")
# print("Signature shape:", signature.shape)
# print("First 8 signature values:", np.round(signature[:8], 4))
#
# # ---------------------------------------------------------------------------
import numpy as np

# -----------------------------------------------------
# 1. INPUT TEXT (can be long)
# -----------------------------------------------------

text = """
Warl0k nano-models do not need to reconstruct the text.
They only need to generate a deterministic compressed signature
that proves the sequence is authentic, consistent, and verified.
This matches the WARL0K principle: "Proof without storing secrets."
"""

# Raw size
raw_bytes = len(text.encode("utf-8"))

# -----------------------------------------------------
# 2. VOCAB & ENCODING
# -----------------------------------------------------

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}

indices = np.array([char_to_idx[ch] for ch in text], dtype=np.int32)
seq_len = len(indices)

# One-hot matrix (not stored — only streamed)
X = np.zeros((seq_len, vocab_size), dtype=np.float32)
X[np.arange(seq_len), indices] = 1.0

# -----------------------------------------------------
# 3. TINY RNN + ATTENTION (WARL0K style)
# -----------------------------------------------------

np.random.seed(42)

hidden_size = 8      # <<< TINY state  === Warl0k model footprint
attn_size   = 8

# RNN weights
W_xh = np.random.randn(vocab_size, hidden_size).astype(np.float32) * 0.05
W_hh = np.random.randn(hidden_size, hidden_size).astype(np.float32) * 0.05
b_h  = np.zeros((hidden_size,), dtype=np.float32)

# Attention weights
W_att = np.random.randn(hidden_size, attn_size).astype(np.float32) * 0.05
v_att = np.random.randn(attn_size).astype(np.float32) * 0.05

# No output layer — no reconstruction allowed in WARL0K
# Signature only


def rnn_forward(X):
    seq_len, _ = X.shape
    H = np.zeros((seq_len, hidden_size), dtype=np.float32)
    h = np.zeros((hidden_size,), dtype=np.float32)

    for t in range(seq_len):
        h = np.tanh(X[t] @ W_xh + h @ W_hh + b_h)
        H[t] = h

    return H


def attention(H):
    scores = np.tanh(H @ W_att) @ v_att
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    alphas = exp_scores / np.sum(exp_scores)
    signature = (alphas[:, None] * H).sum(axis=0)
    return signature, alphas


# -----------------------------------------------------
# 4. RUN SIGNATURE GENERATION
# -----------------------------------------------------

H = rnn_forward(X)
signature, alphas = attention(H)

# -----------------------------------------------------
# 5. FOOTPRINT METRICS
# -----------------------------------------------------

def count_params():
    return W_xh.size + W_hh.size + b_h.size + W_att.size + v_att.size

model_params = count_params()
model_bytes  = model_params * 4  # float32

signature_bytes = signature.size * 4

print("\n=== RAW TEXT ===")
print(f"Raw text size (bytes):        {raw_bytes}")

print("\n=== MODEL FOOTPRINT ===")
print(f"Model parameters:              {model_params}")
print(f"Model weight size (bytes):     {model_bytes}")

print("\n=== SIGNATURE ===")
print(f"Signature shape:               {signature.shape}")
print(f"Signature size (bytes):        {signature_bytes}")
print("Signature (first 8 values):    ", np.round(signature[:8], 6))

total_proof_bytes = model_bytes + signature_bytes

print("\n=== COMPARISON ===")
print(f"Total WARL0K proof size:       {total_proof_bytes} bytes")
print(f"Compression ratio:             {raw_bytes / total_proof_bytes:.4f}")
print(f"Reduction factor:              {total_proof_bytes / raw_bytes:.4f}")
