import numpy as np

# ============================================================
# CONFIG
# ============================================================

MS_DIM      = 4           # master secret length
SEQ_LEN     = 32          # obfuscated secret token length
VOCAB_SIZE  = 16          # using 16 printable/nibble tokens
HIDDEN_DIM  = 48          # RNN hidden size for reconstruction
ATTN_DIM    = 24          # attention size
LR          = 0.01
EPOCHS      = 150
N_SAMPLES   = 100

np.random.seed(42)

# Build a tiny deterministic vocab (0..15 chars)
VOCAB = [chr(65+i) for i in range(VOCAB_SIZE)]    # 'A','B',... 'P'
char_idx = {c:i for i,c in enumerate(VOCAB)}

# ============================================================
# 1. DETERMINISTIC OBFUSCATION (MS → 32 TOKENS)
# ============================================================

def encode_ms_to_tokens(ms_int):
    """
    Convert MS of 4 bytes into 8 nibble tokens + checksum padding.
    Deterministic, stable, easy to invert by neural model.
    """
    tokens = []
    for v in ms_int:
        hi = (v >> 4) & 0xF
        lo = v & 0xF
        tokens.append(VOCAB[hi])
        tokens.append(VOCAB[lo])

    checksum = ms_int.sum() % 16
    pad = VOCAB[int(checksum)]

    while len(tokens) < SEQ_LEN:
        tokens.append(pad)

    return np.array([char_idx[t] for t in tokens], dtype=np.int32)


def generate_master_and_secret():
    ms_int  = np.random.randint(0, 256, size=MS_DIM)
    ms_cont = (ms_int / 255.0) * 1.8 - 0.9        # map to [-0.9, 0.9]
    secret  = encode_ms_to_tokens(ms_int)
    return ms_cont.astype(np.float32), secret


def one_hot(idxs):
    X = np.zeros((len(idxs), VOCAB_SIZE), dtype=np.float32)
    X[np.arange(len(idxs)), idxs] = 1.0
    return X


# ============================================================
# 2. RNN + ATTENTION SECRET RECONSTRUCTION MODEL
# ============================================================

def xavier(a,b):
    limit = np.sqrt(6/(a+b))
    return np.random.uniform(-limit,limit,(a,b)).astype(np.float32)

W_xh = xavier(VOCAB_SIZE, HIDDEN_DIM)
W_hh = xavier(HIDDEN_DIM, HIDDEN_DIM)
b_h  = np.zeros(HIDDEN_DIM, dtype=np.float32)

W_att = xavier(HIDDEN_DIM, ATTN_DIM)
v_att = np.random.uniform(-0.1,0.1,(ATTN_DIM,)).astype(np.float32)

W_out = xavier(HIDDEN_DIM, MS_DIM)
b_out = np.zeros(MS_DIM, dtype=np.float32)

def tanh(x): return np.tanh(x)
def dtanh(x): t=np.tanh(x); return 1-t*t

def fwd_reconstruct(X):
    """
    RNN + attention → MS
    X: (32, VOCAB_SIZE)
    """
    T = X.shape[0]
    h = np.zeros(HIDDEN_DIM, dtype=np.float32)
    H_list, pre_list = [], []

    # RNN
    for t in range(T):
        pre = X[t] @ W_xh + h @ W_hh + b_h
        h   = tanh(pre)
        pre_list.append(pre)
        H_list.append(h)

    H = np.stack(H_list, axis=0)

    # Attention
    scores = np.zeros(T)
    U, preU = [], []
    for t in range(T):
        preu = H[t] @ W_att
        u    = tanh(preu)
        preU.append(preu)
        U.append(u)
        scores[t] = np.dot(u, v_att)

    scores -= np.max(scores)
    alphas = np.exp(scores)/np.sum(np.exp(scores))

    context = np.sum(alphas[:,None] * H, axis=0)

    y = context @ W_out + b_out
    cache = (X, pre_list, H, alphas, preU, U, context)
    return y, cache


def bwd_reconstruct(cache, dy):
    global W_xh, W_hh, b_h, W_att, v_att, W_out, b_out

    (X, pre_list, H, alphas, preU, U, context) = cache
    T = X.shape[0]

    # output
    dW_out = np.outer(context, dy)
    db_out = dy
    dcontext = W_out @ dy

    # attention
    dH = np.zeros_like(H)
    dalpha = np.zeros_like(alphas)

    for t in range(T):
        dH[t] += alphas[t] * dcontext
        dalpha[t] += np.dot(dcontext, H[t])

    # softmax backprop
    dotv = np.dot(dalpha, alphas)
    ds = alphas * (dalpha - dotv)

    dW_att = np.zeros_like(W_att)
    dv_att = np.zeros_like(v_att)

    for t in range(T):
        du = v_att * ds[t]
        dpreu = dtanh(preU[t]) * du
        dW_att += np.outer(H[t], dpreu)
        dH[t]  += W_att @ dpreu
        dv_att += U[t] * ds[t]

    # RNN BPTT
    dW_xh = np.zeros_like(W_xh)
    dW_hh = np.zeros_like(W_hh)
    db_hh = np.zeros_like(b_h)
    dh_next = np.zeros(HIDDEN_DIM)

    for t in reversed(range(T)):
        dh = dH[t] + dh_next
        dpre = dtanh(pre_list[t]) * dh
        dW_xh += np.outer(X[t], dpre)
        if t > 0:
            dW_hh += np.outer(H[t-1], dpre)
        db_hh += dpre
        dh_next = W_hh @ dpre

    # SGD
    W_xh -= LR * dW_xh
    W_hh -= LR * dW_hh
    b_h  -= LR * db_hh
    W_att -= LR * dW_att
    v_att -= LR * dv_att
    W_out -= LR * dW_out
    b_out -= LR * db_out


# ============================================================
# 3. BEHAVIORAL SIGNATURE MODEL (seed, counter, window)
#    A tiny RNN + attention to bind behavior to params
# ============================================================

def behavior_signature(seed, counter, window):
    """
    Convert parameters into a short one-hot "behavior sequence"
    then compute attention signature.
    """
    vec = np.array([seed % 16, counter % 16, window % 16], dtype=np.int32)
    X = np.zeros((3, VOCAB_SIZE))
    for i,v in enumerate(vec): X[i,v]=1.0

    # small RNN
    h = np.zeros(8)
    H = []
    Wxh = np.random.randn(VOCAB_SIZE,8)*0.05
    Whh = np.random.randn(8,8)*0.05
    b   = np.zeros(8)

    for t in range(3):
        h = np.tanh(X[t]@Wxh + h@Whh + b)
        H.append(h)

    H = np.stack(H)
    att = np.random.randn(8)
    scores = H @ att
    scores -= scores.max()
    a = np.exp(scores)/np.sum(np.exp(scores))
    return np.sum(a[:,None]*H,axis=0)


# ============================================================
# 4. TRAIN RECONSTRUCTION MODEL
# ============================================================

masters, secrets = [], []
for _ in range(N_SAMPLES):
    ms, sec = generate_master_and_secret()
    masters.append(ms)
    secrets.append(one_hot(sec))

masters = np.stack(masters,0)

for ep in range(EPOCHS):
    ix = np.random.permutation(N_SAMPLES)
    tot = 0
    for i in ix:
        X = secrets[i]
        ms = masters[i]
        y, cache = fwd_reconstruct(X)
        loss = 0.5*np.mean((y-ms)**2)
        tot += loss
        dy = (y-ms)/MS_DIM
        bwd_reconstruct(cache, dy)

    if ep%20==0: print(f"[Epoch {ep}] loss={tot/N_SAMPLES:.5f}")

print("\nTraining finished.\n")


# ============================================================
# 5. FULL PIM VALIDATION TESTS
# ============================================================

def check_pim(seed, counter, window, tamper=False, shift_seed=None, shift_counter=None, shift_window=None):
    print("\n=== TEST CASE ===")

    ms, sec = generate_master_and_secret()
    X = one_hot(sec)

    # Expected behavioral signature
    sig_ref = behavior_signature(seed, counter, window)

    # Reconstruction
    y,_ = fwd_reconstruct(X)
    recon_err = np.linalg.norm(y-ms)

    # tamper OS?
    if tamper:
        X = X.copy()
        X[0] = np.roll(X[0],1)

    # shift params?
    s = shift_seed if shift_seed is not None else seed
    c = shift_counter if shift_counter is not None else counter
    w = shift_window if shift_window is not None else window

    sig_new = behavior_signature(s,c,w)
    sig_err = np.linalg.norm(sig_new - sig_ref)

    print("Reconstruction L2:", recon_err)
    print("Behavior L2:", sig_err)
    print("TAMPER:", tamper, " SEED SHIFT:", shift_seed,
          " COUNTER SHIFT:", shift_counter, " WINDOW SHIFT:", shift_window)


# Honest
check_pim(seed=10, counter=5, window=3)

# Wrong seed
check_pim(seed=10, counter=5, window=3, shift_seed=14)

# Wrong counter
check_pim(seed=10, counter=5, window=3, shift_counter=9)

# Wrong window
check_pim(seed=10, counter=5, window=3, shift_window=6)

# OS tampering
check_pim(seed=10, counter=5, window=3, tamper=True)
