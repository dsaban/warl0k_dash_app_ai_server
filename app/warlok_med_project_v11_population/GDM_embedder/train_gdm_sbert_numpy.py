#!/usr/bin/env python3
"""
GDM NumPy-only SBERT-lite (slim but deep): 2 gated layers + attention pooling
- Loads 4 local medical docs (GDM related)
- Builds word vocab
- Generates synthetic NLI-like pairs: entailment / neutral / contradiction
- Trains NumPy-only (manual backprop) for similarity + NLI head
- Saves a package for fast inference:
    gdm_sbert_numpy_weights.npz
    gdm_sbert_numpy_vocab.txt
    gdm_sbert_numpy_config.npz

Run:
  python3 train_gdm_sbert_numpy.py

Outputs:
  gdm_sbert_numpy_weights.npz
  gdm_sbert_numpy_vocab.txt
  gdm_sbert_numpy_config.npz
"""
import os, re, math, random, time
import numpy as np

# =========================
# Config
# =========================
SEED = 7
random.seed(SEED)
np.random.seed(SEED)

DOC_PATHS = [
    "/mnt/data/Management_of_Diabetes_in_Pregnancy_A_Review_of_Clinical_Guidelines_Practices.txt",
    "/mnt/data/A Comprehensive_Review_of_Gestational_Diabetes_Mellitus.txt",
    "/mnt/data/Diabetes_mellitus_and_pregnancy.txt",
    "/mnt/data/Gestational_diabetes_update_screening_diagnosis_and_maternal_management.txt",
]

MAX_LEN = 56
VOCAB_MAX = 25000
MIN_FREQ = 2

D_MODEL  = 96
D_HID    = 96
OUT_DIM  = 96
N_LAYERS = 2

BATCH = 32
EPOCHS = 6
LR = 2e-3
WEIGHT_DECAY = 1e-4
LAM_NLI = 0.6
HARD_NEG_RATE = 0.25

PRINT_EVERY_BATCHES = 40
VAL_MAX = 800
CAND_MAX = 400

LABELS = {0: "entail", 1: "neutral", 2: "contradict"}

# =========================
# Logging helpers
# =========================
def now():
    return time.strftime("%H:%M:%S")

def log(msg):
    print(f"[{now()}] {msg}", flush=True)

def fmt_sec(s):
    if s < 60: return f"{s:.1f}s"
    return f"{s/60:.1f}m"

# =========================
# Text + Vocab
# =========================
def normalize_text(t: str) -> str:
    t = t.replace("\r", " ")
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def sent_split(text: str):
    text = normalize_text(text)
    parts = re.split(r"(?<!\d)[\.\?\!;]\s+", text)
    sents = []
    for p in parts:
        p = p.strip()
        if len(p) < 30:
            continue
        if p.lower().startswith(("table", "figure", "open in a new tab", "author", "editors")):
            continue
        sents.append(p)
    return sents

def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9%/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

class Vocab:
    def __init__(self, sentences_tokens, max_vocab=VOCAB_MAX, min_freq=MIN_FREQ):
        freq = {}
        for toks in sentences_tokens:
            for w in toks:
                freq[w] = freq.get(w, 0) + 1

        items = [(w,c) for w,c in freq.items() if c >= min_freq]
        items.sort(key=lambda x: (-x[1], x[0]))
        items = items[:max_vocab-2]

        self.stoi = {"<PAD>": 0, "<UNK>": 1}
        for i, (w,_) in enumerate(items, start=2):
            self.stoi[w] = i
        self.itos = {i:w for w,i in self.stoi.items()}

    def encode(self, toks, max_len=MAX_LEN):
        ids = [self.stoi.get(w, 1) for w in toks[:max_len]]
        if len(ids) < max_len:
            ids += [0]*(max_len-len(ids))
        return np.array(ids, dtype=np.int32)

# =========================
# Synthetic pair generation (GDM-specific)
# =========================
FLIPS = [
    ("increases", "decreases"),
    ("increased", "reduced"),
    ("higher", "lower"),
    ("elevated", "reduced"),
    ("more likely", "less likely"),
    ("associated with", "not associated with"),
    ("recommended", "not recommended"),
    ("should", "should not"),
    ("must", "must not"),
    ("indicated", "contraindicated"),
    ("mainstay", "not used"),
]

CANON_EQUIV = [
    ("gdm", "gestational diabetes"),
    ("ogtt", "oral glucose tolerance test"),
    ("hba1c", "glycated haemoglobin"),
    ("c-section", "cesarean section"),
    ("macrosomic", "large for gestational age"),
]

ENTAIL_TEMPL = [
    ("is associated with", "is linked to"),
    ("is increasing in", "is rising in"),
    ("is recommended", "guidelines recommend"),
    ("the key to management is", "management focuses on"),
    ("first-line treatment", "initial treatment"),
    ("should undergo", "is advised to have"),
]

def apply_equiv(s: str):
    s2 = s
    for a,b in CANON_EQUIV:
        s2 = re.sub(rf"\b{re.escape(a)}\b", b, s2, flags=re.I)
    return s2

def entail_rewrite(s: str):
    s2 = apply_equiv(s)
    low = s2.lower()
    for a,b in ENTAIL_TEMPL:
        if a in low:
            s2 = re.sub(re.escape(a), b, s2, flags=re.I)
            break
    if s2 == s:
        s2 = s + " (as described in clinical guidance)"
    return s2

def contradiction_flip(s: str):
    low = s.lower()
    for a,b in FLIPS:
        if a in low:
            return re.sub(re.escape(a), b, s, flags=re.I)
        if b in low:
            return re.sub(re.escape(b), a, s, flags=re.I)
    s2 = re.sub(r"\b(is|are|was|were|can|may|should|must)\b", r"\1 not", s, count=1, flags=re.I)
    return s2 if s2 != s else ("not " + s)

def neutral_pick(sentences, anchor, max_tries=20):
    a_toks = set(tokenize(anchor))
    a_key = set([w for w in a_toks if w in (
        "gdm","gestational","diabetes","pregnancy","ogtt","hba1c","insulin","metformin",
        "macrosomia","hypoglycaemia","hypoglycemia","screening","postpartum","risk",
        "preeclampsia","guidelines"
    )])
    best = None
    for _ in range(max_tries):
        cand = random.choice(sentences)
        if cand == anchor:
            continue
        c_toks = set(tokenize(cand))
        overlap = len((a_key or a_toks) & c_toks)
        if overlap >= 1:
            best = cand
            break
    return best if best is not None else random.choice(sentences)

def build_pairs(sentences, n_pairs=9000):
    log(f"Synthetic generation: target pairs={n_pairs} (≈ {n_pairs//3} anchors × 3 labels)")
    t0 = time.time()
    pairs = []
    anchors = n_pairs // 3
    for i in range(anchors):
        a = random.choice(sentences)
        pairs.append((a, entail_rewrite(a), 0))
        pairs.append((a, contradiction_flip(a), 2))
        pairs.append((a, neutral_pick(sentences, a), 1))
        if (i+1) % 1000 == 0:
            log(f"  generated anchors={i+1}/{anchors}  (pairs={len(pairs)})")
    random.shuffle(pairs)
    log(f"Synthetic generation done: pairs={len(pairs)} in {fmt_sec(time.time()-t0)}")
    return pairs

# =========================
# Math utils
# =========================
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def dsigmoid(y):
    return y * (1. - y)

def dtanh(y):
    return 1. - y*y

def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu)**2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def cosine(a, b, eps=1e-9):
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def cosine_batch(A, B, eps=1e-9):
    An = np.linalg.norm(A, axis=1, keepdims=True) + eps
    Bn = np.linalg.norm(B, axis=1, keepdims=True) + eps
    return np.sum(A*B, axis=1) / (An[:,0]*Bn[:,0])

# =========================
# Adam optimizer (NumPy)
# =========================
class Adam:
    def __init__(self, params, lr=LR, b1=0.9, b2=0.999, eps=1e-8, wd=WEIGHT_DECAY):
        self.params = params
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self.m = {k: np.zeros_like(v) for k,v in params.items()}
        self.v = {k: np.zeros_like(v) for k,v in params.items()}

    def step(self, grads):
        self.t += 1
        for k in self.params:
            g = grads[k]
            if self.wd > 0 and self.params[k].ndim >= 2:
                g = g + self.wd * self.params[k]
            self.m[k] = self.b1*self.m[k] + (1-self.b1)*g
            self.v[k] = self.b2*self.v[k] + (1-self.b2)*(g*g)
            mhat = self.m[k] / (1 - self.b1**self.t)
            vhat = self.v[k] / (1 - self.b2**self.t)
            self.params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)

# =========================
# Model
# =========================
class GatedAttnEncoder:
    def __init__(self, vocab_size, d_model=D_MODEL, d_hid=D_HID, out_dim=OUT_DIM, n_layers=N_LAYERS):
        rng = np.random.RandomState(SEED)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_hid = d_hid
        self.out_dim = out_dim
        self.n_layers = n_layers

        self.E = (rng.randn(vocab_size, d_model) * 0.02).astype(np.float32)

        self.Wx, self.Wh, self.Wg, self.bg = [], [], [], []
        for _ in range(n_layers):
            self.Wx.append((rng.randn(d_model, d_hid) * 0.02).astype(np.float32))
            self.Wh.append((rng.randn(d_hid, d_hid) * 0.02).astype(np.float32))
            self.Wg.append((rng.randn(d_model + d_hid, d_hid) * 0.02).astype(np.float32))
            self.bg.append(np.zeros((d_hid,), dtype=np.float32))

        self.Watt = (rng.randn(d_hid, d_hid) * 0.02).astype(np.float32)
        self.vatt = (rng.randn(d_hid,) * 0.02).astype(np.float32)

        self.Wo = (rng.randn(d_hid, out_dim) * 0.02).astype(np.float32)
        self.bo = np.zeros((out_dim,), dtype=np.float32)

        self.Wnli = (rng.randn(4*out_dim, 3) * 0.02).astype(np.float32)
        self.bnli = np.zeros((3,), dtype=np.float32)

        self.params = {
            "E": self.E,
            "Watt": self.Watt, "vatt": self.vatt,
            "Wo": self.Wo, "bo": self.bo,
            "Wnli": self.Wnli, "bnli": self.bnli,
        }
        for i in range(n_layers):
            self.params[f"Wx{i}"] = self.Wx[i]
            self.params[f"Wh{i}"] = self.Wh[i]
            self.params[f"Wg{i}"] = self.Wg[i]
            self.params[f"bg{i}"] = self.bg[i]

    def encode_forward(self, ids):
        T = ids.shape[0]
        x0 = self.E[ids]
        caches = {"ids": ids, "x0": x0, "layers": []}

        x_in = x0
        for li in range(self.n_layers):
            h = np.zeros((self.d_hid,), dtype=np.float32)
            H = np.zeros((T, self.d_hid), dtype=np.float32)
            gates = np.zeros((T, self.d_hid), dtype=np.float32)
            cands = np.zeros((T, self.d_hid), dtype=np.float32)

            for t in range(T):
                xt = x_in[t]
                gh = np.concatenate([xt, h], axis=0)
                gate = sigmoid(gh @ self.Wg[li] + self.bg[li])
                cand = np.tanh(xt @ self.Wx[li] + h @ self.Wh[li])
                h = gate * h + (1.0 - gate) * cand
                H[t], gates[t], cands[t] = h, gate, cand

            x_in = H
            caches["layers"].append({"H": H, "gates": gates, "cands": cands})

        Hlast = caches["layers"][-1]["H"]
        A = np.tanh(Hlast @ self.Watt)
        scores = A @ self.vatt
        alpha = softmax(scores.reshape(1,-1), axis=1).reshape(-1)
        s = (alpha[:, None] * Hlast).sum(axis=0)
        s_ln = layer_norm(s)
        e = s_ln @ self.Wo + self.bo
        e_ln = layer_norm(e)

        caches.update({"A": A, "scores": scores, "alpha": alpha, "s": s, "s_ln": s_ln, "e": e, "e_ln": e_ln})
        return e_ln.astype(np.float32), caches

    def pair_forward(self, idsA, idsB):
        eA, cA = self.encode_forward(idsA)
        eB, cB = self.encode_forward(idsB)
        feat = np.concatenate([eA, eB, np.abs(eA-eB), eA*eB], axis=0)
        logits = feat @ self.Wnli + self.bnli
        return eA, eB, feat, logits, cA, cB

# =========================
# Loss + Backprop
# =========================
def targets_for_label(y):
    return 0.90 if y == 0 else (0.35 if y == 1 else -0.10)

def cross_entropy_logits(logits, y):
    p = softmax(logits.reshape(1,-1), axis=1).reshape(-1)
    return -math.log(float(p[y] + 1e-9)), p

def backprop_layer_norm_simple(x, grad_out, eps=1e-5):
    mu = x.mean()
    var = ((x - mu)**2).mean()
    inv = 1.0 / math.sqrt(var + eps)
    N = x.shape[0]
    dxhat = grad_out
    dvar = np.sum(dxhat*(x-mu)) * (-0.5) * (var+eps)**(-1.5)
    dmu = np.sum(dxhat)*(-inv) + dvar*np.mean(-2*(x-mu))
    dx = dxhat*inv + dvar*(2*(x-mu)/N) + dmu/N
    return dx.astype(np.float32)

def backprop_encode(model, cache, grad_e_ln, grads):
    e = cache["e"]
    de = backprop_layer_norm_simple(e, grad_e_ln)

    s_ln = cache["s_ln"]
    grads["Wo"] += np.outer(s_ln, de).astype(np.float32)
    grads["bo"] += de.astype(np.float32)
    ds_ln = model.Wo @ de

    s = cache["s"]
    ds = backprop_layer_norm_simple(s, ds_ln)

    alpha = cache["alpha"]
    Hlast = cache["layers"][-1]["H"]
    T = Hlast.shape[0]

    dHlast = (alpha[:,None] * ds[None,:]).astype(np.float32)
    dalpha = (Hlast @ ds).astype(np.float32)

    sdot = float(np.sum(dalpha*alpha))
    dscores = (alpha * (dalpha - sdot)).astype(np.float32)

    A = cache["A"]
    grads["vatt"] += (A.T @ dscores).astype(np.float32)

    dA = np.outer(dscores, model.vatt).astype(np.float32)
    dZ = dA * dtanh(A)
    grads["Watt"] += (Hlast.T @ dZ).astype(np.float32)
    dHlast += (dZ @ model.Watt.T).astype(np.float32)

    ids = cache["ids"]
    x0 = cache["x0"]

    dH = dHlast
    for li in reversed(range(model.n_layers)):
        layer = cache["layers"][li]
        H = layer["H"]
        gates = layer["gates"]
        cands = layer["cands"]

        x_in = x0 if li == 0 else cache["layers"][li-1]["H"]
        dh_next = np.zeros((model.d_hid,), dtype=np.float32)

        for t in reversed(range(T)):
            dh = dH[t] + dh_next

            gate = gates[t]
            cand = cands[t]
            h_prev = H[t-1] if t > 0 else np.zeros((model.d_hid,), dtype=np.float32)
            xt = x_in[t]

            dgate = dh * (h_prev - cand)
            dcand = dh * (1.0 - gate)

            dcand_pre = dcand * dtanh(cand)
            grads[f"Wx{li}"] += np.outer(xt, dcand_pre).astype(np.float32)
            grads[f"Wh{li}"] += np.outer(h_prev, dcand_pre).astype(np.float32)

            dxt_from_cand = model.Wx[li] @ dcand_pre
            dhprev_from_cand = model.Wh[li] @ dcand_pre

            g_in = np.concatenate([xt, h_prev], axis=0)
            dgate_pre = dgate * dsigmoid(gate)
            grads[f"Wg{li}"] += np.outer(g_in, dgate_pre).astype(np.float32)
            grads[f"bg{li}"] += dgate_pre.astype(np.float32)

            dg_in = model.Wg[li] @ dgate_pre
            dxt_from_gate = dg_in[:model.d_model]
            dhprev_from_gate = dg_in[model.d_model:]

            dxt = dxt_from_cand + dxt_from_gate
            dh_prev = (dh * gate) + dhprev_from_cand + dhprev_from_gate
            dh_next = dh_prev.astype(np.float32)

            if li == 0:
                tid = int(ids[t])
                if tid != 0:
                    grads["E"][tid] += dxt.astype(np.float32)

def train_step(model, batch_idsA, batch_idsB, batch_y, hard_neg=False):
    B = batch_idsA.shape[0]
    grads = {k: np.zeros_like(v) for k,v in model.params.items()}
    loss_sim = 0.0
    loss_nli = 0.0

    eAs, eBs, feats, logits_list, cachesA, cachesB = [], [], [], [], [], []
    for i in range(B):
        eA, eB, feat, logits, cA, cB = model.pair_forward(batch_idsA[i], batch_idsB[i])
        eAs.append(eA); eBs.append(eB); feats.append(feat); logits_list.append(logits)
        cachesA.append(cA); cachesB.append(cB)

    eAs = np.stack(eAs, axis=0)
    eBs = np.stack(eBs, axis=0)

    if hard_neg:
        sim_mat = (eAs @ eBs.T) / ((np.linalg.norm(eAs, axis=1, keepdims=True)+1e-9) * (np.linalg.norm(eBs, axis=1, keepdims=True).T+1e-9))
        k = max(1, int(B * HARD_NEG_RATE))
        idxs = np.argsort(-np.diag(sim_mat))[:k]
        for i in idxs:
            j = int(np.argsort(-sim_mat[i])[1])
            batch_idsB[i] = batch_idsB[j].copy()
            batch_y[i] = 1  # neutral

        eAs, eBs, feats, logits_list, cachesA, cachesB = [], [], [], [], [], []
        for i in range(B):
            eA, eB, feat, logits, cA, cB = model.pair_forward(batch_idsA[i], batch_idsB[i])
            eAs.append(eA); eBs.append(eB); feats.append(feat); logits_list.append(logits)
            cachesA.append(cA); cachesB.append(cB)
        eAs = np.stack(eAs, axis=0)
        eBs = np.stack(eBs, axis=0)

    for i in range(B):
        y = int(batch_y[i])
        eA = eAs[i]; eB = eBs[i]
        feat = feats[i]; logits = logits_list[i]
        cA = cachesA[i]; cB = cachesB[i]

        sim = cosine(eA, eB)
        t = targets_for_label(y)
        Ls = (sim - t)**2
        loss_sim += Ls

        eps = 1e-9
        na = np.linalg.norm(eA) + eps
        nb = np.linalg.norm(eB) + eps
        dot = float(np.dot(eA, eB))
        dsim_deA = (eB/(na*nb)) - (dot/(na*na*na*nb))*eA
        dsim_deB = (eA/(na*nb)) - (dot/(nb*nb*nb*na))*eB
        dLs_dsim = 2.0*(sim - t)
        deA_from_sim = dLs_dsim * dsim_deA
        deB_from_sim = dLs_dsim * dsim_deB

        Lc, p = cross_entropy_logits(logits, y)
        loss_nli += Lc

        dlogits = p.copy()
        dlogits[y] -= 1.0

        grads["Wnli"] += np.outer(feat, dlogits).astype(np.float32)
        grads["bnli"] += dlogits.astype(np.float32)

        dfeat = model.Wnli @ dlogits
        D = model.out_dim
        deA_from_nli = dfeat[:D] + dfeat[2*D:3*D]*np.sign(eA-eB) + dfeat[3*D:4*D]*eB
        deB_from_nli = dfeat[D:2*D] - dfeat[2*D:3*D]*np.sign(eA-eB) + dfeat[3*D:4*D]*eA

        deA = deA_from_sim + LAM_NLI*deA_from_nli
        deB = deB_from_sim + LAM_NLI*deB_from_nli

        backprop_encode(model, cA, deA, grads)
        backprop_encode(model, cB, deB, grads)

    for k in grads:
        grads[k] /= float(B)

    return (loss_sim/float(B)), (loss_nli/float(B)), grads

# =========================
# Data prep
# =========================
def load_all_sentences():
    all_sents = []
    log("Loading documents...")
    for p in DOC_PATHS:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        log(f"  reading: {os.path.basename(p)}")
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        sents = sent_split(text)
        log(f"    extracted sentences: {len(sents)}")
        all_sents.extend(sents)

    log(f"Total extracted sentences (raw): {len(all_sents)}")
    keep = []
    for s in all_sents:
        low = s.lower()
        if any(k in low for k in ["gestational", "gdm", "pregnan", "ogtt", "hba1c", "macrosomia", "hypogly", "insulin", "metformin", "postpartum", "screen", "preeclamp"]):
            keep.append(s)
    log(f"Sentences kept (GDM-focused filter): {len(keep)}")
    return keep

def build_vocab_and_dataset(sentences):
    log("Tokenizing sentences for vocab...")
    t0 = time.time()
    tok_sents = [tokenize(s) for s in sentences]
    log(f"Tokenization done in {fmt_sec(time.time()-t0)}")

    log("Building vocab...")
    t0 = time.time()
    vocab = Vocab(tok_sents)
    log(f"Vocab built: size={len(vocab.stoi)} in {fmt_sec(time.time()-t0)}")

    pairs = build_pairs(sentences, n_pairs=9000)

    log("Encoding dataset pairs -> ids...")
    t0 = time.time()
    X1, X2, Y = [], [], []
    for i, (a,b,y) in enumerate(pairs):
        X1.append(vocab.encode(tokenize(a)))
        X2.append(vocab.encode(tokenize(b)))
        Y.append(y)
        if (i+1) % 3000 == 0:
            log(f"  encoded {i+1}/{len(pairs)} pairs")
    X1 = np.stack(X1, axis=0)
    X2 = np.stack(X2, axis=0)
    Y  = np.array(Y, dtype=np.int32)
    log(f"Encoding done in {fmt_sec(time.time()-t0)}")
    return vocab, X1, X2, Y, pairs

# =========================
# Save package
# =========================
def save_model_package(model, vocab, out_prefix="gdm_sbert_numpy"):
    weights_file = out_prefix + "_weights.npz"
    vocab_file   = out_prefix + "_vocab.txt"
    config_file  = out_prefix + "_config.npz"

    np.savez(weights_file, **model.params)

    max_id = max(vocab.itos.keys())
    with open(vocab_file, "w", encoding="utf-8") as f:
        for i in range(max_id + 1):
            tok = vocab.itos.get(i, "<UNK>")
            f.write(tok + "\n")

    np.savez(config_file,
             MAX_LEN=MAX_LEN,
             D_MODEL=D_MODEL,
             D_HID=D_HID,
             OUT_DIM=OUT_DIM,
             N_LAYERS=N_LAYERS)

    return weights_file, vocab_file, config_file

# =========================
# Train
# =========================
def train():
    log("=== GDM NumPy SBERT-lite training START ===")
    t_all = time.time()

    sents = load_all_sentences()
    vocab, X1, X2, Y, pairs = build_vocab_and_dataset(sents)

    log(f"Dataset ready: pairs={X1.shape[0]} max_len={MAX_LEN} batch={BATCH}")
    log(f"Model dims: d_model={D_MODEL} d_hid={D_HID} out_dim={OUT_DIM} layers={N_LAYERS}")

    n = X1.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.9*n)
    tr, va = idx[:split], idx[split:]
    log(f"Split: train={len(tr)} val={len(va)}")

    model = GatedAttnEncoder(vocab_size=len(vocab.stoi))
    opt = Adam(model.params, lr=LR)

    for ep in range(1, EPOCHS+1):
        log(f"--- Epoch {ep}/{EPOCHS} ---")
        t_ep = time.time()

        np.random.shuffle(tr)
        tr_loss_s, tr_loss_c = 0.0, 0.0
        steps = 0
        bcount = 0

        for i in range(0, len(tr), BATCH):
            bidx = tr[i:i+BATCH]
            if len(bidx) < 4:
                continue

            idsA = X1[bidx].copy()
            idsB = X2[bidx].copy()
            y = Y[bidx].copy()

            hard = (random.random() < 0.5)
            ls, lc, grads = train_step(model, idsA, idsB, y, hard_neg=hard)
            opt.step(grads)

            tr_loss_s += ls
            tr_loss_c += lc
            steps += 1
            bcount += 1

            if (bcount % PRINT_EVERY_BATCHES) == 0:
                avg_s = tr_loss_s / max(1, steps)
                avg_c = tr_loss_c / max(1, steps)
                log(f"  step={steps:4d} avg_loss(sim)={avg_s:.4f} avg_loss(nli)={avg_c:.4f} hard_neg={hard}")

        sims_by = {0: [], 1: [], 2: []}
        v_take = va[:min(VAL_MAX, len(va))]
        for j in v_take:
            eA, _ = model.encode_forward(X1[j])
            eB, _ = model.encode_forward(X2[j])
            sims_by[int(Y[j])].append(cosine(eA, eB))

        val_report = {LABELS[k]: float(np.mean(v)) for k,v in sims_by.items() if len(v)}
        log(f"Epoch {ep} done in {fmt_sec(time.time()-t_ep)}")
        log(f"  train avg loss(sim)={tr_loss_s/max(1,steps):.4f} loss(nli)={tr_loss_c/max(1,steps):.4f}")
        log(f"  val mean cosine: {val_report}")

    wfile, vfile, cfile = save_model_package(model, vocab, out_prefix="gdm_sbert_numpy")
    log(f"Saved model package:\n  {wfile}\n  {vfile}\n  {cfile}")

    log("=== DONE ===")
    log(f"Total time: {fmt_sec(time.time()-t_all)}")

if __name__ == "__main__":
    train()
