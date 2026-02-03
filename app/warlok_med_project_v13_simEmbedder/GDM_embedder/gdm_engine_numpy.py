#!/usr/bin/env python3
"""
Fast NumPy-only inference engine for the saved GDM model package.

Loads:
  gdm_sbert_numpy_weights.npz
  gdm_sbert_numpy_vocab.txt
  gdm_sbert_numpy_config.npz

Provides:
  - embed(text) -> np.ndarray [D]
  - similarity(a,b) -> float (cosine)
  - embed_pipe(list_of_phrases) -> embeddings [N,D]
  - topk(query, phrases, k) -> ranked results
  - build_index(phrases) -> (phrases, embs)
  - topk_from_index(query, phrases, embs, k) -> ranked results

Run a quick test:
  python3 gdm_engine_numpy.py --prefix gdm_sbert_numpy --demo
"""
import re
import argparse
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def layer_norm(x, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu)**2).mean(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def cosine(a, b, eps=1e-9):
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))

def cosine_batch(A, b, eps=1e-9):
    An = np.linalg.norm(A, axis=1) + eps
    bn = np.linalg.norm(b) + eps
    return (A @ b) / (An * bn)

def tokenize(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9%/\-\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.split()

class Vocab:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    @staticmethod
    def load_from_vocab_txt(vocab_file):
        itos = {}
        stoi = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                tok = line.rstrip("\n")
                itos[i] = tok
                if tok not in stoi:
                    stoi[tok] = i
        return Vocab(stoi=stoi, itos=itos)

    def encode(self, toks, max_len):
        ids = [self.stoi.get(w, 1) for w in toks[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return np.array(ids, dtype=np.int32)

class GatedAttnEncoderInfer:
    def __init__(self, params, config):
        self.p = params
        self.MAX_LEN = int(config["MAX_LEN"])
        self.D_MODEL = int(config["D_MODEL"])
        self.D_HID   = int(config["D_HID"])
        self.OUT_DIM = int(config["OUT_DIM"])
        self.N_LAYERS = int(config["N_LAYERS"])

    def encode_ids(self, ids):
        E = self.p["E"]
        x_in = E[ids]  # [T,d_model]
        T = x_in.shape[0]

        for li in range(self.N_LAYERS):
            Wx = self.p[f"Wx{li}"]
            Wh = self.p[f"Wh{li}"]
            Wg = self.p[f"Wg{li}"]
            bg = self.p[f"bg{li}"]

            h = np.zeros((self.D_HID,), dtype=np.float32)
            H = np.zeros((T, self.D_HID), dtype=np.float32)

            for t in range(T):
                xt = x_in[t]
                gh = np.concatenate([xt, h], axis=0)
                gate = sigmoid(gh @ Wg + bg)
                cand = np.tanh(xt @ Wx + h @ Wh)
                h = gate * h + (1.0 - gate) * cand
                H[t] = h

            x_in = H  # assumes D_MODEL == D_HID (true in provided config)

        Watt = self.p["Watt"]
        vatt = self.p["vatt"]
        A = np.tanh(H @ Watt)
        scores = A @ vatt
        alpha = softmax(scores.reshape(1,-1), axis=1).reshape(-1)
        s = (alpha[:, None] * H).sum(axis=0)
        s = layer_norm(s)

        Wo = self.p["Wo"]
        bo = self.p["bo"]
        e = s @ Wo + bo
        e = layer_norm(e)
        return e.astype(np.float32)

class GDMEmbeddingEngine:
    def __init__(self, vocab, model):
        self.vocab = vocab
        self.model = model

    @staticmethod
    def load(prefix="gdm_sbert_numpy"):
        weights_file = prefix + "_weights.npz"
        vocab_file   = prefix + "_vocab.txt"
        config_file  = prefix + "_config.npz"

        w = np.load(weights_file)
        params = {k: w[k].astype(np.float32) for k in w.files}

        c = np.load(config_file)
        config = {k: c[k] for k in c.files}

        vocab = Vocab.load_from_vocab_txt(vocab_file)
        model = GatedAttnEncoderInfer(params=params, config=config)
        return GDMEmbeddingEngine(vocab=vocab, model=model)

    def embed(self, text: str):
        ids = self.vocab.encode(tokenize(text), max_len=self.model.MAX_LEN)
        return self.model.encode_ids(ids)

    def similarity(self, a: str, b: str):
        ea = self.embed(a)
        eb = self.embed(b)
        return cosine(ea, eb)

    def embed_pipe(self, phrases):
        # phrases: list[str] -> embeddings [N,D]
        embs = [self.embed(p) for p in phrases]
        return np.stack(embs, axis=0)

    def build_index(self, phrases):
        embs = self.embed_pipe(phrases)
        return phrases, embs

    def topk_from_index(self, query, phrases, embs, k=5):
        eq = self.embed(query)
        sims = cosine_batch(embs, eq)
        top = np.argsort(-sims)[:k]
        return [(float(sims[i]), phrases[i]) for i in top]

    def topk(self, query: str, phrases, k=5):
        phrases, embs = self.build_index(phrases)
        return self.topk_from_index(query, phrases, embs, k=k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="gdm_sbert_numpy")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--pipe", nargs="*", default=None, help="A pipe of phrases to embed and print shapes")
    args = ap.parse_args()

    eng = GDMEmbeddingEngine.load(prefix=args.prefix)

    if args.pipe:
        E = eng.embed_pipe(args.pipe)
        print("Embedded pipe shape:", E.shape)
        print("First vector (truncated):", E[0][:10])

    if args.demo:
        cands = [
            "women with gestational diabetes should have postpartum glucose testing",
            "maternal hyperglycemia increases fetal insulin and macrosomia risk",
            "metformin is sometimes used to treat gestational diabetes when diet is insufficient",
            "screening may be performed with an oral glucose tolerance test",
            "contradiction example: gestational diabetes decreases risk of macrosomia",
        ]
        q = "how does maternal hyperglycemia lead to fetal macrosomia"
        print("\nQuery:", q)
        for sim, txt in eng.topk(q, cands, k=3):
            print(f"  sim={sim:.3f} | {txt}")

if __name__ == "__main__":
    main()
