from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A, softmax


# ============================================================
# 1) Sentence extraction + vocab
# ============================================================

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")


def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        # keep evidence-sized sentences
        if 60 <= len(s) <= 360 and "http" not in s.lower():
            out.append(s)
    return out


def build_vocab(texts: List[str], max_vocab: int = 12000) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for t in texts:
        for w in t.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%', '-'])
            if not w:
                continue
            freq[w] = freq.get(w, 0) + 1

    vocab = {"<pad>": 0, "<unk>": 1}
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    for w, _ in items[: max(0, max_vocab - 2)]:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


# ============================================================
# 2) Evidence-like bank filtering (critical!)
# ============================================================

EVIDENCE_TERMS = [
    # screening/diagnosis
    "ogtt", "75", "24", "28", "fasting", "hba1c", "threshold", "diagnos",
    # postpartum
    "postpartum", "6", "12", "follow-up", "type 2", "surveillance",
    # management
    "insulin", "metformin", "lifestyle", "diet", "exercise", "monitor", "targets", "smbg", "cgm",
    # complications
    "macrosomia", "hypoglyc", "preeclamps", "shoulder", "nicu", "polycyth", "hyperbilirub",
    # pathophys
    "placent", "insulin resistance", "beta", "hyperglyc", "fetal insulin",
]

GENERIC_BAD_PHRASES = [
    "this review", "in this review", "this article", "we discuss", "we describe",
    "recent years", "in conclusion", "overall", "however", "therefore", "more research",
]


def is_evidence_sentence(s: str) -> bool:
    t = s.lower()
    if len(t) < 70:
        return False
    # remove obvious fluff
    bad_hits = sum(1 for bp in GENERIC_BAD_PHRASES if bp in t)
    if bad_hits >= 2:
        return False
    hits = 0
    for w in EVIDENCE_TERMS:
        if w in t:
            hits += 1
            if hits >= 1:
                return True
    return False


# ============================================================
# 3) Masked-sentence positives (aligned training!)
# ============================================================

STOP = set([
    "the","a","an","and","or","of","to","in","for","with","on","at","by",
    "is","are","was","were","be","been","being","as","it","this","that","these","those"
])

KEY_MED = set([
    "ogtt","insulin","metformin","postpartum","macrosomia","hypoglycemia","preeclampsia",
    "hyperglycemia","screening","diagnosis","surveillance","placental","beta-cell"
])


def mask_sentence_to_question(sent: str, seed: int) -> str:
    rng = random.Random(seed)
    toks = sent.split()

    cand = []
    for i, w in enumerate(toks):
        ww = re.sub(r"[^a-zA-Z0-9%\-]", "", w.lower())
        if not ww or ww in STOP:
            continue

        is_num = any(ch.isdigit() for ch in ww)
        is_long = len(ww) >= 7
        is_key = ww in KEY_MED

        if is_num or is_long or is_key:
            cand.append(i)

    if not cand:
        cand = list(range(min(12, len(toks))))

    rng.shuffle(cand)
    k = min(4, max(2, len(cand) // 7))
    mask_idx = set(cand[:k])

    masked = ["[MASK]" if i in mask_idx else w for i, w in enumerate(toks)]
    masked_sent = " ".join(masked)

    # simple but effective prompt
    return f"Fill the missing clinical details in this statement: {masked_sent}"


def make_positive_pairs(sentences: List[str], n_pairs: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    out = []
    for i in range(n_pairs):
        a = rng.choice(sentences)
        q = mask_sentence_to_question(a, seed=seed * 100000 + i)
        out.append((q, a))
    return out


# ============================================================
# 4) Contrastive training + hard-negative mining
# ============================================================

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def cosine_sim_matrix(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
    return Q @ A.T


def batch_iter(pairs: List[Tuple[str, str]], bs: int, seed: int):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [pairs[j] for j in idx[i:i + bs]]


def top_vocab_ids_by_freq(vocab: Dict[str, int], texts: List[str], top_k: int) -> List[int]:
    freq = np.zeros((len(vocab),), dtype=np.int32)
    for t in texts:
        for w in t.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%', '-'])
            if not w:
                continue
            freq[vocab.get(w, 1)] += 1
    freq[0] = 0
    freq[1] = 0
    ids = np.argsort(-freq)[:top_k]
    return [int(i) for i in ids if freq[i] > 0]


class HardNegMiner:
    """
    Evidence-filtered sentence bank + embeddings for mining hard negatives.
    Mined negative must not equal the positive sentence and must not be “nearby” by bucket.
    """

    def __init__(self, bank_sents: List[str], meta_bucket: List[int], top_k: int = 25):
        self.bank_sents = bank_sents
        self.meta_bucket = meta_bucket
        self.top_k = top_k
        self.bank_embs: Optional[np.ndarray] = None

    def rebuild(self, model: GdmSentenceTransformerV2A, print_every: int = 500):
        embs = []
        for i, s in enumerate(self.bank_sents):
            if print_every > 0 and i > 0 and i % print_every == 0:
                print(f"[mine] embed bank {i}/{len(self.bank_sents)}")
            embs.append(model.embed(s).astype(np.float32))
        self.bank_embs = l2norm_rows(np.vstack(embs).astype(np.float32))
        print(f"[mine] bank ready: {self.bank_embs.shape[0]} sentences")

    def mine(self, q_emb: np.ndarray, a_pos: str, pos_bucket: int) -> str:
        sims = self.bank_embs @ q_emb  # (N,)
        k = min(self.top_k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        for j in idx:
            j = int(j)
            s = self.bank_sents[j]
            if s == a_pos:
                continue
            if self.meta_bucket[j] == pos_bucket:
                continue
            return s

        for j in idx:
            s = self.bank_sents[int(j)]
            if s != a_pos:
                return s

        return self.bank_sents[int(idx[0])]


# ============================================================
# 5) Trainer
# ============================================================

def train(args):
    print("[load] reading docs...")
    doc_texts = []
    all_sents = []
    for p in [Path(x) for x in args.docs]:
        txt = read_text(p)
        doc_texts.append(txt)
        ss = split_sentences(txt)
        all_sents.extend(ss)
        print(f"  - {p.name}: {len(ss)} sentences")

    print(f"[bank] total sentences={len(all_sents)}")

    # split to avoid leakage
    rng = random.Random(args.seed)
    idx = list(range(len(all_sents)))
    rng.shuffle(idx)
    cut = int(0.85 * len(idx))
    train_sents = [all_sents[i] for i in idx[:cut]]
    valid_sents = [all_sents[i] for i in idx[cut:]]
    print(f"[split] train={len(train_sents)} valid={len(valid_sents)}")

    # evidence filter for mining bank + (optionally) for training positives
    train_evidence = [s for s in train_sents if is_evidence_sentence(s)]
    valid_evidence = [s for s in valid_sents if is_evidence_sentence(s)]

    # fallback if filter too aggressive
    if len(train_evidence) < 800:
        print("[warn] evidence filter too aggressive; using unfiltered train sentences for positives.")
        train_evidence = train_sents[:]
    if len(valid_evidence) < 200:
        valid_evidence = valid_sents[:]

    print(f"[evidence] train_evidence={len(train_evidence)} valid_evidence={len(valid_evidence)}")

    # vocab from docs + evidence bank
    vocab = build_vocab(doc_texts + train_evidence + valid_evidence, max_vocab=args.max_vocab)
    print(f"[vocab] size={len(vocab)} cap={args.max_vocab}")

    model = GdmSentenceTransformerV2A.init(
        vocab=vocab,
        seed=args.seed,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        emb_dim=args.emb_dim,
    )
    print(f"[init] layers={args.n_layers} heads={args.n_heads} d_model={args.d_model} emb_dim={args.emb_dim}")

    # positives (masked aligned)
    train_pairs = make_positive_pairs(train_evidence, args.train_pairs, seed=args.seed + 10)
    valid_pairs = make_positive_pairs(valid_evidence, args.valid_pairs, seed=args.seed + 20)
    print(f"[pairs] train={len(train_pairs)} valid={len(valid_pairs)}")

    # trainable embedding rows
    trainable_ids = top_vocab_ids_by_freq(vocab, train_evidence, top_k=args.train_embed_rows)
    trainable_set = set(trainable_ids)
    print(f"[emb] trainable rows={len(trainable_ids)} (requested={args.train_embed_rows})")

    # params to update (fast subset)
    params = {
        "W_proj": model.W_proj,
        "b_proj": model.b_proj,
        "w_pool": model.w_pool,
        "b_pool": np.array([model.b_pool], dtype=np.float32),
    }
    for l in range(model.n_layers):
        params[f"b_gate_g_{l}"] = model.b_gate_g[l]
        params[f"b_gate_h_{l}"] = model.b_gate_h[l]

    # Adam state
    beta1, beta2 = 0.9, 0.999
    tstep = 0
    m = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items()}
    v = {k: np.zeros_like(vv, dtype=np.float32) for k, vv in params.items()}

    # embedding row Adam state (separate)
    emb_m = np.zeros((len(trainable_ids), model.d_model), dtype=np.float32)
    emb_v = np.zeros((len(trainable_ids), model.d_model), dtype=np.float32)

    def adam_step(name: str, grad: np.ndarray, lr: float):
        nonlocal tstep
        tstep += 1
        m[name] = beta1 * m[name] + (1 - beta1) * grad
        v[name] = beta2 * v[name] + (1 - beta2) * (grad * grad)
        mhat = m[name] / (1 - beta1 ** tstep)
        vhat = v[name] / (1 - beta2 ** tstep)
        params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

    def encode_pooled(text: str):
        ids = model.tokenize(text)
        emb, _, pooled = model.forward(ids)
        return emb.astype(np.float32), pooled.astype(np.float32), ids

    def tangent(dE: np.ndarray, E: np.ndarray) -> np.ndarray:
        return dE - (np.sum(dE * E, axis=1, keepdims=True) * E)

    def valid_recall_at_1(n: int = 200) -> float:
        sample = random.sample(valid_pairs, k=min(n, len(valid_pairs)))
        Q = []
        A = []
        for q, a in sample:
            Q.append(model.embed(q))
            A.append(model.embed(a))
        Q = l2norm_rows(np.vstack(Q).astype(np.float32))
        A = l2norm_rows(np.vstack(A).astype(np.float32))
        S = Q @ A.T
        return float(np.mean(np.argmax(S, axis=1) == np.arange(S.shape[0])))

    # -------------------- mining bank --------------------
    bank_sents = train_evidence[:]
    rng.shuffle(bank_sents)
    bank_sents = bank_sents[: args.bank_max_sents]

    # meta buckets: group every stride sentences (prevents near-duplicate “same chunk” mining)
    bank_bucket = [i // args.bank_meta_stride for i in range(len(bank_sents))]

    miner = HardNegMiner(bank_sents, bank_bucket, top_k=args.hard_topk)
    miner.rebuild(model, print_every=500)

    # train loop
    tau = args.tau
    id_to_row = {tid: r for r, tid in enumerate(trainable_ids)}

    print("[train] v2.1 aligned contrastive + evidence-filtered hardneg ...")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        seen = 0

        for step, batch in enumerate(batch_iter(train_pairs, args.batch, seed=args.seed + epoch)):
            B = len(batch)
            if B < 2:
                continue
            global_step += 1

            if global_step % args.mine_refresh_steps == 0:
                print(f"[mine] refresh @gstep={global_step}")
                miner.rebuild(model, print_every=500)

            # Encode Q and A_pos
            Q_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            Apos_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            Q_pool = np.zeros((B, model.d_model), dtype=np.float32)
            Apos_pool = np.zeros((B, model.d_model), dtype=np.float32)
            Q_ids = []
            Apos_ids = []

            a_pos_text = []
            pos_bucket = []

            for i, (q, a_pos) in enumerate(batch):
                qe, qp, qids = encode_pooled(q)
                ae, ap, aids = encode_pooled(a_pos)

                Q_emb[i] = qe
                Apos_emb[i] = ae
                Q_pool[i] = qp
                Apos_pool[i] = ap
                Q_ids.append(qids)
                Apos_ids.append(aids)

                a_pos_text.append(a_pos)
                pos_bucket.append(i // max(1, (B // 8)))

            Q_emb = l2norm_rows(Q_emb)
            Apos_emb = l2norm_rows(Apos_emb)

            # Mine hard negatives (moderate)
            hard_mask = (np.random.random(B) < args.p_hard)
            neg_texts = []
            for i in range(B):
                if not hard_mask[i]:
                    neg_texts.append(None)
                    continue
                neg_texts.append(miner.mine(Q_emb[i], a_pos_text[i], pos_bucket[i]))

            neg_idx = [i for i in range(B) if neg_texts[i] is not None]
            M = len(neg_idx)

            if M > 0:
                Aneg_emb = np.zeros((M, model.emb_dim), dtype=np.float32)
                Aneg_pool = np.zeros((M, model.d_model), dtype=np.float32)
                Aneg_ids = []
                for j, i0 in enumerate(neg_idx):
                    ne, npool, nids = encode_pooled(neg_texts[i0])
                    Aneg_emb[j] = ne
                    Aneg_pool[j] = npool
                    Aneg_ids.append(nids)
                Aneg_emb = l2norm_rows(Aneg_emb)
                A_ext = np.vstack([Apos_emb, Aneg_emb]).astype(np.float32)
            else:
                Aneg_pool = None
                Aneg_ids = None
                A_ext = Apos_emb

            # InfoNCE
            S = Q_emb @ A_ext.T
            logits = S / tau
            P = softmax(logits, axis=1)
            y = np.arange(B, dtype=np.int32)
            loss = -np.log(P[np.arange(B), y] + 1e-9).mean()

            total_loss += float(loss) * B
            seen += B

            # grads
            G = P.copy()
            G[np.arange(B), y] -= 1.0
            G /= B
            dS = G / tau

            dQ = dS @ A_ext
            dAext = dS.T @ Q_emb

            dApos = dAext[:B]
            dAneg = dAext[B:] if M > 0 else None

            dZq = tangent(dQ, Q_emb)
            dZa_pos = tangent(dApos, Apos_emb)
            dZa_neg = tangent(dAneg, Aneg_emb) if M > 0 else None

            # projection updates
            gW = (Q_pool.T @ dZq) + (Apos_pool.T @ dZa_pos)
            gb = dZq.sum(axis=0) + dZa_pos.sum(axis=0)
            if M > 0:
                gW += (Aneg_pool.T @ dZa_neg)
                gb += dZa_neg.sum(axis=0)

            adam_step("W_proj", gW.astype(np.float32), args.lr)
            adam_step("b_proj", gb.astype(np.float32), args.lr)

            # pooling gate shaping (light)
            push = float(loss)
            gw_pool = ((Q_pool.mean(axis=0) + Apos_pool.mean(axis=0)) * min(1.0, push)).astype(np.float32) / 2.0
            gb_pool = np.array([min(1.0, push)], dtype=np.float32)
            adam_step("w_pool", gw_pool, args.lr_pool)
            adam_step("b_pool", gb_pool, args.lr_pool)

            # attention gate bias nudges (small)
            g = np.array([min(0.03, push * 0.015)], dtype=np.float32)
            for l in range(model.n_layers):
                adam_step(f"b_gate_g_{l}", g, args.lr_gate)
                adam_step(f"b_gate_h_{l}", np.full((model.n_heads,), float(g[0]), dtype=np.float32), args.lr_gate)

            # embedding subset update via pooled proxy
            dPoolQ = dZq @ params["W_proj"].T
            dPoolApos = dZa_pos @ params["W_proj"].T
            dPoolAneg = (dZa_neg @ params["W_proj"].T) if M > 0 else None

            gEmb = np.zeros((len(trainable_ids), model.d_model), dtype=np.float32)

            for i in range(B):
                for tid in Q_ids[i]:
                    tid = int(tid)
                    if tid in trainable_set:
                        gEmb[id_to_row[tid]] += dPoolQ[i] / max(1, len(Q_ids[i]))
                for tid in Apos_ids[i]:
                    tid = int(tid)
                    if tid in trainable_set:
                        gEmb[id_to_row[tid]] += dPoolApos[i] / max(1, len(Apos_ids[i]))

            if M > 0:
                for j, i0 in enumerate(neg_idx):
                    for tid in Aneg_ids[j]:
                        tid = int(tid)
                        if tid in trainable_set:
                            gEmb[id_to_row[tid]] += dPoolAneg[j] / max(1, len(Aneg_ids[j]))

            # Adam for embedding rows (separate state)
            emb_m[:] = 0.9 * emb_m + 0.1 * gEmb
            emb_v[:] = 0.999 * emb_v + 0.001 * (gEmb * gEmb)
            emb_t = epoch * 100000 + step + 1
            mhat = emb_m / (1 - (0.9 ** emb_t))
            vhat = emb_v / (1 - (0.999 ** emb_t))
            delta = args.lr_emb * mhat / (np.sqrt(vhat) + 1e-8)

            for r, tid in enumerate(trainable_ids):
                model.W_emb[tid] -= delta[r].astype(np.float32)

            # sync scalar
            model.b_pool = float(params["b_pool"][0])

            if args.print_steps and (step % args.print_every == 0):
                r1 = valid_recall_at_1(n=args.valid_probe)
                print(f"[epoch {epoch:02d} step {step:05d} gstep {global_step:07d}] "
                      f"loss={loss:.4f} hardM={M}/{B} val_r@1={r1:.3f}")

        avg = total_loss / max(1, seen)
        r1 = valid_recall_at_1(n=400)
        print(f"[epoch {epoch:02d}] avg_loss={avg:.4f} val_r@1={r1:.3f}")

    out_dir = Path(args.out_dir)
    model.save(out_dir, name=args.name)
    print(f"[save] -> {out_dir}/{args.name}_weights.npz (+config+vocab)")
    print("[done]")


# ============================================================
# 6) CLI
# ============================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--name", default="gdm_sbert_v2_1_hardneg_aligned")
    ap.add_argument("--seed", type=int, default=7)

    # model
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_vocab", type=int, default=12000)
    ap.add_argument("--d_model", type=int, default=160)
    ap.add_argument("--n_heads", type=int, default=5)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=192)

    # data
    ap.add_argument("--train_pairs", type=int, default=25000)
    ap.add_argument("--valid_pairs", type=int, default=2500)

    # training
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.10)

    ap.add_argument("--lr", type=float, default=1.2e-3)      # proj
    ap.add_argument("--lr_pool", type=float, default=6e-4)   # pooling gate
    ap.add_argument("--lr_gate", type=float, default=3e-4)   # attn gate biases
    ap.add_argument("--lr_emb", type=float, default=5e-4)    # embedding rows

    # request: 8000
    ap.add_argument("--train_embed_rows", type=int, default=8000)

    # mining (moderate, stable)
    ap.add_argument("--p_hard", type=float, default=0.55)
    ap.add_argument("--hard_topk", type=int, default=25)
    ap.add_argument("--mine_refresh_steps", type=int, default=150)
    ap.add_argument("--bank_max_sents", type=int, default=4000)
    ap.add_argument("--bank_meta_stride", type=int, default=20)

    # logging
    ap.add_argument("--print_steps", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--valid_probe", type=int, default=200)

    args = ap.parse_args()
    train(args)
