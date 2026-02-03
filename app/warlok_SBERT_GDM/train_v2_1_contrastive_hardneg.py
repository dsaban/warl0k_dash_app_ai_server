from __future__ import annotations

import argparse, random, re
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A, softmax


# ----------------------------
# Text / sentence processing
# ----------------------------

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")


def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        if 50 <= len(s) <= 340 and "http" not in s.lower():
            out.append(s)
    return out


TOPIC_HINTS = {
    "screening": ["screen", "ogtt", "75", "24-28", "24–28", "hba1c", "first trimester", "fasting", "threshold"],
    "management": ["insulin", "metformin", "lifestyle", "diet", "exercise", "monitoring", "targets", "smbg", "cgm"],
    "complications": ["macrosomia", "hypoglycemia", "preeclampsia", "shoulder dystocia", "nicu", "trauma",
                      "hyperbilirubinaemia", "polycythaemia"],
    "postpartum": ["postpartum", "6", "12", "follow-up", "type 2", "lifelong", "surveillance"],
    "pathophysiology": ["insulin resistance", "placental", "beta", "hyperglycemia", "fetal insulin"],
}

Q_TEMPL = {
    "screening": [
        "When is screening for gestational diabetes typically performed, and what test is used?",
        "What timing and test are used to diagnose gestational diabetes in pregnancy?",
        "Why might early pregnancy testing be performed in high-risk women?"
    ],
    "management": [
        "What is first-line management for gestational diabetes before medications?",
        "When is insulin started in gestational diabetes management?",
        "How is glucose monitoring used in gestational diabetes?"
    ],
    "complications": [
        "How does maternal hyperglycaemia contribute to fetal macrosomia and neonatal hypoglycaemia?",
        "What neonatal complications are associated with diabetes in pregnancy?",
        "Why does macrosomia increase risk of difficult delivery?"
    ],
    "postpartum": [
        "What postpartum follow-up testing is recommended after gestational diabetes?",
        "Why is lifelong surveillance recommended after a GDM pregnancy?",
        "What is the long-term risk after gestational diabetes?"
    ],
    "pathophysiology": [
        "How do placental hormones contribute to insulin resistance in pregnancy?",
        "Why does insulin resistance rise in pregnancy and when does it become gestational diabetes?",
        "What is the role of beta-cell compensation in pregnancy?"
    ],
    "general": [
        "Summarize the key clinical point in this sentence.",
        "Explain the clinical meaning of this statement.",
        "What does evidence suggest here?"
    ]
}


def pick_topic(sentence: str) -> str:
    s = sentence.lower()
    best = "general"
    best_score = 0
    for t, keys in TOPIC_HINTS.items():
        sc = 0
        for k in keys:
            if k in s:
                sc += 1
        if sc > best_score:
            best_score = sc
            best = t
    return best


def make_question(topic: str) -> str:
    return random.choice(Q_TEMPL.get(topic, Q_TEMPL["general"]))


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


# ----------------------------
# Synthetic Q-A pairs (positives)
# ----------------------------

def make_positive_pairs(sentences: List[str], n_pairs: int, seed: int) -> List[Tuple[str, str, str]]:
    """
    Returns list of (q, a_pos, topic)
    Positive means: q should retrieve/support a_pos.
    """
    random.seed(seed)
    by_topic: Dict[str, List[str]] = {}
    for s in sentences:
        by_topic.setdefault(pick_topic(s), []).append(s)
    if "general" not in by_topic:
        by_topic["general"] = sentences

    topics = [t for t in by_topic.keys() if by_topic[t]]
    out = []
    for _ in range(n_pairs):
        t = random.choice(topics)
        a = random.choice(by_topic[t])
        q = make_question(t)
        out.append((q, a, t))
    return out


# ----------------------------
# Contrastive helpers
# ----------------------------

def cosine_sim_matrix(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
    return Q @ A.T


def batch_iter(items: List[Tuple[str, str, str]], bs: int, seed: int):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


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


def l2norm_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


# ----------------------------
# Hard-negative miner
# ----------------------------

class HardNegMiner:
    """
    Maintains a sentence bank + embeddings for mining hard negatives:
      for each q, pick a bank sentence with high cosine that is NOT the positive sentence.
    """

    def __init__(self, bank_sents: List[str], bank_meta: List[int], top_k: int = 25):
        self.bank_sents = bank_sents
        self.bank_meta = bank_meta  # paragraph-ish id or index
        self.top_k = top_k
        self.bank_embs = None

    def rebuild(self, model: GdmSentenceTransformerV2A, *, print_every: int = 500) -> None:
        embs = []
        for i, s in enumerate(self.bank_sents):
            if print_every > 0 and i > 0 and i % print_every == 0:
                print(f"[mine] embed bank {i}/{len(self.bank_sents)}")
            e = model.embed(s).astype(np.float32)
            embs.append(e)
        embs = np.vstack(embs).astype(np.float32)
        self.bank_embs = l2norm_rows(embs)
        print(f"[mine] bank built: {self.bank_embs.shape[0]} x {self.bank_embs.shape[1]}")

    def mine_one(self, q_emb: np.ndarray, a_pos: str, pos_meta: int) -> str:
        """
        q_emb must be normalized row vector (emb_dim,)
        """
        sims = self.bank_embs @ q_emb  # (N,)
        k = min(self.top_k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]

        # choose first good hard negative that isn't identical and isn't from same meta bucket
        for j in idx:
            s = self.bank_sents[int(j)]
            if s == a_pos:
                continue
            if self.bank_meta[int(j)] == pos_meta:
                continue
            return s

        # fallback: best different string
        for j in idx:
            s = self.bank_sents[int(j)]
            if s != a_pos:
                return s
        return self.bank_sents[int(idx[0])]


# ----------------------------
# Training loop (v2.1 + hardneg)
# ----------------------------

def train(args):
    print("[load] docs...")
    doc_texts = []
    sentences = []
    for p in [Path(x) for x in args.docs]:
        txt = read_text(p)
        doc_texts.append(txt)
        ss = split_sentences(txt)
        sentences += ss
        print(f"  - {p.name}: {len(ss)} sentences")
    print(f"[bank] total sentences={len(sentences)}")

    # split sentence bank to avoid leakage
    rng = random.Random(args.seed)
    idx = list(range(len(sentences)))
    rng.shuffle(idx)
    cut = int(0.85 * len(idx))
    train_sents = [sentences[i] for i in idx[:cut]]
    valid_sents = [sentences[i] for i in idx[cut:]]
    print(f"[split] train_sents={len(train_sents)} valid_sents={len(valid_sents)}")

    # vocab
    vocab = build_vocab(doc_texts + train_sents + valid_sents, max_vocab=args.max_vocab)
    print(f"[vocab] size={len(vocab)} cap={args.max_vocab}")

    # init model
    model = GdmSentenceTransformerV2A.init(
        vocab=vocab,
        seed=args.seed,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        emb_dim=args.emb_dim
    )
    print(f"[init] layers={args.n_layers} heads={args.n_heads} d_model={args.d_model} emb_dim={args.emb_dim}")

    # build positive Q-A pairs
    train_pairs = make_positive_pairs(train_sents, n_pairs=args.train_pairs, seed=args.seed + 10)
    valid_pairs = make_positive_pairs(valid_sents, n_pairs=args.valid_pairs, seed=args.seed + 20)
    print(f"[pairs] train={len(train_pairs)} valid={len(valid_pairs)}")

    # choose embedding rows to update
    trainable_emb_ids: List[int] = []
    if args.train_embed_rows > 0:
        trainable_emb_ids = top_vocab_ids_by_freq(vocab, train_sents, top_k=args.train_embed_rows)
        trainable_emb_ids_set = set(trainable_emb_ids)
        print(f"[emb] trainable rows={len(trainable_emb_ids)} (top frequency tokens)")
    else:
        trainable_emb_ids_set = set()

    # params to update
    params: Dict[str, np.ndarray] = {
        "W_proj": model.W_proj,
        "b_proj": model.b_proj,
        "w_pool": model.w_pool,
        "b_pool": np.array([model.b_pool], dtype=np.float32),
    }
    for l in range(model.n_layers):
        params[f"b_gate_g_{l}"] = model.b_gate_g[l]
        params[f"b_gate_h_{l}"] = model.b_gate_h[l]

    # Adam buffers
    emb_m = np.zeros((len(trainable_emb_ids), model.d_model), dtype=np.float32)
    emb_v = np.zeros((len(trainable_emb_ids), model.d_model), dtype=np.float32)

    m = {k: np.zeros_like(v, dtype=np.float32) for k, v in params.items()}
    v = {k: np.zeros_like(vv, dtype=np.float32) for k, vv in params.items()}
    beta1, beta2 = 0.9, 0.999
    tstep = 0

    def adam_step(name: str, grad: np.ndarray, lr: float):
        nonlocal tstep
        tstep += 1
        m[name] = beta1 * m[name] + (1 - beta1) * grad
        v[name] = beta2 * v[name] + (1 - beta2) * (grad * grad)
        mhat = m[name] / (1 - beta1 ** tstep)
        vhat = v[name] / (1 - beta2 ** tstep)
        params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

    tau = args.tau

    def encode_pooled(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = model.tokenize(text)
        emb, _, pooled = model.forward(ids)
        return emb.astype(np.float32), pooled.astype(np.float32), ids

    def valid_recall_at_1(n: int = 200) -> float:
        if not valid_pairs:
            return 0.0
        sample = random.sample(valid_pairs, k=min(n, len(valid_pairs)))
        Q = []
        A = []
        for q, a, _ in sample:
            Q.append(model.embed(q))
            A.append(model.embed(a))
        Q = l2norm_rows(np.vstack(Q).astype(np.float32))
        A = l2norm_rows(np.vstack(A).astype(np.float32))
        S = Q @ A.T
        pred = np.argmax(S, axis=1)
        return float(np.mean(pred == np.arange(S.shape[0])))

    # ----------------------------
    # Build mining bank (subset for speed)
    # ----------------------------
    # Use a limited bank size so mining stays fast on CPU
    bank_sents = train_sents[:]
    rng.shuffle(bank_sents)
    bank_sents = bank_sents[: args.bank_max_sents]

    # meta buckets: simple heuristic (group every N sentences)
    # This prevents picking a "nearby" sentence from the same source chunk.
    bank_meta = [i // args.bank_meta_stride for i in range(len(bank_sents))]

    miner = HardNegMiner(bank_sents=bank_sents, bank_meta=bank_meta, top_k=args.hard_topk)
    miner.rebuild(model, print_every=500)

    # ----------------------------
    # Main training
    # ----------------------------
    print("[train] starting contrastive (InfoNCE) + hard-negative mining ...")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_seen = 0

        for step, batch in enumerate(batch_iter(train_pairs, args.batch, seed=args.seed + epoch)):
            B = len(batch)
            if B < 2:
                continue

            global_step += 1

            # refresh bank periodically so negatives track current embedding space
            if global_step % args.mine_refresh_steps == 0:
                print(f"[mine] refresh @global_step={global_step}")
                miner.rebuild(model, print_every=500)

            # forward Q and A_pos
            Q_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            Apos_emb = np.zeros((B, model.emb_dim), dtype=np.float32)

            Q_pool = np.zeros((B, model.d_model), dtype=np.float32)
            Apos_pool = np.zeros((B, model.d_model), dtype=np.float32)

            Q_ids_list = []
            Apos_ids_list = []

            # store positives for mining exclusion
            a_pos_list = []
            pos_meta_list = []

            for i, (q, a_pos, _) in enumerate(batch):
                qe, qp, qids = encode_pooled(q)
                ae, ap, aids = encode_pooled(a_pos)

                Q_emb[i] = qe
                Apos_emb[i] = ae
                Q_pool[i] = qp
                Apos_pool[i] = ap

                Q_ids_list.append(qids)
                Apos_ids_list.append(aids)

                a_pos_list.append(a_pos)
                # pos_meta: bucket by i (not perfect, but good enough to avoid exact adjacency)
                pos_meta_list.append(i // max(1, (args.batch // 8)))

            Q_emb = l2norm_rows(Q_emb)
            Apos_emb = l2norm_rows(Apos_emb)

            # ----------------------------
            # Hard negatives: mine one per query with prob p_hard
            # Build A_ext = [A_pos (B)] + [A_neg (M)]
            # Targets remain y = 0..B-1 on the first B columns.
            # ----------------------------
            hard_mask = np.random.random(B) < args.p_hard
            neg_texts = []
            for i in range(B):
                if not hard_mask[i]:
                    neg_texts.append(None)
                    continue
                qv = Q_emb[i]
                neg = miner.mine_one(qv, a_pos=a_pos_list[i], pos_meta=pos_meta_list[i])
                neg_texts.append(neg)

            # encode negatives that exist
            neg_indices = [i for i in range(B) if neg_texts[i] is not None]
            M = len(neg_indices)

            Aneg_emb = np.zeros((M, model.emb_dim), dtype=np.float32)
            Aneg_pool = np.zeros((M, model.d_model), dtype=np.float32)
            Aneg_ids_list = []

            for j, i in enumerate(neg_indices):
                ne, npool, nids = encode_pooled(neg_texts[i])
                Aneg_emb[j] = ne
                Aneg_pool[j] = npool
                Aneg_ids_list.append(nids)

            if M > 0:
                Aneg_emb = l2norm_rows(Aneg_emb)
                A_ext = np.vstack([Apos_emb, Aneg_emb]).astype(np.float32)  # (B+M, emb_dim)
            else:
                A_ext = Apos_emb

            # logits: (B, B+M)
            S = Q_emb @ A_ext.T
            logits = S / tau

            y = np.arange(B, dtype=np.int32)
            P = softmax(logits, axis=1)
            loss = -np.log(P[np.arange(B), y] + 1e-9).mean()

            epoch_loss += float(loss) * B
            n_seen += B

            # grads
            G = P.copy()
            G[np.arange(B), y] -= 1.0
            G /= B
            dS = G / tau

            dQ = dS @ A_ext  # (B,emb_dim)
            dAext = dS.T @ Q_emb  # (B+M,emb_dim)

            dApos = dAext[:B]
            dAneg = dAext[B:] if M > 0 else None

            # tangent projection for normalization
            # emb = l2norm(Z), so dZ = dEmb - (dEmb·emb)*emb
            def tangent(dE: np.ndarray, E: np.ndarray) -> np.ndarray:
                return dE - (np.sum(dE * E, axis=1, keepdims=True) * E)

            dZq = tangent(dQ, Q_emb)
            dZa_pos = tangent(dApos, Apos_emb)
            if M > 0:
                dZa_neg = tangent(dAneg, Aneg_emb)
            else:
                dZa_neg = None

            # update projection head
            gW = (Q_pool.T @ dZq) + (Apos_pool.T @ dZa_pos)
            gb = dZq.sum(axis=0) + dZa_pos.sum(axis=0)

            if M > 0:
                gW += (Aneg_pool.T @ dZa_neg)
                gb += dZa_neg.sum(axis=0)

            adam_step("W_proj", gW.astype(np.float32), args.lr)
            adam_step("b_proj", gb.astype(np.float32), args.lr)

            # pool gate shaping
            push = float(loss)
            gw_pool = ((Q_pool.mean(axis=0) + Apos_pool.mean(axis=0)) * min(1.0, push)).astype(np.float32) / 2.0
            gb_pool = np.array([min(1.0, push)], dtype=np.float32)

            adam_step("w_pool", gw_pool, args.lr_pool)
            adam_step("b_pool", gb_pool, args.lr_pool)

            # attention gate biases: small nudge
            g = np.array([min(0.05, push * 0.02)], dtype=np.float32)
            for l in range(model.n_layers):
                adam_step(f"b_gate_g_{l}", g, args.lr_gate)
                adam_step(f"b_gate_h_{l}", np.full((model.n_heads,), float(g[0]), dtype=np.float32), args.lr_gate)

            # embedding row updates (subset) using pooled proxy
            if trainable_emb_ids:
                dPoolQ = dZq @ params["W_proj"].T
                dPoolApos = dZa_pos @ params["W_proj"].T

                if M > 0:
                    dPoolAneg = dZa_neg @ params["W_proj"].T
                else:
                    dPoolAneg = None

                id_to_row = {tid: r for r, tid in enumerate(trainable_emb_ids)}
                gEmb = np.zeros((len(trainable_emb_ids), model.d_model), dtype=np.float32)

                for i in range(B):
                    for tid in Q_ids_list[i]:
                        tid = int(tid)
                        if tid in trainable_emb_ids_set:
                            gEmb[id_to_row[tid]] += dPoolQ[i] / max(1, len(Q_ids_list[i]))
                    for tid in Apos_ids_list[i]:
                        tid = int(tid)
                        if tid in trainable_emb_ids_set:
                            gEmb[id_to_row[tid]] += dPoolApos[i] / max(1, len(Apos_ids_list[i]))

                if M > 0:
                    for j, i_orig in enumerate(neg_indices):
                        for tid in Aneg_ids_list[j]:
                            tid = int(tid)
                            if tid in trainable_emb_ids_set:
                                gEmb[id_to_row[tid]] += dPoolAneg[j] / max(1, len(Aneg_ids_list[j]))

                # Adam update on those rows (separate state)
                emb_m[:] = beta1 * emb_m + (1 - beta1) * gEmb
                emb_v[:] = beta2 * emb_v + (1 - beta2) * (gEmb * gEmb)

                # stable step count (doesn't have to be perfect)
                emb_t = epoch * 100000 + step + 1
                emb_mhat = emb_m / (1 - beta1 ** emb_t)
                emb_vhat = emb_v / (1 - beta2 ** emb_t)
                delta = args.lr_emb * emb_mhat / (np.sqrt(emb_vhat) + 1e-8)

                for r, tid in enumerate(trainable_emb_ids):
                    model.W_emb[tid] -= delta[r].astype(np.float32)

            # sync scalar
            model.b_pool = float(params["b_pool"][0])

            if args.print_steps and (step % args.print_every == 0):
                r1 = valid_recall_at_1(n=args.valid_probe)
                print(f"[epoch {epoch:02d} step {step:05d} gstep {global_step:07d}] "
                      f"loss={loss:.4f} hardM={M}/{B} valid_recall@1={r1:.3f}")

        avg = epoch_loss / max(1, n_seen)
        r1 = valid_recall_at_1(n=400)
        print(f"[epoch {epoch:02d}] avg_loss={avg:.4f} valid_recall@1={r1:.3f}")

    out_dir = Path(args.out_dir)
    model.save(out_dir, name=args.name)
    print(f"[save] -> {out_dir}/{args.name}_weights.npz (+config+vocab)")
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--name", default="gdm_sbert_v2_1_hardneg")
    ap.add_argument("--seed", type=int, default=7)

    # model dims
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_vocab", type=int, default=12000)
    ap.add_argument("--d_model", type=int, default=160)
    ap.add_argument("--n_heads", type=int, default=5)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=192)

    # contrastive data
    ap.add_argument("--train_pairs", type=int, default=45000)
    ap.add_argument("--valid_pairs", type=int, default=3000)

    # training
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.14)

    # learning rates
    ap.add_argument("--lr", type=float, default=1.5e-3)       # W_proj, b_proj
    ap.add_argument("--lr_pool", type=float, default=8e-4)    # pooling gate
    ap.add_argument("--lr_gate", type=float, default=4e-4)    # attention gate biases
    ap.add_argument("--lr_emb", type=float, default=9e-4)     # embedding rows

    # embedding subset rows to train (DEFAULT: 8000 as requested)
    ap.add_argument("--train_embed_rows", type=int, default=8000)

    # hard-negative mining controls
    ap.add_argument("--p_hard", type=float, default=0.85, help="probability to add one mined hard neg per query")
    ap.add_argument("--hard_topk", type=int, default=60, help="top-k bank candidates to consider as hard neg")
    ap.add_argument("--mine_refresh_steps", type=int, default=60, help="rebuild bank embeddings every N steps")
    ap.add_argument("--bank_max_sents", type=int, default=3500, help="cap sentence bank size for mining speed")
    ap.add_argument("--bank_meta_stride", type=int, default=20, help="bucket size to avoid nearby negatives")

    # logging
    ap.add_argument("--print_steps", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--valid_probe", type=int, default=200)

    args = ap.parse_args()
    train(args)
