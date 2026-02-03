from __future__ import annotations

import argparse, random, re, math
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
    Returns list of (q, a, topic)
    Positive means: q should retrieve/support a.
    """
    random.seed(seed)
    # bucket by topic to get variety
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
# Contrastive training helpers
# ----------------------------

def cosine_sim_matrix(Q: np.ndarray, A: np.ndarray) -> np.ndarray:
    # Q: (B,d), A:(B,d) assumed L2-normalized
    return Q @ A.T  # (B,B)


def batch_iter(items: List[Tuple[str, str, str]], bs: int, seed: int):
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i+bs]]


def top_vocab_ids_by_freq(vocab: Dict[str,int], texts: List[str], top_k: int) -> List[int]:
    # choose ids to train in embedding table (fast subset)
    inv = {i:w for w,i in vocab.items()}
    freq = np.zeros((len(vocab),), dtype=np.int32)
    for t in texts:
        for w in t.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%','-'])
            if not w:
                continue
            freq[vocab.get(w, 1)] += 1
    # avoid pad/unk
    freq[0] = 0
    freq[1] = 0
    ids = np.argsort(-freq)[:top_k]
    return [int(i) for i in ids if freq[i] > 0]


# ----------------------------
# Training loop (v2.1)
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

    # choose embedding rows to update (optional but recommended)
    trainable_emb_ids: List[int] = []
    if args.train_embed_rows > 0:
        trainable_emb_ids = top_vocab_ids_by_freq(vocab, train_sents, top_k=args.train_embed_rows)
        trainable_emb_ids_set = set(trainable_emb_ids)
        print(f"[emb] trainable rows={len(trainable_emb_ids)} (top frequency tokens)")
    else:
        trainable_emb_ids_set = set()

    # Adam buffers for params we update
    params: Dict[str, np.ndarray] = {
        "W_proj": model.W_proj,
        "b_proj": model.b_proj,
        "w_pool": model.w_pool,
        "b_pool": np.array([model.b_pool], dtype=np.float32),
    }
    # gate biases (cheap)
    for l in range(model.n_layers):
        params[f"b_gate_g_{l}"] = model.b_gate_g[l]
        params[f"b_gate_h_{l}"] = model.b_gate_h[l]

    # embedding subset (store as direct view and update only selected rows)
    # We'll keep a separate Adam state for these rows.
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

    # --- Contrastive loss setup ---
    # InfoNCE: for each i, positive is A[i], negatives are A[j!=i]
    # logits = sim(Q[i], A[j]) / tau
    tau = args.tau

    def encode_pooled(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # returns (emb, pooled, token_ids)
        ids = model.tokenize(text)
        emb, _, pooled = model.forward(ids)
        return emb.astype(np.float32), pooled.astype(np.float32), ids

    def valid_recall_at_1(n: int = 200) -> float:
        # evaluate on a random subset: is the best match A[i] for Q[i]?
        if not valid_pairs:
            return 0.0
        sample = random.sample(valid_pairs, k=min(n, len(valid_pairs)))
        Q = []
        A = []
        for q, a, _ in sample:
            Q.append(model.embed(q))
            A.append(model.embed(a))
        Q = np.vstack(Q).astype(np.float32)
        A = np.vstack(A).astype(np.float32)
        Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
        S = Q @ A.T
        pred = np.argmax(S, axis=1)
        ok = np.mean(pred == np.arange(S.shape[0]))
        return float(ok)

    # ----------------------------
    # Main training
    # ----------------------------
    print("[train] starting contrastive (InfoNCE) ...")
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_seen = 0

        for step, batch in enumerate(batch_iter(train_pairs, args.batch, seed=args.seed + epoch)):
            B = len(batch)
            if B < 2:
                continue

            # forward: encode Q and A separately
            Q_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            A_emb = np.zeros((B, model.emb_dim), dtype=np.float32)

            Q_pool = np.zeros((B, model.d_model), dtype=np.float32)
            A_pool = np.zeros((B, model.d_model), dtype=np.float32)

            Q_ids_list = []
            A_ids_list = []

            for i, (q, a, _) in enumerate(batch):
                qe, qp, qids = encode_pooled(q)
                ae, ap, aids = encode_pooled(a)
                Q_emb[i] = qe
                A_emb[i] = ae
                Q_pool[i] = qp
                A_pool[i] = ap
                Q_ids_list.append(qids)
                A_ids_list.append(aids)

            # normalize embeddings
            Q_emb = Q_emb / (np.linalg.norm(Q_emb, axis=1, keepdims=True) + 1e-9)
            A_emb = A_emb / (np.linalg.norm(A_emb, axis=1, keepdims=True) + 1e-9)

            # similarity logits
            S = cosine_sim_matrix(Q_emb, A_emb)  # (B,B)
            logits = S / tau

            # targets: diagonal
            y = np.arange(B, dtype=np.int32)

            # CE over rows: p(i->j) = softmax(logits[i])
            P = softmax(logits, axis=1)
            loss = -np.log(P[np.arange(B), y] + 1e-9).mean()
            epoch_loss += float(loss) * B
            n_seen += B

            # gradient wrt logits: dL/dlogits = (P - I)/B
            G = P.copy()
            G[np.arange(B), y] -= 1.0
            G /= B  # (B,B)

            # gradient wrt S: dL/dS = G / tau
            dS = G / tau  # (B,B)

            # gradient wrt Q_emb: dL/dQ = dS @ A
            # gradient wrt A_emb: dL/dA = dS.T @ Q
            dQ = dS @ A_emb            # (B,emb_dim)
            dA = dS.T @ Q_emb          # (B,emb_dim)

            # We want to update W_proj, b_proj and gate params.
            # emb = l2norm( pooled @ W_proj + b_proj )
            # We'll approximate gradient through l2norm with a stable projection:
            # dZ = dEmb projected onto tangent space: dZ = dEmb - (dEmb·emb) * emb
            # and treat that as gradient wrt Z.
            # Z = pooled @ W_proj + b_proj

            # compute Z for Q and A
            Zq = (Q_pool @ params["W_proj"] + params["b_proj"]).astype(np.float32)
            Za = (A_pool @ params["W_proj"] + params["b_proj"]).astype(np.float32)

            # current emb (already normalized) acts as emb vectors:
            eq = Q_emb
            ea = A_emb

            # tangent projection
            dZq = dQ - (np.sum(dQ * eq, axis=1, keepdims=True) * eq)
            dZa = dA - (np.sum(dA * ea, axis=1, keepdims=True) * ea)

            # grads for W_proj and b_proj:
            gW = (Q_pool.T @ dZq) + (A_pool.T @ dZa)
            gb = dZq.sum(axis=0) + dZa.sum(axis=0)

            # scale (optional)
            gW = gW.astype(np.float32)
            gb = gb.astype(np.float32)

            adam_step("W_proj", gW, args.lr)
            adam_step("b_proj", gb, args.lr)

            # pool gate shaping: if loss high, increase pooling gate strength
            # (cheap, helps long phrases)
            push = float(loss)
            gw_pool = ((Q_pool.mean(axis=0) + A_pool.mean(axis=0)) * min(1.0, push)).astype(np.float32) / 2.0
            gb_pool = np.array([min(1.0, push)], dtype=np.float32)

            adam_step("w_pool", gw_pool, args.lr_pool)
            adam_step("b_pool", gb_pool, args.lr_pool)

            # attention gate biases: nudge slightly when loss high
            g = np.array([min(0.05, push * 0.02)], dtype=np.float32)
            for l in range(model.n_layers):
                adam_step(f"b_gate_g_{l}", g, args.lr_gate)
                adam_step(f"b_gate_h_{l}", np.full((model.n_heads,), float(g[0]), dtype=np.float32), args.lr_gate)

            # Optional: embedding row updates (subset)
            if trainable_emb_ids:
                # approximate token gradient by spreading pooled gradient to tokens:
                # pooled ~ weighted average of token states; we approximate that changing word embedding shifts pooled
                # We'll update only vocab embeddings of tokens that appear in Q/A and are in trainable list.
                # We use dPool proxy from projection grads:
                dPoolQ = dZq @ params["W_proj"].T  # (B,d_model)
                dPoolA = dZa @ params["W_proj"].T  # (B,d_model)
                dPool = (dPoolQ + dPoolA) / 2.0

                id_to_row = {tid: r for r, tid in enumerate(trainable_emb_ids)}

                # accumulate per trainable row
                gEmb = np.zeros((len(trainable_emb_ids), model.d_model), dtype=np.float32)

                for i in range(B):
                    for tid in Q_ids_list[i]:
                        tid = int(tid)
                        if tid in trainable_emb_ids_set:
                            gEmb[id_to_row[tid]] += dPool[i] / max(1, len(Q_ids_list[i]))
                    for tid in A_ids_list[i]:
                        tid = int(tid)
                        if tid in trainable_emb_ids_set:
                            gEmb[id_to_row[tid]] += dPool[i] / max(1, len(A_ids_list[i]))

                # Adam update on those rows
                # NOTE: we use separate emb_m/emb_v to not blow RAM
                emb_m[:] = beta1 * emb_m + (1 - beta1) * gEmb
                emb_v[:] = beta2 * emb_v + (1 - beta2) * (gEmb * gEmb)
                emb_mhat = emb_m / (1 - beta1 ** (epoch * 1000 + step + 1))
                emb_vhat = emb_v / (1 - beta2 ** (epoch * 1000 + step + 1))

                delta = args.lr_emb * emb_mhat / (np.sqrt(emb_vhat) + 1e-8)

                # apply to actual embedding table rows
                for r, tid in enumerate(trainable_emb_ids):
                    model.W_emb[tid] -= delta[r].astype(np.float32)

            if args.print_steps and (step % args.print_every == 0):
                # recall@1 quick on a mini valid subset
                r1 = valid_recall_at_1(n=args.valid_probe)
                print(f"[epoch {epoch:02d} step {step:05d}] loss={loss:.4f} valid_recall@1={r1:.3f}")

        model.b_pool = float(params["b_pool"][0])
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
    ap.add_argument("--name", default="gdm_sbert_v2_1")
    ap.add_argument("--seed", type=int, default=7)

    # model dims (must match v2A assumptions: d_model divisible by n_heads)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_vocab", type=int, default=12000)
    ap.add_argument("--d_model", type=int, default=160)
    ap.add_argument("--n_heads", type=int, default=5)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=192)

    # contrastive data size
    ap.add_argument("--train_pairs", type=int, default=20000)
    ap.add_argument("--valid_pairs", type=int, default=3000)

    # training
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.08)

    # learning rates (split so you can tune stability)
    ap.add_argument("--lr", type=float, default=1.5e-3)      # W_proj, b_proj
    ap.add_argument("--lr_pool", type=float, default=6e-4)   # pooling gate
    ap.add_argument("--lr_gate", type=float, default=3e-4)   # attention gate biases
    ap.add_argument("--lr_emb", type=float, default=3e-4)    # embedding subset rows

    # embedding subset rows to train (0 disables)
    ap.add_argument("--train_embed_rows", type=int, default=2500)

    # logging
    ap.add_argument("--print_steps", action="store_true")
    ap.add_argument("--print_every", type=int, default=50)
    ap.add_argument("--valid_probe", type=int, default=200)

    args = ap.parse_args()
    train(args)
