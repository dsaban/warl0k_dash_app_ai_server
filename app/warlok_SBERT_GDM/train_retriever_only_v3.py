from __future__ import annotations

import argparse, random, re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A, softmax


# ----------------------------
# Text utils
# ----------------------------

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")

def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        if 60 <= len(s) <= 360 and "http" not in s.lower():
            out.append(s)
    return out

def build_vocab(texts: List[str], max_vocab: int = 12000) -> Dict[str,int]:
    freq: Dict[str,int] = {}
    for t in texts:
        for w in t.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%','-'])
            if not w:
                continue
            freq[w] = freq.get(w, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    for w,_ in items[:max(0, max_vocab-2)]:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


# ----------------------------
# Evidence bank filtering (light but helpful)
# ----------------------------

EVIDENCE_TERMS = [
    "ogtt","75","24","28","fasting","hba1c","threshold","diagnos",
    "postpartum","6","12","follow-up","type 2","surveillance",
    "insulin","metformin","lifestyle","diet","exercise","monitor","targets","smbg","cgm",
    "macrosomia","hypoglyc","preeclamps","shoulder","nicu","polycyth","hyperbilirub",
    "placent","insulin resistance","beta","hyperglyc","fetal insulin",
]

def is_evidence_sentence(s: str) -> bool:
    t = s.lower()
    if len(t) < 70:
        return False
    hits = 0
    for w in EVIDENCE_TERMS:
        if w in t:
            hits += 1
            if hits >= 1:
                return True
    return False


# ----------------------------
# Masked positives (aligned)
# ----------------------------

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
    for i,w in enumerate(toks):
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
    k = min(4, max(2, len(cand)//7))
    mask_idx = set(cand[:k])
    masked = ["[MASK]" if i in mask_idx else w for i,w in enumerate(toks)]
    return "Fill the missing clinical details: " + " ".join(masked)

def make_pairs(evidence_sents: List[str], n_pairs: int, seed: int) -> List[Tuple[str,str]]:
    rng = random.Random(seed)
    out = []
    for i in range(n_pairs):
        a = rng.choice(evidence_sents)
        q = mask_sentence_to_question(a, seed=seed*100000 + i)
        out.append((q, a))
    return out


# ----------------------------
# Helpers
# ----------------------------

def l2norm_rows(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

def batch_iter(pairs: List[Tuple[str,str]], bs: int, seed: int):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [pairs[j] for j in idx[i:i+bs]]

def top_vocab_ids_by_freq(vocab: Dict[str,int], texts: List[str], top_k: int) -> List[int]:
    freq = np.zeros((len(vocab),), dtype=np.int32)
    for t in texts:
        for w in t.lower().split():
            w = ''.join(ch for ch in w if ch.isalnum() or ch in ['%','-'])
            if not w:
                continue
            freq[vocab.get(w, 1)] += 1
    freq[0] = 0
    freq[1] = 0
    ids = np.argsort(-freq)[:top_k]
    return [int(i) for i in ids if freq[i] > 0]


# ----------------------------
# Optional hard-negative miner
# ----------------------------

class HardNegMiner:
    def __init__(self, bank_sents: List[str], top_k: int = 25):
        self.bank_sents = bank_sents
        self.top_k = top_k
        self.bank_embs: Optional[np.ndarray] = None

    def rebuild(self, model: GdmSentenceTransformerV2A, print_every: int = 800):
        embs = []
        for i,s in enumerate(self.bank_sents):
            if print_every and i and (i % print_every == 0):
                print(f"[mine] bank embed {i}/{len(self.bank_sents)}")
            embs.append(model.embed(s).astype(np.float32))
        self.bank_embs = l2norm_rows(np.vstack(embs).astype(np.float32))
        print(f"[mine] bank ready: {self.bank_embs.shape[0]}")

    def mine(self, q_emb: np.ndarray, a_pos: str) -> str:
        sims = self.bank_embs @ q_emb
        k = min(self.top_k, sims.shape[0])
        idx = np.argpartition(-sims, kth=k-1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        for j in idx:
            s = self.bank_sents[int(j)]
            if s != a_pos:
                return s
        return self.bank_sents[int(idx[0])]


# ----------------------------
# Retrieval eval: Recall@K against bank
# ----------------------------

def recall_at_k(model: GdmSentenceTransformerV2A,
                pairs: List[Tuple[str,str]],
                bank_sents: List[str],
                K: int,
                max_eval: int,
                bank_embs_cache: Optional[np.ndarray] = None) -> Tuple[float, np.ndarray]:
    sample = pairs if len(pairs) <= max_eval else random.sample(pairs, k=max_eval)

    # build bank embeddings once (or reuse)
    if bank_embs_cache is None:
        bank_embs = []
        for s in bank_sents:
            bank_embs.append(model.embed(s).astype(np.float32))
        bank_embs = l2norm_rows(np.vstack(bank_embs).astype(np.float32))
    else:
        bank_embs = bank_embs_cache

    # map sentence->indices (handle duplicates)
    sent_to_idx: Dict[str, List[int]] = {}
    for i,s in enumerate(bank_sents):
        sent_to_idx.setdefault(s, []).append(i)

    hits = 0
    for (q, a_true) in sample:
        qv = model.embed(q).astype(np.float32)
        qv = qv / (np.linalg.norm(qv) + 1e-9)
        sims = bank_embs @ qv
        topk = np.argpartition(-sims, kth=min(K, sims.shape[0]-1))[:K]
        topk_set = set(int(x) for x in topk)
        true_idx_list = sent_to_idx.get(a_true, [])
        if any(i in topk_set for i in true_idx_list):
            hits += 1

    return hits / max(1, len(sample)), bank_embs


# ----------------------------
# Training (retriever-only)
# ----------------------------

def train(args):
    print("[load] docs...")
    docs = [Path(x) for x in args.docs]
    doc_texts = []
    all_sents = []
    for p in docs:
        txt = read_text(p)
        doc_texts.append(txt)
        ss = split_sentences(txt)
        all_sents.extend(ss)
        print(f"  - {p.name}: {len(ss)} sentences")
    print(f"[sentences] total={len(all_sents)}")

    # split (avoid leakage)
    rng = random.Random(args.seed)
    idx = list(range(len(all_sents)))
    rng.shuffle(idx)
    cut = int(0.85 * len(idx))
    train_sents = [all_sents[i] for i in idx[:cut]]
    valid_sents = [all_sents[i] for i in idx[cut:]]

    train_ev = [s for s in train_sents if is_evidence_sentence(s)]
    valid_ev = [s for s in valid_sents if is_evidence_sentence(s)]
    if len(train_ev) < 800:
        print("[warn] evidence filter too small; fallback to train_sents.")
        train_ev = train_sents[:]
    if len(valid_ev) < 200:
        valid_ev = valid_sents[:]

    print(f"[evidence] train_ev={len(train_ev)} valid_ev={len(valid_ev)}")

    # bank used for evaluation + mining
    bank_sents = train_ev[:]
    rng.shuffle(bank_sents)
    bank_sents = bank_sents[: args.bank_max_sents]
    print(f"[bank] eval/mine bank size={len(bank_sents)}")

    vocab = build_vocab(doc_texts + train_ev + valid_ev, max_vocab=args.max_vocab)
    print(f"[vocab] size={len(vocab)}")

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

    # data
    train_pairs = make_pairs(train_ev, args.train_pairs, seed=args.seed + 10)
    valid_pairs = make_pairs(valid_ev, args.valid_pairs, seed=args.seed + 20)
    print(f"[pairs] train={len(train_pairs)} valid={len(valid_pairs)}")

    # trainable embedding rows
    trainable_ids = top_vocab_ids_by_freq(vocab, train_ev, top_k=args.train_embed_rows)
    trainable_set = set(trainable_ids)
    id_to_row = {tid: r for r, tid in enumerate(trainable_ids)}
    print(f"[emb] trainable rows={len(trainable_ids)} (requested={args.train_embed_rows})")

    # params to update (NO gate updates!)
    params = {
        "W_proj": model.W_proj,
        "b_proj": model.b_proj,
    }
    if args.train_pool:
        params["w_pool"] = model.w_pool
        params["b_pool"] = np.array([model.b_pool], dtype=np.float32)

    # Adam state
    beta1, beta2 = 0.9, 0.999
    tstep = 0
    m = {k: np.zeros_like(v, dtype=np.float32) for k,v in params.items()}
    v = {k: np.zeros_like(vv, dtype=np.float32) for k,vv in params.items()}

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

    # miner
    miner = HardNegMiner(bank_sents, top_k=args.hard_topk) if args.p_hard > 0 else None
    bank_embs_cache = None
    if miner is not None:
        miner.rebuild(model, print_every=800)
        bank_embs_cache = miner.bank_embs

    # baseline eval
    r10, bank_embs_cache = recall_at_k(model, valid_pairs, bank_sents, K=10,
                                       max_eval=args.eval_max, bank_embs_cache=bank_embs_cache)
    print(f"[eval:before] recall@10={r10:.3f}")

    tau = args.tau
    best = r10
    no_imp = 0

    print("[train] retriever-only contrastive (no gate training, no hint metrics)")
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        seen = 0

        for step, batch in enumerate(batch_iter(train_pairs, args.batch, seed=args.seed + epoch)):
            B = len(batch)
            if B < 2:
                continue
            global_step += 1

            # refresh miner bank embeddings occasionally
            if miner is not None and (global_step % args.mine_refresh_steps == 0):
                print(f"[mine] refresh @gstep={global_step}")
                miner.rebuild(model, print_every=800)
                bank_embs_cache = miner.bank_embs

            # encode batch
            Q_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            Apos_emb = np.zeros((B, model.emb_dim), dtype=np.float32)
            Q_pool = np.zeros((B, model.d_model), dtype=np.float32)
            Apos_pool = np.zeros((B, model.d_model), dtype=np.float32)
            Q_ids = []
            A_ids = []
            a_texts = []

            for i, (q, a) in enumerate(batch):
                qe, qp, qids = encode_pooled(q)
                ae, ap, aids = encode_pooled(a)
                Q_emb[i] = qe
                Apos_emb[i] = ae
                Q_pool[i] = qp
                Apos_pool[i] = ap
                Q_ids.append(qids)
                A_ids.append(aids)
                a_texts.append(a)

            Q_emb = l2norm_rows(Q_emb)
            Apos_emb = l2norm_rows(Apos_emb)

            # add mined negatives (optional)
            neg_idx = []
            neg_texts = []
            if miner is not None and args.p_hard > 0:
                mask = (np.random.random(B) < args.p_hard)
                for i in range(B):
                    if mask[i]:
                        neg_idx.append(i)
                        neg_texts.append(miner.mine(Q_emb[i], a_texts[i]))

            M = len(neg_idx)
            if M > 0:
                Aneg_emb = np.zeros((M, model.emb_dim), dtype=np.float32)
                Aneg_pool = np.zeros((M, model.d_model), dtype=np.float32)
                Aneg_ids = []
                for j, i0 in enumerate(neg_idx):
                    ne, npool, nids = encode_pooled(neg_texts[j])
                    Aneg_emb[j] = ne
                    Aneg_pool[j] = npool
                    Aneg_ids.append(nids)
                Aneg_emb = l2norm_rows(Aneg_emb)
                A_ext = np.vstack([Apos_emb, Aneg_emb]).astype(np.float32)
            else:
                Aneg_pool = None
                Aneg_ids = None
                A_ext = Apos_emb

            # InfoNCE (targets are diagonal in first B columns)
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

            # projection update
            gW = (Q_pool.T @ dZq) + (Apos_pool.T @ dZa_pos)
            gb = dZq.sum(axis=0) + dZa_pos.sum(axis=0)
            if M > 0:
                gW += (Aneg_pool.T @ dZa_neg)
                gb += dZa_neg.sum(axis=0)

            adam_step("W_proj", gW.astype(np.float32), args.lr_proj)
            adam_step("b_proj", gb.astype(np.float32), args.lr_proj)

            # optional pool update
            if args.train_pool:
                push = float(loss)
                gw_pool = ((Q_pool.mean(axis=0) + Apos_pool.mean(axis=0)) * min(1.0, push)).astype(np.float32) / 2.0
                gb_pool = np.array([min(1.0, push)], dtype=np.float32)
                adam_step("w_pool", gw_pool, args.lr_pool)
                adam_step("b_pool", gb_pool, args.lr_pool)
                model.b_pool = float(params["b_pool"][0])

            # embedding subset update via pooled proxy
            dPoolQ = dZq @ params["W_proj"].T
            dPoolA = dZa_pos @ params["W_proj"].T
            if M > 0:
                dPoolNeg = dZa_neg @ params["W_proj"].T
            else:
                dPoolNeg = None

            gEmb = np.zeros((len(trainable_ids), model.d_model), dtype=np.float32)

            for i in range(B):
                for tid in Q_ids[i]:
                    tid = int(tid)
                    if tid in trainable_set:
                        gEmb[id_to_row[tid]] += dPoolQ[i] / max(1, len(Q_ids[i]))
                for tid in A_ids[i]:
                    tid = int(tid)
                    if tid in trainable_set:
                        gEmb[id_to_row[tid]] += dPoolA[i] / max(1, len(A_ids[i]))

            if M > 0:
                for j in range(M):
                    for tid in Aneg_ids[j]:
                        tid = int(tid)
                        if tid in trainable_set:
                            gEmb[id_to_row[tid]] += dPoolNeg[j] / max(1, len(Aneg_ids[j]))

            emb_m[:] = beta1 * emb_m + (1 - beta1) * gEmb
            emb_v[:] = beta2 * emb_v + (1 - beta2) * (gEmb * gEmb)
            emb_t = epoch * 100000 + step + 1
            mhat = emb_m / (1 - beta1 ** emb_t)
            vhat = emb_v / (1 - beta2 ** emb_t)
            delta = args.lr_emb * mhat / (np.sqrt(vhat) + 1e-8)

            for r, tid in enumerate(trainable_ids):
                model.W_emb[tid] -= delta[r].astype(np.float32)

            if args.print_steps and (step % args.print_every == 0):
                print(f"[epoch {epoch:02d} step {step:05d}] loss={loss:.4f} mined={M}")

        avg_loss = total_loss / max(1, seen)

        # eval
        r10, bank_embs_cache = recall_at_k(model, valid_pairs, bank_sents, K=10,
                                           max_eval=args.eval_max, bank_embs_cache=bank_embs_cache)
        print(f"[epoch {epoch:02d}] avg_loss={avg_loss:.4f}  recall@10={r10:.3f}")

        # early stop
        if r10 > best + 1e-4:
            best = r10
            no_imp = 0
            # quick save best
            model.save(Path(args.out_dir), name=args.name + "_best")
            print(f"[save] best -> {args.out_dir}/{args.name}_best_weights.npz")
        else:
            no_imp += 1
            if no_imp >= args.early_stop_patience:
                print("[early-stop] no improvement")
                break

    # final save
    model.save(Path(args.out_dir), name=args.name)
    print(f"[save] final -> {args.out_dir}/{args.name}_weights.npz")
    print("[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--name", default="gdm_retriever_only_v3")
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
    ap.add_argument("--train_pairs", type=int, default=20000)
    ap.add_argument("--valid_pairs", type=int, default=2500)
    ap.add_argument("--bank_max_sents", type=int, default=4500)

    # training
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.10)

    # updates (retriever only)
    ap.add_argument("--train_embed_rows", type=int, default=8000)
    ap.add_argument("--lr_proj", type=float, default=1.2e-3)
    ap.add_argument("--lr_emb", type=float, default=5e-4)

    # optional pooling update (often helps long phrases, but safe to disable)
    ap.add_argument("--train_pool", action="store_true")
    ap.add_argument("--lr_pool", type=float, default=6e-4)

    # hard-negative mining (stable defaults; set p_hard=0 to disable)
    ap.add_argument("--p_hard", type=float, default=0.45)
    ap.add_argument("--hard_topk", type=int, default=25)
    ap.add_argument("--mine_refresh_steps", type=int, default=200)

    # eval / early stop
    ap.add_argument("--eval_max", type=int, default=400)
    ap.add_argument("--early_stop_patience", type=int, default=3)

    # logging
    ap.add_argument("--print_steps", action="store_true")
    ap.add_argument("--print_every", type=int, default=80)

    args = ap.parse_args()
    train(args)
