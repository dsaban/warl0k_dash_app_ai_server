from __future__ import annotations

import argparse, random, re
from pathlib import Path
from typing import List, Dict
import numpy as np

from src.model_v2a import GdmSentenceTransformerV2A, LABELS, LAB2ID, softmax


# ---------- data utils ----------

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore")


def split_sentences(txt: str) -> List[str]:
    txt = re.sub(r"\s+", " ", txt).strip()
    parts = re.split(r"(?<=[\.\?\!])\s+", txt)
    out = []
    for s in parts:
        s = s.strip()
        if 60 <= len(s) <= 320 and "http" not in s.lower():
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
        "What timing is used for diagnosing gestational diabetes and what test is commonly used?",
        "How is early pregnancy testing performed for hyperglycaemia in pregnancy?"
    ],
    "management": [
        "What is first-line management for gestational diabetes, and when is insulin started?",
        "What role does metformin have in gestational diabetes management?",
        "How should blood glucose be monitored during gestational diabetes?"
    ],
    "complications": [
        "How does maternal hyperglycaemia contribute to fetal macrosomia and neonatal hypoglycaemia?",
        "What maternal and neonatal complications are associated with gestational diabetes?",
        "Why does gestational diabetes increase risk of difficult delivery?"
    ],
    "postpartum": [
        "What postpartum follow-up testing is recommended after gestational diabetes?",
        "Why is lifelong surveillance recommended after a gestational diabetes pregnancy?",
        "What is the risk of progression to type 2 diabetes after gestational diabetes?"
    ],
    "pathophysiology": [
        "Why does insulin resistance increase during pregnancy, and when does it become gestational diabetes?",
        "How do placental hormones contribute to insulin resistance in pregnancy?",
        "What is the role of beta-cell compensation in gestational diabetes?"
    ],
    "general": [
        "Summarize the key clinical facts relevant to this statement.",
        "Explain the mechanism and clinical implications of the following.",
        "What does evidence suggest about this topic?"
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


def make_contradiction(sentence: str) -> str:
    s = sentence
    flips = [
        (r"\brecommended\b", "not recommended"),
        (r"\bnot recommended\b", "recommended"),
        (r"\bincreases\b", "decreases"),
        (r"\bdecreases\b", "increases"),
        (r"\bhigher\b", "lower"),
        (r"\blower\b", "higher"),
        (r"\bshould\b", "should not"),
        (r"\bis\b", "is not"),
        (r"\bare\b", "are not"),
    ]
    applied = 0
    for pat, rep in flips:
        if re.search(pat, s, flags=re.IGNORECASE):
            s2 = re.sub(pat, rep, s, count=1, flags=re.IGNORECASE)
            if s2 != s:
                s = s2
                applied += 1
                if applied >= 2:
                    break
    return s


def make_positive_qa(topic: str) -> str:
    if topic == "screening":
        return "Is a 75 g 2-hour OGTT commonly used to diagnose gestational diabetes around 24–28 weeks?"
    if topic == "postpartum":
        return "After gestational diabetes, is postpartum OGTT follow-up recommended within about 6–12 weeks?"
    if topic == "management":
        return "Is lifestyle modification the first-line approach for gestational diabetes management?"
    if topic == "complications":
        return "Does maternal hyperglycaemia increase risk of fetal macrosomia and neonatal hypoglycaemia?"
    return "Is this statement supported by the clinical evidence described?"


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


def gen_dataset(sent_by_topic: Dict[str, List[str]],
                n_total: int,
                hard_neutral_frac: float,
                seed: int) -> List[dict]:
    random.seed(seed)
    topics = [t for t in sent_by_topic.keys() if sent_by_topic[t]]

    def rt():
        return random.choice(topics)

    per = n_total // 4
    out = []

    # entailment
    while sum(1 for d in out if d["label"] == "entailment") < per:
        t = rt()
        a = random.choice(sent_by_topic[t])
        q = make_question(t)
        out.append({"q": q, "a": a, "label": "entailment", "topic": t})

    # neutral (with hard-neutral)
    while sum(1 for d in out if d["label"] == "neutral") < per:
        t1 = rt()
        q = make_question(t1)
        if len(topics) > 1:
            t2 = random.choice([t for t in topics if t != t1])
        else:
            t2 = t1
        a = random.choice(sent_by_topic[t2])

        # hard-neutral: inject overlap token sometimes
        if random.random() < hard_neutral_frac:
            q2 = q + " OGTT insulin postpartum"  # keyword clutter
            out.append({"q": q2, "a": a, "label": "neutral", "topic": f"{t1}->{t2}(hard)"})
        else:
            out.append({"q": q, "a": a, "label": "neutral", "topic": f"{t1}->{t2}"})

    # contradiction
    while sum(1 for d in out if d["label"] == "contradiction") < per:
        t = rt()
        base = random.choice(sent_by_topic[t])
        q = make_question(t)
        a = make_contradiction(base)
        out.append({"q": q, "a": a, "label": "contradiction", "topic": t})

    # positive_qa
    while sum(1 for d in out if d["label"] == "positive_qa") < per:
        t = rt()
        a = random.choice(sent_by_topic[t])
        q = make_positive_qa(t)
        out.append({"q": q, "a": a, "label": "positive_qa", "topic": t})

    random.shuffle(out)
    return out[:n_total]


def mini_eval_acc(model: GdmSentenceTransformerV2A, data: List[dict], n: int = 200) -> float:
    if not data:
        return 0.0
    sample = random.sample(data, k=min(n, len(data)))
    ok = 0
    for ex in sample:
        pred, _ = model.predict_relation(ex["q"], ex["a"])
        ok += int(pred == ex["label"])
    return ok / len(sample)


# ---------- training (pooled-level analytic grads) ----------

def train(args):
    print("[load] reading docs...")
    doc_texts = []
    sentences = []
    for p in [Path(x) for x in args.docs]:
        txt = read_text(p)
        doc_texts.append(txt)
        ss = split_sentences(txt)
        sentences += ss
        print(f"  - {p.name}: {len(ss)} sentences")

    print(f"[docs] total sentences={len(sentences)}")

    # sentence bank split (leakage control)
    idx = list(range(len(sentences)))
    random.seed(args.seed)
    random.shuffle(idx)
    cut = int(0.8 * len(idx))
    train_sents = [sentences[i] for i in idx[:cut]]
    valid_sents = [sentences[i] for i in idx[cut:]]
    print(f"[split] train_sents={len(train_sents)} valid_sents={len(valid_sents)}")

    # topic buckets
    tr_by_topic: Dict[str, List[str]] = {}
    for s in train_sents:
        tr_by_topic.setdefault(pick_topic(s), []).append(s)
    tr_by_topic.setdefault("general", train_sents)

    va_by_topic: Dict[str, List[str]] = {}
    for s in valid_sents:
        va_by_topic.setdefault(pick_topic(s), []).append(s)
    va_by_topic.setdefault("general", valid_sents)

    # vocab
    vocab = build_vocab(doc_texts + train_sents + valid_sents, max_vocab=args.max_vocab)
    print(f"[vocab] size={len(vocab)} cap={args.max_vocab}")

    # model init
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

    # datasets
    train_data = gen_dataset(tr_by_topic, args.train_samples, args.hard_neutral_frac, seed=args.seed + 11)
    valid_data = gen_dataset(va_by_topic, args.valid_samples, args.hard_neutral_frac, seed=args.seed + 22)
    print(f"[data] train={len(train_data)} valid={len(valid_data)} labels={LABELS}")

    # params to update (fast subset)
    # heads + pooling gate + attention gate biases (stability)
    lr = args.lr

    # Adam state
    def adam_init(arr):
        return np.zeros_like(arr, dtype=np.float32)

    # params dict (views into model arrays)
    params = {
        "W_cls": model.W_cls,
        "b_cls": model.b_cls,
        "W_cls_gate": model.W_cls_gate,
        "b_cls_gate": model.b_cls_gate,
        "w_pool": model.w_pool,
        "b_pool": np.array([model.b_pool], dtype=np.float32),
    }
    # nudge gate biases per layer (cheap and effective)
    for l in range(model.n_layers):
        params[f"b_gate_g_{l}"] = model.b_gate_g[l]
        params[f"b_gate_h_{l}"] = model.b_gate_h[l]

    m = {k: adam_init(v) for k, v in params.items()}
    v = {k: adam_init(v) for k, v in params.items()}
    beta1, beta2 = 0.9, 0.999
    tstep = 0

    def adam_step(name: str, grad: np.ndarray):
        nonlocal tstep
        tstep += 1
        m[name] = beta1 * m[name] + (1 - beta1) * grad
        v[name] = beta2 * v[name] + (1 - beta2) * (grad * grad)
        mhat = m[name] / (1 - beta1 ** tstep)
        vhat = v[name] / (1 - beta2 ** tstep)
        params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)

    def batch_iter(data: List[dict], bs: int):
        ids = list(range(len(data)))
        random.shuffle(ids)
        for i in range(0, len(ids), bs):
            yield [data[j] for j in ids[i:i + bs]]

    for epoch in range(1, args.epochs + 1):
        total_ce = 0.0
        total_n = 0

        for step, batch in enumerate(batch_iter(train_data, args.batch)):
            # forward
            pooled = []
            logits = []
            y = []

            for ex in batch:
                pair = ex["q"] + " [SEP] " + ex["a"]
                ids = model.tokenize(pair)
                _, logit, pool = model.forward(ids)
                pooled.append(pool)
                logits.append(logit)
                y.append(LAB2ID[ex["label"]])

            pooled = np.vstack(pooled).astype(np.float32)  # (B,d)
            logits = np.vstack(logits).astype(np.float32)  # (B,C)
            y = np.array(y, dtype=np.int32)

            probs = softmax(logits, axis=1)
            ce = -np.log(probs[np.arange(y.shape[0]), y] + 1e-9).mean()

            total_ce += float(ce) * y.shape[0]
            total_n += y.shape[0]

            # analytic grads for:
            # logits = (pooled@W_cls + b_cls) * sigmoid(pooled@W_cls_gate + b_cls_gate)
            gate = 1.0 / (1.0 + np.exp(-(pooled @ params["W_cls_gate"] + params["b_cls_gate"])))
            base = pooled @ params["W_cls"] + params["b_cls"]

            grad_logits = probs.copy()
            grad_logits[np.arange(y.shape[0]), y] -= 1.0
            grad_logits /= y.shape[0]

            grad_base = grad_logits * gate
            grad_gate = grad_logits * base
            grad_z = grad_gate * gate * (1.0 - gate)

            gW_cls = pooled.T @ grad_base
            gb_cls = grad_base.sum(axis=0)
            gW_cls_gate = pooled.T @ grad_z
            gb_cls_gate = grad_z.sum(axis=0)

            adam_step("W_cls", gW_cls.astype(np.float32))
            adam_step("b_cls", gb_cls.astype(np.float32))
            adam_step("W_cls_gate", gW_cls_gate.astype(np.float32))
            adam_step("b_cls_gate", gb_cls_gate.astype(np.float32))

            # gate-shaping: push pooling + attn gate biases up when margin low
            top2 = np.argsort(-probs, axis=1)[:, :2]
            correct_prob = probs[np.arange(y.shape[0]), y]
            second_prob = probs[np.arange(y.shape[0]), top2[:, 1]]
            margin = (correct_prob - second_prob).reshape(-1, 1)
            push = np.clip(0.5 - margin, 0.0, 1.0)  # only when margin < 0.5

            gw_pool = (push * pooled).mean(axis=0)
            gb_pool = float(push.mean())

            adam_step("w_pool", gw_pool.astype(np.float32))
            adam_step("b_pool", np.array([gb_pool], dtype=np.float32))

            # attention gate biases nudged too
            g = float(push.mean())
            for l in range(model.n_layers):
                adam_step(f"b_gate_g_{l}", np.array([g], dtype=np.float32))
                adam_step(f"b_gate_h_{l}", np.full((model.n_heads,), g, dtype=np.float32))

            if args.print_steps and step % args.print_every == 0:
                acc = float((np.argmax(probs, axis=1) == y).mean())
                print(f"[epoch {epoch:02d} step {step:04d}] ce={ce:.4f} acc={acc:.3f} push={g:.4f}")

        # sync scalar
        model.b_pool = float(params["b_pool"][0])

        val_acc = mini_eval_acc(model, valid_data, n=200)
        print(f"[epoch {epoch:02d}] avg_ce={total_ce/total_n:.4f} val_acc@200={val_acc:.3f}")

    out_dir = Path(args.out_dir)
    model.save(out_dir, name=args.name)
    print(f"[save] -> {out_dir}/{args.name}_weights.npz (+config+vocab)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("--docs", nargs="+", required=True)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--name", default="gdm_sbert_v2A")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--max_vocab", type=int, default=12000)
    ap.add_argument("--d_model", type=int, default=160)
    ap.add_argument("--n_heads", type=int, default=5)
    ap.add_argument("--d_ff", type=int, default=384)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--emb_dim", type=int, default=192)

    ap.add_argument("--train_samples", type=int, default=8000)
    ap.add_argument("--valid_samples", type=int, default=1200)
    ap.add_argument("--hard_neutral_frac", type=float, default=0.30)

    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)

    ap.add_argument("--print_steps", action="store_true")
    ap.add_argument("--print_every", type=int, default=20)

    args = ap.parse_args()
    train(args)
