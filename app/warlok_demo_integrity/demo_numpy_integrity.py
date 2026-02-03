
import json, re
from pathlib import Path
import numpy as np

CLAIMS_PATH = Path("claims_200.jsonl")

def hash_ngrams(text, n=3, dim=2048):
    text = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    toks = [t for t in text.split() if t]
    v = np.zeros(dim, dtype=np.float32)
    if len(toks) < n:
        toks = toks + [""]*(n-len(toks))
    for i in range(len(toks)-n+1):
        ng = " ".join(toks[i:i+n])
        v[hash(ng) % dim] += 1.0
    for t in toks:
        v[hash(t) % dim] += 0.3
    return v / (np.linalg.norm(v) + 1e-9)

def extract_numbers_units(text):
    nums=[]
    for m in re.finditer(r'(\d{2,3})\s*mg/dL', text, re.I):
        nums.append(("mg/dL", int(m.group(1))))
    for m in re.finditer(r'(\d{1,2})\s*[-–]\s*(\d{1,2})\s*weeks', text, re.I):
        nums.append(("weeks_range", (int(m.group(1)), int(m.group(2)))))
    for m in re.finditer(r'(\d{1,2})\s*weeks', text, re.I):
        if re.search(r'[-–]\s*\d{1,2}\s*weeks', text[m.start()-3:m.end()+10], re.I):
            continue
        nums.append(("weeks", int(m.group(1))))
    return nums

def load_claims():
    claims=[]
    with CLAIMS_PATH.open() as f:
        for line in f:
            claims.append(json.loads(line))
    return claims

def build_matrix(claims):
    C = np.stack([hash_ngrams(c["text"]) for c in claims])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    return C

def retrieve(claims, C, question, k=6):
    q = hash_ngrams(question)
    sims = C @ q
    idx = np.argsort(-sims)[:k]
    return [claims[i] for i in idx], sims[idx]

def compose_answer(claims_k, max_sentences=3):
    kind_order = {"atom":0,"rel_atom":1,"sentence":2,"gen_atom":3}
    uniq=[]
    seen=set()
    for c in sorted(claims_k, key=lambda x:(kind_order.get(x["kind"],9), len(x["text"]))):
        t=c["text"].strip()
        if t.lower() in seen:
            continue
        seen.add(t.lower())
        uniq.append(t)
        if len(uniq) >= max_sentences:
            break
    out=[]
    for t in uniq:
        if not t.endswith(('.', '!', '?')):
            t += '.'
        out.append(t)
    return " ".join(out)

def best_claim_for_sentence(claims, C, sent):
    v = hash_ngrams(sent)
    sims = C @ v
    i = int(np.argmax(sims))
    return claims[i], float(sims[i])

def classify_integrity(claims, C, question, answer, support_thr=0.55):
    sents = [s for s in re.split(r'(?<=[\.\?\!])\s+', answer.strip()) if len(s)>10]
    if not sents:
        return "bad"
    unsupported=0
    contradictory=0
    for s in sents:
        claim, sim = best_claim_for_sentence(claims, C, s)
        if sim < support_thr:
            unsupported += 1
            continue
        a_nums = extract_numbers_units(s)
        c_nums = extract_numbers_units(claim.get("evidence","") + " " + claim.get("text",""))
        if a_nums and c_nums:
            for typ,val in a_nums:
                c_vals=[v for t,v in c_nums if t==typ]
                if c_vals and val not in c_vals:
                    contradictory += 1
                    break
    if unsupported/len(sents) >= 0.34:
        return "bad"
    if contradictory > 0:
        return "contradictory"
    if len(sents) > 3:
        return "entangled"
    return "good" if unsupported==0 else "neutral"

if __name__ == "__main__":
    claims = load_claims()
    C = build_matrix(claims)

    q = input("Question: ").strip()
    top, sims = retrieve(claims, C, q, k=3)
    ans = compose_answer(top, max_sentences=3)
    label = classify_integrity(claims, C, q, ans)

    print("\n--- Evidence-locked answer ---")
    print(ans)
    print("\nIntegrity label:", label)
    print("\nTop evidence claims:")
    for c, s in zip(top, sims):
        print(f"- {c['claim_id']} ({s:.3f}) [{c['doc']} | {c['kind']}]: {c['text']}")
