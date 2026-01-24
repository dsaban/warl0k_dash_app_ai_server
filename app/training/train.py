
import os, json
from core.pipeline import GDMPipeline, DEFAULT_WEIGHTS

def load_trainset(root):
    with open(f"{root}/data/trainset.json","r",encoding="utf-8") as f:
        return json.load(f)

def featurize_path(path):
    f={}
    for n in path:
        f[f"path_has::{n}"]=1.0
    f["path_len"]=float(len(path))
    return f

def train(root, epochs=25, lr=0.2):
    w=dict(DEFAULT_WEIGHTS)
    ds=load_trainset(root)
    # ensure trainable keys exist
    for row in ds:
        for n in row["gold_path"]:
            w.setdefault(f"path_has::{n}", 0.05)

    def score(path):
        feats=featurize_path(path)
        return sum(w.get(k,0.0)*v for k,v in feats.items())

    updates=0
    for _ in range(epochs):
        for row in ds:
            gold=row["gold_path"]
            s_gold=score(gold)
            for bad in row.get("bad_paths",[]):
                s_bad=score(bad)
                margin=0.5
                if s_gold <= s_bad + margin:
                    for k,v in featurize_path(gold).items():
                        w[k]=w.get(k,0.0)+lr*v
                    for k,v in featurize_path(bad).items():
                        w[k]=w.get(k,0.0)-lr*v
                    updates += 1
                    s_gold=score(gold)

    os.makedirs(f"{root}/artifacts", exist_ok=True)
    with open(f"{root}/artifacts/trained_weights.json","w",encoding="utf-8") as f:
        json.dump(w,f,indent=2)
    return w, updates

if __name__=="__main__":
    root=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    w, updates = train(root)
    print("updates=",updates)
