import os, re, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

_TOKEN_PAT = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_PAT.findall(text or "")]

def _build_vocab(docs_tokens: List[List[str]], min_df: int = 2, max_features: int = 60000) -> Tuple[Dict[str,int], np.ndarray]:
    # document frequency
    df = {}
    for toks in docs_tokens:
        seen=set(toks)
        for t in seen:
            df[t]=df.get(t,0)+1
    items=[(t,c) for t,c in df.items() if c>=min_df]
    items.sort(key=lambda x: x[1], reverse=True)
    items=items[:max_features]
    vocab={t:i for i,(t,_) in enumerate(items)}
    df_arr=np.zeros(len(vocab), dtype=np.int32)
    for t,i in vocab.items():
        df_arr[i]=df[t]
    return vocab, df_arr

def _tfidf_matrix(docs_tokens: List[List[str]], vocab: Dict[str,int], df: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N=len(docs_tokens)
    V=len(vocab)
    # idf smooth
    idf = np.log((1.0 + N) / (1.0 + df.astype(np.float32))) + 1.0
    X = np.zeros((N,V), dtype=np.float32)
    for i,toks in enumerate(docs_tokens):
        if not toks: 
            continue
        counts={}
        for t in toks:
            j=vocab.get(t)
            if j is not None:
                counts[j]=counts.get(j,0)+1
        if not counts:
            continue
        # tf = raw count
        for j,c in counts.items():
            X[i,j]=float(c)
        # l2 normalize tf
        norm=np.linalg.norm(X[i])
        if norm>0: X[i]/=norm
    # apply idf
    X *= idf[None,:]
    # l2 normalize tfidf
    norms=np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0]=1.0
    X/=norms
    return X, idf

def _randomized_svd(X: np.ndarray, dim: int = 384, n_iter: int = 2, seed: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Return components (V) and mean (zero mean not applied here). Numpy-only randomized SVD on dense X."""
    rng=np.random.default_rng(seed)
    n, d = X.shape
    k=min(dim, d)
    # random projection
    P = rng.normal(size=(d, k)).astype(np.float32)
    Z = X @ P  # (n,k)
    for _ in range(max(0,n_iter)):
        Z = X @ (X.T @ Z)
    # orthonormalize
    Q, _ = np.linalg.qr(Z, mode='reduced')  # (n,k)
    B = Q.T @ X  # (k,d)
    # SVD on small matrix
    Uhat, S, Vt = np.linalg.svd(B, full_matrices=False)
    V = Vt.T[:, :k]  # (d,k)
    return V.astype(np.float32), S[:k].astype(np.float32)

@dataclass
class SemanticIndex:
    vocab: Dict[str,int]
    idf: np.ndarray
    proj: np.ndarray  # (V,dim)
    doc_emb: np.ndarray  # (N,dim) normalized
    dim: int

    @staticmethod
    def build(passages: List[Dict[str, Any]], dim: int = 384, min_df: int = 2) -> "SemanticIndex":
        docs_tokens=[_tokenize(p.get('text','')) for p in passages]
        vocab, df = _build_vocab(docs_tokens, min_df=min_df)
        X, idf = _tfidf_matrix(docs_tokens, vocab, df)
        V, _S = _randomized_svd(X, dim=dim)
        # project documents
        doc = X @ V  # (N,dim)
        # normalize
        norms=np.linalg.norm(doc, axis=1, keepdims=True)
        norms[norms==0]=1.0
        doc/=norms
        return SemanticIndex(vocab=vocab, idf=idf.astype(np.float32), proj=V.astype(np.float32), doc_emb=doc.astype(np.float32), dim=V.shape[1])

    def embed_query(self, text: str) -> np.ndarray:
        toks=_tokenize(text)
        V=len(self.vocab)
        v=np.zeros((V,), dtype=np.float32)
        counts={}
        for t in toks:
            j=self.vocab.get(t)
            if j is not None:
                counts[j]=counts.get(j,0)+1
        for j,c in counts.items():
            v[j]=float(c)
        norm=np.linalg.norm(v)
        if norm>0: v/=norm
        # idf
        if self.idf.shape[0]==v.shape[0]:
            v*=self.idf
        # tfidf norm
        n2=np.linalg.norm(v)
        if n2>0: v/=n2
        q = v @ self.proj
        nq=np.linalg.norm(q)
        if nq>0: q/=nq
        return q.astype(np.float32)

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int,float]]:
        q=self.embed_query(query)
        sims = self.doc_emb @ q  # cosine since normalized
        # topk
        if top_k >= sims.shape[0]:
            idx=np.argsort(-sims)
        else:
            idx=np.argpartition(-sims, top_k)[:top_k]
            idx=idx[np.argsort(-sims[idx])]
        return [(int(i), float(sims[i])) for i in idx]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # vocab as json
        with open(path + ".vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f)
        np.savez_compressed(path + ".npz", idf=self.idf, proj=self.proj, doc_emb=self.doc_emb, dim=np.array([self.dim], dtype=np.int32))

    @staticmethod
    def load(path: str) -> Optional["SemanticIndex"]:
        try:
            with open(path + ".vocab.json", "r", encoding="utf-8") as f:
                vocab=json.load(f)
            data=np.load(path + ".npz")
            idf=data["idf"]
            proj=data["proj"]
            doc_emb=data["doc_emb"]
            dim=int(data["dim"][0])
            return SemanticIndex(vocab=vocab, idf=idf, proj=proj, doc_emb=doc_emb, dim=dim)
        except Exception:
            return None
