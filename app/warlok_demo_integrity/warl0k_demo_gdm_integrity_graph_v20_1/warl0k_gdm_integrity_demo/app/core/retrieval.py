import re, math, numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict
import os
from core.guards import SlotGuard
from core.semantic_index import SemanticIndex

_STOP = set("the a an and or to of in on for with without by from as is are was were be been being this that these those it its".split())

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9%\.\/\- ]+", " ", text)
    toks = [t for t in text.split() if t and t not in _STOP]
    return toks

class Retriever:
    """
    Hybrid retrieval:
      - BM25 for lexical precision
      - TF-IDF cosine for semantic-ish match (SBERT-lite)
    """
    def __init__(self, passages: List[Dict[str, Any]]):
        self.passages = passages
        self.guard = SlotGuard()
        self.semantic = None
        self._build()
        self._load_or_build_semantic()

    def _build(self):
        self.toks = [_tokenize(p["text"]) for p in self.passages]
        self.N = len(self.passages)
        self.df = Counter()
        for toks in self.toks:
            for w in set(toks):
                self.df[w] += 1
        self.avgdl = sum(len(t) for t in self.toks) / max(1, self.N)

        # TF-IDF matrix (sparse-ish via dict of weights)
        vocab = {w:i for i,w in enumerate(self.df.keys())}
        self.vocab = vocab
        self.idf = np.zeros(len(vocab), dtype=np.float32)
        for w,i in vocab.items():
            self.idf[i] = math.log((self.N + 1) / (self.df[w] + 1)) + 1.0

        self.tfidf_rows = []
        self.norms = np.zeros(self.N, dtype=np.float32)
        for i, toks in enumerate(self.toks):
            tf = Counter(toks)
            row = {}
            for w,c in tf.items():
                if w in vocab:
                    j = vocab[w]
                    val = (c / len(toks)) * self.idf[j]
                    row[j] = float(val)
            norm = math.sqrt(sum(v*v for v in row.values())) if row else 0.0
            self.tfidf_rows.append(row)
            self.norms[i] = norm

    def _bm25(self, query_toks: List[str], idx: int, k1=1.5, b=0.75) -> float:
        tf = Counter(self.toks[idx])
        dl = len(self.toks[idx])
        score = 0.0
        for w in query_toks:
            if w not in tf: 
                continue
            df = self.df.get(w, 0)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            score += idf * (tf[w] * (k1 + 1)) / (tf[w] + k1 * (1 - b + b * dl / self.avgdl))
        return score

    def _tfidf_cos(self, query_toks: List[str], idx: int) -> float:
        # build query vector
        tf = Counter(query_toks)
        qrow = {}
        for w,c in tf.items():
            if w in self.vocab:
                j = self.vocab[w]
                qrow[j] = (c / len(query_toks)) * float(self.idf[j])
        qnorm = math.sqrt(sum(v*v for v in qrow.values())) if qrow else 0.0
        if qnorm == 0.0 or self.norms[idx] == 0.0:
            return 0.0

        drow = self.tfidf_rows[idx]
        # dot product over smaller dict
        if len(qrow) < len(drow):
            dot = sum(v * drow.get(j, 0.0) for j,v in qrow.items())
        else:
            dot = sum(v * qrow.get(j, 0.0) for j,v in drow.items())
        return float(dot / (qnorm * float(self.norms[idx])))

    

    def _load_or_build_semantic(self, dim: int = 384):
        """Build or load a local SBERT-like semantic index (TFIDF+SVD dense embeddings)."""
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.abspath(os.path.join(here, ".."))
            idx_dir = os.path.join(base_dir, "data", "index")
            os.makedirs(idx_dir, exist_ok=True)
            path = os.path.join(idx_dir, "semantic_v10_2")
            sem = SemanticIndex.load(path)
            if sem is None:
                sem = SemanticIndex.build(self.passages, dim=dim)
                sem.save(path)
            self.semantic = sem
        except Exception:
            self.semantic = None

    def semantic_search(self, query: str, k: int = 30) -> List[Dict[str, Any]]:
        """Return top-k passages by semantic similarity."""
        if self.semantic is None:
            return []
        hits = self.semantic.search(query, top_k=k)
        out: List[Dict[str, Any]] = []
        for idx, s in hits:
            p = self.passages[idx]
            e = dict(p)
            e["score_semantic"] = float(s)
            out.append(e)
        return out

    def search(self, query: str, k: int = 8, w_bm25: float = 0.6, w_tfidf: float = 0.4) -> List[Dict[str, Any]]:
        qtoks = _tokenize(query)
        scored = []
        for i in range(self.N):
            s1 = self._bm25(qtoks, i)
            s2 = self._tfidf_cos(qtoks, i)
            score = w_bm25 * s1 + w_tfidf * s2
            if score > 0:
                scored.append((score, s1, s2, i))
        scored.sort(reverse=True)
        out = []
        for rank,(score,s1,s2,i) in enumerate(scored[:k], start=1):
            text = self.passages[i]["text"]
            out.append({
                "rank": rank,
                "score": score,
                "bm25": s1,
                "tfidf": s2,
                "pid": self.passages[i]["pid"],
                "doc": self.passages[i]["doc"],
                "text": text,
                "highlights": self._highlight(text, qtoks),
            })
        return out

    def _highlight(self, text: str, qtoks: List[str]) -> str:
        # crude keyword highlight: wrap matched tokens with ** **
        out = text
        for w in sorted(set(qtoks), key=len, reverse=True):
            if len(w) < 3: 
                continue
            out = re.sub(rf"(?i)\b{re.escape(w)}\b", lambda m: f"**{m.group(0)}**", out)
        return out


    def search_guarded(self, query: str, qtype: str, k: int = 10, pre_k: int = 30) -> List[Dict[str, Any]]:
        """
        Guarded retrieval:
          1) retrieve a larger candidate set (pre_k)
          2) compute slot coverage for qtype
          3) rank: primary first, then support, then others; within each by base score + slot_score
        Returns evidence dicts with additional fields:
          - slot_primary, slot_support, slot_score, slot_hits
        """
        bm = self.search(query, k=pre_k)
        sem = self.semantic_search(query, k=pre_k)
        # merge by (doc,pid)
        merged = {}
        for e in bm:
            key = (str(e.get('doc','')), str(e.get('pid','')))
            merged[key] = dict(e)
            merged[key]['score_bm25'] = float(e.get('score',0.0))
        for e in sem:
            key = (str(e.get('doc','')), str(e.get('pid','')))
            if key in merged:
                merged[key]['score_semantic'] = float(e.get('score_semantic',0.0))
            else:
                merged[key] = dict(e)
                merged[key]['score_bm25'] = 0.0
                merged[key]['score_semantic'] = float(e.get('score_semantic',0.0))
        base = list(merged.values())
        # normalize scores roughly
        if base:
            bmax = max([x.get('score_bm25',0.0) for x in base]) or 1.0
            smax = max([x.get('score_semantic',0.0) for x in base]) or 1.0
            for x in base:
                x['score_bm25_n'] = float(x.get('score_bm25',0.0))/bmax
                x['score_sem_n'] = float(x.get('score_semantic',0.0))/smax
                # keep existing 'score' for UI as hybrid
                x['score'] = 0.55*x['score_bm25_n'] + 0.45*x['score_sem_n']
                # lightweight entity boost: reward passages that contain key query terms
                ql = query.lower()
                key_terms = ['ogtt','oral glucose tolerance','75 g','75g','24-28','24–28','postpartum','6 weeks','4-12','fasting','one-hour','two-hour','cut-off','cutoff','92','180','153','macrosomia','preeclampsia','cesarean','hypoglycemia','metformin','glibenclamide','glyburide','insulin']
                boost = 0.0
                tl = str(x.get('text','')).lower()
                for kt in key_terms:
                    if kt and (kt in ql) and (kt in tl):
                        boost += 0.02
                x['score'] += min(boost, 0.10)

        scored = []
        for e in base:
            sr = self.guard.evaluate_passage(e.get("text",""), qtype=qtype)
            ee = dict(e)
            ee["slot_primary"] = bool(sr.primary)
            ee["slot_support"] = bool(sr.support)
            ee["slot_score"] = float(sr.slot_score)
            ee["slot_hits"] = dict(sr.slot_hits)
            # combined score: base bm25/tfidf score is in e['score']; bump with slot_score
            ee["_combo"] = float(e.get("score",0.0)) + 2.5*float(sr.slot_score)
            scored.append(ee)

        # Sort: primary desc, support desc, combo desc
        scored.sort(key=lambda x: (1 if x["slot_primary"] else 0,
                                  1 if x["slot_support"] else 0,
                                  x["_combo"]), reverse=True)

        # if we have any primary evidence, prefer returning only primary + a few supports
        prim = [x for x in scored if x["slot_primary"]]
        if prim:
            primary_docs = sorted(set([str(x.get('doc','')) for x in prim if x.get('doc')]))
            supp = [x for x in scored if (not x["slot_primary"]) and x["slot_support"]]
            out = prim[:k]
            # add consensus meta
            for x in out:
                x['evidence_docs_primary'] = primary_docs
                x['primary_doc_count'] = len(primary_docs)
            # add up to 2 supports if room
            for s in supp:
                if len(out) >= k:
                    break
                out.append(s)
            return out

        return scored[:k]
