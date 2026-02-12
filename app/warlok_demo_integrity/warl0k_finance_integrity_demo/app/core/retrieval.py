from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class PolicyRetriever:
    def __init__(self, policies_df):
        self.df = policies_df.copy()
        self.df["blob"] = (
            self.df["title"].fillna("") + " " +
            self.df["tag"].fillna("") + " " +
            self.df["keywords"].fillna("") + " " +
            self.df["clause"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.mat = self.vectorizer.fit_transform(self.df["blob"].tolist())

    def search(self, query: str, k: int = 5) -> List[Dict]:
        qv = self.vectorizer.transform([query])
        sims = (self.mat @ qv.T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        out = []
        for i in idx:
            out.append({
                "policy_id": str(self.df.iloc[i]["policy_id"]),
                "title": str(self.df.iloc[i]["title"]),
                "tag": str(self.df.iloc[i]["tag"]),
                "score": float(sims[i]),
                "clause": str(self.df.iloc[i]["clause"]),
            })
        return out
