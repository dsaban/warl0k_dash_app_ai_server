import argparse
import numpy as np

from core.ingest import load_docs, build_chunks, build_claims
from core.schema import ENTITY_LEXICON, EDGE_CUES
from core.utils import norm

def match_entities(text: str):
    s = norm(text)
    out = []
    for k, (etype, aliases) in ENTITY_LEXICON.items():
        for a in aliases:
            if norm(a) in s:
                out.append(k)
                break
    return out

def detect_edges(text: str):
    s = norm(text)
    hits = []
    for ename, cues in EDGE_CUES.items():
        if any(norm(c) in s for c in cues):
            hits.append(ename)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk_tokens", type=int, default=220)
    ap.add_argument("--chunk_overlap", type=int, default=60)
    ap.add_argument("--sent_window", type=int, default=1)
    args = ap.parse_args()

    docs = load_docs(args.docs)
    chunks = build_chunks(docs, target_tokens=args.chunk_tokens, overlap_tokens=args.chunk_overlap)
    claims = build_claims(docs, chunks, match_entities, detect_edges, sent_window=args.sent_window)

    np.savez(
        args.out,
        chunk_id=np.array([c.chunk_id for c in chunks], dtype=object),
        chunk_file=np.array([c.file for c in chunks], dtype=object),
        chunk_text=np.array([c.text for c in chunks], dtype=object),
        chunk_span=np.array([c.span for c in chunks], dtype=object),
        chunk_tok=np.array([c.token_count for c in chunks], dtype=np.int32),
        chunk_section=np.array([c.section for c in chunks], dtype=object),

        claim_id=np.array([c.claim_id for c in claims], dtype=object),
        claim_file=np.array([c.file for c in claims], dtype=object),
        claim_chunk=np.array([c.chunk_id for c in claims], dtype=object),
        claim_sentence=np.array([c.sentence for c in claims], dtype=object),
        claim_span=np.array([c.span for c in claims], dtype=object),
        claim_support=np.array([c.support_text for c in claims], dtype=object),
        claim_entities=np.array([c.entities for c in claims], dtype=object),
        claim_edges=np.array([c.edge_hints for c in claims], dtype=object),
    )

    print(f"Saved index: {args.out}")
    print(f"Chunks: {len(chunks)} | Claims: {len(claims)}")

if __name__ == "__main__":
    main()
