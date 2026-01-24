import os, glob
from dataclasses import dataclass
from typing import List, Dict, Tuple

from .utils import split_sentences, window_text_span
from .chunking import ChunkNode, chunk_by_tokens, detect_section_headers, section_for_span

@dataclass
class ClaimNode:
    claim_id: str
    file: str
    chunk_id: str
    sentence: str
    span: Tuple[int, int]
    support_text: str
    entities: List[str]
    edge_hints: List[str]

def load_docs(folder: str) -> Dict[str, str]:
    paths = sorted(glob.glob(os.path.join(folder, "*.txt")))
    docs = {}
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            docs[os.path.basename(p)] = f.read()
    if not docs:
        raise RuntimeError(f"No .txt files found in {folder}")
    return docs

def build_chunks(docs: Dict[str,str], target_tokens: int = 220, overlap_tokens: int = 60) -> List[ChunkNode]:
    out: List[ChunkNode] = []
    k = 0
    for fn, text in docs.items():
        headers = detect_section_headers(text)
        for ch_text, span in chunk_by_tokens(text, target_tokens, overlap_tokens):
            sec = section_for_span(headers, span[0])
            out.append(ChunkNode(
                chunk_id=f"K{k:05d}",
                file=fn,
                text=ch_text,
                token_count=len(ch_text.split()),
                span=span,
                section=sec
            ))
            k += 1
    return out

def _support_window(sents: List[str], i: int, win: int) -> str:
    lo = max(0, i - win)
    hi = min(len(sents), i + win + 1)
    return " ".join(sents[lo:hi]).strip()

def build_claims(
    docs: Dict[str,str],
    chunks: List[ChunkNode],
    entity_matcher,
    edge_detector,
    sent_window: int = 1
) -> List[ClaimNode]:
    by_file: Dict[str, List[ChunkNode]] = {}
    for c in chunks:
        by_file.setdefault(c.file, []).append(c)

    claims: List[ClaimNode] = []
    cidx = 0

    for fn, text in docs.items():
        sents = split_sentences(text)
        file_chunks = by_file.get(fn, [])

        for i, sent in enumerate(sents):
            span = window_text_span(text, sent)
            support = _support_window(sents, i, sent_window)

            best = file_chunks[0].chunk_id if file_chunks else "K00000"
            best_ov = -1
            for ch in file_chunks:
                a0, a1 = span
                b0, b1 = ch.span
                ov = max(0, min(a1, b1) - max(a0, b0))
                if ov > best_ov:
                    best_ov = ov
                    best = ch.chunk_id

            entities = entity_matcher(support)
            edges = edge_detector(support)

            claims.append(ClaimNode(
                claim_id=f"C{cidx:05d}",
                file=fn,
                chunk_id=best,
                sentence=sent,
                span=span,
                support_text=support,
                entities=entities,
                edge_hints=edges
            ))
            cidx += 1

    return claims
