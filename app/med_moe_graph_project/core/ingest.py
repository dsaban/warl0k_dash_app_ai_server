from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from core.graph import Edge, Node, TypedGraph
from core.ontology import Ontology


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    index: int


_WORD_RE = re.compile(r"[A-Za-z0-9\-αβγδμ]+", re.UNICODE)


def _normalize(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())


def chunk_text(text: str, max_chars: int = 480, overlap: int = 60) -> List[str]:
    text = _normalize(text)
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return chunks


def load_entity_lexicon(path: str | Path) -> List[Dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(data.get("entities", []))


def match_entities(chunk: str, entities: List[Dict]) -> List[Tuple[str, str]]:
    """Returns list of (entity_id, matched_alias). Lexicon-based only."""
    chunk_l = chunk.lower()
    found: List[Tuple[str, str]] = []
    for ent in entities:
        for alias in ent.get("aliases", []):
            a = alias.lower()
            if a and a in chunk_l:
                found.append((ent["id"], alias))
                break
    return found


def infer_tags(text: str, ontology: Ontology) -> List[str]:
    t = text.lower()
    hits: List[str] = []
    for tag, aliases in ontology.tags.items():
        for a in aliases:
            if a.lower() in t:
                hits.append(tag)
                break
    return sorted(set(hits))


def ingest_docs(
    docs_dir: str | Path,
    ontology: Ontology,
    entity_lexicon_path: str | Path,
    max_chars: int = 480,
    overlap: int = 60,
) -> Tuple[TypedGraph, List[Chunk]]:
    docs_dir = Path(docs_dir)
    entities = load_entity_lexicon(entity_lexicon_path)

    g = TypedGraph()
    chunks: List[Chunk] = []

    # Add ontology nodes (fixed)
    for qk, qspec in ontology.qtypes.items():
        g.add_node(Node(id=f"QTYPE::{qk}", ntype="qtype", label=qk, meta={"description": qspec.description}))
    for tk in ontology.tags.keys():
        g.add_node(Node(id=f"TAG::{tk}", ntype="tag", label=tk, meta={}))

    # Add entity nodes (fixed lexicon)
    for ent in entities:
        g.add_node(Node(id=f"ENT::{ent['id']}", ntype="entity", label=ent.get("name", ent["id"]), meta={"type": ent.get("type", "")}))

    # Ingest docs
    for path in sorted(docs_dir.glob("**/*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        ctexts = chunk_text(raw, max_chars=max_chars, overlap=overlap)
        for i, ct in enumerate(ctexts):
            cid = f"CHUNK::{path.name}::{i}"
            chunks.append(Chunk(id=cid, text=ct, source=str(path), index=i))
            g.add_node(Node(id=cid, ntype="chunk", label=f"{path.name}#{i}", meta={"source": str(path)}))

            # Tag edges
            tags = infer_tags(ct, ontology)
            for t in tags:
                g.add_edge(Edge(src=cid, dst=f"TAG::{t}", etype="has_tag", weight=1.0))

            # Entity edges (lexicon-based only)
            ents = match_entities(ct, entities)
            for ent_id, alias in ents:
                g.add_edge(Edge(src=cid, dst=f"ENT::{ent_id}", etype="mentions", weight=1.0, meta={"alias": alias}))

    return g, chunks
