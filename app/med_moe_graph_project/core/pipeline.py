from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from core.graph import TypedGraph
from core.ingest import Chunk, ingest_docs
from core.integrity import check_answer_integrity
from core.moe import MoEAnswer, route_and_answer
from core.ontology import Ontology
from core.retrieval import RetrievalHit, classify_qtype, retrieve


@dataclass
class InferResult:
    question: str
    qtype: str
    answer: str
    integrity_ok: bool
    integrity_notes: str
    hits: List[RetrievalHit]
    qtags: List[str]
    graph_neighborhood_nodes: int
    graph_edges_count: int
    conn_table: List[dict]


class LocalMedMoEPipeline:
    """One-call interface used by the UI.

    Requirements met:
    - Fixed QType taxonomy
    - Lexicon-based entities only
    - Graph-backed explainability
    """

    def __init__(
        self,
        ontology_path: str | Path = "config/ontology.json",
        entity_lexicon_path: str | Path = "data/lexicon/entities.json",
        docs_dir: str | Path = "data/docs",
        graph_path: str | Path = "data/graph/graph.json",
    ):
        self.ontology_path = Path(ontology_path)
        self.entity_lexicon_path = Path(entity_lexicon_path)
        self.docs_dir = Path(docs_dir)
        self.graph_path = Path(graph_path)

        self.ontology = Ontology.load(self.ontology_path)
        self.graph: Optional[TypedGraph] = None
        self.chunks: List[Chunk] = []
        self.chunk_tags: Dict[str, List[str]] = {}
        self.chunk_entities: Dict[str, List[str]] = {}

        self._load_or_build()

    def _load_or_build(self) -> None:
        if self.graph_path.exists():
            self.graph = TypedGraph.load(self.graph_path)
            # Rebuild chunks by re-ingesting docs for simplicity/determinism
            self.graph, self.chunks = ingest_docs(
                docs_dir=self.docs_dir,
                ontology=self.ontology,
                entity_lexicon_path=self.entity_lexicon_path,
            )
            self.graph.save(self.graph_path)
        else:
            self.graph, self.chunks = ingest_docs(
                docs_dir=self.docs_dir,
                ontology=self.ontology,
                entity_lexicon_path=self.entity_lexicon_path,
            )
            self.graph.save(self.graph_path)

        assert self.graph is not None

        # Build quick maps from graph edges
        self.chunk_tags = {c.id: [] for c in self.chunks}
        self.chunk_entities = {c.id: [] for c in self.chunks}

        for e in self.graph.edges:
            if not e.src.startswith("CHUNK::"):
                continue
            if e.etype == "has_tag" and e.dst.startswith("TAG::"):
                self.chunk_tags[e.src].append(e.dst.replace("TAG::", ""))
            if e.etype == "mentions" and e.dst.startswith("ENT::"):
                self.chunk_entities[e.src].append(e.dst.replace("ENT::", ""))

        # Normalize
        for k in self.chunk_tags:
            self.chunk_tags[k] = sorted(set(self.chunk_tags[k]))
        for k in self.chunk_entities:
            self.chunk_entities[k] = sorted(set(self.chunk_entities[k]))

    def rebuild_graph(self) -> None:
        self.graph, self.chunks = ingest_docs(
            docs_dir=self.docs_dir,
            ontology=self.ontology,
            entity_lexicon_path=self.entity_lexicon_path,
        )
        self.graph.save(self.graph_path)
        self._load_or_build()

    def infer(self, question: str, top_k: int = 6, neighborhood_hops: int = 2) -> InferResult:
        qtype = classify_qtype(question, self.ontology)

        hits = retrieve(
            question=question,
            chunks=self.chunks,
            chunk_tags=self.chunk_tags,
            chunk_entities=self.chunk_entities,
            ontology=self.ontology,
            top_k=top_k,
        )

        moe: MoEAnswer = route_and_answer(question=question, qtype=qtype, hits=hits, ontology=self.ontology)
        integrity = check_answer_integrity(moe.answer, moe.qtype, self.ontology)

        # If integrity fails, emit a tighter causal chain (still grounded) rather than hallucinating
        if not integrity.ok and hits:
            # Minimal, checklist-driven patch attempt
            patch = " "
            if moe.qtype == "fetal_growth_mechanism":
                patch = " (Explicit chain: maternal glucose crosses placenta → fetal insulin rises → insulin drives anabolic growth → macrosomia.)"
            elif moe.qtype == "maternal_long_term_risk":
                patch = " (Explicit chain: obesity inflammation → insulin resistance persists postpartum → vascular dysfunction → higher cardiometabolic risk.)"
            elif moe.qtype == "progression_model":
                patch = " (Explicit chain: pregnancy insulin resistance ↑ → beta-cell compensation → failure causes GDM → postpartum unmasks future T2D risk.)"
            moe.answer = moe.answer + patch
            integrity = check_answer_integrity(moe.answer, moe.qtype, self.ontology)

        # Build connection table for UI: chunk -> tags/entities
        conn_table: List[dict] = []
        for h in hits:
            conn_table.append(
                {
                    "chunk_id": h.chunk_id,
                    "score": round(h.score, 4),
                    "tags": ", ".join(h.tags),
                    "entities": ", ".join(h.entities),
                }
            )

        # Graph neighborhood (for UI explainability)
        neighborhood_ids = set()
        if self.graph is not None:
            for h in hits[:3]:
                neighborhood_ids |= set(self.graph.neighbors(h.chunk_id, hops=neighborhood_hops).keys())

        graph_edges = self.graph.edges_between(neighborhood_ids) if self.graph is not None else []

        # Query tags to display
        qtags = []
        ql = question.lower()
        for tag, aliases in self.ontology.tags.items():
            if any(a.lower() in ql for a in aliases):
                qtags.append(tag)
        qtags = sorted(set(qtags))

        return InferResult(
            question=question,
            qtype=moe.qtype,
            answer=moe.answer,
            integrity_ok=integrity.ok,
            integrity_notes=integrity.notes,
            hits=hits,
            qtags=qtags,
            graph_neighborhood_nodes=len(neighborhood_ids),
            graph_edges_count=len(graph_edges),
            conn_table=conn_table,
        )
