from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class Node:
    id: str
    ntype: str
    label: str
    meta: Dict[str, str]


@dataclass
class Edge:
    src: str
    dst: str
    etype: str
    weight: float = 1.0
    meta: Dict[str, str] | None = None


class TypedGraph:
    """A lightweight, dependency-free typed graph.

    Designed for:
    - deterministic serialization
    - simple neighborhood queries
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adj_out: Dict[str, List[int]] = {}
        self.adj_in: Dict[str, List[int]] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            return
        self.nodes[node.id] = node
        self.adj_out[node.id] = []
        self.adj_in[node.id] = []

    def add_edge(self, edge: Edge) -> None:
        if edge.src not in self.nodes or edge.dst not in self.nodes:
            raise KeyError(f"Edge endpoints must exist: {edge.src} -> {edge.dst}")
        idx = len(self.edges)
        self.edges.append(edge)
        self.adj_out[edge.src].append(idx)
        self.adj_in[edge.dst].append(idx)

    def out_edges(self, node_id: str, etype: Optional[str] = None) -> List[Edge]:
        idxs = self.adj_out.get(node_id, [])
        out = [self.edges[i] for i in idxs]
        if etype is None:
            return out
        return [e for e in out if e.etype == etype]

    def in_edges(self, node_id: str, etype: Optional[str] = None) -> List[Edge]:
        idxs = self.adj_in.get(node_id, [])
        out = [self.edges[i] for i in idxs]
        if etype is None:
            return out
        return [e for e in out if e.etype == etype]

    def neighbors(self, node_id: str, hops: int = 1) -> Dict[str, Node]:
        """Returns nodes within N hops (outgoing + incoming)."""
        seen = {node_id}
        frontier = {node_id}
        for _ in range(hops):
            nxt = set()
            for nid in list(frontier):
                for e in self.out_edges(nid):
                    nxt.add(e.dst)
                for e in self.in_edges(nid):
                    nxt.add(e.src)
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return {nid: self.nodes[nid] for nid in seen}

    def edges_between(self, node_ids: Iterable[str]) -> List[Edge]:
        s = set(node_ids)
        out = []
        for e in self.edges:
            if e.src in s and e.dst in s:
                out.append(e)
        return out

    def to_json_dict(self) -> Dict:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges],
        }

    @staticmethod
    def from_json_dict(data: Dict) -> "TypedGraph":
        g = TypedGraph()
        for n in data.get("nodes", []):
            g.add_node(Node(**n))
        for e in data.get("edges", []):
            g.add_edge(Edge(**e))
        return g

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "TypedGraph":
        path = Path(path)
        return TypedGraph.from_json_dict(json.loads(path.read_text(encoding="utf-8")))


def safe_log(x: float) -> float:
    return math.log(x) if x > 0 else 0.0
