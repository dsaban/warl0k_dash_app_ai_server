import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

@dataclass
class Claim:
    id: str
    doc: str
    text: str
    nodes: List[str]
    edge_types: List[str]
    confidence: float = 0.0
    edge_evidence: Optional[Dict[str,int]] = None

class Graph:
    def __init__(self, ontology: Dict, claims: List[Claim]):
        self.ontology=ontology
        self.claims=claims
        self.node_by_id={n["id"]: n for n in ontology.get("nodes", [])}
        self.claims_by_node: Dict[str,List[Claim]]={}
        for c in claims:
            for n in c.nodes:
                self.claims_by_node.setdefault(n, []).append(c)

    @staticmethod
    def load(index_dir: Path) -> "Graph":
        ont = json.loads((index_dir/"ontology.json").read_text(errors="ignore"))
        claims=[]
        for line in (index_dir/"claims.jsonl").read_text(errors="ignore").splitlines():
            if line.strip():
                claims.append(Claim(**json.loads(line)))
        return Graph(ont, claims)

    def node_type(self, node_id: str) -> str:
        return self.node_by_id.get(node_id, {}).get("type", "unknown")

# import json
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, List
#
# @dataclass
# class Claim:
#     id: str
#     doc: str
#     text: str
#     nodes: List[str]
#     edge_types: List[str]
#
# class Graph:
#     def __init__(self, ontology: Dict, claims: List[Claim]):
#         self.ontology = ontology
#         self.claims = claims
#         self.node_by_id = {n["id"]: n for n in ontology.get("nodes", [])}
#         self.claims_by_node: Dict[str, List[Claim]] = {}
#         for c in claims:
#             for n in c.nodes:
#                 self.claims_by_node.setdefault(n, []).append(c)
#
#     @staticmethod
#     def load(index_dir: Path) -> "Graph":
#         ont = json.loads((index_dir / "ontology.json").read_text(errors="ignore"))
#         claims = []
#         for line in (index_dir / "claims.jsonl").read_text(errors="ignore").splitlines():
#             if line.strip():
#                 claims.append(Claim(**json.loads(line)))
#         return Graph(ont, claims)
#
#     def node_type(self, node_id: str) -> str:
#         return self.node_by_id.get(node_id, {}).get("type", "unknown")
