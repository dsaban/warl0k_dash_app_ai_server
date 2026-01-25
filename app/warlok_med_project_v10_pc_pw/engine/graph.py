from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class Claim:
    id: str
    doc: str
    text: str
    edge_types: List[str]
    confidence: float
    edge_evidence: Optional[Dict[str, int]] = None
    node_ids: Optional[List[str]] = None

# from dataclasses import dataclass
# from typing import List, Dict, Optional
#
# @dataclass
# class Claim:
#     id: str
#     doc: str
#     text: str
#     edge_types: List[str]
#     confidence: float
#     edge_evidence: Optional[Dict[str, int]] = None
#
# # from __future__ import annotations
# # import json
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Dict, List, Optional, Any
# #
# # @dataclass
# # class Claim:
# #     id: str
# #     doc: str
# #     text: str
# #     nodes: List[str]
# #     edge_types: List[str]
# #     confidence: float = 0.0
# #     edge_evidence: Optional[Dict[str, int]] = None
# #
# # class Graph:
# #     def __init__(self, ontology: Dict[str, Any], claims: List[Claim]):
# #         self.ontology = ontology
# #         self.claims = claims
# #         self.node_by_id = {n["id"]: n for n in ontology.get("nodes", [])}
# #         self.claims_by_node: Dict[str, List[Claim]] = {}
# #         for c in claims:
# #             for n in (c.nodes or []):
# #                 self.claims_by_node.setdefault(n, []).append(c)
# #
# #     @staticmethod
# #     def load(index_dir: Path) -> "Graph":
# #         ont = json.loads((index_dir / "ontology.json").read_text(errors="ignore"))
# #         claims: List[Claim] = []
# #         for line in (index_dir / "claims.jsonl").read_text(errors="ignore").splitlines():
# #             if not line.strip():
# #                 continue
# #             obj = json.loads(line)
# #             # backward compatible
# #             if "confidence" not in obj:
# #                 obj["confidence"] = 0.0
# #             claims.append(Claim(**obj))
# #         return Graph(ont, claims)
# #
# #     def node_type(self, node_id: str) -> str:
# #         return self.node_by_id.get(node_id, {}).get("type", "unknown")
# #
# #     def node_label(self, node_id: str) -> str:
# #         d = self.node_by_id.get(node_id, {})
# #         return d.get("label", node_id)
