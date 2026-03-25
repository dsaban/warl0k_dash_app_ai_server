"""
peer/mesh.py — Mesh registry and coordinator.

The Mesh holds all PeerNodes, wires them together, and provides
high-level operations:
  dispatch_task(target, task, parents) — send a signed task to a peer
  replicate_block(hash, targets)       — push a block to named peers (or all)
  send_telemetry(target, key, data)    — route raw data to a peer's store
  cross_validate(hash)                 — ask every peer to validate a block
  status()                             — full mesh health snapshot
"""

from __future__ import annotations
import json
from core.events  import log_event
from node    import PeerNode


class Mesh:
    def __init__(self):
        self._nodes: dict[str, PeerNode] = {}

    # ── Node management ───────────────────────────────────────────────────────

    def add_peer(self, peer: PeerNode):
        """Add a peer and wire it to every existing peer bidirectionally."""
        for existing in self._nodes.values():
            peer.register_peer(existing)
        self._nodes[peer.peer_id] = peer
        log_event("MESH_ADD_PEER", {"peer_id": peer.peer_id,
                                     "total": len(self._nodes)})

    def get_peer(self, peer_id: str) -> PeerNode | None:
        return self._nodes.get(peer_id)

    @property
    def peers(self) -> list[str]:
        return list(self._nodes.keys())

    # ── High-level operations ─────────────────────────────────────────────────

    def dispatch_task(self, sender_id: str, target_id: str,
                      task: dict, parents: list | None = None) -> dict:
        """
        Send a signed task from one peer to another.
        Returns the ACK payload including block_hash and store_receipt.
        """
        sender = self._nodes.get(sender_id)
        if not sender:
            raise ValueError(f"Unknown sender: {sender_id}")
        ack = sender.send(target_id, "TASK",
                          {"task": task, "parents": parents or []})
        log_event("MESH_TASK_DISPATCH", {"from": sender_id, "to": target_id,
                                          "task_type": task.get("type", ""),
                                          "status": ack.payload.get("status")})
        return ack.payload

    def replicate_block(self, source_id: str, block_hash: str,
                        targets: list[str] | None = None) -> dict[str, dict]:
        """
        Push a block from source peer to target peers (default: all others).
        Returns dict of target_id -> replication result.
        """
        source  = self._nodes.get(source_id)
        if not source:
            raise ValueError(f"Unknown source: {source_id}")
        bd = source.serialise_block(block_hash)
        if not bd:
            raise ValueError(f"Block {block_hash[:16]} not in {source_id}'s DAG")

        target_ids = targets or [p for p in self.peers if p != source_id]
        results    = {}
        for tid in target_ids:
            ack = source.send(tid, "BLOCK", {"block": bd})
            results[tid] = ack.payload
            log_event("MESH_BLOCK_REPLICATE", {"from": source_id, "to": tid,
                                                "hash": block_hash[:16],
                                                "ok": ack.payload.get("replicated")})
        return results

    def send_telemetry(self, sender_id: str, target_id: str,
                       key: str, data: bytes) -> dict:
        """Route raw telemetry bytes to a peer's storage."""
        sender = self._nodes.get(sender_id)
        if not sender:
            raise ValueError(f"Unknown sender: {sender_id}")
        ack = sender.send(target_id, "TELEMETRY",
                          {"key": key, "data_hex": data.hex()})
        log_event("MESH_TELEMETRY", {"from": sender_id, "to": target_id,
                                      "key": key, "bytes": len(data)})
        return ack.payload

    def cross_validate(self, requestor_id: str,
                       block_hash: str) -> dict[str, dict]:
        """
        Ask every peer (except requestor) to validate a block.
        Returns consensus summary + per-peer results.
        """
        requestor = self._nodes.get(requestor_id)
        if not requestor:
            raise ValueError(f"Unknown requestor: {requestor_id}")

        results = requestor.cross_validate(block_hash)

        # Consensus: how many peers confirmed valid?
        valid_count   = sum(1 for r in results.values()
                            if r.get("status") == "OK" and r.get("valid"))
        total_peers   = len(results)
        consensus_pct = (valid_count / total_peers * 100) if total_peers else 0

        log_event("MESH_CROSS_VALIDATE", {
            "hash": block_hash[:16],
            "valid": valid_count,
            "total": total_peers,
            "consensus_pct": round(consensus_pct, 1),
        })

        return {
            "block_hash":    block_hash,
            "valid_count":   valid_count,
            "total_peers":   total_peers,
            "consensus_pct": round(consensus_pct, 1),
            "peer_results":  results,
        }

    def status(self) -> dict:
        return {
            "peers":       self.peers,
            "peer_count":  len(self._nodes),
            "peer_status": {pid: p.status() for pid, p in self._nodes.items()},
        }
