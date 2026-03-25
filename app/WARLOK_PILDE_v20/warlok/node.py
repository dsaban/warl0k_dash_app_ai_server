"""
peer/node.py — A WARLOK peer node.

Each peer:
  - Has its own HSM (shares root_key with hub — simulates shared hardware root of trust)
  - Has its own local storage
  - Exposes an in-process 'receive' API (stands in for real network transport)
  - Can execute signed tasks and produce blocks
  - Cross-validates blocks from other peers using epoch tokens

Message types (all dicts):
  TASK     — signed execution payload sent to this peer
  BLOCK    — a block produced elsewhere, for DAG replication
  TELEMETRY — raw data to be stored
  ACK      — receipt after processing any of the above
  VALIDATE_REQ — cross-peer validation request
  VALIDATE_RESP — cross-peer validation response
"""

from __future__ import annotations
import hashlib, hmac as _hmac, json, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Callable

from demo_hsm import DemoHSM
from core.block   import Block
from core.dag     import DAG
from core.events  import log_event
from storage.store import MemoryStore, StorageReceipt


# ── Message envelope ──────────────────────────────────────────────────────────

@dataclass
class Message:
    msg_type:  str                  # TASK | BLOCK | TELEMETRY | ACK | VALIDATE_REQ | VALIDATE_RESP
    sender_id: str
    payload:   dict
    msg_id:    str  = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    signature: str  = ""            # hex HMAC of canonical payload

    def canonical(self) -> bytes:
        """Deterministic bytes for signing — sorted JSON of core fields."""
        return json.dumps({
            "msg_id":    self.msg_id,
            "msg_type":  self.msg_type,
            "sender_id": self.sender_id,
            "payload":   self.payload,
            "timestamp": self.timestamp,
        }, sort_keys=True).encode()

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Message":
        return Message(**d)


# ── Peer node ─────────────────────────────────────────────────────────────────

class PeerNode:
    def __init__(self, peer_id: str, root_key: bytes,
                 task_runner: Callable[[dict], bytes] | None = None):
        """
        peer_id    : unique name, e.g. "peer-alpha"
        root_key   : shared HSM root — same as hub (simulates shared root-of-trust)
        task_runner: optional callable that executes a task payload and returns result bytes.
                     Default: echo the task as JSON.
        """
        self.peer_id     = peer_id
        self.hsm         = DemoHSM(root_key)
        self.dag         = DAG()
        self.store       = MemoryStore(peer_id=peer_id)
        self.task_runner = task_runner or (lambda t: json.dumps(t).encode())

        # Known peers: id -> PeerNode (for direct mesh communication)
        self._peers: dict[str, "PeerNode"] = {}

        # Inbox: list of received messages (audit trail)
        self.inbox: list[Message] = []

        # Blocks this peer has originated
        self._originated: list[str] = []

        log_event("PEER_INIT", {"peer_id": peer_id})

    # ── Peer registry ─────────────────────────────────────────────────────────

    def register_peer(self, other: "PeerNode"):
        """Add another peer to the mesh. Bidirectional."""
        self._peers[other.peer_id] = other
        other._peers[self.peer_id] = self
        log_event("PEER_REGISTER", {"from": self.peer_id, "to": other.peer_id})

    # ── Message signing / verification ───────────────────────────────────────

    def _sign_msg(self, msg: Message) -> str:
        """Sign a message with this peer's HSM epoch key (no block hash — transport level)."""
        raw = msg.canonical()
        sig = _hmac.new(self.hsm.root_key, raw, hashlib.sha256).digest()
        return sig.hex()

    def _verify_msg(self, msg: Message, sender: "PeerNode") -> bool:
        """Verify a message came from the claimed sender (same root key in demo)."""
        raw      = msg.canonical()
        expected = _hmac.new(sender.hsm.root_key, raw, hashlib.sha256).digest()
        return _hmac.compare_digest(expected, bytes.fromhex(msg.signature))

    def _make_msg(self, msg_type: str, payload: dict) -> Message:
        msg = Message(msg_type=msg_type, sender_id=self.peer_id, payload=payload)
        msg.signature = self._sign_msg(msg)
        return msg

    # ── Send to a specific peer ───────────────────────────────────────────────

    def send(self, target_id: str, msg_type: str, payload: dict) -> Message:
        """Send a message to a named peer. Returns the ACK."""
        target = self._peers.get(target_id)
        if not target:
            raise ValueError(f"Unknown peer: {target_id}")
        msg = self._make_msg(msg_type, payload)
        log_event("PEER_SEND", {"from": self.peer_id, "to": target_id,
                                 "type": msg_type, "msg_id": msg.msg_id})
        ack = target.receive(msg, sender=self)
        return ack

    def broadcast(self, msg_type: str, payload: dict) -> dict[str, Message]:
        """Broadcast to all known peers. Returns dict of peer_id -> ACK."""
        results = {}
        for pid in list(self._peers):
            try:
                results[pid] = self.send(pid, msg_type, payload)
            except Exception as e:
                log_event("PEER_SEND_ERR", {"from": self.peer_id, "to": pid, "err": str(e)})
        return results

    # ── Receive & dispatch ────────────────────────────────────────────────────

    def receive(self, msg: Message, sender: "PeerNode") -> Message:
        """Entry point for all inbound messages. Verifies, dispatches, returns ACK."""
        # Signature check
        if not self._verify_msg(msg, sender):
            log_event("PEER_AUTH_FAIL", {"from": msg.sender_id, "type": msg.msg_type})
            return self._make_msg("ACK", {"status": "AUTH_FAIL", "ref": msg.msg_id})

        self.inbox.append(msg)
        log_event("PEER_RECV", {"at": self.peer_id, "from": msg.sender_id,
                                  "type": msg.msg_type, "msg_id": msg.msg_id})

        handlers = {
            "TASK":          self._handle_task,
            "BLOCK":         self._handle_block,
            "TELEMETRY":     self._handle_telemetry,
            "VALIDATE_REQ":  self._handle_validate_req,
        }

        handler = handlers.get(msg.msg_type)
        if handler:
            result = handler(msg)
        else:
            result = {"status": "UNKNOWN_TYPE"}

        return self._make_msg("ACK", {"status": "OK", "ref": msg.msg_id, **result})

    # ── Message handlers ──────────────────────────────────────────────────────

    def _handle_task(self, msg: Message) -> dict:
        """
        Execute a signed task payload.
        Creates a block recording the task + result, signs it, stores result.
        """
        task    = msg.payload.get("task", {})
        parents = msg.payload.get("parents", [])

        # Execute the task
        try:
            result_bytes = self.task_runner(task)
            status = "OK"
        except Exception as e:
            result_bytes = str(e).encode()
            status = "ERROR"

        # Store result in local store
        store_key     = f"task:{msg.msg_id}"
        receipt       = self.store.put(store_key, result_bytes,
                                        meta={"task": task, "from": msg.sender_id})

        # Create a block for this execution
        payload_str = json.dumps({"task": task, "result": result_bytes.hex()[:64],
                                   "store_key": store_key, "executor": self.peer_id})
        b = self._make_block(parents, payload_str,
                              {"agent": self.peer_id, "task_type": task.get("type", ""),
                               "store_key": store_key, "msg_id": msg.msg_id})

        self._originated.append(b.hash)
        log_event("PEER_TASK_EXEC", {"peer": self.peer_id, "task": task,
                                      "block": b.hash, "status": status})

        return {"block_hash": b.hash, "store_key": store_key,
                "store_receipt": receipt.to_dict(), "exec_status": status}

    def _handle_block(self, msg: Message) -> dict:
        """Replicate a block from another peer into local DAG + store."""
        bd = msg.payload.get("block", {})
        try:
            b           = Block(bd["parents"], bytes.fromhex(bd["payload_hex"]), bd["meta"])
            b.hash      = bd["hash"]
            b.signature = bytes.fromhex(bd["signature_hex"])
            b.status    = bd.get("status", "ACTIVE")
            b.timestamp = bd.get("timestamp", time.time())

            # Verify the block signature using the originating epoch
            epoch = bd["meta"].get("_epoch", self.hsm.epoch_id)
            ok    = self.hsm.verify_epoch(b.hash.encode(), b.signature, b.hash, epoch)

            if ok:
                # Only add if parents are present (or no parents — genesis)
                can_add = all(p in self.dag.nodes for p in b.parents)
                if can_add:
                    self.dag.nodes[b.hash] = b      # direct insert (already validated)
                    # Store raw block data
                    self.store.put(f"block:{b.hash}",
                                   json.dumps(bd).encode(),
                                   meta={"from": msg.sender_id, "epoch": epoch})
                    log_event("PEER_BLOCK_REPLICATED",
                              {"peer": self.peer_id, "hash": b.hash[:16],
                               "from": msg.sender_id})
                    return {"replicated": True, "hash": b.hash}
                else:
                    log_event("PEER_BLOCK_MISSING_PARENTS",
                              {"peer": self.peer_id, "hash": b.hash[:16]})
                    return {"replicated": False, "reason": "missing_parents"}
            else:
                log_event("PEER_BLOCK_INVALID_SIG",
                          {"peer": self.peer_id, "hash": b.hash[:16]})
                return {"replicated": False, "reason": "bad_signature"}
        except Exception as e:
            log_event("PEER_BLOCK_ERR", {"peer": self.peer_id, "err": str(e)})
            return {"replicated": False, "reason": str(e)}

    def _handle_telemetry(self, msg: Message) -> dict:
        """Store raw telemetry data. Returns storage receipt."""
        data_hex = msg.payload.get("data_hex", "")
        key      = msg.payload.get("key", f"telemetry:{msg.msg_id}")
        data     = bytes.fromhex(data_hex)
        receipt  = self.store.put(key, data,
                                   meta={"from": msg.sender_id,
                                         "ts": msg.timestamp})
        log_event("PEER_TELEMETRY", {"peer": self.peer_id, "key": key,
                                      "bytes": len(data)})
        return {"stored": True, "receipt": receipt.to_dict()}

    def _handle_validate_req(self, msg: Message) -> dict:
        """Cross-peer validation. Validate a block hash from our local DAG."""
        block_hash = msg.payload.get("hash", "")
        b          = self.dag.nodes.get(block_hash)
        if not b:
            return {"valid": False, "reason": "not_in_local_dag"}
        epoch = b.meta.get("_epoch", self.hsm.epoch_id)
        ok    = self.hsm.verify_epoch(b.hash.encode(), b.signature, b.hash, epoch)
        log_event("PEER_CROSS_VALIDATE", {"peer": self.peer_id,
                                           "hash": block_hash[:16], "ok": ok})
        return {"valid": ok, "epoch": epoch, "validator": self.peer_id,
                "token": self.hsm.epoch_token(epoch).hex()}

    # ── Block creation ────────────────────────────────────────────────────────

    def _make_block(self, parents: list, payload: str, meta: dict) -> Block:
        b   = Block(parents, payload, meta)
        h   = b.compute_hash()
        b.signature = self.hsm.sign(h.encode(), block_hash=h)
        b.meta["_epoch"] = self.hsm.epoch_id
        try:
            self.dag.add(b)
        except Exception:
            # Parent may be in hub's DAG but not peer's — store anyway
            self.dag.nodes[b.hash] = b
        return b

    # ── Convenience: cross-validate a hash on all peers ─────────────────────

    def cross_validate(self, block_hash: str) -> dict[str, dict]:
        """Ask all known peers to validate a block. Returns peer -> result."""
        results = {}
        for pid, peer in self._peers.items():
            ack = self.send(pid, "VALIDATE_REQ", {"hash": block_hash})
            results[pid] = ack.payload
        return results

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "peer_id":     self.peer_id,
            "hsm":         self.hsm.status(),
            "dag_blocks":  len(self.dag.nodes),
            "store":       self.store.stats(),
            "peers":       list(self._peers.keys()),
            "inbox_count": len(self.inbox),
            "originated":  len(self._originated),
        }

    def serialise_block(self, block_hash: str) -> dict | None:
        """Serialise a block for wire transfer."""
        b = self.dag.nodes.get(block_hash)
        if not b:
            return None
        return {
            "hash":          b.hash,
            "parents":       b.parents,
            "payload_hex":   b.payload.hex(),
            "signature_hex": b.signature.hex() if b.signature else "",
            "meta":          b.meta,
            "status":        b.status,
            "timestamp":     b.timestamp,
        }
