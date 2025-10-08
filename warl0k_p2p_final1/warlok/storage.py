# warlok/storage.py
import os, pickle, json, time, hashlib

class TicketedAdapters:
    def __init__(self, dirpath=".adapters"):
        self.dir = dirpath
        os.makedirs(self.dir, exist_ok=True)

    def _peer_dir(self, peer_id):
        p = os.path.join(self.dir, peer_id)
        os.makedirs(p, exist_ok=True)
        return p

    def path(self, peer_id, n):
        base = os.path.join(self._peer_dir(peer_id), f"n_{n:08d}")
        return base + ".pkl"

    def path_audit(self, peer_id, n):
        base = os.path.join(self._peer_dir(peer_id), f"n_{n:08d}")
        return base + ".json"

    def exists(self, peer_id, n):
        return os.path.exists(self.path(peer_id, n))

    def _sha256_hex(self, b: bytes) -> str:
        return hashlib.sha256(b).hexdigest()

    def _write_audit(self, peer_id, n, d):
        # Build safe audit payload (no secrets, only hashes)
        pkl_path = self.path(peer_id, n)
        with open(pkl_path, "rb") as f:
            pkl_bytes = f.read()
        pkl_sha256 = self._sha256_hex(pkl_bytes)

        W_bytes = d.W if getattr(d, "W", None) is not None else b""
        w_sha256 = self._sha256_hex(W_bytes) if W_bytes else None

        obf = getattr(d, "forced_obf", None)   # safe to expose
        target = getattr(d, "forced_target", None)
        target_sha256 = self._sha256_hex(target.encode()) if target else None

        lti = getattr(d, "last_training_info", None) or {}
        saved_at = getattr(d, "_saved_at_epoch_ms", None) or time.time()

        audit = {
            "peer_id": d.peer_id,
            "n": n,
            "w_sha256": w_sha256,
            "obf_hex": obf,
            "target_sha256": target_sha256,
            "training": {
                "epochs_run": lti.get("epochs_run"),
                "early_stopped": lti.get("early_stopped"),
                "meta_used": lti.get("meta_used"),
            },
            "saved_at_epoch_ms": saved_at,
            "artifacts": {
                "adapter_pkl": os.path.basename(pkl_path),
                "adapter_pkl_sha256": pkl_sha256,
            },
        }

        # Self-hash the JSON (after setting all fields)
        json_bytes = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
        audit["audit_json_sha256"] = self._sha256_hex(json_bytes)

        # Write pretty JSON for humans
        with open(self.path_audit(peer_id, n), "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, sort_keys=True)

    def save(self, peer_id, n, drnn):
        # pickle model
        with open(self.path(peer_id, n), "wb") as f:
            pickle.dump({
                "peer_id": drnn.peer_id, "target_len": drnn._target_len, "W": drnn.W,
                "Wxh": drnn.Wxh, "Whh": drnn.Whh, "Why": drnn.Why, "bh": drnn.bh, "by": drnn.by,
                "forced_obf": getattr(drnn, "forced_obf", None),
                "forced_target": getattr(drnn, "forced_target", None),
                "last_training_info": getattr(drnn, "last_training_info", None),
                "saved_at_epoch_ms": __import__("time").time(),
            }, f)

        # write/update audit JSON
        self._write_audit(peer_id, n, drnn)

    def load(self, peer_id, n, ctor):
        p = self.path(peer_id, n)
        with open(p, "rb") as f:
            s = pickle.load(f)
        d = ctor()
        d.peer_id = s["peer_id"]; d._target_len = s["target_len"]; d.W = s["W"]
        d.Wxh = s["Wxh"]; d.Whh = s["Whh"]; d.Why = s["Why"]; d.bh = s["bh"]; d.by = s["by"]
        d.forced_obf = s.get("forced_obf"); d.forced_target = s.get("forced_target")
        d.last_training_info = s.get("last_training_info")
        d._saved_at_epoch_ms = s.get("saved_at_epoch_ms")
        # keep audit JSON as separate artifact; no need to read it here
        return d



# """
# Per-counter ticketed adapters. DEMO ONLY (pickle, no encryption).
# In production: wrap with device-bound key (OS keystore/TEE), add MAC & version.
# """
# import os, pickle
#
# class TicketedAdapters:
#     def __init__(self, dirpath=".adapters"):
#         self.dir = dirpath
#         os.makedirs(self.dir, exist_ok=True)
#
#     def _peer_dir(self, peer_id):
#         p = os.path.join(self.dir, peer_id)
#         os.makedirs(p, exist_ok=True)
#         return p
#
#     def path(self, peer_id, n):
#         return os.path.join(self._peer_dir(peer_id), f"n_{n:08d}.pkl")
#
#     def exists(self, peer_id, n):
#         return os.path.exists(self.path(peer_id, n))
#
#     # ADD two fields when saving:
#     def save(self, peer_id, n, drnn):
#         with open(self.path(peer_id, n), "wb") as f:
#             pickle.dump({
#                 "peer_id": drnn.peer_id, "target_len": drnn._target_len, "W": drnn.W,
#                 "Wxh": drnn.Wxh, "Whh": drnn.Whh, "Why": drnn.Why, "bh": drnn.bh, "by": drnn.by,
#                 # NEW:
#                 "forced_obf": getattr(drnn, "forced_obf", None),
#                 "forced_target": getattr(drnn, "forced_target", None),
#                 # NEW:
#                 "last_training_info": getattr(drnn, "last_training_info", None),
#                 "saved_at_epoch_ms": __import__("time").time(),
#             }, f)
#
#     # On load, restore the forced mapping if present:
#     def load(self, peer_id, n, ctor):
#         p = self.path(peer_id, n)
#         with open(p, "rb") as f:
#             s = pickle.load(f)
#         d = ctor()
#         d.peer_id = s["peer_id"];
#         d._target_len = s["target_len"];
#         d.W = s["W"]
#         d.Wxh = s["Wxh"];
#         d.Whh = s["Whh"];
#         d.Why = s["Why"];
#         d.bh = s["bh"];
#         d.by = s["by"]
#         # NEW:
#         d.forced_obf = s.get("forced_obf")
#         d.forced_target = s.get("forced_target")
#         # NEW:
#         d.last_training_info = s.get("last_training_info")
#         d.saved_at_epoch_ms = s.get("saved_at_epoch_ms")
#         return d
#
#     # def save(self, peer_id, n, drnn):
#     #     with open(self.path(peer_id, n), "wb") as f:
#     #         pickle.dump({
#     #             "peer_id": drnn.peer_id, "target_len": drnn._target_len, "W": drnn.W,
#     #             "Wxh": drnn.Wxh, "Whh": drnn.Whh, "Why": drnn.Why, "bh": drnn.bh, "by": drnn.by
#     #         }, f)
#     #
#     # def load(self, peer_id, n, ctor):
#     #     p = self.path(peer_id, n)
#     #     with open(p, "rb") as f:
#     #         s = pickle.load(f)
#     #     d = ctor()
#     #     d.peer_id = s["peer_id"]; d._target_len = s["target_len"]; d.W = s["W"]
#     #     d.Wxh = s["Wxh"]; d.Whh = s["Whh"]; d.Why = s["Why"]; d.bh = s["bh"]; d.by = s["by"]
#     #     return d
