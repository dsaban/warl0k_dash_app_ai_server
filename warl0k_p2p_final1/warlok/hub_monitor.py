# warlok/hub_monitor.py
import os, ssl, socket, json, time, random, threading, hashlib
from typing import Dict, Callable, Optional
from .net import send_msg, recv_msg
from .storage import TicketedAdapters
from .pretrain import Pretrainer, obf_ticket, master_seedpath
from .models.sess2master_drnn import Sess2MasterDRNN

class HubMonitor:
    """
    Background thread that:
      - polls hub for W(peer_id) for a configured list of peers
      - detects init/rotation by W.sha256 change
      - (re)trains adapters for the next N tickets proactively
      - optionally keeps overlap tickets during rotation (graceful mode)
    """
    def __init__(
        self,
        my_device_id: str,
        hub_host: str,
        hub_port: int,
        ca_cert: str = "ca.crt",
        client_cert: Optional[str] = "peer.crt",
        client_key: Optional[str] = "peer.key",
        adapters_dir: str = ".adapters",
        obf_len: int = 16,
        master_len: int = 32,
        drnn_hidden: int = 48,
        drnn_lr: float = 0.05,
        poll_interval: int = 5,
        jitter_sec: int = 2,
        pretrain_window: int = 8,
        rollout_mode: str = "graceful",          # graceful|force
        overlap_tickets: int = 4,
        peers_to_watch: Optional[list] = None,
        get_next_counter: Optional[Callable[[], int]] = None,
        log: Optional[Callable[[str], None]] = print,
    ):
        self.my_id = my_device_id
        self.hub_host = hub_host
        self.hub_port = hub_port
        self.ca_cert = ca_cert
        self.client_cert = client_cert
        self.client_key = client_key
        self.adapters_dir = adapters_dir
        self.obf_len = obf_len
        self.master_len = master_len
        self.drnn_hidden = drnn_hidden
        self.drnn_lr = drnn_lr
        self.poll_interval = poll_interval
        self.jitter_sec = jitter_sec
        self.pretrain_window = pretrain_window
        self.rollout_mode = rollout_mode
        self.overlap_tickets = overlap_tickets
        self.peers = peers_to_watch or []
        self.get_next_counter = get_next_counter or (lambda: 1)
        self.log = log or (lambda *_: None)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._w_cache_path = os.path.join(self.adapters_dir, "_w_cache.json")
        self._w_cache = self._load_cache()

        self.store = TicketedAdapters(dirpath=self.adapters_dir)
        self.pre = Pretrainer(self.store, self.obf_len, self.master_len, self.drnn_hidden, self.drnn_lr)

    # -------- hub calls --------
    def _hub_ctx(self):
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=self.ca_cert)
        if self.client_cert and self.client_key and os.path.exists(self.client_cert) and os.path.exists(self.client_key):
            ctx.load_cert_chain(certfile=self.client_cert, keyfile=self.client_key)
        return ctx

    def _hub_rpc(self, req: dict) -> dict:
        s = socket.socket()
        tls = self._hub_ctx().wrap_socket(s, server_hostname="hub")
        tls.connect((self.hub_host, self.hub_port))
        send_msg(tls, req)
        res = recv_msg(tls)
        tls.close()
        return res

    def _get_W_hex(self, device_id: str) -> Optional[str]:
        r = self._hub_rpc({"cmd": "get_seed2master_vec", "target_device_id": device_id})
        if r.get("status") != "ok":
            return None
        return r["W_hex"]

    # -------- cache --------
    def _load_cache(self) -> Dict[str, dict]:
        try:
            with open(self._w_cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cache(self):
        os.makedirs(self.adapters_dir, exist_ok=True)
        tmp = self._w_cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._w_cache, f, indent=2, sort_keys=True)
        os.replace(tmp, self._w_cache_path)

    # -------- lifecycle --------
    def start(self):
        self._thread.start()
        self.log(f"[monitor] started for peers={self.peers}, window={self.pretrain_window}, mode={self.rollout_mode}")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    # -------- core loop --------
    def _run(self):
        while not self._stop.is_set():
            try:
                # jitter to avoid sync storms
                time.sleep(self.poll_interval + random.uniform(0, self.jitter_sec))
                next_n = self.get_next_counter()

                for peer_id in self.peers:
                    W_hex = self._get_W_hex(peer_id)
                    if not W_hex:
                        self.log(f"[monitor] hub has no W for {peer_id} yet")
                        continue
                    W_bytes = bytes.fromhex(W_hex)
                    W_sha = hashlib.sha256(W_bytes).hexdigest()

                    cache = self._w_cache.get(peer_id)
                    if cache is None:
                        # first sighting: init
                        self._w_cache[peer_id] = {"W_sha256": W_sha, "seen_at": time.time()}
                        self._save_cache()
                        self.log(f"[monitor] init W for {peer_id} sha={W_sha[:12]}..; pretraining n=[{next_n}..{next_n+self.pretrain_window-1}]")
                        self._pretrain_window(peer_id, W_bytes, next_n)
                        continue

                    if cache.get("W_sha256") != W_sha:
                        # rotation detected
                        old_sha = cache.get("W_sha256")
                        self._w_cache[peer_id]["W_sha256"] = W_sha
                        self._w_cache[peer_id]["rotated_at"] = time.time()
                        self._save_cache()
                        self.log(f"[monitor] ROTATION for {peer_id} old={old_sha[:12]}.. new={W_sha[:12]}..")

                        # Pretrain upcoming window for new W
                        self._pretrain_window(peer_id, W_bytes, next_n)

                        # Optionally retain overlap tickets for the old generation (graceful)
                        if self.rollout_mode == "graceful" and self.overlap_tickets > 0:
                            self.log(f"[monitor] graceful overlap keeps last {self.overlap_tickets} tickets of old generation")
                            # do nothing else; your runtime verification will accept both during overlap
                        else:
                            # force: drop old adapters immediately (only for this peer)
                            self._purge_old_generation(peer_id, keep_last=self.overlap_tickets)

                    else:
                        # steady state: ensure window is ready
                        self._pretrain_window(peer_id, W_bytes, next_n)

            except Exception as e:
                self.log(f"[monitor] error: {e}")

    # -------- helpers --------
    def _pretrain_window(self, peer_id: str, W_peer: bytes, start_n: int):
        # builds or self-heals adapters for [start_n .. start_n+window-1]
        self.pre.schedule_window(owner_id=self.my_id, peer_id=peer_id, W_peer=W_peer,
                                 start_n=start_n, window=self.pretrain_window, meta=(8, 40))

    def _purge_old_generation(self, peer_id: str, keep_last: int = 0):
        """
        Optional: remove adapters that belong to a different W generation.
        For simplicity, we keep the last N tickets regardless; safe in demos.
        """
        # In a production system, you'd tag adapters with W_sha in filename or metadata and
        # remove those that don't match the current cache W_sha, except last 'keep_last' counters.
        # Here we no-op to keep the demo simple.
        self.log(f"[monitor] (info) purge_old_generation is a NO-OP in demo. Implement as needed.")
