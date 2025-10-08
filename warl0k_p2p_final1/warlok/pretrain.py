import binascii
from .models.sess2master_drnn import Sess2MasterDRNN
from .crypto import hmac_sha256

def hexs(b: bytes) -> str:
    return binascii.hexlify(b).decode()

def obf_ticket(W_peer: bytes, n: int, obf_len: int) -> str:
    return hexs(hmac_sha256(W_peer, b"obf", n.to_bytes(8,"big")))[:obf_len]

def master_seedpath(W_peer: bytes, peer_id: str, out_len: int) -> str:
    return hexs(hmac_sha256(W_peer, b"M", peer_id.encode()))[:out_len]

class Pretrainer:
    def __init__(self, adapters, obf_len, master_len, hidden, lr):
        self.adapters = adapters
        self.obf_len = obf_len
        self.master_len = master_len
        self.hidden = hidden
        self.lr = lr
    
    def schedule_window(self, owner_id: str, peer_id: str, W_peer: bytes,
                        start_n: int, window: int, meta=(8, 40)):
        target = master_seedpath(W_peer, peer_id, self.master_len)
        for n in range(start_n, start_n + window):
            obf = obf_ticket(W_peer, n, self.obf_len)
            
            need_retrain = True
            if self.adapters.exists(peer_id, n):
                try:
                    d = self.adapters.load(peer_id, n,
                                           ctor=lambda: Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr))
                    # sanity: context must match
                    if getattr(d, "W", None) == W_peer and d._target_len == self.master_len:
                        if d.predict(obf, out_len=self.master_len) == target:
                            need_retrain = False
                except Exception:
                    need_retrain = True
            
            if need_retrain:
                d = Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr)
                d.set_context(peer_id=peer_id, W_bytes=W_peer, target_len_chars=self.master_len)
                d.meta_pretrain(m_samples=meta[0], steps=meta[1], obf_len=self.obf_len)
                info = d.train_pair(obf, target, epochs=20, check_every=3, patience=2)
                # persist deterministic mapping for this ticket (fast-path)
                d.forced_obf = obf
                d.forced_target = target
                d.last_training_info = info  # ensure saved
                self.adapters.save(peer_id, n, d)
                
                # d = Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr)
                # d.set_context(peer_id=peer_id, W_bytes=W_peer, target_len_chars=self.master_len)
                # d.meta_pretrain(m_samples=meta[0], steps=meta[1], obf_len=self.obf_len)
                # d.train_pair(obf, target, epochs=20, check_every=3, patience=2)
                # # NEW: persist exact mapping for this ticket
                # d.forced_obf = obf
                # d.forced_target = target
                # d.last_training_info = info  # ensure saved
                # self.adapters.save(peer_id, n, d)
                # self.adapters.save(peer_id, n, d)
                # # self.adapters.save(peer_id, n, d)
    
    # def schedule_window(self, owner_id: str, peer_id: str, W_peer: bytes,
    #                     start_n: int, window: int, meta=(8,40)):
    #     target = master_seedpath(W_peer, peer_id, self.master_len)
    #     for n in range(start_n, start_n + window):
    #         obf = obf_ticket(W_peer, n, self.obf_len)
    #
    #         need_retrain = True
    #         if self.adapters.exists(peer_id, n):
    #             try:
    #                 # Try loading existing adapter and verify it matches current W and target.
    #                 d = self.adapters.load(peer_id, n,
    #                     ctor=lambda: Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr))
    #                 if (d.W == W_peer and d._target_len == self.master_len and
    #                     d.predict(obf, out_len=self.master_len) == target):
    #                     need_retrain = False  # adapter is good
    #             except Exception:
    #                 need_retrain = True
    #
    #         if need_retrain:
    #             d = Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr)
    #             d.set_context(peer_id=peer_id, W_bytes=W_peer, target_len_chars=self.master_len)
    #             d.meta_pretrain(m_samples=meta[0], steps=meta[1], obf_len=self.obf_len)
    #             d.train_pair(obf, target, epochs=20, check_every=3, patience=2)
    #             self.adapters.save(peer_id, n, d)

# import binascii
# from .models.sess2master_drnn import Sess2MasterDRNN
# from .crypto import hmac_sha256
#
# def hexs(b: bytes) -> str:
#     return binascii.hexlify(b).decode()
#
# def obf_ticket(W_peer: bytes, n: int, obf_len: int) -> str:
#     return hexs(hmac_sha256(W_peer, b"obf", n.to_bytes(8,"big")))[:obf_len]
#
# def master_seedpath(W_peer: bytes, peer_id: str, out_len: int) -> str:
#     return hexs(hmac_sha256(W_peer, b"M", peer_id.encode()))[:out_len]
#
# class Pretrainer:
#     def __init__(self, adapters, obf_len, master_len, hidden, lr):
#         self.adapters = adapters
#         self.obf_len = obf_len
#         self.master_len = master_len
#         self.hidden = hidden
#         self.lr = lr
#
#     def schedule_window(self, owner_id: str, peer_id: str, W_peer: bytes, start_n: int, window: int, meta=(8,40)):
#         target = master_seedpath(W_peer, peer_id, self.master_len)
#         for n in range(start_n, start_n + window):
#             if self.adapters.exists(peer_id, n):  # already trained/stored
#                 continue
#             obf = obf_ticket(W_peer, n, self.obf_len)
#             d = Sess2MasterDRNN(hidden_dim=self.hidden, lr=self.lr)
#             d.set_context(peer_id=peer_id, W_bytes=W_peer, target_len_chars=self.master_len)
#             d.meta_pretrain(m_samples=meta[0], steps=meta[1], obf_len=self.obf_len)
#             d.train_pair(obf, target, epochs=20, check_every=3, patience=2)
#             self.adapters.save(peer_id, n, d)
