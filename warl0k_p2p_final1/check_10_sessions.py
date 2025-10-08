import os, random, yaml, binascii
from warlok.peer_core import PeerCore
from warlok.crypto import rand, gen_x25519_keypair, pub_bytes_x25519, hmac_sha256
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.pretrain import Pretrainer, obf_ticket, master_seedpath, hexs
from warlok.storage import TicketedAdapters

DEFAULT_CFG = {
    "session": {"policy_id":"PAYMENT:limit=100","obf_len":16,"counter_start":1,"pretrain_window":8},
    "models": {"master_len_chars":32,"drnn_hidden_dim":48,"drnn_lr":0.05,"drnn_epochs_per_session":20,"drnn_meta_samples":8,"drnn_meta_steps":40},
    "general": {"verbose": True, "demo_plaintext": "transfer 25 units"},
}

def load_cfg():
    try:
        with open("config.yaml","r") as f:
            base = yaml.safe_load(f)
        for k,v in DEFAULT_CFG.items():
            base.setdefault(k,v)
            if isinstance(v,dict): base[k] = {**v, **base[k]}
        return base
    except Exception:
        return DEFAULT_CFG

CFG = load_cfg()
POLICY  = CFG["session"]["policy_id"]
OBF_LEN = CFG["session"]["obf_len"]
MASTER_LEN = CFG["models"]["master_len_chars"]
M_SAMP = CFG["models"]["drnn_meta_samples"]; M_STEPS = CFG["models"]["drnn_meta_steps"]
WIN = CFG["session"]["pretrain_window"]
VERBOSE = CFG["general"]["verbose"]

def log(*a):
    if VERBOSE: print(*a)

class InMemHub:
    def __init__(self): self.registry={}
    def enroll(self, dev): self.registry[dev]=rand(32)
    def get_W(self, target_id):
        from warlok.crypto import hmac_sha256
        seed0 = self.registry.get(target_id)
        return None if seed0 is None else hmac_sha256(seed0, b"W", target_id.encode())

def run_session(hub, storeA, storeB, n, A="device-A", B="device-B") -> bool:
    a_priv,a_pub = gen_x25519_keypair()
    b_priv,b_pub = gen_x25519_keypair()
    A_pub = pub_bytes_x25519(a_pub); B_pub = pub_bytes_x25519(b_pub)
    challenge = rand(16)

    pcA = PeerCore(A); pcB = PeerCore(B)

    W_B = hub.get_W(B); W_A = hub.get_W(A)
    preA = Pretrainer(storeA, OBF_LEN, MASTER_LEN, CFG["models"]["drnn_hidden_dim"], CFG["models"]["drnn_lr"])
    preB = Pretrainer(storeB, OBF_LEN, MASTER_LEN, CFG["models"]["drnn_hidden_dim"], CFG["models"]["drnn_lr"])
    preA.schedule_window(A, B, W_B, start_n=n, window=WIN, meta=(M_SAMP, M_STEPS))
    preB.schedule_window(B, A, W_A, start_n=n, window=WIN, meta=(M_SAMP, M_STEPS))

    mB = master_seedpath(W_B, B, MASTER_LEN); mA = master_seedpath(W_A, A, MASTER_LEN)
    obf_train_AB = obf_ticket(W_B, n, OBF_LEN); obf_train_BA = obf_ticket(W_A, n, OBF_LEN)
    dA = storeA.load(B, n, ctor=lambda: Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"]))
    dB = storeB.load(A, n, ctor=lambda: Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"]))
    if dA.predict(obf_train_AB, MASTER_LEN) != mB: return False
    if dB.predict(obf_train_BA, MASTER_LEN) != mA: return False

    kA = pcA.derive_k_session(a_priv, B_pub, A,B,POLICY,n,challenge)
    kB = pcB.derive_k_session(b_priv, A_pub, B,A,POLICY,n,challenge)
    if kA != kB: return False
    obf_real = pcA.obf_from_k(kA, OBF_LEN)

    transcript = f"{A}|{B}|{POLICY}|{n}|{binascii.hexlify(challenge).decode()}".encode()
    K_tag_A = hmac_sha256(W_B, b"sessK", n.to_bytes(8,"big"), obf_real.encode())
    Tag_A = hexs(hmac_sha256(K_tag_A, b"TAG", transcript))
    K_tag_B = hmac_sha256(W_A, b"sessK", n.to_bytes(8,"big"), obf_real.encode())
    Tag_B = hexs(hmac_sha256(K_tag_B, b"TAG", transcript))

    if Tag_A != hexs(hmac_sha256(K_tag_A, b"TAG", transcript)): return False
    if Tag_B != hexs(hmac_sha256(K_tag_B, b"TAG", transcript)): return False

    env = pcA.build_envelope(A,B,a_priv,B_pub,POLICY,n,challenge, CFG["general"]["demo_plaintext"].encode())
    ok, _ = pcB.verify_envelope(env,B,A,b_priv,A_pub)
    return bool(ok)

def main():
    random.seed(99)
    hub = InMemHub(); hub.enroll("device-A"); hub.enroll("device-B")
    storeA = TicketedAdapters(".adapters_A"); storeB = TicketedAdapters(".adapters_B")
    N=10; succ=0
    for i in range(N):
        ok = run_session(hub, storeA, storeB, n=CFG["session"]["counter_start"]+i)
        succ += 1 if ok else 0
        print(f"session {i+1:02d}: {'OK' if ok else 'FAIL'}")
    print(f"\nSummary: {succ}/{N} successful sessions ({(succ/N)*100:.1f}% success rate)")

if __name__=="__main__":
    main()
