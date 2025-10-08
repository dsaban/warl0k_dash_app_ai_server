import os, random, yaml
from warlok.peer_core import PeerCore
from warlok.crypto import rand, gen_x25519_keypair, pub_bytes_x25519, hexlify
from warlok.models.seed2master_model import Seed2MasterModel
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.crypto import hmac_sha256  # used in InMemHub

DEFAULT_CFG = {
    "session": {"policy_id":"PAYMENT:limit=100","obf_len":12,"counter_start":1},
    "models": {"master_len_chars":32,"drnn_hidden_dim":32,"drnn_lr":0.02,"drnn_epochs_per_session":400},
    "general": {"verbose": True, "demo_plaintext": "transfer 25 units"},
}

def load_cfg():
    try:
        with open("config.yaml","r") as f:
            base = yaml.safe_load(f)
        def deep_merge(a,b):
            for k,v in b.items():
                if k not in a: a[k]=v
                elif isinstance(v,dict): deep_merge(a[k],v)
        deep_merge(base, DEFAULT_CFG); return base
    except Exception:
        return DEFAULT_CFG

CFG = load_cfg()
VERBOSE = CFG["general"]["verbose"]
POLICY  = CFG["session"]["policy_id"]
OBF_LEN = CFG["session"]["obf_len"]
MASTER_LEN = CFG["models"]["master_len_chars"]
EPOCHS = CFG["models"]["drnn_epochs_per_session"]

def log(*a):
    if VERBOSE: print(*a)

# In-memory Hub (seed0 registry and Seed→Master vector service)
class InMemHub:
    def __init__(self): self.registry={}
    def enroll(self, dev):
        s = rand(32); self.registry[dev]=s; return s
    def get_W(self, target_id):
        seed0 = self.registry.get(target_id)
        return None if seed0 is None else hmac_sha256(seed0, b"W", target_id.encode())

def run_session(hub, A="device-A", B="device-B") -> bool:
    # Ephemeral handshake
    a_priv,a_pub = gen_x25519_keypair()
    b_priv,b_pub = gen_x25519_keypair()
    A_pub = pub_bytes_x25519(a_pub)
    B_pub = pub_bytes_x25519(b_pub)
    counter = CFG["session"]["counter_start"]; challenge = rand(16)

    pcA = PeerCore(A); pcB = PeerCore(B)
    kA = pcA.derive_k_session(a_priv, B_pub, A, B, POLICY, counter, challenge)
    kB = pcB.derive_k_session(b_priv, A_pub, B, A, POLICY, counter, challenge)
    if kA != kB: log("[x] shared mismatch"); return False
    obf = pcA.obf_from_k(kA, OBF_LEN)

    # Seed-path masters
    W_B = hub.get_W(B); W_A = hub.get_W(A)
    if W_A is None or W_B is None: log("[x] missing W"); return False
    m_seed_B = Seed2MasterModel(B, W_B, MASTER_LEN).compute_master()
    m_seed_A = Seed2MasterModel(A, W_A, MASTER_LEN).compute_master()

    # Sess-path DRNNs (early-stop)
    # drnnA = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
    # drnnB = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
    # infoA = drnnA.train_pair(obf, m_seed_B, epochs=EPOCHS)
    # infoB = drnnB.train_pair(obf, m_seed_A, epochs=EPOCHS)
    # m_sess_B = drnnA.predict(obf, out_len=MASTER_LEN)
    # m_sess_A = drnnB.predict(obf, out_len=MASTER_LEN)
    # Sess-path (deterministic)
    drnnA = Sess2MasterDRNN()
    drnnA.set_context(peer_id=B, W_bytes=W_B, target_len_chars=MASTER_LEN)
    infoA = drnnA.train_pair(obf, m_seed_B, epochs=1)
    m_sess_B = drnnA.predict(obf, out_len=MASTER_LEN)
    
    drnnB = Sess2MasterDRNN()
    drnnB.set_context(peer_id=A, W_bytes=W_A, target_len_chars=MASTER_LEN)
    infoB = drnnB.train_pair(obf, m_seed_A, epochs=1)
    m_sess_A = drnnB.predict(obf, out_len=MASTER_LEN)
    
    if VERBOSE:
        print("A DRNN:", infoA, "| eq:", m_sess_B==m_seed_B)
        print("B DRNN:", infoB, "| eq:", m_sess_A==m_seed_A)

    if not (m_sess_B==m_seed_B and m_sess_A==m_seed_A):
        log(f"[!] map fail A:{m_sess_B==m_seed_B} B:{m_sess_A==m_seed_A}")
        return False

    # AEAD envelope test
    env = pcA.build_envelope(A,B,a_priv,B_pub,POLICY,counter,challenge, CFG["general"]["demo_plaintext"].encode())
    ok, pt = pcB.verify_envelope(env,B,A,b_priv,A_pub)
    return bool(ok)

def main():
    random.seed(1337)
    hub = InMemHub(); hub.enroll("device-A"); hub.enroll("device-B")
    N=10; succ=0
    for i in range(1,N+1):
        ok = run_session(hub); succ += 1 if ok else 0
        print(f"session {i:02d}: {'OK' if ok else 'FAIL'}")
    print(f"\nSummary: {succ}/{N} successful sessions ({(succ/N)*100:.1f}% success rate)")

if __name__=="__main__":
    main()
    
