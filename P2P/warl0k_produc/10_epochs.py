import os, random, yaml
from warlok.peer_core import PeerCore
from warlok.crypto import rand, gen_x25519_keypair, pub_bytes_x25519
from warlok.models.seed2master_model import Seed2MasterModel
from warlok.models.sess2master_drnn import Sess2MasterDRNN
from warlok.crypto import hmac_sha256

DEFAULT_CFG = {
    "session": {"policy_id":"PAYMENT:limit=100","obf_len":12,"counter_start":1},
    "models": {"master_len_chars":32,"drnn_hidden_dim":32,"drnn_lr":0.02,"drnn_epochs_per_session":400},
    "general": {"verbose": False, "demo_plaintext": "transfer 25 units"},
}

def load_cfg():
    try:
        with open("config.yaml","r") as f:
            base = yaml.safe_load(f)
        def deep_merge(a,b):
            for k,v in b.items():
                if k not in a: a[k]=v
                elif isinstance(v,dict): deep_merge(a[k],v)
        deep_merge(base, DEFAULT_CFG)
        return base
    except Exception:
        return DEFAULT_CFG

CFG = load_cfg()
POLICY  = CFG["session"]["policy_id"]
OBF_LEN = CFG["session"]["obf_len"]
MASTER_LEN = CFG["models"]["master_len_chars"]
EPOCHS = CFG["models"]["drnn_epochs_per_session"]

# In-memory hub
class InMemHub:
    def __init__(self): self.registry={}
    def enroll(self, dev): self.registry[dev]=rand(32)
    def get_W(self, target_id):
        seed0 = self.registry.get(target_id)
        return None if seed0 is None else hmac_sha256(seed0, b"W", target_id.encode())

def run_session(hub, A="device-A", B="device-B"):
    a_priv,a_pub = gen_x25519_keypair()
    b_priv,b_pub = gen_x25519_keypair()
    A_pub = pub_bytes_x25519(a_pub); B_pub = pub_bytes_x25519(b_pub)
    counter = CFG["session"]["counter_start"]; challenge = rand(16)

    pcA = PeerCore(A); pcB = PeerCore(B)
    kA = pcA.derive_k_session(a_priv, B_pub, A,B,POLICY,counter,challenge)
    kB = pcB.derive_k_session(b_priv, A_pub, B,A,POLICY,counter,challenge)
    obf = pcA.obf_from_k(kA, OBF_LEN)

    W_B = hub.get_W(B); W_A = hub.get_W(A)
    m_seed_B = Seed2MasterModel(B, W_B, MASTER_LEN).compute_master()
    m_seed_A = Seed2MasterModel(A, W_A, MASTER_LEN).compute_master()

    drnnA = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])
    drnnB = Sess2MasterDRNN(hidden_dim=CFG["models"]["drnn_hidden_dim"], lr=CFG["models"]["drnn_lr"])

    infoA = drnnA.train_pair(obf, m_seed_B, epochs=EPOCHS)
    infoB = drnnB.train_pair(obf, m_seed_A, epochs=EPOCHS)

    return infoA["epochs_run"], infoB["epochs_run"], infoA["early_stopped"], infoB["early_stopped"]

def main():
    random.seed(42)
    hub = InMemHub(); hub.enroll("device-A"); hub.enroll("device-B")

    N=10
    totalA=0; totalB=0
    print("Session | Epochs(A) ES? | Epochs(B) ES?")
    print("----------------------------------------")
    for i in range(1,N+1):
        eA,eB,esA,esB = run_session(hub)
        totalA += eA; totalB += eB
        print(f"{i:>7} | {eA:>8} {'✓' if esA else ''} | {eB:>8} {'✓' if esB else ''}")
    print("----------------------------------------")
    print(f"Avg epochs A: {totalA/N:.1f}, B: {totalB/N:.1f}")

if __name__=="__main__":
    main()
