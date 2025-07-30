
import streamlit as st
import uuid, torch, socket, pickle, time, os, random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from new_RNN_Tiny_models import GatedRNNBroker, TinyModelMCU
from model import train_secret_regenerator, evaluate_secret_regenerator, inject_patterned_noise, add_noise_to_tensor
from utils import generate_secret, aead_encrypt, aead_decrypt, log, log_client

# --- Setup ---
st.set_page_config("WARL0K Unified Dashboard", layout="wide")
vocab = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*=+[]{}|;:<>?~")

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    seed = st.number_input("Seed", value=42, step=1)
    secret_len = st.slider("Secret Length", 2, 16, 8)
    epochs = st.slider("Training Epochs", 100, 2000, 1000, 100)
    broker_dim = st.slider("Broker Hidden", 8, 128, 64)
    mcu_dim = st.slider("MCU Hidden", 4, 32, 8)
    if st.button("▶ Run Joint Process"):
        st.session_state.run = True
    if st.button("🔁 Reset"):
        st.session_state.clear()
        st.rerun()

# --- Session Setup ---
session_id = str(uuid.uuid4())[:8]
if "run" in st.session_state:

    col1, col2 = st.columns(2)

    # --- LEFT COLUMN: MCU + BROKER ---
    with col1:
        st.header("🔐 MCU + Gateway")
        mcu = TinyModelMCU(vocab, hidden_dim=mcu_dim, seed=seed)
        fragments, frag_strs = mcu.generate_3_secrets(length=secret_len)
        full_secret = ''.join(frag_strs)
        broker = GatedRNNBroker(vocab, hidden_dim=broker_dim, lr=0.01)
        broker.train(full_secret, epochs=epochs)
        recon = broker.fingerprint_verify(full_secret)
        #  generate a validation flag for iteration
        valid = recon == full_secret
        print (f"Validation: {valid}")

        st.subheader("🔒 MCU Secret Fragments")
        for i, frag in enumerate(frag_strs):
            st.code(f"Fragment {i+1}: {frag}")
        st.markdown(f"**Concatenated:** `{full_secret}`")

        st.subheader("📡 Gateway Verification")
        st.code(f"Reconstructed: {recon}")
        st.markdown(f"**Verification Result:** {'✅ Match' if valid else '❌ Mismatch'}")
        # st.success("✅ Match") if valid is True else st.error("❌ Mismatch")
        

        st.caption(f"MCU RAM: {mcu.ram_usage_kb():.2f} KB | Broker RAM: {broker.ram_usage_kb():.2f} KB")

    # --- RIGHT COLUMN: Session Simulation ---
    with col2:
        st.header("🧬 Session Auth Pipeline")

        master = generate_secret()
        obfs = generate_secret()
        model_master = train_secret_regenerator(master, vocab)
        model_obfs = train_secret_regenerator(master, vocab, input_override=obfs)

        st.code(f"Obfuscated Secret: {obfs}")

        torch.manual_seed(int(session_id, 16))
        noisy = inject_patterned_noise(torch.tensor([vocab.index(c) for c in obfs]), len(vocab), 0.25, 0.6, session_id)
        noisy_text = ''.join([vocab[i] for i in noisy.tolist()])
        fingerprint = add_noise_to_tensor(torch.tensor([vocab.index(c) for c in obfs]), len(vocab)).unsqueeze(1)

        st.subheader("🔍 Fingerprint Alignment")
        df = pd.DataFrame({
            "Index": list(range(len(obfs))),
            "Obfuscated": [vocab.index(c) for c in obfs],
            "Noisy": noisy.tolist(),
            "Fingerprint": fingerprint.squeeze(1).tolist()
        })
        st.line_chart(df.set_index("Index"))

        st.subheader("📥 Encrypted Payload")
        key = fingerprint.numpy().tobytes()[:16]
        encrypted = aead_encrypt(key, b"Payload: Hello WARL0K")
        recovered = evaluate_secret_regenerator(model_obfs, fingerprint, vocab)
        status = "✅ AUTH OK" if recovered == master else "❌ AUTH FAIL"

        st.code(f"Recovered: {recovered}")
        st.code(encrypted, language="bash")
        st.markdown(f"**Status:** {status}")
        # st.success(status) if "OK" in status else st.error(status)

        # Network map for session
        st.subheader("🌐 Network Map")
        net_df = pd.DataFrame({
            "Client": [f"Client-{i}" for i in range(5)],
            "IP": [f"192.168.1.{i+100}" for i in range(5)],
            "Status": random.choices(["Active", "Idle", "Dropped"], k=5),
            "Latency": [random.randint(10, 90) for _ in range(5)]
        })
        fig = px.scatter(net_df, x="Latency", y="Client", color="Status", size="Latency", hover_data=["IP"])
        st.plotly_chart(fig, use_container_width=True)
