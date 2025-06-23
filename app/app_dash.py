import streamlit as st
import os
import pandas as pd
import uuid
import random
import time
from utils import log

st.set_page_config("WARL0K Dashboard", layout="wide")
st.title("📡 WARL0K Verification Dashboard")

st.info("Simulated dashboard view")

# Simulated logs
if "log" not in st.session_state:
    st.session_state.log = []

if st.button("Simulate Session"):
    st.session_state.log.append({
        "session": str(uuid.uuid4())[:8],
        "master_recovered": True,
        "payload_verified": True,
        "noise_level": f"{random.randint(15, 25)}%",
        "timestamp": pd.Timestamp.now()
    })

if st.session_state.log:
    df = pd.DataFrame(st.session_state.log)
    st.dataframe(df)
    st.metric("Total Sessions", len(df))
    st.bar_chart(df["noise_level"].str.replace('%','').astype(int))
else:
    st.warning("No session logs yet.")
