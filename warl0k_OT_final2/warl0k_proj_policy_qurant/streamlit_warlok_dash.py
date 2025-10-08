
import re
import json
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="WARL0K Live Dash", layout="wide")
st.title("WARL0K – Session Interaction Dashboard")
st.caption("Device ▶ Interceptor ▶ Gateway | Ephemeral secrets • AEAD policy-binding • On-the-fly validation")

# Sidebar – log selection
default_log = Path("warlok_sample.log")
st.sidebar.header("Log Source")
log_path = st.sidebar.text_input("Path to log file", value=str(default_log))
auto_refresh = st.sidebar.checkbox("Auto-refresh (every 2s)", value=False)
uploaded = st.sidebar.file_uploader("Or upload a log file", type=["log", "txt"])

if uploaded is not None:
    log_text = uploaded.read().decode("utf-8")
else:
    try:
        log_text = Path(log_path).read_text()
    except Exception as e:
        st.error(f"Could not read log: {e}")
        log_text = ""

# Raw log viewer
with st.expander("Raw log (tail)", expanded=True):
    st.text("\n".join(log_text.strip().splitlines()[-200:]))

# --- Parse helpers ---
# We support three major line shapes:
# 1) [ts] [DEVICE] ▶ Published warlok/secrets: seq=1 device=... secret_digest=... cmd=...
# 2) [ts] [INTERCEPTOR] (✅ AUTH | 🚫 QUAR) seq=... device=... cmd=... secret_digest=...
# 3) [ts] [GATEWAY] warlok/(ingest|quarantine): {json}

def parse_lines(lines: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, line in enumerate(lines):
        ts_match = re.match(r"\[(\d+)\]\s+\[(\w+)\]\s+(.*)$", line)
        if not ts_match:
            continue
        ts_int = int(ts_match.group(1))
        actor = ts_match.group(2)
        rest = ts_match.group(3)

        if actor == "DEVICE":
            m = re.search(r"Published\s+([^\s:]+):\s+seq=(\d+)\s+device=([^\s]+)\s+secret_digest=([^\s]+)\s+cmd=(.+)$", rest)
            if m:
                topic, seq, device, digest, cmd = m.groups()
                rows.append({
                    "ts": ts_int, "stage": "DEVICE", "event": "PUBLISHED",
                    "topic": topic, "seq": int(seq), "device": device,
                    "secret_digest": digest, "cmd": cmd, "validated": None, "reason": None
                })
            else:
                # e.g., Using session_seed=...
                m2 = re.search(r"Using session_seed=(\d+)", rest)
                if m2:
                    rows.append({
                        "ts": ts_int, "stage": "DEVICE", "event": "SEED",
                        "topic": None, "seq": None, "device": None,
                        "secret_digest": None, "cmd": None, "validated": None, "reason": None,
                        "session_seed": int(m2.group(1))
                    })

        elif actor == "INTERCEPTOR":
            m = re.search(r"(✅ AUTH|🚫 QUAR)\s+seq=(\d+)\s+device=([^\s]+)\s+cmd=(.+?)\s+secret_digest=([^\s]+)$", rest)
            if m:
                tag, seq, device, cmd, digest = m.groups()
                validated = True if "AUTH" in tag else False
                rows.append({
                    "ts": ts_int, "stage": "INTERCEPTOR", "event": "VALIDATION",
                    "topic": None, "seq": int(seq), "device": device,
                    "secret_digest": digest, "cmd": cmd, "validated": validated, "reason": None if validated else "policy_violation"
                })
            else:
                # Regenerated secret... (keep for context)
                pass

        elif actor == "GATEWAY":
            m = re.search(r"(warlok/ingest|warlok/quarantine):\s+(\{.*\})\s*$", rest)
            if m:
                topic, payload = m.groups()
                try:
                    data = json.loads(payload)
                except Exception:
                    data = {}
                rows.append({
                    "ts": ts_int, "stage": "GATEWAY", "event": "FORWARDED",
                    "topic": topic, "seq": data.get("meta", {}).get("seq"),
                    "device": data.get("meta", {}).get("device_id"),
                    "secret_digest": data.get("secret_digest"),
                    "cmd": data.get("plaintext"),
                    "validated": data.get("validated"),
                    "reason": data.get("reason")
                })
    if not rows:
        return pd.DataFrame(columns=["ts","stage","event","topic","seq","device","secret_digest","cmd","validated","reason"])
    df = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    return df

df = parse_lines(log_text.splitlines())

st.subheader("Parsed Events")
st.dataframe(df, use_container_width=True, hide_index=True)

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
total_msgs = len(df)
total_devices = df["device"].nunique() if "device" in df else 0
auth_rate = float((df["validated"] == True).sum()) / max(1, (df["stage"]=="INTERCEPTOR").sum())
quarantined = int((df["validated"] == False).sum())

col1.metric("Total Events", f"{total_msgs}")
col2.metric("Unique Devices", f"{total_devices}")
col3.metric("Auth Success Rate", f"{auth_rate*100:.1f}%")
col4.metric("Quarantined", f"{quarantined}")

# --- Chart: Secrets vs Devices at each Stage (counts) ---
st.subheader("Secrets vs Devices per Stage")
if not df.empty:
    pivot = df.pivot_table(index="device", columns="stage", values="secret_digest", aggfunc="count", fill_value=0)
    fig1, ax1 = plt.subplots()
    pivot.plot(kind="bar", ax=ax1)
    ax1.set_xlabel("Device")
    ax1.set_ylabel("Event Count")
    ax1.set_title("Event Counts by Stage")
    st.pyplot(fig1)
else:
    st.info("No data to chart.")

# --- Chart: Validation outcomes per device ---
st.subheader("Validation Outcomes per Device")
if not df.empty:
    by_dev = df[df["stage"]=="INTERCEPTOR"].groupby(["device","validated"]).size().unstack(fill_value=0)
    fig2, ax2 = plt.subplots()
    by_dev.plot(kind="bar", ax=ax2)
    ax2.set_xlabel("Device")
    ax2.set_ylabel("Count")
    ax2.set_title("AUTH vs QUAR")
    st.pyplot(fig2)

# --- Timeline chart (seq over time) ---
st.subheader("Timeline – Sequence by Stage")
if not df.empty:
    fig3, ax3 = plt.subplots()
    for stage, dsub in df.dropna(subset=["seq"]).groupby("stage"):
        ax3.plot(dsub["ts"], dsub["seq"], marker="o", linestyle="-", label=stage)
    ax3.set_xlabel("Timestamp")
    ax3.set_ylabel("Sequence")
    ax3.set_title("Per-Stage Sequence Timeline")
    ax3.legend()
    st.pyplot(fig3)

# --- Intuitive Insights ---
st.subheader("Intuitive Insights")
insights = []

if (df["stage"]=="INTERCEPTOR").any():
    # Who is noisy? (highest QUAR rate)
    val_df = df[df["stage"]=="INTERCEPTOR"]
    rate = val_df.groupby("device")["validated"].apply(lambda s: 1.0 - (s.sum()/len(s)))
    noisiest = rate.sort_values(ascending=False).head(3)
    for dev, r in noisiest.items():
        insights.append(f"Device **{dev}** shows a quarantine rate of **{r*100:.1f}%** — likely out-of-policy commands or drift.")
    # Repeated secret digest reuse?
    dup = val_df.groupby(["device","secret_digest"]).size().reset_index(name="n").query("n>1")
    if len(dup):
        insights.append("Repeated **secret_digest** reuse detected (possible replay risk if seeds are weak):")
        for _, row in dup.iterrows():
            insights.append(f"- {row['device']} digest {row['secret_digest']} seen {row['n']}×")
else:
    insights.append("No interceptor validations detected. Ensure the interceptor is active and logging.")

if not insights:
    insights.append("Traffic looks healthy. No obvious policy hotspots detected.")

for i in insights:
    st.markdown(f"- {i}")

# Footer controls
st.sidebar.download_button("Download current parsed CSV", data=df.to_csv(index=False), file_name="warlok_parsed.csv", mime="text/csv")
st.sidebar.download_button("Download sample log", data=Path("warlok_sample.log").read_text(), file_name="warlok_sample.log", mime="text/plain")

if auto_refresh:
    import time as _time
    _time.sleep(2)
    st.rerun()
