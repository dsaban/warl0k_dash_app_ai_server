# app_pim_inspector.py
# WARL0K PIM + MLEI Demo (Enhanced PIM Understanding UI)
#
# Drop this file into the project root (same folder as run_demo.py/config.py),
# then run:
#   pip install -r requirements.txt
#   pip install streamlit pandas
#   streamlit run app_pim_inspector.py

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import pandas as pd
import streamlit as st

from config import CFG
from common.crypto import CryptoBox
from common.protocol import SecureChannel
from common.util import canon_json, sha256_hex, now_ts
from common.nano_gate import NanoGate
from db.csv_store import CsvStore
from cloud.llm_cloud_mock import llm_agent_plan
from attacks.injector import (
    attack_prompt_injection,
    attack_tool_swap_to_unauthorized,
    attack_tamper_args,
    attack_delay,
)

# -----------------------------
# PIM Core (enhanced verifier report)
# -----------------------------
@dataclass
class PIMState:
    sid: str
    ctr: int = 0
    last_hash: str = "GENESIS"
    anchor: str = "ANCHOR0"  # for window sealing/rotation
    window_idx: int = 0
    window_hashes: List[str] = None

    def __post_init__(self):
        if self.window_hashes is None:
            self.window_hashes = []

def pim_core_from_env(env: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sid": env["sid"],
        "ctr": env["ctr"],
        "ts": env["ts"],
        "prev": env["prev"],
        "payload": env["payload"],
    }

def build_env(state: PIMState, payload: Dict[str, Any]) -> Dict[str, Any]:
    ts = now_ts()
    ctr = state.ctr + 1
    core = {
        "sid": state.sid,
        "ctr": ctr,
        "ts": ts,
        "prev": state.last_hash,
        "payload": payload,
    }
    h = sha256_hex(canon_json(core))
    env = dict(core)
    env["h"] = h
    return env

def advance_state(state: PIMState, env: Dict[str, Any]) -> None:
    state.ctr = env["ctr"]
    state.last_hash = env["h"]

def verify_env_report(state: PIMState, env: Dict[str, Any], max_skew_s: float) -> Dict[str, Any]:
    """
    Structured verifier report to make PIM understandable in UI.
    """
    report: Dict[str, Any] = {
        "ok": True,
        "checks": {"sid": True, "ctr": True, "prev": True, "skew": True, "hash": True},
        "expected": {},
        "got": {},
        "skew_s": None,
        "reason": "OK",
        "computed_hash": None,
        "canonical_core": None,
    }

    # Defensive reads
    got_sid = env.get("sid")
    got_ctr = env.get("ctr")
    got_prev = env.get("prev")
    got_ts = float(env.get("ts", 0.0))
    got_h = env.get("h")

    report["got"] = {"sid": got_sid, "ctr": got_ctr, "prev": got_prev, "h": got_h, "ts": got_ts}
    report["expected"] = {"sid": state.sid, "ctr": state.ctr + 1, "prev": state.last_hash}

    # sid
    if got_sid != state.sid:
        report["ok"] = False
        report["checks"]["sid"] = False

    # ctr
    if got_ctr != state.ctr + 1:
        report["ok"] = False
        report["checks"]["ctr"] = False

    # prev
    if got_prev != state.last_hash:
        report["ok"] = False
        report["checks"]["prev"] = False

    # skew
    skew = abs(now_ts() - got_ts)
    report["skew_s"] = float(skew)
    if skew > max_skew_s:
        report["ok"] = False
        report["checks"]["skew"] = False

    # hash check
    try:
        core = pim_core_from_env(env)
        report["canonical_core"] = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        computed = sha256_hex(canon_json(core))
        report["computed_hash"] = computed
        if computed != got_h:
            report["ok"] = False
            report["checks"]["hash"] = False
    except Exception:
        report["ok"] = False
        report["checks"]["hash"] = False

    # reason (first failure)
    if not report["ok"]:
        for k in ["sid", "ctr", "prev", "skew", "hash"]:
            if not report["checks"][k]:
                if k == "sid":
                    report["reason"] = "PIM: session_id mismatch"
                elif k == "ctr":
                    report["reason"] = f"PIM: counter mismatch (expected {state.ctr + 1}, got {got_ctr})"
                elif k == "prev":
                    report["reason"] = "PIM: prev-hash mismatch"
                elif k == "skew":
                    report["reason"] = f"PIM: timestamp skew too large ({skew:.3f}s)"
                else:
                    report["reason"] = "PIM: hash mismatch"
                break

    return report

# -----------------------------
# Window sealing / anchor rotation (Proof-in-Motion flavor)
# -----------------------------
def maybe_window_seal(state: PIMState, window_size: int) -> Optional[Dict[str, Any]]:
    """
    Every window_size messages, produce a WINDOW_SEAL event and rotate anchor:
      window_hash = H(h1||h2||...||hN)
      anchor_next = H(anchor || window_hash)
    """
    if state.ctr == 0:
        return None
    if state.ctr % window_size != 0:
        return None
    # Use last N message hashes in the window (tracked externally); if not present, skip.
    if not state.window_hashes:
        return None
    # seal last window_size hashes
    last = state.window_hashes[-window_size:]
    wblob = ("".join(last)).encode("utf-8")
    w_hash = sha256_hex(wblob)
    anchor_next = sha256_hex((state.anchor + "|" + w_hash).encode("utf-8"))

    seal_event = {
        "window_idx": state.window_idx,
        "window_size": window_size,
        "window_hash": w_hash,
        "anchor_before": state.anchor,
        "anchor_after": anchor_next,
        "last_ctr": state.ctr,
        "last_h_prefix": state.last_hash[:12],
    }

    state.anchor = anchor_next
    state.window_idx += 1
    return seal_event

# -----------------------------
# DB Tooling (same semantics as FarPeer)
# -----------------------------
def exec_tool(db: CsvStore, payload: Dict[str, Any]) -> Any:
    tool = payload.get("tool")
    args = payload.get("args") or {}

    if tool == "read_db":
        limit = int(args.get("limit", 5))
        return {"rows": db.read_rows(limit=limit)}

    if tool == "write_db":
        row = args.get("row") or {}
        for k in ["id", "task", "result", "ts"]:
            if k not in row:
                return {"error": f"missing field {k}"}
        db.append_row(row)
        return {"written": True, "row": row}

    if tool == "summarize":
        rows = args.get("rows") or []
        return {"summary": f"{len(rows)} rows, last_id={rows[-1].get('id') if rows else 'n/a'}"}

    if tool == "llm_query":
        q = str(args.get("q", ""))
        return {"answer": f"(mock) model answered safely for: {q[:80]}"}

    return {"error": f"unknown tool {tool}"}

# -----------------------------
# Attacks (MLEI + PIM rule demos)
# -----------------------------
def pim_attack_counter_replay(env: Dict[str, Any]) -> Dict[str, Any]:
    e = dict(env)
    e["ctr"] = max(1, int(e["ctr"]) - 1)  # replay-ish: lower counter
    # keep h unchanged to show verifier catches ctr (and likely hash too)
    return e

def pim_attack_prev_rewrite(env: Dict[str, Any]) -> Dict[str, Any]:
    e = dict(env)
    e["prev"] = "BAD_PREV_" + (str(e["prev"])[:8] if e.get("prev") else "X")
    return e

def pim_attack_payload_mutation(env: Dict[str, Any]) -> Dict[str, Any]:
    e = json.loads(json.dumps(env))
    # mutate payload but keep hash -> should trigger hash mismatch
    p = e.get("payload", {})
    if isinstance(p, dict):
        p["args"] = p.get("args", {})
        if isinstance(p["args"], dict):
            p["args"]["__tampered__"] = True
        p["text"] = (p.get("text", "") + " [tampered]").strip()
    e["payload"] = p
    return e

def pim_attack_delay(env: Dict[str, Any]) -> Dict[str, Any]:
    # Delay happens before far verification; timestamp becomes stale by skew
    # We'll sleep outside by calling attack_delay on payload, so for PIM we just return env.
    return env

# "Reorder" and "Replay" are handled at the plan level.

MLEI_ATTACKS = {
    "None": None,
    "Prompt injection (tool text)": attack_prompt_injection,
    "Tool swap to unauthorized exec": attack_tool_swap_to_unauthorized,
    "Tamper tool args": attack_tamper_args,
    "Delay (timing skew)": lambda m: attack_delay(m, seconds=3.5),
}

PIM_ATTACKS = {
    "None": None,
    "Counter replay": pim_attack_counter_replay,
    "Prev-hash rewrite": pim_attack_prev_rewrite,
    "Payload mutation (hash mismatch)": pim_attack_payload_mutation,
    "Reorder (swap step 1<->2)": "REORDER",
    "Replay (duplicate step 1)": "REPLAY",
}

# -----------------------------
# UI Helpers
# -----------------------------
def reset_db_bootstrap():
    with open(CFG.db_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("id,task,result,ts\nr0,bootstrap,ok,0\nr1,task1,ok,0\n")

def load_db_df() -> pd.DataFrame:
    store = CsvStore(CFG.db_path)
    rows = store.read_rows(limit=2000)
    return pd.DataFrame(rows)

def style_status(val: str) -> str:
    v = str(val)
    if v in ("ALLOW", "OK"):
        return "background-color: rgba(0, 200, 0, 0.12)"
    if v == "ATTACK":
        return "background-color: rgba(255, 160, 0, 0.18)"
    if v == "BLOCK":
        return "background-color: rgba(255, 0, 0, 0.12)"
    if v == "DROP":
        return "background-color: rgba(160, 0, 255, 0.10)"
    if v == "RUN":
        return "background-color: rgba(120, 120, 120, 0.10)"
    return ""

def transcript_to_df(tr: List[Dict[str, Any]]) -> pd.DataFrame:
    flat = []
    for ev in tr:
        row = dict(ev)
        for k, v in list(row.items()):
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, ensure_ascii=False)
        flat.append(row)
    return pd.DataFrame(flat)

def kpis_from_df(df: pd.DataFrame) -> Dict[str, int]:
    if df is None or df.empty:
        return {"events": 0, "near_blocks": 0, "far_blocks": 0, "pim_drops": 0, "db_writes": 0}

    events = len(df)
    near_blocks = int(((df.get("who") == "NEAR") & (df.get("status") == "BLOCK")).sum()) if "who" in df else 0
    far_blocks = int(((df.get("who") == "FAR") & (df.get("status") == "BLOCK")).sum()) if "who" in df else 0
    pim_drops = int(((df.get("status") == "DROP")).sum()) if "status" in df else 0
    db_writes = 0
    if "tool" in df and "status" in df:
        db_writes = int(((df["tool"].astype(str) == "write_db") & (df["status"] == "OK")).sum())
    return {"events": events, "near_blocks": near_blocks, "far_blocks": far_blocks, "pim_drops": pim_drops, "db_writes": db_writes}

def status_for_event(ev: Dict[str, Any]) -> str:
    e = ev.get("event", "")
    who = ev.get("who", "")
    if who == "UI" and e == "RUN":
        return "RUN"
    if who == "ATTACK":
        return "ATTACK"
    if e == "NEAR_GATE":
        return "ALLOW" if ev.get("allow") else "BLOCK"
    if e == "FAR_GATE":
        return "ALLOW" if ev.get("allow") else "BLOCK"
    if e in ("PIM_VERIFY_NEAR", "PIM_VERIFY_FAR"):
        return "OK" if ev.get("ok") else "DROP"
    if e == "EXEC_RESULT":
        return "OK" if not ev.get("is_error") else "BLOCK"
    if e == "WINDOW_SEAL":
        return "INFO"
    return "INFO"

def build_chain_edges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clear view: prev -> h (hash prefixes) with ctr/tool.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    rows = []
    for _, r in df.iterrows():
        if str(r.get("event")) not in ("PIM_VERIFY_NEAR", "PIM_VERIFY_FAR"):
            continue
        env = r.get("env", "")
        try:
            env_obj = json.loads(env) if isinstance(env, str) else None
        except Exception:
            env_obj = None
        if not env_obj:
            continue
        rows.append({
            "who": r.get("who"),
            "ctr": env_obj.get("ctr"),
            "tool": (env_obj.get("payload") or {}).get("tool"),
            "prev": str(env_obj.get("prev", ""))[:12],
            "h": str(env_obj.get("h", ""))[:12],
            "ok": r.get("ok"),
            "reason": r.get("reason", ""),
        })
    return pd.DataFrame(rows)

def build_timeline_chart(df: pd.DataFrame) -> pd.DataFrame:
    """
    Numeric timeline for a simple chart:
      OK/ALLOW -> 1
      INFO -> 0
      ATTACK -> -0.5
      BLOCK/DROP -> -1
    """
    if df is None or df.empty:
        return pd.DataFrame()
    sev = {"OK": 1.0, "ALLOW": 1.0, "INFO": 0.0, "RUN": 0.0, "ATTACK": -0.5, "BLOCK": -1.0, "DROP": -1.0}
    out = df.copy()
    if "step" not in out.columns:
        return pd.DataFrame()
    out["step_n"] = pd.to_numeric(out["step"], errors="coerce")
    out = out.dropna(subset=["step_n"])
    out["step_n"] = out["step_n"].astype(int)
    out["sev"] = out["status"].map(lambda s: sev.get(str(s), 0.0))
    # Take minimum per step (worst outcome)
    g = out.groupby("step_n", as_index=False)["sev"].min()
    g = g.rename(columns={"step_n": "step"})
    return g

# -----------------------------
# Core simulation runner (enhanced PIM logging)
# -----------------------------
def run_flow(
    task_prompt: str,
    mle_i_attack: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
    pim_attack: Optional[Any],
    near_gate: NanoGate,
    far_gate: NanoGate,
    near_threshold: float,
    far_threshold: float,
    max_skew_s: float,
    window_size: int,
) -> List[Dict[str, Any]]:
    """
    Implements the whole session in-app so we can log every PIM internal.
    """
    sid = f"SID-{uuid.uuid4().hex[:8]}"
    box = CryptoBox.new()
    chan = SecureChannel(crypto=box)

    db = CsvStore(CFG.db_path)

    near = PIMState(sid=sid)
    far = PIMState(sid=sid)

    t0 = datetime.utcnow().isoformat() + "Z"
    transcript: List[Dict[str, Any]] = [{
        "who": "UI",
        "event": "RUN",
        "sid": sid,
        "task_prompt": task_prompt,
        "mlei_attack": getattr(mle_i_attack, "__name__", str(mle_i_attack)) if mle_i_attack else "None",
        "pim_attack": pim_attack if isinstance(pim_attack, str) else (getattr(pim_attack, "__name__", "None") if pim_attack else "None"),
        "started_utc": t0,
        "step": 0,
    }]

    # LLM "plan" tool calls
    plan = llm_agent_plan(task_prompt, db_preview=[])

    # Plan-level PIM attacks
    if pim_attack == "REORDER" and len(plan) >= 2:
        plan[0], plan[1] = plan[1], plan[0]
        transcript.append({"who": "ATTACK", "event": "PLAN_REORDER", "detail": "Swapped step 1 and 2", "step": 0})

    if pim_attack == "REPLAY" and len(plan) >= 1:
        plan = [plan[0]] + plan  # duplicate first
        transcript.append({"who": "ATTACK", "event": "PLAN_REPLAY", "detail": "Duplicated step 1", "step": 0})

    # Execute each tool call across near->far->near
    for i, tool_call in enumerate(plan, start=1):
        # ---------- NEAR gate ----------
        allow_n, score_n, _ = near_gate.decide(tool_call, near_threshold)
        transcript.append({
            "who": "NEAR",
            "event": "NEAR_GATE",
            "step": i,
            "tool": tool_call.get("tool"),
            "allow": bool(allow_n),
            "score": float(score_n),
            "payload": json.dumps(tool_call, ensure_ascii=False),
        })
        if not allow_n:
            continue

        # Apply MLEI attack at agent layer (before PIM envelope)
        injected_call = tool_call
        if mle_i_attack is not None:
            injected_call = mle_i_attack(tool_call)
            transcript.append({
                "who": "ATTACK",
                "event": "MLEI_INJECT",
                "step": i,
                "tool_before": tool_call.get("tool"),
                "tool_after": injected_call.get("tool"),
                "payload_after": json.dumps(injected_call, ensure_ascii=False),
            })

        # ---------- Build env ----------
        env = build_env(near, injected_call)
        advance_state(near, env)
        near.window_hashes.append(env["h"])

        # Apply PIM attack at envelope layer (before encryption)
        env_sent = env
        if callable(pim_attack):
            env_sent = pim_attack(env)
            transcript.append({
                "who": "ATTACK",
                "event": "PIM_TAMPER",
                "step": i,
                "tamper": pim_attack.__name__,
                "env_before": json.dumps(env, ensure_ascii=False),
                "env_after": json.dumps(env_sent, ensure_ascii=False),
            })

        # Optional PIM skew attack: delay the envelope delivery
        if mle_i_attack is not None and getattr(mle_i_attack, "__name__", "") == "<lambda>":
            # (we already delayed at payload stage), but keep logic explicit
            pass

        # Encrypt and send
        blob = chan.seal(env_sent)

        # ---------- FAR verify PIM ----------
        env_recv = chan.open(blob)
        rep_far = verify_env_report(far, env_recv, max_skew_s=max_skew_s)
        transcript.append({
            "who": "FAR",
            "event": "PIM_VERIFY_FAR",
            "step": i,
            "ok": rep_far["ok"],
            "reason": rep_far["reason"],
            "checks": rep_far["checks"],
            "skew_s": rep_far["skew_s"],
            "computed_hash": rep_far["computed_hash"],
            "env": json.dumps(env_recv, ensure_ascii=False),
            "canonical_core": rep_far["canonical_core"],
        })
        if not rep_far["ok"]:
            # Drop message (far does not advance state)
            continue

        # If PIM ok, advance FAR state + track window hashes
        advance_state(far, env_recv)
        far.window_hashes.append(env_recv["h"])

        # ---------- FAR gate ----------
        payload = env_recv["payload"]
        allow_f, score_f, _ = far_gate.decide(payload, far_threshold)
        transcript.append({
            "who": "FAR",
            "event": "FAR_GATE",
            "step": i,
            "tool": payload.get("tool"),
            "allow": bool(allow_f),
            "score": float(score_f),
            "payload": json.dumps(payload, ensure_ascii=False),
        })
        if not allow_f:
            # Policy choice: advance far PIM already done; execution blocked.
            reply_payload = {"tool": "error", "text": f"FAR BLOCK by gate score={score_f:.3f}", "args": {}}
        else:
            result = exec_tool(db, payload)
            is_error = isinstance(result, dict) and "error" in result
            transcript.append({
                "who": "FAR",
                "event": "EXEC_RESULT",
                "step": i,
                "tool": payload.get("tool"),
                "is_error": bool(is_error),
                "result": json.dumps(result, ensure_ascii=False),
            })
            reply_payload = {"tool": "result", "text": "ok", "args": {"result": result}}

        # ---------- FAR reply env ----------
        reply_env = build_env(far, reply_payload)
        advance_state(far, reply_env)
        far.window_hashes.append(reply_env["h"])
        reply_blob = chan.seal(reply_env)

        # ---------- NEAR verify reply ----------
        reply_recv = chan.open(reply_blob)
        rep_near = verify_env_report(near, reply_recv, max_skew_s=max_skew_s)
        transcript.append({
            "who": "NEAR",
            "event": "PIM_VERIFY_NEAR",
            "step": i,
            "ok": rep_near["ok"],
            "reason": rep_near["reason"],
            "checks": rep_near["checks"],
            "skew_s": rep_near["skew_s"],
            "computed_hash": rep_near["computed_hash"],
            "env": json.dumps(reply_recv, ensure_ascii=False),
            "canonical_core": rep_near["canonical_core"],
        })
        if rep_near["ok"]:
            advance_state(near, reply_recv)
            near.window_hashes.append(reply_recv["h"])

        # ---------- Window sealing (both sides) ----------
        seal_n = maybe_window_seal(near, window_size=window_size)
        if seal_n:
            transcript.append({"who": "NEAR", "event": "WINDOW_SEAL", "step": i, **seal_n})
        seal_f = maybe_window_seal(far, window_size=window_size)
        if seal_f:
            transcript.append({"who": "FAR", "event": "WINDOW_SEAL", "step": i, **seal_f})

    t1 = datetime.utcnow().isoformat() + "Z"
    transcript.append({"who": "UI", "event": "RUN_END", "sid": sid, "finished_utc": t1, "step": 999999})
    for ev in transcript:
        ev["status"] = status_for_event(ev)
    return transcript

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="WARL0K PIM Inspector Demo", layout="wide")
st.title("WARL0K PIM Inspector — Proof-in-Motion + MLEI Nano-Gates")
st.caption("This UI is built specifically to make PIM understandable: per-check results, canonical hash core, chain edges, windows, and a timeline.")

# Session init
if "authed" not in st.session_state:
    st.session_state.authed = False
if "transcript" not in st.session_state:
    st.session_state.transcript = []
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

# Sidebar
with st.sidebar:
    st.header("Controls")

    st.subheader("Login (demo)")
    u = st.text_input("Username", value="demo")
    p = st.text_input("Password", value="demo", type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Authenticate"):
            st.session_state.authed = (u == "demo" and p == "demo")
            if st.session_state.authed:
                st.success("Auth granted ✅")
            else:
                st.error("Auth denied ❌")
    with c2:
        if st.button("Reset transcript"):
            st.session_state.transcript = []
            st.session_state.selected_idx = 0
            st.success("Cleared.")

    st.divider()

    st.subheader("Scenario")
    task = st.selectbox(
        "Task",
        ["Task 1: read DB and summarize", "Task 2: write validated result row"],
        index=0,
        disabled=not st.session_state.authed,
    )
    mlei_attack_label = st.selectbox(
        "MLEI attack (agent layer)",
        list(MLEI_ATTACKS.keys()),
        index=0,
        disabled=not st.session_state.authed,
    )
    pim_attack_label = st.selectbox(
        "PIM attack (chain rules)",
        list(PIM_ATTACKS.keys()),
        index=0,
        disabled=not st.session_state.authed,
    )

    st.divider()

    st.subheader("Parameters")
    max_skew_s = st.slider("PIM max skew (s)", 0.5, 10.0, float(CFG.max_skew_s), 0.5)
    window_size = st.slider("Window seal every N messages", 2, 32, int(min(8, CFG.window_size)), 1)
    near_thr = st.slider("NEAR nano-gate threshold", 0.40, 0.95, float(CFG.near_threshold), 0.01)
    far_thr = st.slider("FAR nano-gate threshold", 0.40, 0.95, float(CFG.far_threshold), 0.01)

    st.divider()

    st.subheader("DB")
    if st.button("Reset demo_db.csv"):
        reset_db_bootstrap()
        st.success("DB reset ✅")

    run_btn = st.button("▶ Run", type="primary", disabled=not st.session_state.authed)

# Main layout
left, right = st.columns([1.35, 1.0], gap="large")

with right:
    st.subheader("CSV DB (tail)")
    db_df = load_db_df()
    if db_df.empty:
        st.info("DB is empty.")
    else:
        st.dataframe(db_df.tail(25), use_container_width=True)

with left:
    if run_btn:
        # Train nano-gates (nano model tuning)
        near_gate = NanoGate.train_synthetic(seed=9)
        far_gate = NanoGate.train_synthetic(seed=11)

        mle = MLEI_ATTACKS[mlei_attack_label]
        pim_att = PIM_ATTACKS[pim_attack_label]
        transcript = run_flow(
            task_prompt=task,
            mle_i_attack=mle,
            pim_attack=pim_att,
            near_gate=near_gate,
            far_gate=far_gate,
            near_threshold=near_thr,
            far_threshold=far_thr,
            max_skew_s=max_skew_s,
            window_size=window_size,
        )
        st.session_state.transcript = transcript
        st.session_state.selected_idx = 0
        st.success("Run complete ✅")

    tr = st.session_state.transcript
    if not tr:
        st.info("Authenticate, choose scenario, and click **Run**.")
    else:
        df = transcript_to_df(tr)

        # KPIs
        k = kpis_from_df(df)
        a, b, c, d, e = st.columns(5)
        a.metric("Events", k["events"])
        b.metric("NEAR blocks", k["near_blocks"])
        c.metric("FAR blocks", k["far_blocks"])
        d.metric("PIM drops", k["pim_drops"])
        e.metric("DB writes", k["db_writes"])

        st.divider()

        # Timeline chart
        st.subheader("Timeline (worst severity per step)")
        tl = build_timeline_chart(df)
        if not tl.empty:
            st.line_chart(tl.set_index("step"))
        else:
            st.caption("No step-indexed events to chart.")

        st.divider()

        # Chain edges view
        st.subheader("PIM Chain Edges (prev → h)")
        edges = build_chain_edges(df)
        if edges.empty:
            st.caption("No PIM verify events captured yet.")
        else:
            st.dataframe(edges, use_container_width=True, height=220)

        st.divider()

        # Filter + event log
        st.subheader("Event Log (click to inspect)")
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        with f1:
            who_filter = st.multiselect("who", sorted(df["who"].dropna().unique().tolist()), default=[])
        with f2:
            event_filter = st.multiselect("event", sorted(df["event"].dropna().unique().tolist()), default=[])
        with f3:
            status_filter = st.multiselect("status", sorted(df["status"].dropna().unique().tolist()), default=[])
        with f4:
            q = st.text_input("search", value="")

        view = df.copy()
        if who_filter:
            view = view[view["who"].isin(who_filter)]
        if event_filter:
            view = view[view["event"].isin(event_filter)]
        if status_filter:
            view = view[view["status"].isin(status_filter)]
        if q.strip():
            mask = False
            for col in view.columns:
                mask = mask | view[col].astype(str).str.contains(q, case=False, na=False)
            view = view[mask]

        # Put key cols first
        preferred = [c for c in ["who", "event", "status", "step", "tool", "allow", "score", "ok", "reason"] if c in view.columns]
        rest = [c for c in view.columns if c not in preferred]
        view = view[preferred + rest]

        # selection index
        max_idx = max(0, len(view) - 1)
        sel = st.number_input("Selected row (index in filtered view)", min_value=0, max_value=max_idx, value=min(st.session_state.selected_idx, max_idx), step=1)
        st.session_state.selected_idx = int(sel)

        styled = view.style
        if "status" in view.columns:
            styled = styled.applymap(style_status, subset=["status"])
        st.dataframe(styled, use_container_width=True, height=360)

        # Inspector panel
        st.divider()
        st.subheader("PIM Inspector")

        if len(view) == 0:
            st.caption("No rows match filter.")
        else:
            row = view.iloc[st.session_state.selected_idx].to_dict()
            st.write(f"**Event:** `{row.get('event')}` | **Who:** `{row.get('who')}` | **Status:** `{row.get('status')}` | **Step:** `{row.get('step')}`")

            # Show gate info
            if str(row.get("event")) in ("NEAR_GATE", "FAR_GATE"):
                st.markdown("**Gate decision**")
                st.json({
                    "allow": row.get("allow"),
                    "score": row.get("score"),
                    "tool": row.get("tool"),
                    "payload": row.get("payload"),
                })

            # Show PIM verify info
            if str(row.get("event")) in ("PIM_VERIFY_NEAR", "PIM_VERIFY_FAR"):
                st.markdown("**PIM Verification Checklist**")
                checks = row.get("checks")
                try:
                    checks_obj = json.loads(checks) if isinstance(checks, str) else checks
                except Exception:
                    checks_obj = checks

                cA, cB = st.columns([1, 1])
                with cA:
                    st.json({
                        "ok": row.get("ok"),
                        "reason": row.get("reason"),
                        "skew_s": row.get("skew_s"),
                        "checks": checks_obj,
                    })
                with cB:
                    st.markdown("**Hash comparison**")
                    st.code(f"computed_hash: {row.get('computed_hash')}\nreceived_hash:  (see env.h inside env JSON)", language="text")

                st.markdown("**Envelope (received)**")
                try:
                    env_obj = json.loads(row.get("env")) if isinstance(row.get("env"), str) else row.get("env")
                except Exception:
                    env_obj = row.get("env")
                st.json(env_obj)

                st.markdown("**Canonical core (what is hashed)**")
                st.code(row.get("canonical_core", ""), language="json")

            # Window seal info
            if str(row.get("event")) == "WINDOW_SEAL":
                st.markdown("**Window seal / anchor rotation**")
                st.json({
                    "window_idx": row.get("window_idx"),
                    "window_size": row.get("window_size"),
                    "window_hash": row.get("window_hash"),
                    "anchor_before": row.get("anchor_before"),
                    "anchor_after": row.get("anchor_after"),
                    "last_ctr": row.get("last_ctr"),
                    "last_h_prefix": row.get("last_h_prefix"),
                })

        # Download transcript
        st.download_button(
            "⬇ Download transcript.json",
            data=json.dumps(tr, indent=2, ensure_ascii=False),
            file_name="transcript.json",
            mime="application/json",
        )

st.divider()
st.subheader("Run command")
st.code("streamlit run app_pim_inspector.py", language="bash")
# # -------V2--------------------------
# # app1.py
# import json
# import uuid
# from datetime import datetime
#
# import pandas as pd
# import streamlit as st
#
# from common.crypto import CryptoBox
# from common.protocol import SecureChannel
# from peers.peer_near import NearPeer
# from peers.peer_far import FarPeer
# from attacks.injector import (
#     attack_prompt_injection,
#     attack_tool_swap_to_unauthorized,
#     attack_tamper_args,
#     attack_delay,
# )
# from config import CFG
# from db.csv_store import CsvStore
#
#
# # -----------------------------
# # Helpers / Constants
# # -----------------------------
# ATTACKS = {
#     "None (baseline)": None,
#     "Prompt injection": attack_prompt_injection,
#     "Tool swap to unauthorized exec": attack_tool_swap_to_unauthorized,
#     "Tamper write args": attack_tamper_args,
#     "Delay (timing skew)": lambda m: attack_delay(m, seconds=3.5),
# }
#
# TASKS = {
#     "Task 1: read DB and summarize": "Task 1: read DB and summarize",
#     "Task 2: write validated result row": "Task 2: write validated result row",
# }
#
#
# def load_db_df() -> pd.DataFrame:
#     store = CsvStore(CFG.db_path)
#     rows = store.read_rows(limit=2000)
#     return pd.DataFrame(rows)
#
#
# def reset_db_bootstrap():
#     with open(CFG.db_path, "w", encoding="utf-8", newline="\n") as f:
#         f.write("id,task,result,ts\nr0,bootstrap,ok,0\nr1,task1,ok,0\n")
#
#
# def new_session_bundle():
#     """
#     Creates a fresh near/far pair with a fresh session id and shared key.
#     """
#     sid = f"SID-{uuid.uuid4().hex[:8]}"
#     box = CryptoBox.new()
#     channel = SecureChannel(crypto=box)
#
#     near = NearPeer(sid=sid, channel=channel)
#     far = FarPeer(sid=sid, channel=channel)
#     return sid, near, far
#
#
# def transcript_to_df(transcript):
#     if not transcript:
#         return pd.DataFrame()
#     flat = []
#     for ev in transcript:
#         row = dict(ev)
#         # stringify nested objects for dataframe display
#         for k, v in list(row.items()):
#             if isinstance(v, (dict, list)):
#                 row[k] = json.dumps(v, ensure_ascii=False)
#         flat.append(row)
#     return pd.DataFrame(flat)
#
#
# def compute_kpis(df: pd.DataFrame) -> dict:
#     if df is None or df.empty:
#         return {
#             "events": 0,
#             "near_blocks": 0,
#             "far_blocks": 0,
#             "pim_drops": 0,
#             "db_writes": 0,
#         }
#
#     events = len(df)
#
#     near_blocks = 0
#     if "who" in df.columns and "status" in df.columns:
#         near_blocks = int(((df["who"] == "NEAR") & (df["status"] == "BLOCK")).sum())
#
#     # FAR blocks are signaled inside reply payload text like "FAR BLOCK ..."
#     far_blocks = 0
#     if "payload" in df.columns:
#         far_blocks = int(df["payload"].astype(str).str.contains("FAR BLOCK", case=False, na=False).sum())
#
#     pim_drops = 0
#     if "payload" in df.columns:
#         pim_drops += int(df["payload"].astype(str).str.contains("PIM:", case=False, na=False).sum())
#     if "reason" in df.columns:
#         pim_drops += int(df["reason"].astype(str).str.contains("PIM:", case=False, na=False).sum())
#     if "event" in df.columns and "ok" in df.columns:
#         pim_drops += int(((df["event"] == "REPLY_VERIFY") & (df["ok"].astype(str).str.lower() == "false")).sum())
#
#     db_writes = 0
#     if "payload" in df.columns:
#         db_writes = int(df["payload"].astype(str).str.contains('"written": true', case=False, na=False).sum())
#
#     return {
#         "events": events,
#         "near_blocks": near_blocks,
#         "far_blocks": far_blocks,
#         "pim_drops": pim_drops,
#         "db_writes": db_writes,
#     }
#
#
# def classify_status(row: dict) -> str:
#     ev = str(row.get("event", ""))
#     who = str(row.get("who", ""))
#
#     if who == "UI" and ev == "RUN":
#         return "RUN"
#     if who == "ATTACK":
#         return "ATTACK"
#
#     if ev == "GATE":
#         return "ALLOW" if str(row.get("decision", "")).upper() == "ALLOW" else "BLOCK"
#
#     if ev in ("BLOCKED_LOCALLY",):
#         return "BLOCK"
#
#     if ev == "REPLY_VERIFY":
#         return "OK" if str(row.get("ok", "")).lower() == "true" else "DROP"
#
#     if ev in ("DROP_REPLY",):
#         return "DROP"
#
#     if ev == "REPLY":
#         payload = row.get("payload", "")
#         if isinstance(payload, str) and '"tool": "error"' in payload:
#             return "BLOCK"
#         return "OK"
#
#     return "INFO"
#
#
# def build_timeline(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Build per-step status rollup:
#       - each step may have multiple events (GATE, INJECT, REPLY_VERIFY, REPLY)
#       - we compute a single "final_status" for that step with severity priority
#     """
#     if df is None or df.empty:
#         return pd.DataFrame()
#
#     if "step" not in df.columns:
#         return pd.DataFrame()
#
#     tmp = df.copy()
#     tmp["step_num"] = pd.to_numeric(tmp["step"], errors="coerce")
#     tmp = tmp.dropna(subset=["step_num"])
#     tmp["step_num"] = tmp["step_num"].astype(int)
#
#     # Priority order: BLOCK/DROP > ATTACK > OK/ALLOW > INFO/RUN
#     priority = {"BLOCK": 4, "DROP": 4, "ATTACK": 3, "OK": 2, "ALLOW": 2, "INFO": 1, "RUN": 0}
#     tmp["prio"] = tmp["status"].map(lambda s: priority.get(str(s), 1)).astype(int)
#
#     # Choose max priority per step, then pick one row (first with that prio)
#     out_rows = []
#     for step_num, g in tmp.groupby("step_num"):
#         g2 = g.sort_values(["prio"], ascending=False)
#         top = g2.iloc[0].to_dict()
#         out_rows.append(
#             {
#                 "step": step_num,
#                 "final_status": top.get("status", "INFO"),
#                 "who": top.get("who", ""),
#                 "event": top.get("event", ""),
#                 "tool": top.get("tool", ""),
#                 "reason": top.get("reason", ""),
#             }
#         )
#
#     return pd.DataFrame(out_rows).sort_values("step")
#
#
# def style_status(val: str) -> str:
#     v = str(val)
#     if v == "ALLOW":
#         return "background-color: rgba(0, 200, 0, 0.12)"
#     if v == "OK":
#         return "background-color: rgba(0, 180, 255, 0.10)"
#     if v == "ATTACK":
#         return "background-color: rgba(255, 160, 0, 0.18)"
#     if v == "BLOCK":
#         return "background-color: rgba(255, 0, 0, 0.12)"
#     if v == "DROP":
#         return "background-color: rgba(160, 0, 255, 0.10)"
#     if v == "RUN":
#         return "background-color: rgba(120, 120, 120, 0.10)"
#     return ""
#
#
# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="WARL0K PIM + MLEI Demo", layout="wide")
#
# st.title("WARL0K PIM + MLEI — Cloud LLM Agent Session Demo")
# st.caption("Two-peer session chain (PIM) + nano-gate enforcement (MLEI) + encrypted tunnel + CSV DB read/write")
#
#
# # -----------------------------
# # Sidebar controls
# # -----------------------------
# with st.sidebar:
#     st.header("Session Controls")
#
#     if "authed" not in st.session_state:
#         st.session_state.authed = False
#     if "transcript" not in st.session_state:
#         st.session_state.transcript = []
#
#     st.subheader("Login")
#     user = st.text_input("Username", value="demo")
#     pwd = st.text_input("Password", value="demo", type="password")
#
#     c_login1, c_login2 = st.columns(2)
#     with c_login1:
#         if st.button("Authenticate"):
#             sid, near, far = new_session_bundle()
#             ok = near.login_and_auth(user, pwd)
#             st.session_state.authed = bool(ok)
#             if ok:
#                 st.session_state.sid = sid
#                 st.session_state.near = near
#                 st.session_state.far = far
#                 st.success("Auth granted ✅")
#             else:
#                 st.error("Auth denied ❌")
#
#     with c_login2:
#         if st.button("New session (reset chain)", disabled=not st.session_state.authed):
#             sid, near, far = new_session_bundle()
#             st.session_state.sid = sid
#             st.session_state.near = near
#             st.session_state.far = far
#             st.session_state.transcript = []
#             st.success("New session created ✅")
#
#     st.divider()
#     st.subheader("Run")
#     task_label = st.selectbox("Task", list(TASKS.keys()), index=0)
#     attack_label = st.selectbox("Attack", list(ATTACKS.keys()), index=0)
#
#     st.caption("PIM timing window (seconds)")
#     st.write(f"max_skew_s = **{CFG.max_skew_s}**")
#
#     run_btn = st.button("▶ Run Demo Flow", type="primary", disabled=not st.session_state.authed)
#
#     st.divider()
#     st.subheader("DB Utilities")
#     if st.button("Reset demo_db.csv (bootstrap rows)"):
#         reset_db_bootstrap()
#         st.success("DB reset ✅")
#
#     st.caption(f"DB path: {CFG.db_path}")
#
#
# # -----------------------------
# # Main layout
# # -----------------------------
# colA, colB = st.columns([1.25, 1.0], gap="large")
#
#
# # -----------------------------
# # Right: DB view
# # -----------------------------
# with colB:
#     st.subheader("CSV DB (last rows)")
#     db_df = load_db_df()
#     if db_df.empty:
#         st.info("DB is empty.")
#     else:
#         st.dataframe(db_df.tail(25), use_container_width=True)
#
#
# # -----------------------------
# # Left: Transcript + KPIs + Timeline
# # -----------------------------
# with colA:
#     st.subheader("Transcript")
#
#     if run_btn:
#         if "near" not in st.session_state or "far" not in st.session_state:
#             sid, near, far = new_session_bundle()
#             st.session_state.sid, st.session_state.near, st.session_state.far = sid, near, far
#
#         near: NearPeer = st.session_state.near
#         far: FarPeer = st.session_state.far
#
#         inject = ATTACKS[attack_label]
#         prompt = TASKS[task_label]
#
#         t0 = datetime.utcnow().isoformat() + "Z"
#         transcript = near.run_task_flow(
#             far_peer_execute=far.recv_and_execute,
#             task_prompt=prompt,
#             inject=inject,
#         )
#         t1 = datetime.utcnow().isoformat() + "Z"
#
#         header = {
#             "who": "UI",
#             "event": "RUN",
#             "sid": st.session_state.sid,
#             "task": task_label,
#             "attack": attack_label,
#             "started_utc": t0,
#             "finished_utc": t1,
#         }
#         st.session_state.transcript = [header] + transcript
#         st.success("Run complete ✅")
#
#     tr = st.session_state.transcript
#     if not tr:
#         st.info("Authenticate, then click **Run Demo Flow**.")
#     else:
#         df = transcript_to_df(tr)
#         if not df.empty:
#             # Add status column
#             df["status"] = df.apply(lambda r: classify_status(r.to_dict()), axis=1)
#
#         # KPIs
#         kpis = compute_kpis(df)
#         k1, k2, k3, k4, k5 = st.columns(5)
#         k1.metric("Events", kpis["events"])
#         k2.metric("NEAR blocks", kpis["near_blocks"])
#         k3.metric("FAR blocks", kpis["far_blocks"])
#         k4.metric("PIM drops", kpis["pim_drops"])
#         k5.metric("DB writes", kpis["db_writes"])
#
#         st.divider()
#
#         # Timeline (per step)
#         tl = build_timeline(df)
#         if not tl.empty:
#             st.subheader("Step Timeline")
#             tl_view = tl.copy()
#             tl_styled = tl_view.style.applymap(style_status, subset=["final_status"])
#             st.dataframe(tl_styled, use_container_width=True, height=200)
#             st.caption("Final status per step (severity rollup).")
#
#         st.divider()
#
#         # Filters
#         c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
#         with c1:
#             who_filter = st.multiselect(
#                 "Filter: who",
#                 sorted(df["who"].dropna().unique().tolist()) if "who" in df else [],
#                 default=[],
#             )
#         with c2:
#             event_filter = st.multiselect(
#                 "Filter: event",
#                 sorted(df["event"].dropna().unique().tolist()) if "event" in df else [],
#                 default=[],
#             )
#         with c3:
#             status_filter = st.multiselect(
#                 "Filter: status",
#                 sorted(df["status"].dropna().unique().tolist()) if "status" in df else [],
#                 default=[],
#             )
#         with c4:
#             text_search = st.text_input("Search text", value="")
#
#         view = df.copy()
#         if who_filter:
#             view = view[view["who"].isin(who_filter)]
#         if event_filter:
#             view = view[view["event"].isin(event_filter)]
#         if status_filter:
#             view = view[view["status"].isin(status_filter)]
#         if text_search.strip():
#             mask = False
#             for col in view.columns:
#                 mask = mask | view[col].astype(str).str.contains(text_search, case=False, na=False)
#             view = view[mask]
#
#         # Column order for readability
#         preferred = [
#             c
#             for c in [
#                 "who",
#                 "event",
#                 "status",
#                 "step",
#                 "tool",
#                 "decision",
#                 "score",
#                 "ok",
#                 "reason",
#                 "reply_tool",
#                 "mutated_tool",
#                 "payload",
#             ]
#             if c in view.columns
#         ]
#         rest = [c for c in view.columns if c not in preferred]
#         view = view[preferred + rest]
#
#         # Styled transcript table
#         styled = view.style
#         if "status" in view.columns:
#             styled = styled.applymap(style_status, subset=["status"])
#
#         st.subheader("Event Log")
#         st.dataframe(styled, use_container_width=True, height=520)
#
#         # Download transcript JSON
#         st.download_button(
#             "⬇ Download transcript.json",
#             data=json.dumps(tr, indent=2, ensure_ascii=False),
#             file_name="transcript.json",
#             mime="application/json",
#         )
#
#         with st.expander("Quick summary"):
#             st.write(
#                 f"- Events: **{kpis['events']}**\n"
#                 f"- NEAR blocks: **{kpis['near_blocks']}**\n"
#                 f"- FAR blocks: **{kpis['far_blocks']}**\n"
#                 f"- PIM drops: **{kpis['pim_drops']}**\n"
#                 f"- DB writes: **{kpis['db_writes']}**\n"
#             )
#
# st.divider()
# st.subheader("How to run")
# st.code(
#     "pip install -r requirements.txt\n"
#     "streamlit run app1.py\n",
#     language="bash",
# )
# # ------------V1--------------------------------------
# # import json
# # import uuid
# # from datetime import datetime
# #
# # import pandas as pd
# # import streamlit as st
# #
# # from common.crypto import CryptoBox
# # from common.protocol import SecureChannel
# # from peers.peer_near import NearPeer
# # from peers.peer_far import FarPeer
# # from attacks.injector import (
# #     attack_prompt_injection,
# #     attack_tool_swap_to_unauthorized,
# #     attack_tamper_args,
# #     attack_delay,
# # )
# # from config import CFG
# # from db.csv_store import CsvStore
# #
# #
# # # -----------------------------
# # # Helpers
# # # -----------------------------
# # ATTACKS = {
# #     "None (baseline)": None,
# #     "Prompt injection": attack_prompt_injection,
# #     "Tool swap to unauthorized exec": attack_tool_swap_to_unauthorized,
# #     "Tamper write args": attack_tamper_args,
# #     "Delay (timing skew)": lambda m: attack_delay(m, seconds=3.5),
# # }
# #
# # TASKS = {
# #     "Task 1: read DB and summarize": "Task 1: read DB and summarize",
# #     "Task 2: write validated result row": "Task 2: write validated result row",
# # }
# #
# # def load_db_df() -> pd.DataFrame:
# #     store = CsvStore(CFG.db_path)
# #     rows = store.read_rows(limit=2000)
# #     return pd.DataFrame(rows)
# #
# # def new_session_bundle():
# #     """
# #     Creates a fresh near/far pair with a fresh session id and shared key.
# #     """
# #     sid = f"SID-{uuid.uuid4().hex[:8]}"
# #     box = CryptoBox.new()
# #     channel = SecureChannel(crypto=box)
# #
# #     near = NearPeer(sid=sid, channel=channel)
# #     far = FarPeer(sid=sid, channel=channel)
# #     return sid, near, far
# #
# # def transcript_to_df(transcript):
# #     if not transcript:
# #         return pd.DataFrame()
# #     # Normalize keys a bit
# #     flat = []
# #     for ev in transcript:
# #         row = dict(ev)
# #         # stringify nested
# #         for k, v in list(row.items()):
# #             if isinstance(v, (dict, list)):
# #                 row[k] = json.dumps(v, ensure_ascii=False)
# #         flat.append(row)
# #     return pd.DataFrame(flat)
# #
# #
# # # -----------------------------
# # # Streamlit UI
# # # -----------------------------
# # st.set_page_config(page_title="WARL0K PIM + MLEI Demo", layout="wide")
# #
# # st.title("WARL0K PIM + MLEI — Cloud LLM Agent Session Demo")
# # st.caption("Two-peer session chain (PIM) + nano-gate enforcement (MLEI) + encrypted tunnel + CSV DB read/write")
# #
# # # Sidebar controls
# # with st.sidebar:
# #     st.header("Session Controls")
# #
# #     if "authed" not in st.session_state:
# #         st.session_state.authed = False
# #
# #     st.subheader("Login")
# #     user = st.text_input("Username", value="demo")
# #     pwd = st.text_input("Password", value="demo", type="password")
# #
# #     if st.button("Authenticate"):
# #         # Create temp peers just to validate credentials with NearPeer logic
# #         sid, near, far = new_session_bundle()
# #         ok = near.login_and_auth(user, pwd)
# #         st.session_state.authed = bool(ok)
# #         if ok:
# #             st.session_state.sid = sid
# #             st.session_state.near = near
# #             st.session_state.far = far
# #             st.success("Auth granted ✅")
# #         else:
# #             st.error("Auth denied ❌")
# #
# #     st.divider()
# #     st.subheader("Run")
# #     task_label = st.selectbox("Task", list(TASKS.keys()), index=0)
# #     attack_label = st.selectbox("Attack", list(ATTACKS.keys()), index=0)
# #
# #     st.caption("PIM timing window (seconds)")
# #     st.write(f"max_skew_s = **{CFG.max_skew_s}**")
# #
# #     run_btn = st.button("▶ Run Demo Flow", type="primary", disabled=not st.session_state.authed)
# #
# #     st.divider()
# #     st.subheader("DB Utilities")
# #     if st.button("Reset demo_db.csv (bootstrap rows)"):
# #         # Hard reset the CSV file
# #         with open(CFG.db_path, "w", encoding="utf-8", newline="\n") as f:
# #             f.write("id,task,result,ts\nr0,bootstrap,ok,0\nr1,task1,ok,0\n")
# #         st.success("DB reset.")
# #
# # # Main content
# # colA, colB = st.columns([1.2, 1.0], gap="large")
# #
# # with colB:
# #     st.subheader("CSV DB (last rows)")
# #     db_df = load_db_df()
# #     if db_df.empty:
# #         st.info("DB is empty.")
# #     else:
# #         st.dataframe(db_df.tail(25), use_container_width=True)
# #
# #     st.caption(f"DB path: {CFG.db_path}")
# #
# # with colA:
# #     st.subheader("Transcript")
# #
# #     if "transcript" not in st.session_state:
# #         st.session_state.transcript = []
# #
# #     if run_btn:
# #         # Use existing session peers if present; else create
# #         if "near" not in st.session_state or "far" not in st.session_state:
# #             sid, near, far = new_session_bundle()
# #             st.session_state.sid, st.session_state.near, st.session_state.far = sid, near, far
# #
# #         near: NearPeer = st.session_state.near
# #         far: FarPeer = st.session_state.far
# #
# #         # Run selected flow
# #         inject = ATTACKS[attack_label]
# #         prompt = TASKS[task_label]
# #
# #         t0 = datetime.utcnow().isoformat() + "Z"
# #         transcript = near.run_task_flow(
# #             far_peer_execute=far.recv_and_execute,
# #             task_prompt=prompt,
# #             inject=inject,
# #         )
# #         t1 = datetime.utcnow().isoformat() + "Z"
# #
# #         # annotate run metadata
# #         header = {
# #             "who": "UI",
# #             "event": "RUN",
# #             "sid": st.session_state.sid,
# #             "task": task_label,
# #             "attack": attack_label,
# #             "started_utc": t0,
# #             "finished_utc": t1,
# #         }
# #         st.session_state.transcript = [header] + transcript
# #
# #         st.success("Run complete.")
# #
# #     # Filters + display
# #     tr = st.session_state.transcript
# #     if not tr:
# #         st.info("Authenticate, then click **Run Demo Flow**.")
# #     else:
# #         df = transcript_to_df(tr)
# #
# #         # Simple filters
# #         c1, c2, c3 = st.columns([1, 1, 1])
# #         with c1:
# #             who_filter = st.multiselect(
# #                 "Filter: who",
# #                 sorted(df["who"].dropna().unique().tolist()) if "who" in df else [],
# #                 default=[],
# #             )
# #         with c2:
# #             event_filter = st.multiselect(
# #                 "Filter: event",
# #                 sorted(df["event"].dropna().unique().tolist()) if "event" in df else [],
# #                 default=[],
# #             )
# #         with c3:
# #             text_search = st.text_input("Search text (any column)", value="")
# #
# #         view = df.copy()
# #         if who_filter:
# #             view = view[view["who"].isin(who_filter)]
# #         if event_filter:
# #             view = view[view["event"].isin(event_filter)]
# #         if text_search.strip():
# #             mask = False
# #             for col in view.columns:
# #                 mask = mask | view[col].astype(str).str.contains(text_search, case=False, na=False)
# #             view = view[mask]
# #
# #         st.dataframe(view, use_container_width=True, height=520)
# #
# #         # Download transcript JSON
# #         st.download_button(
# #             "⬇ Download transcript.json",
# #             data=json.dumps(tr, indent=2, ensure_ascii=False),
# #             file_name="transcript.json",
# #             mime="application/json",
# #         )
# #
# # st.divider()
# # st.subheader("How to run")
# # st.code(
# #     "pip install -r requirements.txt\n"
# #     "streamlit run app1.py\n",
# #     language="bash",
# # )
