# app_pim_mlei_visuals.py
# WARL0K PIM + MLEI — Visual Demo (Full “Explainability” UI)
#
# What you get in this single file:
# 1) Dual-lane Mermaid sequence diagram (NEAR ↔ FAR ↔ Cloud LLM) with status markers
# 2) PIM chain “ribbon” (GENESIS → h1 → h2 ...) with click-to-inspect
# 3) Window blocks + anchor rotation “ledger”
# 4) Two gauges (Integrity + Authority) + KPIs
# 5) Steps × Checks heatmap (PIM + MLEI)
# 6) Attack replay mode slider (focus any step; DB + inspector update)
# 7) 4-outcome bucket panel (Policy / Integrity / Exec error / Success)
# 8) “Who knew what” boundary view (Near/Far/Cloud/Attacker)
#
# Run:
#   pip install -r requirements.txt
#   pip install streamlit pandas
#   streamlit run app_pim_mlei_visuals.py

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import CFG
from common.crypto import CryptoBox
from common.protocol import SecureChannel
from common.nano_gate import NanoGate
from common.util import canon_json, sha256_hex, now_ts
from db.csv_store import CsvStore
from cloud.llm_cloud_mock import llm_agent_plan
from attacks.injector import (
	attack_prompt_injection,
	attack_tool_swap_to_unauthorized,
	attack_tamper_args,
	attack_delay,
)

# -----------------------------
# PIM Core (structured verifier report)
# -----------------------------
@dataclass
class PIMState:
	sid: str
	ctr: int = 0
	last_hash: str = "GENESIS"
	anchor: str = "ANCHOR0"
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
	report: Dict[str, Any] = {
		"ok": True,
		"checks": {"sid": True, "ctr": True, "prev": True, "skew": True, "hash": True},
		"expected": {"sid": state.sid, "ctr": state.ctr + 1, "prev": state.last_hash},
		"got": {},
		"skew_s": None,
		"reason": "OK",
		"computed_hash": None,
		"canonical_core": None,
	}

	got_sid = env.get("sid")
	got_ctr = env.get("ctr")
	got_prev = env.get("prev")
	got_ts = float(env.get("ts", 0.0))
	got_h = env.get("h")
	report["got"] = {"sid": got_sid, "ctr": got_ctr, "prev": got_prev, "h": got_h, "ts": got_ts}

	if got_sid != state.sid:
		report["ok"] = False
		report["checks"]["sid"] = False

	if got_ctr != state.ctr + 1:
		report["ok"] = False
		report["checks"]["ctr"] = False

	if got_prev != state.last_hash:
		report["ok"] = False
		report["checks"]["prev"] = False

	skew = abs(now_ts() - got_ts)
	report["skew_s"] = float(skew)
	if skew > max_skew_s:
		report["ok"] = False
		report["checks"]["skew"] = False

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
# Proof-in-Motion window sealing / anchor rotation
# -----------------------------
def maybe_window_seal(state: PIMState, window_size: int) -> Optional[Dict[str, Any]]:
	if state.ctr == 0 or (state.ctr % window_size != 0):
		return None
	if not state.window_hashes or len(state.window_hashes) < window_size:
		return None

	last = state.window_hashes[-window_size:]
	w_hash = sha256_hex(("".join(last)).encode("utf-8"))
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
# Tool execution (CSV DB)
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
# Attacks: MLEI layer + PIM layer
# -----------------------------
def pim_attack_counter_replay(env: Dict[str, Any]) -> Dict[str, Any]:
	e = dict(env)
	e["ctr"] = max(1, int(e["ctr"]) - 1)
	return e

def pim_attack_prev_rewrite(env: Dict[str, Any]) -> Dict[str, Any]:
	e = dict(env)
	e["prev"] = "BAD_PREV_" + (str(e["prev"])[:8] if e.get("prev") else "X")
	return e

def pim_attack_payload_mutation(env: Dict[str, Any]) -> Dict[str, Any]:
	e = json.loads(json.dumps(env))
	p = e.get("payload", {})
	if isinstance(p, dict):
		p["args"] = p.get("args", {})
		if isinstance(p["args"], dict):
			p["args"]["__tampered__"] = True
		p["text"] = (p.get("text", "") + " [tampered]").strip()
	e["payload"] = p
	return e

MLEI_ATTACKS: Dict[str, Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]] = {
	"None": None,
	"Prompt injection (tool text)": attack_prompt_injection,
	"Tool swap to unauthorized exec": attack_tool_swap_to_unauthorized,
	"Tamper tool args": attack_tamper_args,
	"Delay (timing skew)": lambda m: attack_delay(m, seconds=3.5),
}

PIM_ATTACKS: Dict[str, Any] = {
	"None": None,
	"Counter replay": pim_attack_counter_replay,
	"Prev-hash rewrite": pim_attack_prev_rewrite,
	"Payload mutation (hash mismatch)": pim_attack_payload_mutation,
	"Reorder (swap step 1<->2)": "REORDER",
	"Replay (duplicate step 1)": "REPLAY",
}

# -----------------------------
# Transcript schema + utilities
# -----------------------------
def status_of_event(ev: Dict[str, Any]) -> str:
	who, event = ev.get("who"), ev.get("event")
	if who == "UI" and event in ("RUN", "RUN_END"):
		return "RUN"
	if who == "ATTACK":
		return "ATTACK"
	if event in ("NEAR_GATE", "FAR_GATE"):
		return "ALLOW" if ev.get("allow") else "BLOCK"
	if event in ("PIM_VERIFY_FAR", "PIM_VERIFY_NEAR"):
		return "OK" if ev.get("ok") else "DROP"
	if event == "EXEC_RESULT":
		return "OK" if not ev.get("is_error") else "BLOCK"
	if event == "WINDOW_SEAL":
		return "INFO"
	if event in ("PLAN_REORDER", "PLAN_REPLAY", "MLEI_INJECT", "PIM_TAMPER"):
		return "ATTACK"
	return "INFO"

def reset_db_bootstrap():
	with open(CFG.db_path, "w", encoding="utf-8", newline="\n") as f:
		f.write("id,task,result,ts\nr0,bootstrap,ok,0\nr1,task1,ok,0\n")

def load_db_df() -> pd.DataFrame:
	store = CsvStore(CFG.db_path)
	rows = store.read_rows(limit=2000)
	return pd.DataFrame(rows)

def transcript_to_df(tr: List[Dict[str, Any]]) -> pd.DataFrame:
	flat = []
	for ev in tr:
		row = dict(ev)
		for k, v in list(row.items()):
			if isinstance(v, (dict, list)):
				row[k] = json.dumps(v, ensure_ascii=False)
		flat.append(row)
	return pd.DataFrame(flat)

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

def kpis(df: pd.DataFrame) -> Dict[str, int]:
	if df is None or df.empty:
		return {"events": 0, "near_blocks": 0, "far_blocks": 0, "pim_drops": 0, "db_writes": 0}
	near_blocks = int(((df.get("who") == "NEAR") & (df.get("status") == "BLOCK")).sum())
	far_blocks = int(((df.get("who") == "FAR") & (df.get("status") == "BLOCK")).sum())
	pim_drops = int((df.get("status") == "DROP").sum())
	db_writes = 0
	if "tool" in df.columns:
		db_writes = int(((df["tool"].astype(str) == "write_db") & (df["status"] == "OK") & (df["who"] == "FAR")).sum())
	return {"events": len(df), "near_blocks": near_blocks, "far_blocks": far_blocks, "pim_drops": pim_drops, "db_writes": db_writes}

# -----------------------------
# Runner (single-machine simulation with rich logging)
# -----------------------------
def run_flow(
	task_prompt: str,
	mle_i_attack: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]],
	pim_attack: Any,
	near_threshold: float,
	far_threshold: float,
	max_skew_s: float,
	window_size: int,
) -> List[Dict[str, Any]]:
	sid = f"SID-{uuid.uuid4().hex[:8]}"
	box = CryptoBox.new()
	chan = SecureChannel(crypto=box)

	db = CsvStore(CFG.db_path)

	# Nano models (tuned deterministically)
	near_gate = NanoGate.train_synthetic(seed=9)
	far_gate = NanoGate.train_synthetic(seed=11)

	near = PIMState(sid=sid)
	far = PIMState(sid=sid)

	transcript: List[Dict[str, Any]] = [{
		"who": "UI",
		"event": "RUN",
		"sid": sid,
		"task_prompt": task_prompt,
		"mlei_attack": getattr(mle_i_attack, "__name__", str(mle_i_attack)) if mle_i_attack else "None",
		"pim_attack": pim_attack if isinstance(pim_attack, str) else (getattr(pim_attack, "__name__", "None") if pim_attack else "None"),
		"started_utc": datetime.utcnow().isoformat() + "Z",
		"step": 0,
	}]

	# "Cloud LLM plan" output (agent tool calls)
	plan = llm_agent_plan(task_prompt, db_preview=[])

	# Plan-level PIM attacks
	if pim_attack == "REORDER" and len(plan) >= 2:
		plan[0], plan[1] = plan[1], plan[0]
		transcript.append({"who": "ATTACK", "event": "PLAN_REORDER", "detail": "Swapped step 1 and 2", "step": 0})

	if pim_attack == "REPLAY" and len(plan) >= 1:
		plan = [plan[0]] + plan
		transcript.append({"who": "ATTACK", "event": "PLAN_REPLAY", "detail": "Duplicated step 1", "step": 0})

	# Execute each step
	for i, tool_call in enumerate(plan, start=1):
		# NEAR gate
		allow_n, score_n, _ = near_gate.decide(tool_call, near_threshold)
		transcript.append({
			"who": "NEAR", "event": "NEAR_GATE", "step": i,
			"tool": tool_call.get("tool"), "allow": bool(allow_n), "score": float(score_n),
			"payload": tool_call,
		})
		if not allow_n:
			continue

		# MLEI injection (agent layer)
		injected_call = tool_call
		if mle_i_attack is not None:
			injected_call = mle_i_attack(tool_call)
			transcript.append({
				"who": "ATTACK", "event": "MLEI_INJECT", "step": i,
				"tool_before": tool_call.get("tool"),
				"tool_after": injected_call.get("tool"),
				"payload_after": injected_call,
			})

		# Build PIM envelope on NEAR and advance NEAR state
		env = build_env(near, injected_call)
		advance_state(near, env)
		near.window_hashes.append(env["h"])

		# PIM envelope tamper
		env_sent = env
		if callable(pim_attack):
			env_sent = pim_attack(env)
			transcript.append({
				"who": "ATTACK", "event": "PIM_TAMPER", "step": i,
				"tamper": pim_attack.__name__,
				"env_before": env,
				"env_after": env_sent,
			})

		# Encrypt and send
		blob = chan.seal(env_sent)

		# FAR decrypt and PIM verify
		env_recv = chan.open(blob)
		rep_far = verify_env_report(far, env_recv, max_skew_s=max_skew_s)
		transcript.append({
			"who": "FAR", "event": "PIM_VERIFY_FAR", "step": i,
			"ok": rep_far["ok"], "reason": rep_far["reason"], "checks": rep_far["checks"],
			"skew_s": rep_far["skew_s"], "computed_hash": rep_far["computed_hash"],
			"env": env_recv, "canonical_core": rep_far["canonical_core"],
		})
		if not rep_far["ok"]:
			continue

		# FAR advances state (accepted message)
		advance_state(far, env_recv)
		far.window_hashes.append(env_recv["h"])

		# FAR gate
		payload = env_recv["payload"]
		allow_f, score_f, _ = far_gate.decide(payload, far_threshold)
		transcript.append({
			"who": "FAR", "event": "FAR_GATE", "step": i,
			"tool": payload.get("tool"), "allow": bool(allow_f), "score": float(score_f),
			"payload": payload,
		})

		if not allow_f:
			reply_payload = {"tool": "error", "text": f"FAR BLOCK by gate score={score_f:.3f}", "args": {}}
			transcript.append({
				"who": "FAR", "event": "EXEC_RESULT", "step": i, "tool": payload.get("tool"),
				"is_error": True, "result": {"error": "blocked_by_gate"},
			})
		else:
			result = exec_tool(db, payload)
			is_error = isinstance(result, dict) and ("error" in result)
			transcript.append({
				"who": "FAR", "event": "EXEC_RESULT", "step": i,
				"tool": payload.get("tool"), "is_error": bool(is_error), "result": result,
			})
			reply_payload = {"tool": "result", "text": "ok", "args": {"result": result}}

		# FAR reply envelope
		reply_env = build_env(far, reply_payload)
		advance_state(far, reply_env)
		far.window_hashes.append(reply_env["h"])
		reply_blob = chan.seal(reply_env)

		# NEAR verify reply (PIM)
		reply_recv = chan.open(reply_blob)
		rep_near = verify_env_report(near, reply_recv, max_skew_s=max_skew_s)
		transcript.append({
			"who": "NEAR", "event": "PIM_VERIFY_NEAR", "step": i,
			"ok": rep_near["ok"], "reason": rep_near["reason"], "checks": rep_near["checks"],
			"skew_s": rep_near["skew_s"], "computed_hash": rep_near["computed_hash"],
			"env": reply_recv, "canonical_core": rep_near["canonical_core"],
		})
		if rep_near["ok"]:
			advance_state(near, reply_recv)
			near.window_hashes.append(reply_recv["h"])

		# Window seals
		seal_n = maybe_window_seal(near, window_size=window_size)
		if seal_n:
			transcript.append({"who": "NEAR", "event": "WINDOW_SEAL", "step": i, **seal_n})
		seal_f = maybe_window_seal(far, window_size=window_size)
		if seal_f:
			transcript.append({"who": "FAR", "event": "WINDOW_SEAL", "step": i, **seal_f})

	transcript.append({
		"who": "UI", "event": "RUN_END", "sid": sid,
		"finished_utc": datetime.utcnow().isoformat() + "Z",
		"step": 999999,
	})

	# Assign status labels
	for ev in transcript:
		ev["status"] = status_of_event(ev)

	# Add “attacker view” (what an eavesdropper sees)
	# We can't reproduce real blobs for each event in this transcript without storing them,
	# but for the "knowledge boundary" visualization we include a representative example.
	transcript.append({
		"who": "ATTACK",
		"event": "EAVESDROP_SAMPLE",
		"step": 0,
		"attacker_sees": {
			"encrypted_blob": {"nonce": "<hex>", "ct": "<hex>"},
			"note": "Attacker sees only encrypted nonce+ct; cannot modify without AES-GCM failure.",
		},
		"status": "ATTACK",
	})

	return transcript

# -----------------------------
# Visual builders
# -----------------------------
def mermaid_component(mermaid_code: str, height: int = 520) -> None:
	"""
	Render Mermaid in Streamlit using a CDN.
	"""
	html = f"""
	<div class="mermaid">
	{mermaid_code}
	</div>
	<script type="module">
	  import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
	  mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
	</script>
	"""
	components.html(html, height=height, scrolling=True)

def icon_for_status(status: str) -> str:
	s = str(status)
	if s in ("OK", "ALLOW"):
		return "✅"
	if s == "ATTACK":
		return "⚠️"
	if s == "BLOCK":
		return "🛑"
	if s == "DROP":
		return "🟣"
	return "ℹ️"

def build_sequence_mermaid(tr: List[Dict[str, Any]], focus_step: Optional[int] = None) -> str:
	"""
	Dual-lane sequence diagram:
	  NEAR -> FAR -> CloudLLM
	  and FAR -> NEAR replies
	"""
	# Participants
	lines = [
		"sequenceDiagram",
		"participant N as NEAR",
		"participant T as Tunnel(AES-GCM)",
		"participant F as FAR",
		"participant C as Cloud LLM",
	]

	# We show plan as Cloud->Near
	lines.append("C-->>N: Plan (tool calls)")
	last_step = None

	# For each step, show key events (gate, pim verify, exec, reply)
	steps = sorted({int(e["step"]) for e in tr if isinstance(e.get("step"), int) and 1 <= e["step"] < 999999})

	for step in steps:
		if focus_step is not None and step != focus_step:
			continue

		# Extract relevant events for step
		evs = [e for e in tr if e.get("step") == step]
		# Find gate outcomes
		ng = next((e for e in evs if e.get("event") == "NEAR_GATE"), None)
		fg = next((e for e in evs if e.get("event") == "FAR_GATE"), None)
		pvf = next((e for e in evs if e.get("event") == "PIM_VERIFY_FAR"), None)
		pvn = next((e for e in evs if e.get("event") == "PIM_VERIFY_NEAR"), None)
		ex = next((e for e in evs if e.get("event") == "EXEC_RESULT"), None)
		inj = [e for e in evs if e.get("event") in ("MLEI_INJECT", "PIM_TAMPER")]

		tool = None
		if ng and isinstance(ng.get("payload"), dict):
			tool = ng["payload"].get("tool")
		tool = tool or (ng.get("tool") if ng else None) or "tool"

		# Near gate
		if ng:
			lines.append(f"Note over N: Step {step} | {tool} | NEAR gate {icon_for_status(ng['status'])} score={ng.get('score',0):.3f}")

		# Attacks
		for a in inj:
			if a["event"] == "MLEI_INJECT":
				lines.append(f"Note over N: MLEI inject {icon_for_status('ATTACK')} {a.get('tool_before')}→{a.get('tool_after')}")
			if a["event"] == "PIM_TAMPER":
				lines.append(f"Note over T: PIM tamper {icon_for_status('ATTACK')} {a.get('tamper')}")

		# Near to far
		if ng and ng.get("allow"):
			lines.append(f"N->>T: seal(env ctr={extract_ctr_from_step(tr, step, side='NEAR')})")
			lines.append("T->>F: deliver(ciphertext)")

		# Far PIM verify
		if pvf:
			lines.append(f"Note over F: FAR PIM verify {icon_for_status(pvf['status'])} ({pvf.get('reason','')})")

		# Far gate + exec + reply
		if fg:
			lines.append(f"Note over F: FAR gate {icon_for_status(fg['status'])} score={fg.get('score',0):.3f}")
		if ex:
			lines.append(f"F-->>F: exec {tool} {icon_for_status(ex['status'])}")

		# Reply
		if pvf and pvf.get("ok"):
			lines.append("F->>T: seal(reply)")
			lines.append("T->>N: deliver(reply ciphertext)")
		if pvn:
			lines.append(f"Note over N: NEAR PIM verify(reply) {icon_for_status(pvn['status'])} ({pvn.get('reason','')})")

		# Optional window seals
		wn = next((e for e in evs if e.get("event") == "WINDOW_SEAL" and e.get("who") == "NEAR"), None)
		wf = next((e for e in evs if e.get("event") == "WINDOW_SEAL" and e.get("who") == "FAR"), None)
		if wn:
			lines.append(f"Note over N: Window seal #{wn.get('window_idx')} anchor→{str(wn.get('anchor_after',''))[:8]}..")
		if wf:
			lines.append(f"Note over F: Window seal #{wf.get('window_idx')} anchor→{str(wf.get('anchor_after',''))[:8]}..")

	return "\n".join(lines)

def extract_ctr_from_step(tr: List[Dict[str, Any]], step: int, side: str = "NEAR") -> str:
	"""
	Best-effort: pull ctr from an envelope in a PIM verify event (far) for that step.
	"""
	if side == "NEAR":
		# Use FAR verify env as "the sent envelope"
		pvf = next((e for e in tr if e.get("step") == step and e.get("event") == "PIM_VERIFY_FAR"), None)
		if pvf and isinstance(pvf.get("env"), dict):
			return str(pvf["env"].get("ctr", "?"))
	return "?"

def build_chain_ribbon(tr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""
	Build a compact chain list from FAR PIM verify events:
	each accepted envelope has (ctr, prev, h, tool, ok, reason).
	"""
	out = []
	for e in tr:
		if e.get("event") != "PIM_VERIFY_FAR":
			continue
		env = e.get("env")
		if not isinstance(env, dict):
			continue
		payload = env.get("payload") or {}
		out.append({
			"step": e.get("step"),
			"ctr": env.get("ctr"),
			"tool": payload.get("tool"),
			"prev": str(env.get("prev", "")),
			"h": str(env.get("h", "")),
			"ok": bool(e.get("ok")),
			"reason": e.get("reason", ""),
			"status": e.get("status"),
		})
	return out

def ribbon_html(nodes: List[Dict[str, Any]], focus_step: Optional[int] = None) -> str:
	"""
	Minimal HTML ribbon. Uses inline styles (no external deps).
	"""
	if not nodes:
		return "<div style='padding:8px;color:#555'>No chain nodes (no FAR PIM verify events).</div>"

	def color(status: str, ok: bool) -> str:
		if status == "OK" and ok:
			return "rgba(0,200,0,0.18)"
		if status == "DROP" or not ok:
			return "rgba(160,0,255,0.14)"
		return "rgba(120,120,120,0.10)"

	items = []
	# genesis
	items.append(f"""
	  <div style="display:inline-block;padding:6px 10px;border-radius:10px;background:rgba(120,120,120,0.10);margin-right:8px;">
		GENESIS
	  </div>
	  <span style="margin-right:8px;">→</span>
	""")

	for n in nodes:
		step = n.get("step")
		is_focus = (focus_step is not None and step == focus_step)
		bg = color(n.get("status","INFO"), n.get("ok", False))
		border = "2px solid rgba(0,0,0,0.28)" if is_focus else "1px solid rgba(0,0,0,0.12)"
		label = f"ctr {n.get('ctr')} | {n.get('tool')} | {n.get('h','')[:8]}"
		items.append(f"""
		  <div style="display:inline-block;padding:6px 10px;border-radius:10px;background:{bg};border:{border};margin-right:8px;">
			{label}
		  </div>
		  <span style="margin-right:8px;">→</span>
		""")

	return f"<div style='white-space:nowrap;overflow-x:auto;padding:8px;border:1px solid rgba(0,0,0,0.08);border-radius:12px;'>{''.join(items)}</div>"

def build_windows_ledger(tr: List[Dict[str, Any]]) -> pd.DataFrame:
	rows = []
	for e in tr:
		if e.get("event") == "WINDOW_SEAL":
			rows.append({
				"who": e.get("who"),
				"step": e.get("step"),
				"window_idx": e.get("window_idx"),
				"window_size": e.get("window_size"),
				"window_hash": str(e.get("window_hash",""))[:12],
				"anchor_before": str(e.get("anchor_before",""))[:12],
				"anchor_after": str(e.get("anchor_after",""))[:12],
				"last_ctr": e.get("last_ctr"),
			})
	return pd.DataFrame(rows)

def build_steps_checks_heatmap(tr: List[Dict[str, Any]]) -> pd.DataFrame:
	"""
	Rows=steps; columns=checks:
	  PIM: sid/ctr/prev/skew/hash  (from FAR verify)
	  MLEI: near_allow, far_allow, tool_allowed_proxy
	Values: 1=pass, 0=fail, NaN=missing
	"""
	steps = sorted({int(e["step"]) for e in tr if isinstance(e.get("step"), int) and 1 <= e["step"] < 999999})
	cols = ["pim_sid","pim_ctr","pim_prev","pim_skew","pim_hash","near_allow","far_allow"]
	data = {c: [] for c in cols}
	data["step"] = []

	for s in steps:
		data["step"].append(s)

		ng = next((e for e in tr if e.get("step") == s and e.get("event") == "NEAR_GATE"), None)
		fg = next((e for e in tr if e.get("step") == s and e.get("event") == "FAR_GATE"), None)
		pvf = next((e for e in tr if e.get("step") == s and e.get("event") == "PIM_VERIFY_FAR"), None)

		# near_allow/far_allow
		data["near_allow"].append(1 if (ng and ng.get("allow")) else 0 if ng else None)
		data["far_allow"].append(1 if (fg and fg.get("allow")) else 0 if fg else None)

		# PIM checks (FAR side)
		if pvf and isinstance(pvf.get("checks"), dict):
			ch = pvf["checks"]
			data["pim_sid"].append(1 if ch.get("sid") else 0)
			data["pim_ctr"].append(1 if ch.get("ctr") else 0)
			data["pim_prev"].append(1 if ch.get("prev") else 0)
			data["pim_skew"].append(1 if ch.get("skew") else 0)
			data["pim_hash"].append(1 if ch.get("hash") else 0)
		else:
			data["pim_sid"].append(None)
			data["pim_ctr"].append(None)
			data["pim_prev"].append(None)
			data["pim_skew"].append(None)
			data["pim_hash"].append(None)

	return pd.DataFrame(data).set_index("step")

# def heatmap_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
#     def cell_style(v):
#         if pd.isna(v):
#             return "background-color: rgba(120,120,120,0.06); color: rgba(0,0,0,0.35)"
#         if float(v) >= 1:
#             return "background-color: rgba(0,200,0,0.12)"
#         return "background-color: rgba(255,0,0,0.12)"
#     return df.style.applymap(cell_style)
try:
	from pandas.io.formats.style import Styler
except Exception:
	Styler = object

def heatmap_style(df: pd.DataFrame) -> "Styler":
	def cell_style(v):
		if pd.isna(v):
			return "background-color: rgba(120,120,120,0.06); color: rgba(0,0,0,0.35)"
		if float(v) >= 1:
			return "background-color: rgba(0,200,0,0.12)"
		return "background-color: rgba(255,0,0,0.12)"
	return df.style.applymap(cell_style)

def build_outcome_buckets(tr: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
	"""
	4 buckets:
	  1) Rejected by policy (MLEI) -> NEAR_GATE false or FAR_GATE false
	  2) Dropped by integrity (PIM) -> PIM_VERIFY_FAR false or PIM_VERIFY_NEAR false
	  3) Executed but returned error -> EXEC_RESULT is_error true (not gate block)
	  4) Succeeded -> EXEC_RESULT OK and reply verified OK
	"""
	steps = sorted({int(e["step"]) for e in tr if isinstance(e.get("step"), int) and 1 <= e["step"] < 999999})
	buckets = {"Policy rejected (MLEI)": [], "Integrity dropped (PIM)": [], "Executed with error": [], "Succeeded": []}

	for s in steps:
		ng = next((e for e in tr if e.get("step") == s and e.get("event") == "NEAR_GATE"), None)
		fg = next((e for e in tr if e.get("step") == s and e.get("event") == "FAR_GATE"), None)
		pvf = next((e for e in tr if e.get("step") == s and e.get("event") == "PIM_VERIFY_FAR"), None)
		pvn = next((e for e in tr if e.get("step") == s and e.get("event") == "PIM_VERIFY_NEAR"), None)
		ex = next((e for e in tr if e.get("step") == s and e.get("event") == "EXEC_RESULT"), None)

		tool = None
		if ng and isinstance(ng.get("payload"), dict):
			tool = ng["payload"].get("tool")
		tool = tool or (ng.get("tool") if ng else None) or "tool"

		# Policy reject if near gate denies OR far gate denies
		if (ng and not ng.get("allow")) or (fg and not fg.get("allow")):
			why = "NEAR gate" if (ng and not ng.get("allow")) else "FAR gate"
			buckets["Policy rejected (MLEI)"].append({"step": s, "tool": tool, "why": why})
			continue

		# Integrity drop
		if (pvf and not pvf.get("ok")) or (pvn and not pvn.get("ok")):
			why = pvf.get("reason") if (pvf and not pvf.get("ok")) else pvn.get("reason")
			buckets["Integrity dropped (PIM)"].append({"step": s, "tool": tool, "why": why})
			continue

		# Executed error
		if ex and ex.get("is_error"):
			buckets["Executed with error"].append({"step": s, "tool": tool, "why": "execution_error"})
			continue

		# Success
		if ex and (not ex.get("is_error")) and (pvn is None or pvn.get("ok", True)):
			buckets["Succeeded"].append({"step": s, "tool": tool, "why": "ok"})
			continue

	return buckets

def knowledge_boundary_view(tr: List[Dict[str, Any]], focus_step: Optional[int]) -> Dict[str, Any]:
	"""
	Show “who knew what” for the focused step.
	"""
	s = focus_step
	if s is None:
		return {}
	ng = next((e for e in tr if e.get("step") == s and e.get("event") == "NEAR_GATE"), None)
	pvf = next((e for e in tr if e.get("step") == s and e.get("event") == "PIM_VERIFY_FAR"), None)
	blob_sample = next((e for e in tr if e.get("event") == "EAVESDROP_SAMPLE"), None)

	near_payload = ng.get("payload") if (ng and isinstance(ng.get("payload"), dict)) else None
	far_env = pvf.get("env") if (pvf and isinstance(pvf.get("env"), dict)) else None

	cloud_plan = "Cloud LLM returns tool calls (plan), not session secrets."
	attacker = blob_sample.get("attacker_sees") if blob_sample else {}

	return {
		"Near sees (agent + policy layer)": {
			"tool_call": near_payload,
			"note": "Near gate sees the tool call before sealing; decides ALLOW/BLOCK."
		},
		"Tunnel carries": {
			"ciphertext": {"nonce": "<hex>", "ct": "<hex>"},
			"note": "AES-GCM sealed envelope; tampering fails decrypt/auth."
		},
		"Far sees (after decrypt)": {
			"envelope": far_env,
			"note": "Far verifies PIM chain before executing tools; applies FAR gate."
		},
		"Cloud LLM sees": {
			"note": cloud_plan
		},
		"Attacker sees": attacker or {"note": "Encrypted blobs only."}
	}

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="WARL0K PIM+MLEI Visual Demo", layout="wide")
st.title("WARL0K PIM + MLEI — Visual Explainability Demo")
st.caption("Sequence diagram • Chain ribbon • Window ledger • Gauges • Steps×Checks heatmap • Replay slider • Outcome buckets • Knowledge boundaries")

# Session state
if "authed" not in st.session_state:
	st.session_state.authed = False
if "transcript" not in st.session_state:
	st.session_state.transcript = []
if "focus_step" not in st.session_state:
	st.session_state.focus_step = 1

# Sidebar
with st.sidebar:
	st.header("Controls")

	st.subheader("Login (demo/demo)")
	u = st.text_input("Username", value="demo")
	p = st.text_input("Password", value="demo", type="password")
	col1, col2 = st.columns(2)
	with col1:
		if st.button("Authenticate"):
			st.session_state.authed = (u == "demo" and p == "demo")
			st.success("Auth granted ✅" if st.session_state.authed else "Auth denied ❌")
	with col2:
		if st.button("Reset transcript"):
			st.session_state.transcript = []
			st.session_state.focus_step = 1

	st.divider()

	st.subheader("Scenario")
	task = st.selectbox(
		"Task",
		["Task 1: read DB and summarize", "Task 2: write validated result row"],
		disabled=not st.session_state.authed,
	)
	mlei_label = st.selectbox(
		"MLEI attack (agent layer)",
		list(MLEI_ATTACKS.keys()),
		disabled=not st.session_state.authed,
	)
	pim_label = st.selectbox(
		"PIM attack (chain rules)",
		list(PIM_ATTACKS.keys()),
		disabled=not st.session_state.authed,
	)

	st.divider()
	st.subheader("Parameters")
	max_skew_s = st.slider("PIM max skew (s)", 0.5, 10.0, float(CFG.max_skew_s), 0.5)
	window_size = st.slider("Window seal every N messages", 2, 32, int(min(8, CFG.window_size)), 1)
	near_thr = st.slider("NEAR gate threshold", 0.40, 0.95, float(CFG.near_threshold), 0.01)
	far_thr = st.slider("FAR gate threshold", 0.40, 0.95, float(CFG.far_threshold), 0.01)

	st.divider()
	st.subheader("DB")
	if st.button("Reset demo_db.csv"):
		reset_db_bootstrap()
		st.success("DB reset ✅")

	run_btn = st.button("▶ Run", type="primary", disabled=not st.session_state.authed)

# Run
if run_btn:
	tr = run_flow(
		task_prompt=task,
		mle_i_attack=MLEI_ATTACKS[mlei_label],
		pim_attack=PIM_ATTACKS[pim_label],
		near_threshold=near_thr,
		far_threshold=far_thr,
		max_skew_s=max_skew_s,
		window_size=window_size,
	)
	st.session_state.transcript = tr
	# Focus first real step if exists
	steps = sorted({int(e["step"]) for e in tr if isinstance(e.get("step"), int) and 1 <= e["step"] < 999999})
	st.session_state.focus_step = steps[0] if steps else 1
	st.success("Run complete ✅")

tr = st.session_state.transcript

# Layout
top_left, top_right = st.columns([1.35, 1.0], gap="large")

with top_right:
	st.subheader("CSV DB (tail)")
	db_df = load_db_df()
	if db_df.empty:
		st.info("DB is empty.")
	else:
		st.dataframe(db_df.tail(25), use_container_width=True)

with top_left:
	if not tr:
		st.info("Authenticate, choose scenario, click **Run**.")
	else:
		df = transcript_to_df(tr)
		if "status" not in df.columns:
			df["status"] = [e.get("status","INFO") for e in tr]

		# KPIs + Gauges
		kk = kpis(df)
		integrity_total = int(((df["event"] == "PIM_VERIFY_FAR") | (df["event"] == "PIM_VERIFY_NEAR")).sum())
		integrity_ok = int(((df["event"].isin(["PIM_VERIFY_FAR","PIM_VERIFY_NEAR"])) & (df["status"] == "OK")).sum())
		integrity_pct = int((integrity_ok / integrity_total) * 100) if integrity_total else 100

		authority_total = int(((df["event"] == "NEAR_GATE") | (df["event"] == "FAR_GATE")).sum())
		authority_ok = int(((df["event"].isin(["NEAR_GATE","FAR_GATE"])) & (df["status"] == "ALLOW")).sum())
		authority_pct = int((authority_ok / authority_total) * 100) if authority_total else 100

		k1, k2, k3, k4, k5 = st.columns(5)
		k1.metric("Events", kk["events"])
		k2.metric("NEAR blocks", kk["near_blocks"])
		k3.metric("FAR blocks", kk["far_blocks"])
		k4.metric("PIM drops", kk["pim_drops"])
		k5.metric("DB writes", kk["db_writes"])

		g1, g2 = st.columns(2)
		with g1:
			st.metric("PIM Integrity Score", f"{integrity_pct}%")
			st.progress(integrity_pct / 100.0)
		with g2:
			st.metric("MLEI Authority Score", f"{authority_pct}%")
			st.progress(authority_pct / 100.0)

		st.divider()

		# Replay slider
		steps = sorted({int(e["step"]) for e in tr if isinstance(e.get("step"), int) and 1 <= e["step"] < 999999})
		# Replay control
		if steps:
			if len(steps) == 1:
				focus = steps[0]
				st.info(f"Only one step in this run (step={focus}). Replay slider disabled.")
			else:
				focus = st.slider(
					"Replay step (focus)",
					min_value=min(steps),
					max_value=max(steps),
					value=int(st.session_state.focus_step) if int(st.session_state.focus_step) in steps else steps[0],
					step=1,
				)
			st.session_state.focus_step = focus
		else:
			focus = None
			
		# if steps:
		#     focus = st.slider("Replay step (focus)", min_value=min(steps), max_value=max(steps), value=int(st.session_state.focus_step), step=1)
		#     st.session_state.focus_step = focus
		# else:
		#     focus = None

		# Mermaid sequence diagram (full + focused)
		st.subheader("1) Sequence Diagram (Mermaid)")
		mode = st.radio("View", ["Full run", "Focused step only"], horizontal=True)
		m = build_sequence_mermaid(tr, focus_step=(focus if mode == "Focused step only" else None))
		mermaid_component(m, height=520 if mode == "Full run" else 360)

		st.divider()

		# PIM chain ribbon
		st.subheader("2) PIM Chain Ribbon (FAR accepted envelopes)")
		nodes = build_chain_ribbon(tr)
		st.markdown(ribbon_html(nodes, focus_step=focus), unsafe_allow_html=True)

		st.divider()

		# Windows ledger
		st.subheader("3) Window Ledger + Anchor Rotation")
		ledger = build_windows_ledger(tr)
		if ledger.empty:
			st.caption("No window seals (try smaller window size like 2–4).")
		else:
			st.dataframe(ledger, use_container_width=True, height=220)

		st.divider()

		# Steps × Checks heatmap
		st.subheader("4) Steps × Checks Heatmap (PIM + MLEI)")
		hm = build_steps_checks_heatmap(tr)
		if hm.empty:
			st.caption("No steps captured.")
		else:
			st.dataframe(heatmap_style(hm), use_container_width=True, height=260)
			st.caption("Green=pass, Red=fail, Gray=missing (not reached).")

		st.divider()

		# Outcome buckets
		st.subheader("5) Outcome Buckets (Why it stopped)")
		buckets = build_outcome_buckets(tr)
		b1, b2, b3, b4 = st.columns(4)
		with b1:
			st.markdown("**Policy rejected (MLEI)**")
			st.write(buckets["Policy rejected (MLEI)"][:10] if buckets["Policy rejected (MLEI)"] else "—")
		with b2:
			st.markdown("**Integrity dropped (PIM)**")
			st.write(buckets["Integrity dropped (PIM)"][:10] if buckets["Integrity dropped (PIM)"] else "—")
		with b3:
			st.markdown("**Executed with error**")
			st.write(buckets["Executed with error"][:10] if buckets["Executed with error"] else "—")
		with b4:
			st.markdown("**Succeeded**")
			st.write(buckets["Succeeded"][:10] if buckets["Succeeded"] else "—")

		st.divider()

		# Knowledge boundary
		st.subheader("6) Who Knew What (Knowledge Boundary)")
		if focus is not None:
			kb = knowledge_boundary_view(tr, focus)
			kba, kbb = st.columns(2)
			with kba:
				st.markdown("**Near / Far**")
				st.json({
					"Near": kb.get("Near sees (agent + policy layer)", {}),
					"Far": kb.get("Far sees (after decrypt)", {}),
				})
			with kbb:
				st.markdown("**Cloud / Attacker**")
				st.json({
					"Cloud": kb.get("Cloud LLM sees", {}),
					"Attacker": kb.get("Attacker sees", {}),
				})
		else:
			st.caption("No focus step available.")

		st.divider()

		# Event log + Inspector
		st.subheader("7) Event Log + Inspector")
		filt1, filt2, filt3 = st.columns(3)
		with filt1:
			who_filter = st.multiselect("who", sorted(df["who"].dropna().unique().tolist()), default=[])
		with filt2:
			event_filter = st.multiselect("event", sorted(df["event"].dropna().unique().tolist()), default=[])
		with filt3:
			status_filter = st.multiselect("status", sorted(df["status"].dropna().unique().tolist()), default=[])

		view = df.copy()
		if who_filter:
			view = view[view["who"].isin(who_filter)]
		if event_filter:
			view = view[view["event"].isin(event_filter)]
		if status_filter:
			view = view[view["status"].isin(status_filter)]

		# Put important cols first
		preferred = [c for c in ["who","event","status","step","tool","allow","score","ok","reason"] if c in view.columns]
		rest = [c for c in view.columns if c not in preferred]
		view = view[preferred + rest]

		# Select row index
		max_idx = max(0, len(view) - 1)
		sel_idx = st.number_input("Inspect row (index in filtered view)", 0, max_idx, 0, 1)

		styled = view.style
		if "status" in view.columns:
			styled = styled.applymap(style_status, subset=["status"])
		st.dataframe(styled, use_container_width=True, height=320)

		if len(view) > 0:
			row = view.iloc[int(sel_idx)].to_dict()
			st.markdown(f"**Inspect:** `{row.get('who')}` / `{row.get('event')}` / status **{row.get('status')}** / step **{row.get('step')}**")

			# Gate inspector
			if row.get("event") in ("NEAR_GATE","FAR_GATE"):
				st.json({
					"allow": row.get("allow"),
					"score": row.get("score"),
					"tool": row.get("tool"),
					"payload": row.get("payload"),
				})

			# PIM inspector
			if row.get("event") in ("PIM_VERIFY_FAR","PIM_VERIFY_NEAR"):
				st.markdown("**PIM checks**")
				st.json({
					"ok": row.get("ok"),
					"reason": row.get("reason"),
					"skew_s": row.get("skew_s"),
					"computed_hash": row.get("computed_hash"),
					"checks": row.get("checks"),
				})
				st.markdown("**Envelope (received)**")
				st.json(row.get("env"))
				st.markdown("**Canonical core (what is hashed)**")
				st.code(row.get("canonical_core",""), language="json")

			# Exec inspector
			if row.get("event") == "EXEC_RESULT":
				st.json({
					"tool": row.get("tool"),
					"is_error": row.get("is_error"),
					"result": row.get("result"),
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
st.code("streamlit run app_pim_mlei_visuals.py", language="bash")
