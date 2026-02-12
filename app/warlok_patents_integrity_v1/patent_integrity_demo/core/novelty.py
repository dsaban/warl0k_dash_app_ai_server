from __future__ import annotations
from typing import Dict, Any, List
from collections import defaultdict

def novelty_summary(selected_atomic: List[Dict[str, Any]], all_atomic: List[Dict[str, Any]]) -> Dict[str, Any]:
    bundle = defaultdict(set)
    for a in selected_atomic:
        for cat, tags in (a.get("tags") or {}).items():
            for t in tags:
                bundle[cat].add(t)

    tag_counts = defaultdict(int)
    for a in all_atomic:
        for cat, tags in (a.get("tags") or {}).items():
            for t in tags:
                tag_counts[(cat, t)] += 1

    rare = []
    for cat, tags in bundle.items():
        for t in tags:
            rare.append(((cat, t), tag_counts[(cat, t)]))
    rare.sort(key=lambda x: x[1])

    return {
        "bundle": {k: sorted(list(v)) for k, v in bundle.items()},
        "rare_tags": [{"category": c, "tag": t, "count": n} for ((c, t), n) in rare[:12]],
    }

def draft_claim_skeleton(bundle: Dict[str, List[str]]) -> str:
    mech = bundle.get("mechanism", [])
    crypto = bundle.get("crypto", [])
    constraints = bundle.get("constraint", [])
    threats = bundle.get("threat", [])

    lines = []
    lines.append("Independent claim skeleton (method):")
    lines.append("1) generating, by a verifier, a challenge bound to a session context and a freshness condition;")
    lines.append("2) deriving, by a device, a proof value based on at least one device-identity substrate and the challenge;")
    if "MAC_AEAD" in crypto:
        lines.append("3) computing, by the device, an authenticator using AEAD/MAC over the challenge and/or payload;")
    else:
        lines.append("3) computing, by the device, an authenticator over the challenge and/or payload;")
    lines.append("4) transmitting the proof to the verifier;")
    lines.append("5) validating, by the verifier, the proof within the freshness condition;")
    if "ANOMALY_SCORING" in mech:
        lines.append("6) computing an anomaly/risk score over observed proof behavior and enforcing a policy based on the score;")
    lines.append("7) granting, maintaining, limiting, or revoking access based on validation results.")
    if constraints:
        lines.append("")
        lines.append("Constraints / deployment hooks:")
        lines.append("- " + "; ".join(constraints))
    if threats:
        lines.append("")
        lines.append("Threat focus:")
        lines.append("- " + "; ".join(threats))
    return "\n".join(lines)
