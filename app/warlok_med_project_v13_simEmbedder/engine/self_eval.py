from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any
from .qa import answer
from .domain_pack import DomainPack


def eval_questions(index_dir: Path, q_path: Path, out_path: Path, domain: DomainPack, limit: int = 600) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    soft = 0
    high_drift = 0

    rows = []
    for i, line in enumerate(q_path.read_text(errors="ignore").splitlines()):
        if i >= limit:
            break
        obj = json.loads(line)
        q = obj["q"]
        res = answer(index_dir, q, domain)
        total += 1
        if res.debug.get("soft_fallback_used"):
            soft += 1
        if float(res.debug.get("drift_ratio") or 0.0) >= 0.35:
            high_drift += 1

        rows.append({
            "q": q,
            "frame": res.frame.get("id"),
            "soft": bool(res.debug.get("soft_fallback_used")),
            "drift_ratio": float(res.debug.get("drift_ratio") or 0.0),
            "covered_edges": res.debug.get("covered_edges", []),
            "missing_edges": res.debug.get("missing_edges", []),
        })

    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    return {"evaluated": total, "soft": soft, "high_drift": high_drift, "out_path": str(out_path)}

# from __future__ import annotations
# import json
# from pathlib import Path
# from typing import Dict, Any, List
#
# from .qa import answer
# from .domain_pack import DomainPack
#
# def eval_questions(index_dir: Path, q_path: Path, out_path: Path,
#                    domain: DomainPack,
#                    limit: int = 2000) -> Dict[str, Any]:
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#
#     total = 0
#     soft = 0
#     high_drift = 0
#     missing_edge_counts = {}
#
#     rows: List[Dict[str, Any]] = []
#
#     for i, line in enumerate(q_path.read_text(errors="ignore").splitlines()):
#         if i >= limit:
#             break
#         if not line.strip():
#             continue
#         obj = json.loads(line)
#         q = obj["q"]
#
#         res = answer(index_dir, q, domain)
#         total += 1
#         if res.debug.get("soft_fallback_used"):
#             soft += 1
#         if (res.debug.get("drift_ratio") or 0) >= 0.35:
#             high_drift += 1
#
#         frame_id = res.frame.get("id")
#         frame = domain.frames.get(frame_id)
#         required = frame.required_edges if frame else []
#         covered = set(res.debug.get("covered_edges", []))
#         missing = [e for e in required if e not in covered]
#
#         for e in missing:
#             missing_edge_counts[e] = missing_edge_counts.get(e, 0) + 1
#
#         rows.append({
#             "q": q,
#             "frame": frame_id,
#             "soft": bool(res.debug.get("soft_fallback_used")),
#             "drift_ratio": float(res.debug.get("drift_ratio") or 0.0),
#             "covered_edges": list(covered),
#             "missing_edges": missing,
#             "retrieval": res.debug.get("retrieval", {}),
#             "notes": obj.get("source", ""),
#         })
#
#     out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
#
#     return {
#         "evaluated": total,
#         "soft_fallback": soft,
#         "high_drift": high_drift,
#         "top_missing_edges": sorted(missing_edge_counts.items(), key=lambda x: x[1], reverse=True)[:20],
#         "out_path": str(out_path)
#     }
