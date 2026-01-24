import json
from pathlib import Path
from .qa import answer

def eval_questions(index_dir: Path, q_path: Path, out_path: Path, limit=200):
    results = []
    for i, line in enumerate(q_path.read_text().splitlines()):
        if i >= limit:
            break
        q = json.loads(line)["q"]
        res = answer(index_dir, q)
        results.append({
            "q": q,
            "frame": res.frame["id"],
            "soft": res.debug["soft_fallback_used"],
            "covered_edges": res.debug["covered_edges"],
            "drift_ratio": res.debug["drift_ratio"]
        })
    out_path.write_text("\n".join(json.dumps(r) for r in results))
