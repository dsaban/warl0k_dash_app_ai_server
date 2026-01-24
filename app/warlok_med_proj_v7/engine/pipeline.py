from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from .ingest import build_index
from .question_gen import generate_questions
from .self_eval import eval_questions

@dataclass
class BuildResult:
    ok: bool
    message: str
    stats: Dict[str, Any]

def ensure_dirs(root: Path) -> Dict[str, Path]:
    docs_dir = root / "docs"
    index_dir = root / "data" / "index"
    eval_dir = root / "data" / "eval"
    lexicon_path = root / "data" / "lexicon.json"

    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {
        "docs_dir": docs_dir,
        "index_dir": index_dir,
        "eval_dir": eval_dir,
        "lexicon_path": lexicon_path,
    }

def save_uploaded_docs(docs_dir: Path, uploaded_files: List, overwrite: bool = True) -> Dict[str, Any]:
    """
    uploaded_files: streamlit file_uploader list of UploadedFile objects
    """
    saved = []
    skipped = []
    for uf in uploaded_files or []:
        name = Path(uf.name).name  # sanitize
        if not name.lower().endswith(".txt"):
            skipped.append({"file": name, "reason": "not .txt"})
            continue
        dest = docs_dir / name
        if dest.exists() and not overwrite:
            skipped.append({"file": name, "reason": "exists"})
            continue
        dest.write_bytes(uf.getbuffer())
        saved.append(str(dest))
    return {"saved": saved, "skipped": skipped}

def build_all(root: Path, run_eval: bool = True, eval_limit: int = 600) -> BuildResult:
    paths = ensure_dirs(root)
    docs_dir = paths["docs_dir"]
    index_dir = paths["index_dir"]
    eval_dir = paths["eval_dir"]
    lexicon_path = paths["lexicon_path"]

    txts = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
    if not txts:
        return BuildResult(False, "No .txt docs found in ./docs. Upload docs first.", {})

    stats = {}
    idx_stats = build_index(docs_dir, index_dir, lexicon_path)
    stats["index"] = idx_stats

    if run_eval:
        q_path = eval_dir / "generated_questions.jsonl"
        q_stats = generate_questions(index_dir, q_path)
        stats["qgen"] = q_stats

        res_path = eval_dir / "results.jsonl"
        eval_questions(index_dir, q_path, res_path, limit=eval_limit)
        stats["eval"] = {"results_path": str(res_path), "limit": eval_limit}

    return BuildResult(True, "Build completed.", stats)
