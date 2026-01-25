from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

from .domain_pack import DomainPack
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

    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    return {"docs_dir": docs_dir, "index_dir": index_dir, "eval_dir": eval_dir}


def save_uploaded_docs(docs_dir: Path, uploaded_files: List, overwrite: bool = True) -> Dict[str, Any]:
    saved = []
    skipped = []
    for uf in uploaded_files or []:
        name = Path(uf.name).name
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


def build_all(root: Path, pack_dir: Path, run_eval: bool = True, eval_limit: int = 600) -> BuildResult:
    paths = ensure_dirs(root)
    docs_dir = paths["docs_dir"]
    index_dir = paths["index_dir"]
    eval_dir = paths["eval_dir"]

    txts = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
    if not txts:
        return BuildResult(False, "No .txt docs found in ./docs. Upload docs first.", {})

    if not (pack_dir / "manifest.json").exists():
        return BuildResult(False, f"Selected pack missing manifest.json: {pack_dir}", {})

    try:
        domain = DomainPack.load(pack_dir)
    except Exception as e:
        return BuildResult(False, f"DomainPack load failed: {e}", {})

    stats: Dict[str, Any] = {
        "domain_pack": {
            "name": domain.manifest.get("name"),
            "version": domain.manifest.get("version"),
            "dir": str(pack_dir),
        }
    }

    idx_stats = build_index(docs_dir, index_dir, domain)
    stats["index"] = idx_stats

    if run_eval:
        q_path = eval_dir / "generated_questions.jsonl"
        q_stats = generate_questions(index_dir=index_dir, out_path=q_path, domain=domain, max_total=3000)
        stats["qgen"] = q_stats

        res_path = eval_dir / "results.jsonl"
        ev_stats = eval_questions(index_dir=index_dir, q_path=q_path, out_path=res_path, domain=domain, limit=eval_limit)
        stats["eval"] = ev_stats

    return BuildResult(True, "Build completed.", stats)

# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import List, Dict, Any
#
# from .domain_pack import DomainPack
# from .ingest import build_index
# from .question_gen import generate_questions
# from .self_eval import eval_questions
#
#
# @dataclass
# class BuildResult:
#     ok: bool
#     message: str
#     stats: Dict[str, Any]
#
#
# def ensure_dirs(root: Path) -> Dict[str, Path]:
#     docs_dir = root / "docs"
#     index_dir = root / "data" / "index"
#     eval_dir = root / "data" / "eval"
#     lexicon_path = root / "data" / "lexicon.json"
#
#     docs_dir.mkdir(parents=True, exist_ok=True)
#     index_dir.mkdir(parents=True, exist_ok=True)
#     eval_dir.mkdir(parents=True, exist_ok=True)
#     lexicon_path.parent.mkdir(parents=True, exist_ok=True)
#
#     return {
#         "docs_dir": docs_dir,
#         "index_dir": index_dir,
#         "eval_dir": eval_dir,
#         "lexicon_path": lexicon_path,
#     }
#
#
# def save_uploaded_docs(docs_dir: Path, uploaded_files: List, overwrite: bool = True) -> Dict[str, Any]:
#     saved = []
#     skipped = []
#     for uf in uploaded_files or []:
#         name = Path(uf.name).name
#         if not name.lower().endswith(".txt"):
#             skipped.append({"file": name, "reason": "not .txt"})
#             continue
#         dest = docs_dir / name
#         if dest.exists() and not overwrite:
#             skipped.append({"file": name, "reason": "exists"})
#             continue
#         dest.write_bytes(uf.getbuffer())
#         saved.append(str(dest))
#     return {"saved": saved, "skipped": skipped}
#
#
# def build_all(root: Path, pack_dir: Path, run_eval: bool = True, eval_limit: int = 600) -> BuildResult:
#     # IMPORTANT: no DomainPack.load at import time. Only here.
#     paths = ensure_dirs(root)
#     docs_dir = paths["docs_dir"]
#     index_dir = paths["index_dir"]
#     eval_dir = paths["eval_dir"]
#     lexicon_path = paths["lexicon_path"]
#
#     txts = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
#     if not txts:
#         return BuildResult(False, "No .txt docs found in ./docs. Upload docs first.", {})
#
#     # Validate pack dir
#     if not (pack_dir / "manifest.json").exists():
#         return BuildResult(False, f"Selected pack missing manifest.json: {pack_dir}", {})
#
#     # Load domain pack
#     try:
#         domain = DomainPack.load(pack_dir)
#     except Exception as e:
#         return BuildResult(False, f"DomainPack load failed: {e}", {})
#
#     stats: Dict[str, Any] = {}
#     stats["domain_pack"] = {
#         "name": domain.manifest.get("name"),
#         "version": domain.manifest.get("version"),
#         "dir": str(pack_dir),
#     }
#
#     # 1) Build index/ontology/claims
#     idx_stats = build_index(docs_dir, index_dir, lexicon_path, domain)
#     stats["index"] = idx_stats
#
#     # 2) Optional: generate questions + self-eval
#     if run_eval:
#         q_path = eval_dir / "generated_questions.jsonl"
#         q_stats = generate_questions(index_dir=index_dir, out_path=q_path, domain=domain)
#         stats["qgen"] = q_stats
#
#         res_path = eval_dir / "results.jsonl"
#         eval_stats = eval_questions(index_dir=index_dir, q_path=q_path, out_path=res_path, domain=domain, limit=eval_limit)
#         stats["eval"] = eval_stats
#
#     return BuildResult(True, "Build completed.", stats)
#
# # from __future__ import annotations
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import List, Dict, Any
# #
# # from .domain_pack import DomainPack
# # from .ingest import build_index
# # from .question_gen import generate_questions
# # from .self_eval import eval_questions
# #
# # # from .question_gen import generate_questions
# # # from .domain_pack import DomainPack
# #
# # # domain = DomainPack.load(Path("./domain_packs"))
# # from pathlib import Path
# # from .domain_pack import DomainPack
# #
# # def load_domain(pack_dir: Path) -> DomainPack:
# #     return DomainPack.load(pack_dir)
# #
# # def build_all(root: Path, pack_dir: Path):
# #     paths = ensure_dirs(root)
# #     domain = DomainPack.load(pack_dir)
# #     print("Building index...")
# #     print("Building questions...")
# #     print("Running evaluation...")
# #     # print domain path
# #     print(f"Using domain pack from: {pack_dir}")
# #
# #     # Further implementation would go here
# #
# # domain = DomainPack.load(Path("./domain_pack"))
# # index_dir = Path("./data/index")
# # questions_path = Path("./data/eval/generated_questions.jsonl")
# #
# # stats = generate_questions(
# #     index_dir=index_dir,
# #     out_path=questions_path,
# #     domain=domain
# # )
# #
# # @dataclass
# # class BuildResult:
# #     ok: bool
# #     message: str
# #     stats: Dict[str, Any]
# #
# # def ensure_dirs(root: Path) -> Dict[str, Path]:
# #     docs_dir = root / "docs"
# #     index_dir = root / "data" / "index"
# #     eval_dir = root / "data" / "eval"
# #     lexicon_path = root / "data" / "lexicon.json"
# #
# #     docs_dir.mkdir(parents=True, exist_ok=True)
# #     index_dir.mkdir(parents=True, exist_ok=True)
# #     eval_dir.mkdir(parents=True, exist_ok=True)
# #
# #     return {
# #         "docs_dir": docs_dir,
# #         "index_dir": index_dir,
# #         "eval_dir": eval_dir,
# #         "lexicon_path": lexicon_path,
# #     }
# #
# # def save_uploaded_docs(docs_dir: Path, uploaded_files: List, overwrite: bool = True) -> Dict[str, Any]:
# #     saved = []
# #     skipped = []
# #     for uf in uploaded_files or []:
# #         name = Path(uf.name).name
# #         if not name.lower().endswith(".txt"):
# #             skipped.append({"file": name, "reason": "not .txt"})
# #             continue
# #         dest = docs_dir / name
# #         if dest.exists() and not overwrite:
# #             skipped.append({"file": name, "reason": "exists"})
# #             continue
# #         dest.write_bytes(uf.getbuffer())
# #         saved.append(str(dest))
# #     return {"saved": saved, "skipped": skipped}
# #
# # def build_all(root: Path, pack_dir: Path, run_eval: bool = True, eval_limit: int = 600) -> BuildResult:
# #     paths = ensure_dirs(root)
# #     docs_dir = paths["docs_dir"]
# #     index_dir = paths["index_dir"]
# #     eval_dir = paths["eval_dir"]
# #     lexicon_path = paths["lexicon_path"]
# #
# #     txts = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
# #     if not txts:
# #         return BuildResult(False, "No .txt docs found in ./docs. Upload docs first.", {})
# #
# #     domain = DomainPack.load(pack_dir)
# #
# #     stats = {}
# #     idx_stats = build_index(docs_dir, index_dir, lexicon_path, domain)
# #     stats["index"] = idx_stats
# #     stats["domain_pack"] = {"name": domain.manifest.get("name"), "version": domain.manifest.get("version"), "dir": str(pack_dir)}
# #
# #     if run_eval:
# #         q_path = eval_dir / "generated_questions.jsonl"
# #         q_stats = generate_questions(index_dir, q_path)  # can later use domain templates too
# #         stats["qgen"] = q_stats
# #
# #         res_path = eval_dir / "results.jsonl"
# #         eval_questions(index_dir, q_path, res_path, limit=eval_limit)  # later: pass domain
# #         stats["eval"] = {"results_path": str(res_path), "limit": eval_limit}
# #
# #     return BuildResult(True, "Build completed.", stats)
