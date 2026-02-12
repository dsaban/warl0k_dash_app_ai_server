from __future__ import annotations
import sys, subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    docs_dir = root / "docs"
    index_dir = root / "data" / "index"
    eval_dir = root / "data" / "eval"
    lexicon_path = root / "data" / "lexicon.json"

    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    txts = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
    if not txts:
        print("\n[STOP] No .txt docs found in ./docs\n")
        print("Put your docs here:\n  ./docs/<yourfile>.txt\n")
        sys.exit(1)

    from engine.ingest import build_index
    stats = build_index(docs_dir, index_dir, lexicon_path)
    print(f"[Index] docs={stats['docs']} nodes={stats['nodes']} claims={stats['claims']}")

    from engine.question_gen import generate_questions
    q_path = eval_dir / "generated_questions.jsonl"
    qstats = generate_questions(index_dir, q_path)
    print(f"[QGen] questions={qstats['questions']}")

    from engine.self_eval import eval_questions
    res_path = eval_dir / "results.jsonl"
    eval_questions(index_dir, q_path, res_path, limit=600)
    print("[Eval] done")

    # Safe auto-improve: expand lexicon from failure-question vocabulary, rebuild twice
    from engine.auto_improve import suggest_terms_from_failures, apply_lexicon_updates
    for i in range(2):
        sugg = suggest_terms_from_failures(res_path, top_n=30)
        upd = apply_lexicon_updates(lexicon_path, sugg)
        if upd["added_terms"] <= 0:
            break
        stats = build_index(docs_dir, index_dir, lexicon_path)
        eval_questions(index_dir, q_path, res_path, limit=600)
        print(f"[AutoImprove] iter={i+1} added_terms={upd['added_terms']}")

    print("\nLaunching UI...\n")
    cmd = [sys.executable, "-m", "streamlit", "run", str(root / "app1.py")]
    raise SystemExit(subprocess.call(cmd))

if __name__ == "__main__":
    main()
