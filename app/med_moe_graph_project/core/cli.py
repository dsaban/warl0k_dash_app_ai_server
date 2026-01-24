from __future__ import annotations

import argparse
from pathlib import Path

from core.ingest import ingest_docs
from core.ontology import Ontology


def cmd_ingest(args: argparse.Namespace) -> int:
    ontology = Ontology.load(args.ontology)
    g, _ = ingest_docs(
        docs_dir=args.docs,
        ontology=ontology,
        entity_lexicon_path=args.entities,
    )
    g.save(args.out)
    print(f"Wrote graph: {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("med_moe_graph")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest docs into graph")
    ing.add_argument("--docs", type=str, default="data/docs")
    ing.add_argument("--ontology", type=str, default="config/ontology.json")
    ing.add_argument("--entities", type=str, default="data/lexicon/entities.json")
    ing.add_argument("--out", type=str, default="data/graph/graph.json")
    ing.set_defaults(func=cmd_ingest)
    return p


def main() -> int:
    p = build_parser()
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
