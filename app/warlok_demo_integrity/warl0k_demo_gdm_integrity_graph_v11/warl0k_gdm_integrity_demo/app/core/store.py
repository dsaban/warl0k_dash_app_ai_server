import os, csv, json
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataStore:
    base_dir: str
    passages: List[Dict[str, Any]]
    claims: Dict[str, Dict[str, Any]]
    claim_packs: List[Dict[str, Any]]
    lexicon: Any
    questions: List[Dict[str, Any]]
    patients: List[Dict[str, Any]]
    state_checks: List[Dict[str, Any]]

    @staticmethod
    def load_default():
        here = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(here, ".."))
        data_dir = os.path.join(base_dir, "data")

        passages: List[Dict[str, Any]] = []
        with open(os.path.join(data_dir, "passages.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))

        claims: Dict[str, Dict[str, Any]] = {}
        claims_path = os.path.join(data_dir, "claims", "claims_seed.jsonl")
        with open(claims_path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                claims[c["claim_id"]] = c

        # claim packs (topic bundles)
        claim_packs: List[Dict[str, Any]] = []
        packs_path = os.path.join(data_dir, "claims", "claim_packs.jsonl")
        if os.path.exists(packs_path):
            with open(packs_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        claim_packs.append(json.loads(line))

        # lexicon index
        lexicon = None
        try:
            from core.lexicon import LexiconIndex
            lex_path = os.path.join(data_dir, "lexicon", "lexicon.csv")
            lexicon = LexiconIndex.load(lex_path)
        except Exception:
            lexicon = None

        questions: List[Dict[str, Any]] = []
        qpath = os.path.join(data_dir, "gold", "questions.csv")
        with open(qpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)

        patients: List[Dict[str, Any]] = []
        with open(os.path.join(data_dir, "patients", "patients_demo.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                patients.append(json.loads(line))

        state_checks: List[Dict[str, Any]] = []
        with open(os.path.join(data_dir, "state_checks", "state_checks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                state_checks.append(json.loads(line))

        return DataStore(
            base_dir=base_dir,
            passages=passages,
            claims=claims,
            claim_packs=claim_packs,
            lexicon=lexicon,
            questions=questions,
            patients=patients,
            state_checks=state_checks,
        )
