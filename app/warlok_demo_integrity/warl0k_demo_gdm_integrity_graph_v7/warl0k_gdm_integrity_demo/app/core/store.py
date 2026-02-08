import os, csv, json
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class DataStore:
    base_dir: str
    passages: List[Dict[str, Any]]
    claims: Dict[str, Dict[str, Any]]
    questions: List[Dict[str, Any]]
    patients: List[Dict[str, Any]]
    state_checks: List[Dict[str, Any]]

    @staticmethod
    def load_default():
        here = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.abspath(os.path.join(here, ".."))
        data_dir = os.path.join(base_dir, "data")

        passages = []
        with open(os.path.join(data_dir, "passages.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                passages.append(json.loads(line))

        claims = {}
        claims_path = os.path.join(data_dir, "claims", "claims_seed.jsonl")
        with open(claims_path, "r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                claims[c["claim_id"]] = c

        questions = []
        qpath = os.path.join(data_dir, "gold", "questions.csv")
        with open(qpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                questions.append(row)

        patients = []
        with open(os.path.join(data_dir, "patients", "patients_demo.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                patients.append(json.loads(line))

        state_checks = []
        with open(os.path.join(data_dir, "state_checks", "state_checks.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                state_checks.append(json.loads(line))

        return DataStore(
            base_dir=base_dir,
            passages=passages,
            claims=claims,
            questions=questions,
            patients=patients,
            state_checks=state_checks,
        )
