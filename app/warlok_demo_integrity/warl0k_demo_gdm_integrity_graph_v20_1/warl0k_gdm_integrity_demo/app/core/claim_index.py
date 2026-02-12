import json
from pathlib import Path
from typing import Dict, Any, List

CLAIMS_DIR = Path(__file__).resolve().parents[1] / "data" / "claims"

def load_claim_packs() -> List[Dict[str, Any]]:
    packs_path = CLAIMS_DIR / "claim_packs.jsonl"
    packs = []
    if packs_path.exists():
        with open(packs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    packs.append(json.loads(line))
    return packs

def build_claim_index() -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for pack in load_claim_packs():
        if isinstance(pack, dict) and isinstance(pack.get("claims"), list):
            for c in pack["claims"]:
                cid = str(c.get("claim_id") or c.get("id") or "")
                if cid:
                    idx[cid] = c
        else:
            cid = str(pack.get("claim_id") or pack.get("id") or "")
            if cid:
                idx[cid] = pack
    return idx
