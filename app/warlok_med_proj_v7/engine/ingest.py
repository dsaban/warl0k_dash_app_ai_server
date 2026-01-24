from __future__ import annotations
import json, re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from .utils import split_sents, norm_space, uniq, tokens

DEFAULT_SEEDS = {
    "hormone": [
        "human placental lactogen","hpl","placental lactogen","placental hormones",
        "progesterone","estrogen","cortisol","prolactin","growth hormone",
        "tnf","tnf-alpha","il-6","adiponectin","leptin"
    ],
    "pathway": ["insulin signaling","gluconeogenesis","lipolysis","adipogenesis","oxidative stress","inflammation"],
    "tissue": ["placenta","pancreas","beta cell","pancreatic beta-cells","liver","skeletal muscle","adipose","fetus","foetus","fetal pancreas","foetal pancreas"],
    "condition": [
        "gestational diabetes mellitus","gdm","type 1 diabetes","type 2 diabetes","t2dm",
        "insulin resistance","hyperglycemia","hyperglycaemia","hypoglycemia","hypoglycaemia",
        "prediabetes","metabolic syndrome","obesity","overweight","hypertension","preeclampsia"
    ],
    "test": ["ogtt","oral glucose tolerance test","gct","glucose challenge test","fpg","fasting plasma glucose","hba1c","cgm","continuous glucose monitoring","smbg"],
    "guideline_org": ["ada","nice","who","iadpsg","idf","easd"],
    "intervention": ["diet","exercise","lifestyle","medical nutrition therapy","insulin","metformin","glyburide","glibenclamide"],
    "drug": ["insulin","metformin","glyburide","glibenclamide"],
    "outcome_maternal": ["preeclampsia","hypertension","type 2 diabetes","postpartum diabetes","cesarean","caesarean","c-section"],
    "outcome_fetal": ["macrosomia","large for gestational age","lga","shoulder dystocia","neonatal hypoglycemia","nicu","preterm birth","stillbirth"],
    "outcome_longterm": ["childhood obesity","offspring obesity","type 2 diabetes","cardiovascular disease"],
    "risk_factor": ["maternal age","age","bmi","family history","ethnicity","pcos","previous gdm","previous infant with macrosomia"],
}

def norm(t: str) -> str:
    return re.sub(r"\s+"," ", t.strip().lower())

THRESH_PAT = re.compile(
    r'(\b\d{1,2}\s*-\s*\d{1,2}\s*weeks\b|\b\d{1,2}\s*weeks\b|\b\d{2,3}\s*mg/dl\b|\b\d{1,2}\.\d\s*mmol\/l\b)',
    re.I
)

# Edge patterns return both: (hit?, strength 1..3)
EDGE_PATTERNS: Dict[str, List[Tuple[re.Pattern, int]]] = {
    "INDUCES_RESISTANCE": [
        (re.compile(r'(placental .*? hormones?|progesterone|estrogen|cortisol|placental lactogen).*?(induces?|causes?).*insulin resistance', re.I), 3),
        (re.compile(r'(placental .*? hormones?|progesterone|estrogen|cortisol|placental lactogen).*insulin resistance', re.I), 2),
    ],
    "REQUIRES_COMPENSATION": [
        (re.compile(r'insulin resistance.*?(requires?|necessitates?|demands?).*?(increase|compensatory).*insulin', re.I), 3),
        (re.compile(r'insulin resistance.*?(increase|compensatory).*insulin', re.I), 2),
    ],
    "FAILURE_LEADS_TO_GDM": [
        (re.compile(r'(beta\s*[- ]?cells?|pancreatic beta).*?(fail|cannot).*?(adapt|compensate).*?(gdm|gestational diabetes)', re.I), 3),
        (re.compile(r'(fail|cannot).*?(adapt|compensate).*?(gdm|gestational diabetes)', re.I), 2),
    ],
    "CROSSES_PLACENTA": [
        (re.compile(r'glucose.*?(freely|readily)\s+(crosses|cross).*placenta', re.I), 3),
        (re.compile(r'glucose.*?(cross(es)?|transfer(red)?|pass(es)?|diffus(es)?).*placenta', re.I), 2),
    ],
    "STIMULATES_INSULIN": [
        (re.compile(r'(fetal|foetal).*?(hyperinsulin|hyperinsulinaemia|hyperinsulinemia).*', re.I), 3),
        (re.compile(r'(fetal|foetal).*(pancreas|beta).*insulin', re.I), 2),
    ],
    "PROMOTES_GROWTH": [
        (re.compile(r'(fetal|foetal)\s+insulin.*?(anabolic|promotes|drives).*?(growth|adiposity|macrosomia|lga)', re.I), 3),
        (re.compile(r'(fetal|foetal)\s+insulin.*?(growth|adiposity|macrosomia|lga|large for gestational)', re.I), 2),
    ],
    "INCREASES_RISK": [
        (re.compile(r'(obesity|bmi|ethnicity|age|family history|pcos).*?(risk factor|independent risk|increases? risk|associated with).*?(gdm|gestational diabetes)', re.I), 3),
        (re.compile(r'(obesity|bmi|ethnicity|age|family history|pcos).*?(increases? risk|associated with)', re.I), 2),
    ],
    "VARIES_WITH_BACKGROUND_T2DM": [
        (re.compile(r'(gdm|gestational diabetes).*?(varies|in direct proportion).*?(type 2 diabetes|t2dm)', re.I), 3),
        (re.compile(r'(varies|in direct proportion).*?(type 2 diabetes|t2dm)', re.I), 2),
    ],
    "SCREEN_24_28_WEEKS": [
        (re.compile(r'(screening|screen).*?(24\s*-\s*28\s*weeks|24–28\s*weeks|between\s+24\s*-\s*28\s*weeks)', re.I), 3),
        (re.compile(r'(24\s*-\s*28\s*weeks|24–28\s*weeks)', re.I), 2),
    ],
    "POSTPARTUM_OGTT": [
        (re.compile(r'postpartum.*?(ogtt|oral glucose tolerance test).*?(4\s*-\s*12\s*weeks|6\s*weeks|6–12\s*weeks)', re.I), 3),
        (re.compile(r'postpartum.*?(ogtt|oral glucose tolerance test)', re.I), 2),
    ],
    "THRESHOLD_CRITERIA": [
        (re.compile(r'(diagnostic|criteria|threshold|cut[- ]?off).*?(ogtt|glucose).*?(mg/dl|mmol)', re.I), 3),
        (re.compile(r'(criteria|threshold|cut[- ]?off).*?(mg/dl|mmol)', re.I), 2),
    ],
    "FOLLOWUP_GAP": [
        (re.compile(r'(postpartum|after delivery).*?(follow[- ]?up|attendance|visit|testing).*?(low|poor|suboptimal|gap|missed|fails?)', re.I), 3),
        (re.compile(r'(follow[- ]?up|attendance|testing).*?(low|poor|gap|missed)', re.I), 2),
    ],
    "MISSED_DETECTION": [
        (re.compile(r'(missed|delayed).*?(diagnos|detect).*?(diabetes|dysglyc|hyperglyc)', re.I), 3),
        (re.compile(r'(missed|delayed).*?(diagnos|detect)', re.I), 2),
    ],
    "MONITORING_TARGETS": [
        (re.compile(r'(target|goal).*?(fasting|postprandial|preprandial).*?\b\d{2,3}\s*mg/dl\b', re.I), 3),
        (re.compile(r'(fasting|postprandial|preprandial).*?\b\d{2,3}\s*mg/dl\b', re.I), 2),
    ],
    "TREATED_BY": [
        (re.compile(r'(treated|managed|therapy|intervention|started on|initiated|prescribed|administered)'
                    r'.{0,120}?(diet|exercise|insulin|metformin|glyburide|glibenclamide)', re.I), 3),
        (re.compile(r'(treated|managed).{0,120}?(diet|exercise|insulin|metformin)', re.I), 2),
    ],
    "RESOLVES_AFTER_DELIVERY": [
        (re.compile(r'(gdm|gestational diabetes).{0,140}?(usually|often|typically).{0,40}?(resolves|resolve).{0,120}?(after delivery|postpartum|after giving birth)', re.I), 3),
        (re.compile(r'(resolves|resolve).{0,120}?(after delivery|postpartum|after giving birth)', re.I), 2),
    ],
    "POSTPARTUM_T2DM_RISK": [
        (re.compile(r'(gdm|gestational diabetes).{0,200}?(increases|elevated|higher).{0,80}?risk.{0,200}?(type 2 diabetes|t2dm)', re.I), 3),
        (re.compile(r'(history of gdm|after gdm).{0,200}?(risk).{0,200}?(type 2 diabetes|t2dm)', re.I), 2),
    ],
    "BETA_CELL_LIMITATION": [
        (re.compile(r'(beta[- ]?cell|beta cell).{0,120}?(reserve|limitation|limited|declin|deteriorat|failure).{0,260}?(over time|postpartum|chronic|long[- ]?term)', re.I), 3),
        (re.compile(r'(declin|deteriorat).{0,120}?(beta[- ]?cell|beta cell)', re.I), 2),
    ],
    "PERSISTENT_INSULIN_RESISTANCE": [
        (re.compile(r'(insulin resistance).{0,200}?(persists|persistent|underlying|baseline|chronic)', re.I), 3),
        (re.compile(r'underlying.{0,80}?insulin resistance', re.I), 2),
    ],

    # "TREATED_BY": [
    #     (re.compile(r'(managed|treated).*?(diet|exercise|insulin|metformin|glyburide|glibenclamide)', re.I), 3),
    #     (re.compile(r'(diet|exercise|insulin|metformin|glyburide|glibenclamide)', re.I), 1),
    # ],
    "IMPROVES_OUTCOMES": [
        (re.compile(r'(improves?|reduces?).*?(macrosomia|lga|preeclampsia|complications|outcomes)', re.I), 3),
        (re.compile(r'(improves?|reduces?).*?(complications|outcomes)', re.I), 2),
    ],
}

def load_lexicon(lexicon_path: Path) -> Dict[str, List[str]]:
    if not lexicon_path.exists():
        return {}
    obj = json.loads(lexicon_path.read_text(errors="ignore"))
    return {k: [norm(x) for x in v] for k, v in obj.get("seeds", {}).items()}

def save_lexicon(lexicon_path: Path, seeds: Dict[str, List[str]]):
    lexicon_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon_path.write_text(json.dumps({"seeds": seeds}, ensure_ascii=False, indent=2))

def extract_concepts(sent: str, seed_norm: Dict[str, set]) -> List[Tuple[str,str]]:
    found=[]
    low=sent.lower()
    for cat, terms in seed_norm.items():
        for t in terms:
            if t and t in low:
                found.append((cat,t))
    for m in THRESH_PAT.finditer(sent):
        found.append(("threshold_or_timing", m.group(0).strip()))
    return found

def extract_edges_with_strength(sent: str) -> Tuple[List[str], Dict[str,int]]:
    edges=[]
    evidence={}
    for et, pats in EDGE_PATTERNS.items():
        best=0
        for pat, strength in pats:
            if pat.search(sent):
                best = max(best, strength)
        if best>0:
            edges.append(et)
            evidence[et]=best
    return edges, evidence

def compute_confidence(sent: str, nodes_n: int, edge_evidence: Dict[str,int]) -> float:
    """
    0.0..1.0 (heuristic, stable)
    Boosts:
      - strong edge matches
      - more edges
      - presence of numeric thresholds/timing
      - more recognized nodes (but capped)
    Penalizes:
      - extremely short/long noisy sentences
      - very low node+edge signal
    """
    length = len(tokens(sent))
    # base
    score = 0.15

    # edges strength
    if edge_evidence:
        score += 0.12 * min(6, len(edge_evidence))  # count
        score += 0.08 * sum(edge_evidence.values())  # strength sum

    # nodes signal
    score += 0.03 * min(10, nodes_n)

    # threshold/timing
    if THRESH_PAT.search(sent):
        score += 0.10

    # length sanity
    if length < 6:
        score -= 0.10
    elif length > 60:
        score -= 0.08

    # clamp
    score = max(0.0, min(1.0, score))
    return round(score, 3)

def build_index(docs_dir: Path, index_dir: Path, lexicon_path: Path) -> Dict[str,int]:
    docs = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
    if not docs:
        raise FileNotFoundError(f"No .txt docs found in {docs_dir}")

    seeds = {k: [norm(x) for x in v] for k,v in DEFAULT_SEEDS.items()}
    user_lex = load_lexicon(lexicon_path)
    for k,v in user_lex.items():
        seeds.setdefault(k, [])
        seeds[k] = uniq(seeds[k] + v)

    seed_norm = {cat:set(map(norm,terms)) for cat,terms in seeds.items()}

    concept_index: Dict[str,Dict[str,Any]] = {}
    claims=[]
    def add_concept(cat: str, label: str) -> str:
        key=f"{cat}:{label}"
        if key not in concept_index:
            concept_index[key]={"id":key,"type":cat,"label":label,"aliases":[label]}
        return key

    cid=0
    for doc in docs:
        text = doc.read_text(errors="ignore")
        for sent in split_sents(text):
            s = sent.strip()
            low = s.lower()
            if low.startswith("keywords:"):
                continue
            if len(tokens(s)) < 6:
                continue
            
            concepts = extract_concepts(sent, seed_norm)
            edges, edge_evidence = extract_edges_with_strength(sent)
            if not concepts and not edges:
                continue
            nodes=[add_concept(c,t) for c,t in concepts]
            conf = compute_confidence(sent, len(nodes), edge_evidence)

            claims.append({
                "id": f"c{cid:07d}",
                "doc": doc.name,
                "text": norm_space(sent),
                "nodes": nodes,
                "edge_types": edges,
                "edge_evidence": edge_evidence,
                "confidence": conf,
            })
            cid += 1

    ontology = {
        "meta": {
            "name": "GDM ChainGraph v3",
            "source_docs": [d.name for d in docs],
            "node_count": len(concept_index),
            "claim_count": len(claims),
        },
        "node_types": sorted(list(seeds.keys()) + ["threshold_or_timing"]),
        "edge_types": sorted(list(EDGE_PATTERNS.keys())),
        "nodes": list(concept_index.values()),
        "seeds": seeds
    }

    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir/"ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2))
    (index_dir/"claims.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in claims))

    save_lexicon(lexicon_path, seeds)
    return {"docs": len(docs), "nodes": len(concept_index), "claims": len(claims)}

# from __future__ import annotations
# import json, re
# from pathlib import Path
# from typing import Dict, List, Tuple
# from .utils import split_sents, norm_space, uniq
#
# DEFAULT_SEEDS = {
#     "hormone": [
#         "human placental lactogen","hpl","placental lactogen","placental hormones",
#         "progesterone","estrogen","cortisol","prolactin","growth hormone",
#         "tnf","tnf-alpha","il-6","adiponectin","leptin"
#     ],
#     "pathway": ["insulin signaling","gluconeogenesis","lipolysis","adipogenesis","oxidative stress","inflammation"],
#     "tissue": ["placenta","pancreas","beta cell","pancreatic beta-cells","liver","skeletal muscle","adipose","fetus","foetus","fetal pancreas"],
#     "condition": [
#         "gestational diabetes mellitus","gdm","type 1 diabetes","type 2 diabetes","t2dm",
#         "insulin resistance","hyperglycemia","hyperglycaemia","prediabetes","metabolic syndrome","obesity"
#     ],
#     "test": ["ogtt","oral glucose tolerance test","gct","glucose challenge test","fpg","hba1c","cgm","smbg"],
#     "guideline_org": ["ada","nice","who","iadpsg"],
#     "intervention": ["diet","exercise","lifestyle","insulin","metformin","glyburide","glibenclamide"],
#     "drug": ["insulin","metformin","glyburide","glibenclamide"],
#     "outcome_maternal": ["preeclampsia","hypertension","type 2 diabetes","postpartum diabetes"],
#     "outcome_fetal": ["macrosomia","large for gestational age","lga","shoulder dystocia","neonatal hypoglycemia","nicu"],
#     "outcome_longterm": ["childhood obesity","type 2 diabetes","cardiovascular disease"],
#     "risk_factor": ["maternal age","bmi","family history","ethnicity","pcos","previous gdm"],
# }
#
# def norm(t: str) -> str:
#     return re.sub(r"\s+"," ", t.strip().lower())
#
# THRESH_PAT = re.compile(
#     r'(\b\d{1,2}\s*-\s*\d{1,2}\s*weeks\b|\b\d{1,2}\s*weeks\b|\b\d{2,3}\s*mg/dl\b|\b\d{1,2}\.\d\s*mmol\/l\b)',
#     re.I
# )
#
# EDGE_PATTERNS = {
#     "INDUCES_RESISTANCE": re.compile(
#         r'(placental .*? hormones?|progesterone|estrogen|cortisol|placental lactogen)[^\.]{0,220}?(induces?|causes?)\s+.*insulin resistance', re.I),
#     "REQUIRES_COMPENSATION": re.compile(
#         r'insulin resistance[^\.]{0,220}?(compensatory|increase|increased)[^\.]{0,120}?insulin', re.I),
#     "FAILURE_LEADS_TO_GDM": re.compile(
#         r'(beta\s*[- ]?cells?).{0,240}?(fail|cannot).{0,200}?(adapt|compensate).{0,200}?(gdm|gestational diabetes)', re.I),
#
#     "CROSSES_PLACENTA": re.compile(
#         r'glucose[^\.]{0,260}?(cross(es)?|transfer(red)?|pass(es)?|diffus(es)?)[^\.]{0,160}?placenta', re.I),
#     "STIMULATES_INSULIN": re.compile(
#         r'(fetal|foetal)[^\.]{0,260}?(hyperinsulin|hyperinsulinaemia|hyperinsulinemia|insulin\s+secretion|increased\s+insulin)', re.I),
#     "PROMOTES_GROWTH": re.compile(
#         r'(fetal|foetal)\s+insulin[^\.]{0,340}?(anabolic|adiposity|fat|macrosomia|lga|large\s+for\s+gestational|overgrowth)', re.I),
#
#     "INCREASES_RISK": re.compile(
#         r'(obesity|bmi|ethnicity|age|family history|pcos)[^\.]{0,260}?(risk factor|associated|increases?)', re.I),
#     "VARIES_WITH_BACKGROUND_T2DM": re.compile(
#         r'(gdm|gestational diabetes)[^\.]{0,260}?(varies|in direct proportion)[^\.]{0,260}?(type 2 diabetes|t2dm)', re.I),
#
#     "SCREEN_24_28_WEEKS": re.compile(r'(24\s*-\s*28\s*weeks|24–28\s*weeks)', re.I),
#     "POSTPARTUM_OGTT": re.compile(
#         r'postpartum[^\.]{0,320}?(ogtt|oral glucose tolerance test)[^\.]{0,220}?(4\s*-\s*12\s*weeks|6\s*weeks|6–12\s*weeks)', re.I),
#     "THRESHOLD_CRITERIA": re.compile(r'(criteria|threshold|cut[- ]?off)[^\.]{0,260}?(mg/dl|mmol)', re.I),
#
#     "FOLLOWUP_GAP": re.compile(
#         r'(postpartum|after delivery)[^\.]{0,280}?(follow[- ]?up|attendance|testing)[^\.]{0,280}?(low|poor|suboptimal|gap|missed|fails?)', re.I),
#     "MISSED_DETECTION": re.compile(
#         r'(missed|delayed)[^\.]{0,140}?(diagnos|detect)[^\.]{0,220}?(diabetes|dysglyc|hyperglyc)', re.I),
#
#     "MONITORING_TARGETS": re.compile(r'(fasting|postprandial|preprandial)[^\.]{0,140}?\b\d{2,3}\s*mg/dl\b', re.I),
#     "TREATED_BY": re.compile(r'(treated|managed)[^\.]{0,180}?(diet|exercise|insulin|metformin|glyburide|glibenclamide)', re.I),
#     "IMPROVES_OUTCOMES": re.compile(r'(improves?|reduces?)[^\.]{0,220}?(macrosomia|lga|preeclampsia|complications|outcomes)', re.I),
# }
#
# def load_lexicon(lexicon_path: Path) -> Dict[str, List[str]]:
#     if not lexicon_path.exists():
#         return {}
#     obj = json.loads(lexicon_path.read_text(errors="ignore"))
#     return {k: [norm(x) for x in v] for k, v in obj.get("seeds", {}).items()}
#
# def save_lexicon(lexicon_path: Path, seeds: Dict[str, List[str]]):
#     lexicon_path.parent.mkdir(parents=True, exist_ok=True)
#     lexicon_path.write_text(json.dumps({"seeds": seeds}, ensure_ascii=False, indent=2))
#
# def extract_concepts(sent: str, seed_norm: Dict[str, set]) -> List[Tuple[str,str]]:
#     found=[]
#     low=sent.lower()
#     for cat, terms in seed_norm.items():
#         for t in terms:
#             if t and t in low:
#                 found.append((cat,t))
#     for m in THRESH_PAT.finditer(sent):
#         found.append(("threshold_or_timing", m.group(0).strip()))
#     return found
#
# def extract_edges(sent: str) -> List[str]:
#     out=[]
#     for et, pat in EDGE_PATTERNS.items():
#         if pat.search(sent):
#             out.append(et)
#     return out
#
# def build_index(docs_dir: Path, index_dir: Path, lexicon_path: Path) -> Dict[str,int]:
#     docs = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
#     if not docs:
#         raise FileNotFoundError(f"No .txt docs found in {docs_dir}")
#
#     seeds = {k: [norm(x) for x in v] for k,v in DEFAULT_SEEDS.items()}
#     user_lex = load_lexicon(lexicon_path)
#     for k,v in user_lex.items():
#         seeds.setdefault(k, [])
#         seeds[k] = uniq(seeds[k] + v)
#
#     seed_norm = {cat:set(map(norm,terms)) for cat,terms in seeds.items()}
#
#     concept_index: Dict[str,Dict] = {}
#     claims=[]
#     def add_concept(cat: str, label: str) -> str:
#         key=f"{cat}:{label}"
#         if key not in concept_index:
#             concept_index[key]={"id":key,"type":cat,"label":label,"aliases":[label]}
#         return key
#
#     cid=0
#     for doc in docs:
#         text = doc.read_text(errors="ignore")
#         for sent in split_sents(text):
#             concepts = extract_concepts(sent, seed_norm)
#             edges = extract_edges(sent)
#             if not concepts and not edges:
#                 continue
#             nodes=[add_concept(c,t) for c,t in concepts]
#             claims.append({
#                 "id": f"c{cid:07d}",
#                 "doc": doc.name,
#                 "text": norm_space(sent),
#                 "nodes": nodes,
#                 "edge_types": edges
#             })
#             cid += 1
#
#     ontology = {
#         "meta": {
#             "name": "GDM ChainGraph v3",
#             "source_docs": [d.name for d in docs],
#             "node_count": len(concept_index),
#             "claim_count": len(claims),
#         },
#         "node_types": sorted(list(seeds.keys()) + ["threshold_or_timing"]),
#         "edge_types": sorted(list(EDGE_PATTERNS.keys())),
#         "nodes": list(concept_index.values()),
#         "seeds": seeds
#     }
#
#     index_dir.mkdir(parents=True, exist_ok=True)
#     (index_dir/"ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2))
#     (index_dir/"claims.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in claims))
#
#     save_lexicon(lexicon_path, seeds)
#     return {"docs": len(docs), "nodes": len(concept_index), "claims": len(claims)}
