from __future__ import annotations
import json
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from .graph import Graph

EDGE_TEMPLATES: Dict[str, List[str]] = {
    "CROSSES_PLACENTA": [
        "Explain how maternal glucose exposure reaches the fetus via placental transfer.",
        "Why does maternal hyperglycemia lead to fetal hyperglycemia (placental glucose transfer)?",
        "Describe the maternal-to-fetal glucose transfer step in the modified Pedersen hypothesis."
    ],
    "STIMULATES_INSULIN": [
        "Why does fetal hyperglycemia trigger fetal hyperinsulinemia?",
        "Explain the fetal pancreatic insulin response to increased glucose exposure.",
        "Connect fetal glucose exposure to fetal insulin secretion in GDM."
    ],
    "PROMOTES_GROWTH": [
        "How does fetal insulin promote macrosomia/overgrowth?",
        "Explain why fetal hyperinsulinemia is anabolic and drives fetal adiposity.",
        "Connect fetal insulin to macrosomia in the modified Pedersen hypothesis."
    ],
    "INDUCES_RESISTANCE": [
        "How do placental hormones induce insulin resistance during pregnancy?",
        "Explain why insulin resistance rises in late pregnancy (hormonal mechanisms)."
    ],
    "REQUIRES_COMPENSATION": [
        "Why must beta cells increase insulin secretion during pregnancy?",
        "Explain how insulin resistance increases insulin demand in pregnancy."
    ],
    "FAILURE_LEADS_TO_GDM": [
        "Explain how inadequate beta-cell adaptation leads to GDM.",
        "Why does beta-cell compensation failure produce gestational diabetes?"
    ],
    "INCREASES_RISK": [
        "Explain how obesity increases GDM risk (mechanistic linkage).",
        "Why do age/BMI/family history increase the incidence of GDM?"
    ],
    "VARIES_WITH_BACKGROUND_T2DM": [
        "Why does GDM prevalence vary across populations (background T2DM linkage)?",
        "Explain the relationship between background T2DM rates and GDM prevalence."
    ],
    "SCREEN_24_28_WEEKS": [
        "Why is routine GDM screening commonly performed at 24–28 weeks?",
        "Explain the rationale for the 24–28 week screening window."
    ],
    "POSTPARTUM_OGTT": [
        "Why is postpartum OGTT recommended after GDM, and when should it be done?",
        "Explain how postpartum OGTT detects persistent dysglycemia after GDM."
    ],
    "THRESHOLD_CRITERIA": [
        "How do diagnostic thresholds influence GDM prevalence and management?",
        "Explain why different criteria/thresholds produce different GDM rates."
    ],
    "FOLLOWUP_GAP": [
        "How does postpartum follow-up failure represent a systemic gap (not only noncompliance)?",
        "Explain system-level reasons postpartum testing after GDM is missed."
    ],
    "MISSED_DETECTION": [
        "How can missed postpartum testing delay detection of persistent diabetes after GDM?",
        "Explain how gaps in follow-up cause missed detection of postpartum dysglycemia."
    ],
    "MONITORING_TARGETS": [
        "What glucose targets are used in GDM monitoring and why do they matter?",
        "Explain how fasting and postprandial targets guide therapy in GDM."
    ],
    "TREATED_BY": [
        "Describe the stepwise management of GDM (diet/exercise → medication).",
        "When is pharmacotherapy indicated in GDM and what options are used?"
    ],
    "IMPROVES_OUTCOMES": [
        "How does treating hyperglycemia reduce adverse maternal/fetal outcomes?",
        "Explain how glycemic control improves pregnancy outcomes in GDM."
    ],
}

NODE_TEMPLATES: Dict[str, List[str]] = {
    "risk_factor": [
        "Explain how {x} increases the risk of gestational diabetes mellitus (GDM).",
        "Why is {x} considered a risk factor for GDM?"
    ],
    "outcome_fetal": [
        "Explain how GDM contributes to {x} and the causal steps involved.",
        "What mechanisms link maternal dysglycemia to {x}?"
    ],
    "outcome_maternal": [
        "How is GDM associated with maternal outcome: {x}?",
        "Explain mechanistic or clinical links between GDM and {x}."
    ],
    "test": [
        "What is the role of {x} in screening/diagnosing/monitoring GDM?",
        "Explain when and why {x} is used in GDM care."
    ],
    "hormone": [
        "Explain how {x} contributes to insulin resistance in pregnancy.",
        "What role does {x} play in pregnancy metabolism relevant to GDM?"
    ],
    "intervention": [
        "Explain how {x} is used to manage GDM and what outcomes it targets.",
        "When is {x} recommended in GDM management?"
    ],
    "threshold_or_timing": [
        "What is the significance of the threshold/timing '{x}' in GDM diagnosis or management?",
        "Explain how '{x}' is used in guideline timing/targets for GDM."
    ],
}

CHAIN_2HOP_TEMPLATES: List[str] = [
    "Explain the causal chain: {e1} → {e2}, and why this progression matters clinically.",
    "Build a stepwise explanation connecting {e1} to {e2} in GDM physiology/pathophysiology.",
    "Using a mechanistic chain, connect {e1} and {e2} with intermediate steps."
]

CORE_QUESTIONS = [
    "Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia.",
    "Explain how normal pregnancy physiology progresses to GDM when beta-cell compensation fails.",
    "How does failure to address postpartum follow-up represent a systemic gap in GDM care rather than individual noncompliance?",
    "Compare guideline approaches to screening and diagnostic criteria for GDM.",
    "How do monitoring targets and interventions reduce GDM-related complications?",
]

def _dedupe(items: List[Dict]) -> List[Dict]:
    seen=set(); out=[]
    for it in items:
        k = it["q"].strip().lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out

def generate_questions(index_dir: Path, out_path: Path,
                       max_total: int = 6000,
                       max_per_edge: int = 250,
                       max_per_node_type: int = 900,
                       max_2hop: int = 1200) -> Dict[str,int]:
    g = Graph.load(index_dir)

    # Stats we’ll use to prioritize what the docs actually cover
    edge_counts = Counter()
    node_counts_by_type = defaultdict(Counter)
    threshold_samples = Counter()

    # co-occurrence graph for 2-hop: edge→nodes and node→edges
    edge_to_nodes = defaultdict(Counter)
    node_to_edges = defaultdict(Counter)

    for c in g.claims:
        for e in c.edge_types:
            edge_counts[e] += 1
        for n in c.nodes:
            t = g.node_type(n)
            label = n.split(":", 1)[1] if ":" in n else n
            node_counts_by_type[t][label] += 1
            if t == "threshold_or_timing":
                threshold_samples[label] += 1
        # co-occurrence
        for e in c.edge_types:
            for n in c.nodes:
                edge_to_nodes[e][n] += 1
                node_to_edges[n][e] += 1

    questions: List[Dict] = []

    # Core anchors
    for q in CORE_QUESTIONS:
        questions.append({"q": q, "source": "core"})

    # Edge-driven: for edges present in docs, generate many variants
    for e, tmpls in EDGE_TEMPLATES.items():
        if edge_counts.get(e, 0) == 0:
            continue
        # allocate based on coverage, capped
        budget = min(max_per_edge, 40 + 10 * edge_counts[e])
        for i in range(budget):
            t = tmpls[i % len(tmpls)]
            # light perturbation with “exam-style” wrappers
            style = i % 5
            if style == 1:
                q = f"In mechanistic steps, {t[0].lower()}{t[1:]}"
            elif style == 2:
                q = f"{t} Include key intermediates and directionality."
            elif style == 3:
                q = f"{t} Use a concise 4–6 step causal chain."
            else:
                q = t
            questions.append({"q": q, "edge": e, "source": "edge"})

    # Node-driven: generate by common nodes in ontology (risk factors, outcomes, tests, etc.)
    for node_type, tmpl_list in NODE_TEMPLATES.items():
        # pick top-K labels from docs
        labels = [lbl for lbl, _ in node_counts_by_type.get(node_type, {}).most_common(200)]
        budget = min(max_per_node_type, len(labels) * len(tmpl_list))
        j = 0
        for lbl in labels:
            for t in tmpl_list:
                questions.append({"q": t.format(x=lbl), "node_type": node_type, "source": "node"})
                j += 1
                if j >= budget:
                    break
            if j >= budget:
                break

    # Threshold/timing extra: generate more variants for frequent thresholds
    thr_labels = [lbl for lbl, _ in threshold_samples.most_common(200)]
    for lbl in thr_labels:
        for t in NODE_TEMPLATES["threshold_or_timing"]:
            questions.append({"q": t.format(x=lbl), "node_type": "threshold_or_timing", "source": "threshold"})

    # 2-hop chain questions: find frequent edge pairs connected by shared nodes
    # Build candidate pairs (e1,e2) if they share nodes
    pair_scores = Counter()
    edges_present = list(edge_counts.keys())
    for e1 in edges_present:
        nodes1 = edge_to_nodes.get(e1, {})
        if not nodes1:
            continue
        for n, w in nodes1.most_common(40):  # top shared nodes only
            for e2, w2 in node_to_edges[n].most_common(20):
                if e1 == e2:
                    continue
                pair_scores[(e1, e2)] += min(w, w2)

    for (e1, e2), _score in pair_scores.most_common(max_2hop):
        for t in CHAIN_2HOP_TEMPLATES:
            questions.append({"q": t.format(e1=e1, e2=e2), "source": "2hop", "edges": [e1, e2]})

    # Dedupe & hard cap
    questions = _dedupe(questions)
    if len(questions) > max_total:
        questions = questions[:max_total]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in questions))

    return {
        "questions": len(questions),
        "edges_present": len([e for e in EDGE_TEMPLATES.keys() if edge_counts.get(e,0)>0]),
        "unique_nodes_seen": sum(len(c) for c in node_counts_by_type.values()),
    }

# import json
# from pathlib import Path
#
# CORE_QUESTIONS = [
#     "Using the modified Pedersen hypothesis, explain how maternal hyperglycemia leads to fetal macrosomia.",
#     "How does failure of beta-cell compensation lead to gestational diabetes mellitus?",
#     "How does postpartum follow-up failure represent a systemic gap in GDM care?",
#     "Compare diagnostic criteria for GDM across major guidelines.",
#     "Why does obesity increase the risk of gestational diabetes?"
# ]
#
# def generate_questions(index_dir: Path, out_path: Path):
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with out_path.open("w") as f:
#         for q in CORE_QUESTIONS:
#             f.write(json.dumps({"q": q}) + "\n")
#     return {"questions": len(CORE_QUESTIONS)}
