from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from .utils import norm_space, tokens
from .graph import Claim
from .domain_pack import DomainPack


_ABBR = {
    "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "fig.", "ref.",
    "no.", "approx.", "al.", "et al.", "cf."
}

_LISTY_RE = re.compile(r"(\s{2,}|\t|•|\u2022| - | \| )")


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    t = text.replace("\u00a0", " ").replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", " ")
    t = norm_space(t)
    if not t:
        return []

    protected = t
    for ab in _ABBR:
        protected = protected.replace(ab, ab.replace(".", "<DOT>"))
    protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)

    parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\[])", protected)

    out = []
    for p in parts:
        p = p.replace("<DOT>", ".")
        p = norm_space(p)
        if p:
            out.append(p)
    return out


def _load_list(junk: Dict[str, Any], key: str) -> List[str]:
    v = (junk or {}).get(key)
    if not v:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str):
        return [v]
    return []


def _clean_sentence(s: str, junk: Dict[str, Any]) -> str:
    s = s or ""
    s = norm_space(s)

    # Drop common UI artifacts if present (pack can also add drop_contains)
    drop_contains = [x.lower() for x in _load_list(junk, "drop_contains")]
    for dc in drop_contains:
        if dc and dc in s.lower():
            s = re.sub(re.escape(dc), " ", s, flags=re.IGNORECASE)
            s = norm_space(s)

    # Strip leading junk until any causal marker appears (pack-driven)
    strip_markers = [x.lower() for x in _load_list(junk, "strip_leading_until_any")]
    if strip_markers:
        orig = s
        low = s.lower()
        hits = [(low.find(m), m) for m in strip_markers if low.find(m) != -1]
        if hits:
            pos = min(h[0] for h in hits)
            if pos > 15:
                candidate = norm_space(s[pos:])
                # accept only if it looks like a clean start
                if candidate and (candidate[0].isupper() or candidate.lower().startswith(("fetal", "foetal", "maternal", "glucose"))):
                    s = candidate
                else:
                    s = orig

    # If it's still listy and long, keep last segment (generic salvage)
    if _LISTY_RE.search(s) and len(s.split()) > int((junk or {}).get("listy_salvage_min_tokens", 25)):
        for sep in ["|", "\t", "  "]:
            if sep in s:
                s = norm_space(s.split(sep)[-1])

    return s


def _is_junk_sentence(s: str, junk: Dict[str, Any]) -> bool:
    low = (s or "").strip().lower()
    if not low:
        return True

    skip_prefixes = [x.lower() for x in _load_list(junk, "skip_prefixes")]
    if any(low.startswith(pref) for pref in skip_prefixes):
        return True

    drop_contains = [x.lower() for x in _load_list(junk, "drop_contains")]
    if any(dc in low for dc in drop_contains):
        return True

    # listy header dump
    listy_drop_min = int((junk or {}).get("listy_drop_min_tokens", 18))
    if _LISTY_RE.search(s) and len(s.split()) >= listy_drop_min:
        keep_if = [x.lower() for x in _load_list(junk, "keep_if_contains")]
        if keep_if and any(k in low for k in keep_if):
            return False
        return True

    return False


def _claim_confidence(edge_ev: Dict[str, int], sent_tokens: int, score_cfg: Dict[str, Any]) -> float:
    import math

    base = float(score_cfg.get("base", 0.18))
    w_strength = float(score_cfg.get("w_strength", 0.16))
    w_edges = float(score_cfg.get("w_edges", 0.12))
    w_len = float(score_cfg.get("w_len", 0.07))

    ideal_min = int(score_cfg.get("ideal_len_min", 12))
    ideal_max = int(score_cfg.get("ideal_len_max", 40))

    if not edge_ev:
        return 0.01

    best = max(edge_ev.values())
    n_edges = len(edge_ev)

    if sent_tokens < ideal_min:
        len_q = sent_tokens / max(1, ideal_min)
    elif sent_tokens > ideal_max:
        len_q = max(0.0, 1.0 - (sent_tokens - ideal_max) / max(1, ideal_max))
    else:
        len_q = 1.0

    raw = base + w_strength * best + w_edges * n_edges + w_len * len_q
    conf = 1.0 / (1.0 + math.exp(-2.6 * (raw - 0.5)))
    return float(max(0.01, min(0.99, conf)))


def _extract_nodes_pack_driven(text: str, domain: DomainPack) -> List[Dict[str, Any]]:
    """
    Generic node extraction executed from pack's ontology_spec.json.
    Returns list of {"type": node_type, "label": label, "strength": int}
    """
    spec = getattr(domain, "ontology_extractors", None) or []
    low = (text or "").lower()

    hits: List[Dict[str, Any]] = []
    for ex in spec:
        ntype = ex.get("node_type")
        if not ntype:
            continue

        # substring matches
        for term in ex.get("match", []) or []:
            tl = str(term).lower()
            if tl and tl in low:
                hits.append({"type": ntype, "label": str(term), "strength": 2})

        # regex matches
        for rp in ex.get("regex", []) or []:
            if rp.regex.search(text):
                # if regex extractor has a label, use it; else use pattern string
                label = ex.get("label") or ex.get("name") or ntype
                hits.append({"type": ntype, "label": str(label), "strength": int(getattr(rp, "strength", 2))})

    # de-dupe stable
    seen = set()
    uniq = []
    for h in hits:
        key = (h["type"], h["label"].lower())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)
    return uniq


def build_index(docs_dir: Path, index_dir: Path, domain: DomainPack) -> Dict[str, Any]:
    index_dir.mkdir(parents=True, exist_ok=True)

    junk = domain.junk_filters or {}
    min_token_len = int(junk.get("min_token_len", 7))
    max_token_len = int(junk.get("max_token_len", 80))

    scoring = domain.scoring or {}
    score_cfg = scoring.get(
        "claim_confidence",
        {
            "base": 0.18,
            "w_strength": 0.16,
            "w_edges": 0.12,
            "w_len": 0.07,
            "ideal_len_min": 12,
            "ideal_len_max": 40,
        },
    )

    claims: List[Claim] = []
    cid = 0

    # Ontology nodes: pack-driven only
    node_id = 0
    node_index: Dict[Tuple[str, str], str] = {}
    nodes: List[Dict[str, Any]] = []

    def add_node(ntype: str, label: str) -> str:
        nonlocal node_id
        key = (ntype, label.lower())
        if key in node_index:
            return node_index[key]
        node_id += 1
        nid = f"n{node_id:06d}"
        node_index[key] = nid
        nodes.append({"id": nid, "type": ntype, "label": label})
        return nid

    for doc_path in sorted(docs_dir.glob("*.txt")):
        doc_name = doc_path.name
        raw = doc_path.read_text(errors="ignore")
        sents = split_sentences(raw)

        for s in sents:
            s = _clean_sentence(s, junk)
            if not s:
                continue
            if _is_junk_sentence(s, junk):
                continue

            tok_n = len(tokens(s))
            if tok_n < min_token_len or tok_n > max_token_len:
                continue

            edge_types, edge_ev = domain.extract_edges_with_strength(s)
            inf_edges, inf_ev = domain.infer_edges_with_strength(s)
            all_edges = list(dict.fromkeys((edge_types or []) + (inf_edges or [])))
            if not all_edges:
                continue

            # Merge evidence (max per edge)
            merged_ev: Dict[str, int] = {}
            for e, v in (edge_ev or {}).items():
                merged_ev[e] = max(merged_ev.get(e, 0), int(v))
            for e, v in (inf_ev or {}).items():
                merged_ev[e] = max(merged_ev.get(e, 0), int(v))

            conf = _claim_confidence(merged_ev, tok_n, score_cfg)

            # Pack-driven nodes
            node_ids: List[str] = []
            for h in _extract_nodes_pack_driven(s, domain):
                node_ids.append(add_node(h["type"], h["label"]))

            cid += 1
            claims.append(
                Claim(
                    id=f"c{cid:07d}",
                    doc=doc_name,
                    text=s if s.endswith((".", "!", "?")) else (s + "."),
                    edge_types=all_edges,
                    confidence=conf,
                    edge_evidence=merged_ev,
                    node_ids=node_ids if node_ids else None,
                )
            )

    # Write claims.jsonl
    claims_path = index_dir / "claims.jsonl"
    with claims_path.open("w", encoding="utf-8") as f:
        for c in claims:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")

    # Write ontology.json
    ontology = {
        "nodes": nodes,
        "edges": [],
        "meta": {
            "docs": [p.name for p in docs_dir.glob("*.txt")],
            "claim_count": len(claims),
            "node_count": len(nodes),
        },
    }
    (index_dir / "ontology.json").write_text(
        json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "claims": len(claims),
        "docs": len(list(docs_dir.glob("*.txt"))),
        "nodes": len(nodes),
        "index_dir": str(index_dir),
        "claims_path": str(claims_path),
    }

# # engine/ingest.py
# from __future__ import annotations
#
# import json
# import re
# from pathlib import Path
# from typing import Dict, Any, List, Tuple
#
# from .utils import norm_space, tokens
# from .graph import Claim
# from .domain_pack import DomainPack
#
#
# _ABBR = {
#     "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "fig.", "ref.",
#     "no.", "approx.", "al.", "et al.", "cf."
# }
#
# _TABLE_JUNK_MARKERS = [
#     "open in a new tab",
#     "open in new tab",
#     "table",
#     "figure",
#     "fig.",
#     "doi:",
# ]
#
# # Heuristic: if a sentence contains many list-like tokens, it's probably a table/header dump.
# _LISTY_RE = re.compile(r"(\s{2,}|\t|•|\u2022| - | \| )")
#
#
# def split_sentences(text: str) -> List[str]:
#     """
#     Lightweight sentence splitter:
#     - preserves common abbreviations
#     - avoids splitting decimals (e.g., 3.5)
#     - splits on [.?!] followed by whitespace and a likely sentence start
#     """
#     if not text:
#         return []
#
#     t = text.replace("\u00a0", " ")
#     t = t.replace("\r\n", "\n").replace("\r", "\n")
#
#     # Keep newlines as spaces for splitting; later cleaning will handle remnants.
#     t = t.replace("\n", " ")
#     t = norm_space(t)
#     if not t:
#         return []
#
#     protected = t
#     for ab in _ABBR:
#         protected = protected.replace(ab, ab.replace(".", "<DOT>"))
#     protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
#
#     parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\[])", protected)
#
#     out = []
#     for p in parts:
#         p = p.replace("<DOT>", ".")
#         p = norm_space(p)
#         if p:
#             out.append(p)
#     return out
#
#
# def _load_list(junk: Dict[str, Any], key: str) -> List[str]:
#     v = junk.get(key)
#     if not v:
#         return []
#     if isinstance(v, list):
#         return [str(x) for x in v if str(x).strip()]
#     if isinstance(v, str):
#         return [v]
#     return []
#
#
# def _clean_sentence(s: str, junk: Dict[str, Any]) -> str:
#     """
#     Aggressive cleaning to avoid table/header leakage into evidence.
#     - removes known markers like 'Open in a new tab'
#     - strips bracketed UI remnants
#     - collapses whitespace
#     """
#     s = s or ""
#     low = s.lower()
#
#     # Remove common "UI" / HTML-ish fragments
#     for marker in _TABLE_JUNK_MARKERS:
#         if marker in low:
#             # remove marker occurrences case-insensitively
#             s = re.sub(re.escape(marker), " ", s, flags=re.IGNORECASE)
#
#     # Remove repeated "Open in a new tab" style artifacts even if merged
#     s = re.sub(r"open\s+in\s+(a\s+)?new\s+tab", " ", s, flags=re.IGNORECASE)
#
#     # Remove stray bracket artifacts like [36] ONLY if they're dense lists; keep normal citations.
#     # Here: just normalize multiple bracket blocks.
#     s = re.sub(r"\s*\[\s*\]\s*", " ", s)
#
#     # Collapse whitespace
#     s = norm_space(s)
#
#     # If configured, strip leading junk until a causal marker appears (pack-driven)
#     strip_markers = [x.lower() for x in (junk.get("strip_leading_until_any") or [])]
#     if strip_markers:
#         low2 = s.lower()
#         hits = [(low2.find(m), m) for m in strip_markers if low2.find(m) != -1]
#         if hits:
#             pos = min(h[0] for h in hits)
#             # only strip if we remove a meaningful prefix
#             if pos > 15:
#                 s = norm_space(s[pos:])
#
#     # If sentence starts with a colon-terminated header ("Neonatal complications ..."), try to keep tail after it.
#     # e.g. "Neonatal complications ... Open ... Foetal hyperinsulinaemia may ..."
#     if ":" in s and len(s.split()) > 20:
#         head, tail = s.split(":", 1)
#         # keep tail if it looks like a real sentence
#         if len(tail.split()) >= 8:
#             s = norm_space(tail)
#
#     # If it still looks like a list dump, keep only the last clause after a long run of tokens.
#     # This salvages the actual causal sentence often appearing at the end.
#     if _LISTY_RE.search(s) and len(s.split()) > 25:
#         # Take last segment after common separators
#         for sep in ["  ", "\t", "|"]:
#             if sep in s:
#                 s = norm_space(s.split(sep)[-1])
#         # Also try after "tab" residues
#         if "tab" in s.lower():
#             s = norm_space(re.split(r"tab", s, flags=re.IGNORECASE)[-1])
#
#     return s
#
#
# def _is_junk_sentence(s: str, junk: Dict[str, Any]) -> bool:
#     low = (s or "").strip().lower()
#
#     skip_prefixes = [x.lower() for x in _load_list(junk, "skip_prefixes")]
#     if any(low.startswith(pref) for pref in skip_prefixes):
#         return True
#
#     drop_contains = [x.lower() for x in _load_list(junk, "drop_contains")]
#     if any(dc in low for dc in drop_contains):
#         return True
#
#     if low.startswith("keywords:"):
#         return True
#
#     # Drop very list-like "header dumps"
#     if _LISTY_RE.search(s) and len(s.split()) >= int(junk.get("listy_drop_min_tokens", 18)):
#         # Allow if it contains a strong causal keyword likely to be a real sentence
#         keep_if = [x.lower() for x in _load_list(junk, "keep_if_contains")]
#         if keep_if and any(k in low for k in keep_if):
#             return False
#         return True
#
#     return False
#
#
# def _claim_confidence(edge_ev: Dict[str, int], sent_tokens: int, score_cfg: Dict[str, Any]) -> float:
#     """
#     Confidence scoring:
#     - strongest pattern strength
#     - number of distinct edges
#     - length quality
#     """
#     import math
#
#     base = float(score_cfg.get("base", 0.18))
#     w_strength = float(score_cfg.get("w_strength", 0.16))
#     w_edges = float(score_cfg.get("w_edges", 0.12))
#     w_len = float(score_cfg.get("w_len", 0.07))
#
#     ideal_min = int(score_cfg.get("ideal_len_min", 12))
#     ideal_max = int(score_cfg.get("ideal_len_max", 40))
#
#     if not edge_ev:
#         return 0.01
#
#     best = max(edge_ev.values())
#     n_edges = len(edge_ev)
#
#     if sent_tokens < ideal_min:
#         len_q = sent_tokens / max(1, ideal_min)
#     elif sent_tokens > ideal_max:
#         len_q = max(0.0, 1.0 - (sent_tokens - ideal_max) / max(1, ideal_max))
#     else:
#         len_q = 1.0
#
#     raw = base + w_strength * best + w_edges * n_edges + w_len * len_q
#
#     # Less steep than before to avoid all ~0.85 saturation
#     conf = 1.0 / (1.0 + math.exp(-2.6 * (raw - 0.5)))
#     return float(max(0.01, min(0.99, conf)))
#
#
# def _load_entities(domain: DomainPack) -> Dict[str, List[str]]:
#     entities = getattr(domain, "entities", None)
#     if not isinstance(entities, dict):
#         return {}
#     out: Dict[str, List[str]] = {}
#     for k, v in entities.items():
#         if isinstance(v, list):
#             out[k] = [str(x) for x in v if str(x).strip()]
#     return out
#
#
# def _extract_entity_mentions(text: str, entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
#     low = (text or "").lower()
#     hits: List[Dict[str, str]] = []
#     for ntype, vocab in entities.items():
#         for term in vocab:
#             tl = term.lower()
#             if tl and tl in low:
#                 hits.append({"type": ntype, "label": term})
#     seen = set()
#     uniq = []
#     for h in hits:
#         key = (h["type"], h["label"].lower())
#         if key in seen:
#             continue
#         seen.add(key)
#         uniq.append(h)
#     return uniq
#
#
# def build_index(docs_dir: Path, index_dir: Path, domain: DomainPack) -> Dict[str, Any]:
#     """
#     Builds:
#       - claims.jsonl: sentences -> typed edges (+ inferred edges) + confidence
#       - ontology.json: entity nodes (pack-driven) collected from mentions
#     """
#     index_dir.mkdir(parents=True, exist_ok=True)
#
#     junk = domain.junk_filters or {}
#     min_token_len = int(junk.get("min_token_len", 7))
#     max_token_len = int(junk.get("max_token_len", 80))
#
#     scoring = domain.scoring or {}
#     score_cfg = scoring.get(
#         "claim_confidence",
#         {
#             "base": 0.18,
#             "w_strength": 0.16,
#             "w_edges": 0.12,
#             "w_len": 0.07,
#             "ideal_len_min": 12,
#             "ideal_len_max": 40,
#         },
#     )
#
#     entities = _load_entities(domain)
#
#     claims: List[Claim] = []
#     cid = 0
#
#     # Ontology nodes
#     node_id = 0
#     node_index: Dict[Tuple[str, str], str] = {}
#     nodes: List[Dict[str, Any]] = []
#
#     def add_node(ntype: str, label: str) -> str:
#         nonlocal node_id
#         key = (ntype, label.lower())
#         if key in node_index:
#             return node_index[key]
#         node_id += 1
#         nid = f"n{node_id:06d}"
#         node_index[key] = nid
#         nodes.append({"id": nid, "type": ntype, "label": label})
#         return nid
#
#     for doc_path in sorted(docs_dir.glob("*.txt")):
#         doc_name = doc_path.name
#         raw = doc_path.read_text(errors="ignore")
#         sents = split_sentences(raw)
#
#         for s in sents:
#             s = _clean_sentence(s, junk)
#             if not s:
#                 continue
#             if _is_junk_sentence(s, junk):
#                 continue
#
#             tok_n = len(tokens(s))
#             if tok_n < min_token_len or tok_n > max_token_len:
#                 continue
#
#             edge_types, edge_ev = domain.extract_edges_with_strength(s)
#             inf_edges, inf_ev = domain.infer_edges_with_strength(s)
#
#             all_edges = list(dict.fromkeys((edge_types or []) + (inf_edges or [])))
#             if not all_edges:
#                 continue
#
#             # Merge evidence (max per edge)
#             merged_ev: Dict[str, int] = {}
#             for e, v in (edge_ev or {}).items():
#                 merged_ev[e] = max(merged_ev.get(e, 0), int(v))
#             for e, v in (inf_ev or {}).items():
#                 merged_ev[e] = max(merged_ev.get(e, 0), int(v))
#
#             conf = _claim_confidence(merged_ev, tok_n, score_cfg)
#
#             # Entity nodes
#             node_ids: List[str] = []
#             if entities:
#                 mentions = _extract_entity_mentions(s, entities)
#                 for m in mentions:
#                     node_ids.append(add_node(m["type"], m["label"]))
#
#             cid += 1
#             claims.append(
#                 Claim(
#                     id=f"c{cid:07d}",
#                     doc=doc_name,
#                     text=s if s.endswith((".", "!", "?")) else (s + "."),
#                     edge_types=all_edges,
#                     confidence=conf,
#                     edge_evidence=merged_ev,
#                     node_ids=node_ids if node_ids else None,
#                 )
#             )
#
#     # Write claims.jsonl
#     claims_path = index_dir / "claims.jsonl"
#     with claims_path.open("w", encoding="utf-8") as f:
#         for c in claims:
#             f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
#
#     # Write ontology.json
#     ontology = {
#         "nodes": nodes,
#         "edges": [],
#         "meta": {
#             "docs": [p.name for p in docs_dir.glob("*.txt")],
#             "claim_count": len(claims),
#             "node_count": len(nodes),
#         },
#     }
#     (index_dir / "ontology.json").write_text(
#         json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
#     )
#
#     return {
#         "claims": len(claims),
#         "docs": len(list(docs_dir.glob("*.txt"))),
#         "nodes": len(nodes),
#         "index_dir": str(index_dir),
#         "claims_path": str(claims_path),
#     }
#
# # # engine/ingest.py
# # from __future__ import annotations
# #
# # import json
# # import re
# # from pathlib import Path
# # from typing import Dict, Any, List, Tuple, Optional
# #
# # from .utils import norm_space, tokens
# # from .graph import Claim
# # from .domain_pack import DomainPack
# #
# #
# # # Common abbreviations that should not trigger sentence breaks.
# # _ABBR = {
# #     "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "fig.", "ref.",
# #     "no.", "approx.", "al.", "et al.", "cf."
# # }
# #
# #
# # def split_sentences(text: str) -> List[str]:
# #     """
# #     Lightweight sentence splitter:
# #     - preserves common abbreviations
# #     - avoids splitting decimals (e.g., 3.5)
# #     - splits on [.?!] followed by whitespace and a likely sentence start
# #     """
# #     if not text:
# #         return []
# #
# #     t = text.replace("\u00a0", " ")
# #     t = t.replace("\r\n", "\n").replace("\r", "\n")
# #     t = norm_space(t)
# #
# #     if not t:
# #         return []
# #
# #     protected = t
# #
# #     # Protect abbreviations: replace '.' with <DOT> inside known abbreviations
# #     for ab in _ABBR:
# #         protected = protected.replace(ab, ab.replace(".", "<DOT>"))
# #
# #     # Protect decimals: 3.5 -> 3<DOT>5
# #     protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
# #
# #     # Split on sentence end punctuation + whitespace + likely start (Capital/number/bracket)
# #     parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\[])", protected)
# #
# #     out = []
# #     for p in parts:
# #         p = p.replace("<DOT>", ".")
# #         p = norm_space(p)
# #         if p:
# #             out.append(p)
# #     return out
# #
# #
# # def _is_junk_sentence(s: str, junk: Dict[str, Any]) -> bool:
# #     low = (s or "").strip().lower()
# #
# #     # Explicit prefix blacklist
# #     skip_prefixes = [x.lower() for x in (junk.get("skip_prefixes") or [])]
# #     if any(low.startswith(pref) for pref in skip_prefixes):
# #         return True
# #
# #     # Common junk patterns in your docs
# #     if low.startswith("keywords:"):
# #         return True
# #
# #     return False
# #
# #
# # def _claim_confidence(
# #     edge_ev: Dict[str, int],
# #     sent_tokens: int,
# #     score_cfg: Dict[str, Any],
# # ) -> float:
# #     """
# #     Confidence scoring:
# #     - prefers strong pattern matches (strength)
# #     - prefers multi-edge sentences
# #     - prefers sentence lengths in an ideal range
# #     Returns float in (0.01..0.99)
# #     """
# #     import math
# #
# #     base = float(score_cfg.get("base", 0.20))
# #     w_strength = float(score_cfg.get("w_strength", 0.18))
# #     w_edges = float(score_cfg.get("w_edges", 0.10))
# #     w_len = float(score_cfg.get("w_len", 0.06))
# #
# #     ideal_min = int(score_cfg.get("ideal_len_min", 12))
# #     ideal_max = int(score_cfg.get("ideal_len_max", 40))
# #
# #     if not edge_ev:
# #         return 0.01
# #
# #     best = max(edge_ev.values())
# #     n_edges = len(edge_ev)
# #
# #     # length quality: peak inside [ideal_min, ideal_max]
# #     if sent_tokens < ideal_min:
# #         len_q = sent_tokens / max(1, ideal_min)
# #     elif sent_tokens > ideal_max:
# #         len_q = max(0.0, 1.0 - (sent_tokens - ideal_max) / max(1, ideal_max))
# #     else:
# #         len_q = 1.0
# #
# #     raw = base + w_strength * best + w_edges * n_edges + w_len * len_q
# #
# #     # Squash
# #     conf = 1.0 / (1.0 + math.exp(-3.5 * (raw - 0.5)))
# #     return float(max(0.01, min(0.99, conf)))
# #
# #
# # def _load_entities(domain: DomainPack) -> Dict[str, List[str]]:
# #     """
# #     DomainPack may or may not expose entities. Keep robust.
# #     entities.json format:
# #       { "condition":[...], "process":[...], ... }
# #     """
# #     entities = getattr(domain, "entities", None)
# #     if not isinstance(entities, dict):
# #         return {}
# #     out: Dict[str, List[str]] = {}
# #     for k, v in entities.items():
# #         if isinstance(v, list):
# #             out[k] = [str(x) for x in v if str(x).strip()]
# #     return out
# #
# #
# # def _extract_entity_mentions(text: str, entities: Dict[str, List[str]]) -> List[Dict[str, str]]:
# #     """
# #     Extract simple substring-based mentions from the domain vocab.
# #     Returns list of {"type":..., "label":...}.
# #     Keeps it deterministic and pack-driven.
# #     """
# #     low = (text or "").lower()
# #     hits: List[Dict[str, str]] = []
# #     for ntype, vocab in entities.items():
# #         for term in vocab:
# #             tl = term.lower()
# #             if tl and tl in low:
# #                 hits.append({"type": ntype, "label": term})
# #     # De-dupe while preserving order
# #     seen = set()
# #     uniq = []
# #     for h in hits:
# #         key = (h["type"], h["label"].lower())
# #         if key in seen:
# #             continue
# #         seen.add(key)
# #         uniq.append(h)
# #     return uniq
# #
# #
# # def build_index(docs_dir: Path, index_dir: Path, domain: DomainPack) -> Dict[str, Any]:
# #     """
# #     Builds:
# #       data/index/claims.jsonl  (Claim objects with edges + evidence + confidence)
# #       data/index/ontology.json (typed nodes from entity vocab mentions; edges left empty for now)
# #     """
# #     index_dir.mkdir(parents=True, exist_ok=True)
# #
# #     junk = domain.junk_filters or {}
# #     min_token_len = int(junk.get("min_token_len", 7))
# #     max_token_len = int(junk.get("max_token_len", 70))
# #
# #     scoring = domain.scoring or {}
# #     score_cfg = scoring.get(
# #         "claim_confidence",
# #         {
# #             "base": 0.20,
# #             "w_strength": 0.18,
# #             "w_edges": 0.10,
# #             "w_len": 0.06,
# #             "ideal_len_min": 12,
# #             "ideal_len_max": 40,
# #         },
# #     )
# #
# #     entities = _load_entities(domain)
# #
# #     claims: List[Claim] = []
# #     cid = 0
# #
# #     # Ontology node store (id stable for build; can be improved later)
# #     node_id = 0
# #     node_index: Dict[Tuple[str, str], str] = {}
# #     nodes: List[Dict[str, Any]] = []
# #
# #     def add_node(ntype: str, label: str) -> str:
# #         nonlocal node_id
# #         key = (ntype, label.lower())
# #         if key in node_index:
# #             return node_index[key]
# #         node_id += 1
# #         nid = f"n{node_id:06d}"
# #         node_index[key] = nid
# #         nodes.append({"id": nid, "type": ntype, "label": label})
# #         return nid
# #
# #     def ensure_nodes_for_mentions(mentions: List[Dict[str, str]]) -> List[str]:
# #         nids = []
# #         for m in mentions:
# #             nids.append(add_node(m["type"], m["label"]))
# #         return nids
# #
# #     # Process documents
# #     for doc_path in sorted(docs_dir.glob("*.txt")):
# #         doc_name = doc_path.name
# #         raw = doc_path.read_text(errors="ignore")
# #         sents = split_sentences(raw)
# #
# #         for s in sents:
# #             s = norm_space(s)
# #             if not s:
# #                 continue
# #             if _is_junk_sentence(s, junk):
# #                 continue
# #
# #             tok_n = len(tokens(s))
# #             if tok_n < min_token_len or tok_n > max_token_len:
# #                 continue
# #
# #             # Hard extraction edges
# #             edge_types, edge_ev = domain.extract_edges_with_strength(s)
# #
# #             # Soft inference edges
# #             inf_edges, inf_ev = domain.infer_edges_with_strength(s)
# #
# #             # Merge edges, keep order, avoid duplicates
# #             all_edges = list(dict.fromkeys((edge_types or []) + (inf_edges or [])))
# #             if not all_edges:
# #                 continue
# #
# #             # Merge evidence maps (max strength per edge)
# #             merged_ev: Dict[str, int] = {}
# #             for e, v in (edge_ev or {}).items():
# #                 merged_ev[e] = max(merged_ev.get(e, 0), int(v))
# #             for e, v in (inf_ev or {}).items():
# #                 merged_ev[e] = max(merged_ev.get(e, 0), int(v))
# #
# #             conf = _claim_confidence(merged_ev, tok_n, score_cfg)
# #
# #             # Entity mentions from pack vocab (optional)
# #             mentions = _extract_entity_mentions(s, entities) if entities else []
# #             node_ids = ensure_nodes_for_mentions(mentions) if mentions else []
# #
# #             cid += 1
# #             claims.append(
# #                 Claim(
# #                     id=f"c{cid:07d}",
# #                     doc=doc_name,
# #                     text=s if s.endswith((".", "!", "?")) else (s + "."),
# #                     edge_types=all_edges,
# #                     confidence=conf,
# #                     edge_evidence=merged_ev,
# #                 )
# #             )
# #
# #             # Attach nodes to claim (optional field; safe if Claim dataclass allows extras)
# #             # If your Claim dataclass is strict, comment this out.
# #             try:
# #                 setattr(claims[-1], "node_ids", node_ids)
# #             except Exception:
# #                 pass
# #
# #     # Write claims.jsonl
# #     claims_path = index_dir / "claims.jsonl"
# #     with claims_path.open("w", encoding="utf-8") as f:
# #         for c in claims:
# #             # If Claim is a dataclass, __dict__ works; include node_ids if attached
# #             f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
# #
# #     # Write ontology.json (nodes only for now)
# #     ontology = {
# #         "nodes": nodes,
# #         "edges": [],
# #         "meta": {
# #             "docs": [p.name for p in docs_dir.glob("*.txt")],
# #             "claim_count": len(claims),
# #             "node_count": len(nodes),
# #         },
# #     }
# #     (index_dir / "ontology.json").write_text(
# #         json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8"
# #     )
# #
# #     return {
# #         "claims": len(claims),
# #         "docs": len(list(docs_dir.glob("*.txt"))),
# #         "nodes": len(nodes),
# #         "index_dir": str(index_dir),
# #         "claims_path": str(claims_path),
# #     }
# #
# # # from __future__ import annotations
# # # import json
# # # import re
# # # from pathlib import Path
# # # from typing import Dict, Any, List, Tuple
# # #
# # # from .utils import norm_space, tokens
# # # from .graph import Claim
# # # from .domain_pack import DomainPack
# # #
# # #
# # # _ABBR = {
# # #     "e.g.", "i.e.", "vs.", "etc.", "dr.", "mr.", "mrs.", "ms.", "fig.", "ref.",
# # #     "no.", "approx.", "al.", "et al."
# # # }
# # #
# # # def split_sentences(text: str) -> List[str]:
# # #     """
# # #     Lightweight sentence splitter:
# # #     - avoids splitting on common abbreviations
# # #     - avoids splitting decimals (3.5)
# # #     - handles newlines
# # #     """
# # #     t = norm_space(text.replace("\u00a0", " "))
# # #     if not t:
# # #         return []
# # #
# # #     # Protect abbreviations and decimals
# # #     protected = t
# # #     for ab in _ABBR:
# # #         protected = protected.replace(ab, ab.replace(".", "<DOT>"))
# # #     protected = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected)
# # #
# # #     # Split on sentence end punctuation followed by space + capital/number/bracket
# # #     parts = re.split(r"(?<=[\.\!\?])\s+(?=[A-Z0-9\[])","{}".format(protected))
# # #     out = []
# # #     for p in parts:
# # #         p = p.replace("<DOT>", ".").strip()
# # #         p = norm_space(p)
# # #         if p:
# # #             out.append(p)
# # #     return out
# # #
# # #
# # # def _is_junk_sentence(s: str, junk: Dict[str, Any]) -> bool:
# # #     low = s.lower()
# # #     skip_prefixes = [x.lower() for x in (junk.get("skip_prefixes") or [])]
# # #     if any(low.startswith(pref) for pref in skip_prefixes):
# # #         return True
# # #     # Drop pure “Keywords:” / “Introduction and background” type lines
# # #     if low.startswith("keywords:"):
# # #         return True
# # #     return False
# # #
# # #
# # # def _confidence(edge_ev: Dict[str, int], sent_tokens: int, score_cfg: Dict[str, Any]) -> float:
# # #     """
# # #     Confidence scoring: designed to rank clean, causal claims above noisy paragraphs.
# # #     """
# # #     import math
# # #     base = float(score_cfg.get("base", 0.20))
# # #     w_strength = float(score_cfg.get("w_strength", 0.18))
# # #     w_edges = float(score_cfg.get("w_edges", 0.10))
# # #     w_len = float(score_cfg.get("w_len", 0.06))
# # #     ideal_min = int(score_cfg.get("ideal_len_min", 12))
# # #     ideal_max = int(score_cfg.get("ideal_len_max", 40))
# # #
# # #     if not edge_ev:
# # #         return 0.0
# # #
# # #     best = max(edge_ev.values())
# # #     n_edges = len(edge_ev)
# # #
# # #     # length quality: peak in [ideal_min, ideal_max]
# # #     if sent_tokens < ideal_min:
# # #         len_q = sent_tokens / max(1, ideal_min)
# # #     elif sent_tokens > ideal_max:
# # #         len_q = max(0.0, 1.0 - (sent_tokens - ideal_max) / max(1, ideal_max))
# # #     else:
# # #         len_q = 1.0
# # #
# # #     raw = base + w_strength * best + w_edges * n_edges + w_len * len_q
# # #     # squash into (0,1)
# # #     conf = 1.0 / (1.0 + math.exp(-3.5 * (raw - 0.5)))
# # #     return float(max(0.01, min(0.99, conf)))
# # #
# # #
# # # def build_index(docs_dir: Path, index_dir: Path, domain: DomainPack) -> Dict[str, Any]:
# # #     index_dir.mkdir(parents=True, exist_ok=True)
# # #
# # #     junk = domain.junk_filters or {}
# # #     min_token_len = int(junk.get("min_token_len", 7))
# # #     max_token_len = int(junk.get("max_token_len", 70))
# # #
# # #     scoring = domain.scoring or {}
# # #     score_cfg = scoring.get("claim_confidence", {
# # #         "base": 0.20, "w_strength": 0.18, "w_edges": 0.10, "w_len": 0.06,
# # #         "ideal_len_min": 12, "ideal_len_max": 40
# # #     })
# # #
# # #     claims: List[Claim] = []
# # #     cid = 0
# # #
# # #     for doc_path in sorted(docs_dir.glob("*.txt")):
# # #         doc_name = doc_path.name
# # #         raw = doc_path.read_text(errors="ignore")
# # #         sents = split_sentences(raw)
# # #
# # #         for s in sents:
# # #             s = norm_space(s)
# # #             if not s:
# # #                 continue
# # #             if _is_junk_sentence(s, junk):
# # #                 continue
# # #
# # #             tok_n = len(tokens(s))
# # #             if tok_n < min_token_len or tok_n > max_token_len:
# # #                 continue
# # #
# # #             # Extract typed edges (hard patterns)
# # #             edge_types, edge_ev = domain.extract_edges_with_strength(s)
# # #
# # #             # Runtime infer edges (looser patterns) can be included too
# # #             inf_edges, inf_ev = domain.infer_edges_with_strength(s)
# # #             all_edges = list(dict.fromkeys((edge_types or []) + (inf_edges or [])))
# # #
# # #             # Merge evidence (take max strength per edge)
# # #             merged_ev: Dict[str, int] = {}
# # #             for d in (edge_ev or {}).items():
# # #                 merged_ev[d[0]] = max(merged_ev.get(d[0], 0), int(d[1]))
# # #             for d in (inf_ev or {}).items():
# # #                 merged_ev[d[0]] = max(merged_ev.get(d[0], 0), int(d[1]))
# # #
# # #             if not all_edges:
# # #                 continue
# # #
# # #             conf = _confidence(merged_ev, tok_n, score_cfg)
# # #             cid += 1
# # #             claims.append(Claim(
# # #                 id=f"c{cid:07d}",
# # #                 doc=doc_name,
# # #                 text=s if s.endswith((".", "!", "?")) else (s + "."),
# # #                 edge_types=all_edges,
# # #                 confidence=conf,
# # #                 edge_evidence=merged_ev
# # #             ))
# # #
# # #     # Write claims.jsonl
# # #     claims_path = index_dir / "claims.jsonl"
# # #     with claims_path.open("w", encoding="utf-8") as f:
# # #         for c in claims:
# # #             f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
# # #
# # #     # Minimal ontology placeholder (B step will expand this)
# # #     ontology = {
# # #         "nodes": [],
# # #         "edges": [],
# # #         "meta": {
# # #             "docs": [p.name for p in docs_dir.glob("*.txt")],
# # #             "claim_count": len(claims)
# # #         }
# # #     }
# # #     (index_dir / "ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2), encoding="utf-8")
# # #
# # #     return {
# # #         "claims": len(claims),
# # #         "docs": len(list(docs_dir.glob("*.txt"))),
# # #         "index_dir": str(index_dir)
# # #     }
# # #
# # # # from __future__ import annotations
# # # # import json
# # # # from pathlib import Path
# # # # from typing import Dict, Any
# # # # from .utils import norm_space
# # # # from .graph import Claim
# # # # from .domain_pack import DomainPack
# # # #
# # # #
# # # # def build_index(docs_dir: Path, index_dir: Path, domain: DomainPack) -> Dict[str, Any]:
# # # #     index_dir.mkdir(parents=True, exist_ok=True)
# # # #
# # # #     junk = domain.junk_filters or {}
# # # #     skip_prefixes = [s.lower() for s in (junk.get("skip_prefixes") or [])]
# # # #     min_token_len = int(junk.get("min_token_len", 6))
# # # #
# # # #     claims = []
# # # #     cid = 0
# # # #
# # # #     for doc_path in sorted(docs_dir.glob("*.txt")):
# # # #         doc_name = doc_path.name
# # # #         text = doc_path.read_text(errors="ignore")
# # # #         # cheap sentence-ish split
# # # #         parts = [p.strip() for p in text.replace("\n", " ").split(".") if p.strip()]
# # # #         for p in parts:
# # # #             s = norm_space(p)
# # # #             if not s:
# # # #                 continue
# # # #             low = s.lower()
# # # #             if any(low.startswith(pref) for pref in skip_prefixes):
# # # #                 continue
# # # #             if len(s.split()) < min_token_len:
# # # #                 continue
# # # #
# # # #             edge_types, edge_ev = domain.extract_edges_with_strength(s)
# # # #             if not edge_types:
# # # #                 continue
# # # #
# # # #             # simple confidence: normalized by best edge strength
# # # #             best = max(edge_ev.values()) if edge_ev else 1
# # # #             conf = min(1.0, 0.25 + 0.25 * best)
# # # #
# # # #             cid += 1
# # # #             claims.append(Claim(
# # # #                 id=f"c{cid:07d}",
# # # #                 doc=doc_name,
# # # #                 text=s + ". ",
# # # #                 edge_types=edge_types,
# # # #                 confidence=conf,
# # # #                 edge_evidence=edge_ev
# # # #             ))
# # # #
# # # #     # write claims.jsonl
# # # #     claims_path = index_dir / "claims.jsonl"
# # # #     with claims_path.open("w", encoding="utf-8") as f:
# # # #         for c in claims:
# # # #             f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
# # # #
# # # #     # minimal ontology (you can expand later)
# # # #     ontology = {"nodes": [], "edges": [], "meta": {"docs": [p.name for p in docs_dir.glob("*.txt")]}}
# # # #     (index_dir / "ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2))
# # # #
# # # #     return {"claims": len(claims), "docs": len(list(docs_dir.glob("*.txt"))), "index_dir": str(index_dir)}
# # # #
# # # # # from __future__ import annotations
# # # # # import json, re
# # # # # from pathlib import Path
# # # # # from typing import Dict, List, Tuple, Any
# # # # # from .utils import split_sents, norm_space, uniq, tokens
# # # # # from .domain_pack import DomainPack
# # # # #
# # # # # def norm(t: str) -> str:
# # # # #     return re.sub(r"\s+"," ", t.strip().lower())
# # # # #
# # # # # THRESH_PAT = re.compile(
# # # # #     r'(\b\d{1,2}\s*-\s*\d{1,2}\s*weeks\b|\b\d{1,2}\s*weeks\b|\b\d{2,3}\s*mg/dl\b|\b\d{1,2}\.\d\s*mmol\/l\b)',
# # # # #     re.I
# # # # # )
# # # # #
# # # # # def load_lexicon(lexicon_path: Path) -> Dict[str, List[str]]:
# # # # #     if not lexicon_path.exists():
# # # # #         return {}
# # # # #     obj = json.loads(lexicon_path.read_text(errors="ignore"))
# # # # #     return {k: [norm(x) for x in v] for k, v in obj.get("seeds", {}).items()}
# # # # #
# # # # # def save_lexicon(lexicon_path: Path, seeds: Dict[str, List[str]]):
# # # # #     lexicon_path.parent.mkdir(parents=True, exist_ok=True)
# # # # #     lexicon_path.write_text(json.dumps({"seeds": seeds}, ensure_ascii=False, indent=2))
# # # # #
# # # # # def extract_concepts(sent: str, seed_norm: Dict[str, set]) -> List[Tuple[str,str]]:
# # # # #     found=[]
# # # # #     low=sent.lower()
# # # # #     for cat, terms in seed_norm.items():
# # # # #         for t in terms:
# # # # #             if t and t in low:
# # # # #                 found.append((cat,t))
# # # # #     for m in THRESH_PAT.finditer(sent):
# # # # #         found.append(("threshold_or_timing", m.group(0).strip()))
# # # # #     return found
# # # # #
# # # # # def compute_confidence(sent: str, nodes_n: int, edge_evidence: Dict[str,int], domain: DomainPack) -> float:
# # # # #     # from scoring.json
# # # # #     sc = domain.scoring or {}
# # # # #     base = float(sc.get("base", 0.15))
# # # # #     edge_count_w = float(sc.get("edge_count_weight", 0.12))
# # # # #     edge_strength_w = float(sc.get("edge_strength_weight", 0.08))
# # # # #     node_w = float(sc.get("node_weight", 0.03))
# # # # #     thresh_bonus = float(sc.get("threshold_bonus", 0.10))
# # # # #     short_pen = float(sc.get("short_penalty", 0.10))
# # # # #     long_pen = float(sc.get("long_penalty", 0.08))
# # # # #     short_len = int(sc.get("short_len", 6))
# # # # #     long_len = int(sc.get("long_len", 60))
# # # # #     max_edges = int(sc.get("max_edges", 6))
# # # # #     max_nodes = int(sc.get("max_nodes", 10))
# # # # #
# # # # #     s = sent.strip().lower()
# # # # #     for pref in domain.junk_filters.get("skip_prefixes", ["keywords:"]):
# # # # #         if s.startswith(pref):
# # # # #             return 0.05
# # # # #
# # # # #     length = len(tokens(sent))
# # # # #     score = base
# # # # #
# # # # #     if edge_evidence:
# # # # #         score += edge_count_w * min(max_edges, len(edge_evidence))
# # # # #         score += edge_strength_w * sum(edge_evidence.values())
# # # # #
# # # # #     score += node_w * min(max_nodes, nodes_n)
# # # # #
# # # # #     if THRESH_PAT.search(sent):
# # # # #         score += thresh_bonus
# # # # #
# # # # #     if length < short_len:
# # # # #         score -= short_pen
# # # # #     elif length > long_len:
# # # # #         score -= long_pen
# # # # #
# # # # #     score = max(0.0, min(1.0, score))
# # # # #     return round(score, 3)
# # # # #
# # # # # def build_index(docs_dir: Path, index_dir: Path, lexicon_path: Path, domain: DomainPack) -> Dict[str,int]:
# # # # #     docs = sorted([p for p in docs_dir.glob("*.txt") if p.is_file()])
# # # # #     if not docs:
# # # # #         raise FileNotFoundError(f"No .txt docs found in {docs_dir}")
# # # # #
# # # # #     # seeds come from domain pack + optional lexicon extension
# # # # #     seeds = {k: [norm(x) for x in v] for k,v in domain.seeds.items()}
# # # # #     user_lex = load_lexicon(lexicon_path)
# # # # #     for k,v in user_lex.items():
# # # # #         seeds.setdefault(k, [])
# # # # #         seeds[k] = uniq(seeds[k] + v)
# # # # #
# # # # #     seed_norm = {cat:set(map(norm,terms)) for cat,terms in seeds.items()}
# # # # #
# # # # #     concept_index: Dict[str,Dict[str,Any]] = {}
# # # # #     claims=[]
# # # # #     def add_concept(cat: str, label: str) -> str:
# # # # #         key=f"{cat}:{label}"
# # # # #         if key not in concept_index:
# # # # #             concept_index[key]={"id":key,"type":cat,"label":label,"aliases":[label]}
# # # # #         return key
# # # # #
# # # # #     min_len = int(domain.junk_filters.get("min_token_len", 6))
# # # # #     skip_prefixes = [p.lower() for p in domain.junk_filters.get("skip_prefixes", ["keywords:"])]
# # # # #
# # # # #     cid=0
# # # # #     for doc in docs:
# # # # #         text = doc.read_text(errors="ignore")
# # # # #         for sent in split_sents(text):
# # # # #             s = sent.strip()
# # # # #             low = s.lower()
# # # # #
# # # # #             if any(low.startswith(p) for p in skip_prefixes):
# # # # #                 continue
# # # # #             if len(tokens(s)) < min_len:
# # # # #                 continue
# # # # #
# # # # #             concepts = extract_concepts(s, seed_norm)
# # # # #             edges, edge_evidence = domain.extract_edges_with_strength(s)
# # # # #             if not concepts and not edges:
# # # # #                 continue
# # # # #
# # # # #             nodes=[add_concept(c,t) for c,t in concepts]
# # # # #             conf = compute_confidence(s, len(nodes), edge_evidence, domain)
# # # # #
# # # # #             claims.append({
# # # # #                 "id": f"c{cid:07d}",
# # # # #                 "doc": doc.name,
# # # # #                 "text": norm_space(s),
# # # # #                 "nodes": nodes,
# # # # #                 "edge_types": edges,
# # # # #                 "edge_evidence": edge_evidence,
# # # # #                 "confidence": conf,
# # # # #             })
# # # # #             cid += 1
# # # # #
# # # # #     ontology = {
# # # # #         "meta": {
# # # # #             "name": domain.manifest.get("name","DomainPack"),
# # # # #             "version": domain.manifest.get("version",""),
# # # # #             "source_docs": [d.name for d in docs],
# # # # #             "node_count": len(concept_index),
# # # # #             "claim_count": len(claims),
# # # # #         },
# # # # #         "node_types": sorted(list(seeds.keys()) + ["threshold_or_timing"]),
# # # # #         "edge_types": sorted(list(domain.edge_patterns_raw.keys())),
# # # # #         "nodes": list(concept_index.values()),
# # # # #         "seeds": seeds
# # # # #     }
# # # # #
# # # # #     index_dir.mkdir(parents=True, exist_ok=True)
# # # # #     (index_dir/"ontology.json").write_text(json.dumps(ontology, ensure_ascii=False, indent=2))
# # # # #     (index_dir/"claims.jsonl").write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in claims))
# # # # #
# # # # #     save_lexicon(lexicon_path, seeds)
# # # # #     return {"docs": len(docs), "nodes": len(concept_index), "claims": len(claims)}
