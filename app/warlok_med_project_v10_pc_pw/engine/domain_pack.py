# from __future__ import annotations
# import json
# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Any, List, Tuple
#
#
# @dataclass
# class CompiledEdgePattern:
# 	regex: re.Pattern
# 	strength: int = 2
#
#
# class DomainPack:
# 	def __init__(
# 		self,
# 		pack_dir: Path,
# 		manifest: Dict[str, Any],
# 		frames_raw: List[Dict[str, Any]],
# 		frames: Dict[str, Dict[str, Any]],
# 		entities: Dict[str, List[str]],
# 		edges_meta: Dict[str, Dict[str, Any]],
# 		edge_patterns_raw: Dict[str, Any],
# 		infer_rules_raw: Dict[str, Any],
# 		routing_rules: List[Dict[str, Any]],
# 		question_templates: Dict[str, Any],
# 		junk_filters: Dict[str, Any],
# 		scoring: Dict[str, Any],
# 		edge_patterns: Dict[str, List[CompiledEdgePattern]],
# 		infer_rules: Dict[str, List[CompiledEdgePattern]],
# 	):
# 		self.pack_dir = pack_dir
# 		self.manifest = manifest
#
# 		self.frames_raw = frames_raw
# 		self.frames = frames  # dict id -> dict (stable API)
# 		self.entities = entities
# 		self.edges_meta = edges_meta
#
# 		self.edge_patterns_raw = edge_patterns_raw
# 		self.infer_rules_raw = infer_rules_raw
# 		self.routing_rules = routing_rules
#
# 		self.question_templates = question_templates or {"by_frame": {}, "by_edge": {}, "chain_2hop": []}
# 		self.junk_filters = junk_filters or {}
# 		self.scoring = scoring or {}
#
# 		self.edge_patterns = edge_patterns
# 		self.infer_rules = infer_rules
#
# 	# @staticmethod
# 	# def _load_json(path: Path, default):
# 	#     if not path.exists():
# 	#         return default
# 	#     return json.loads(path.read_text(errors="ignore"))
# 	@staticmethod
# 	def _load_json(path: Path, default):
# 		if not path.exists():
# 			return default
# 		try:
# 			return json.loads(path.read_text(errors="ignore"))
# 		except Exception as e:
# 			raise ValueError(f"Bad JSON in {path}: {e}")
#
# 	@staticmethod
# 	def _compile_map(raw: Dict[str, Any]) -> Dict[str, List[CompiledEdgePattern]]:
# 		out: Dict[str, List[CompiledEdgePattern]] = {}
# 		for edge, arr in (raw or {}).items():
# 			compiled: List[CompiledEdgePattern] = []
# 			if isinstance(arr, str):
# 				compiled.append(CompiledEdgePattern(regex=re.compile(arr, re.I), strength=2))
# 			else:
# 				for item in arr:
# 					pat = item["pattern"]
# 					strength = int(item.get("strength", 2))
# 					compiled.append(CompiledEdgePattern(regex=re.compile(pat, re.I), strength=strength))
# 			out[edge] = compiled
# 		return out
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


@dataclass
class CompiledPattern:
    regex: re.Pattern
    strength: int = 2


class DomainPack:
    """
    Engine-generic domain pack.
    All domain knowledge (frames, edges, routing, templates, ontology_spec...) lives in JSON files.
    The engine only loads and executes pack-provided specs.
    """

    def __init__(
        self,
        pack_dir: Path,
        manifest: Dict[str, Any],
        frames_raw: List[Dict[str, Any]],
        frames: Dict[str, Dict[str, Any]],
        edges_meta: Dict[str, Dict[str, Any]],
        edge_patterns_raw: Dict[str, Any],
        infer_rules_raw: Dict[str, Any],
        routing_rules: List[Dict[str, Any]],
        question_templates: Dict[str, Any],
        junk_filters: Dict[str, Any],
        scoring: Dict[str, Any],
        ontology_spec_raw: Dict[str, Any],
        edge_patterns: Dict[str, List[CompiledPattern]],
        infer_rules: Dict[str, List[CompiledPattern]],
        ontology_extractors: List[Dict[str, Any]],
    ):
        self.pack_dir = pack_dir
        self.manifest = manifest

        self.frames_raw = frames_raw
        self.frames = frames  # dict id -> dict

        self.edges_meta = edges_meta

        self.edge_patterns_raw = edge_patterns_raw
        self.infer_rules_raw = infer_rules_raw
        self.routing_rules = routing_rules

        self.question_templates = question_templates or {"by_frame": {}, "by_edge": {}, "chain_2hop": []}
        self.junk_filters = junk_filters or {}
        self.scoring = scoring or {}

        # Ontology is entirely pack-driven:
        self.ontology_spec_raw = ontology_spec_raw or {}
        self.ontology_extractors = ontology_extractors or []

        self.edge_patterns = edge_patterns
        self.infer_rules = infer_rules

    @staticmethod
    def _load_json(path: Path, default):
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(errors="ignore"))
        except Exception as e:
            raise ValueError(f"Bad JSON in {path}: {e}")

    @staticmethod
    def _compile_pattern_map(raw: Dict[str, Any]) -> Dict[str, List[CompiledPattern]]:
        """
        raw can be:
          { "EDGE": "regex" }
        or
          { "EDGE": [ {"pattern": "...", "strength": 3}, ... ] }
        """
        out: Dict[str, List[CompiledPattern]] = {}
        for key, arr in (raw or {}).items():
            compiled: List[CompiledPattern] = []
            if isinstance(arr, str):
                compiled.append(CompiledPattern(regex=re.compile(arr, re.I), strength=2))
            elif isinstance(arr, list):
                for item in arr:
                    pat = item["pattern"]
                    strength = int(item.get("strength", 2))
                    compiled.append(CompiledPattern(regex=re.compile(pat, re.I), strength=strength))
            else:
                continue
            out[key] = compiled
        return out

    @staticmethod
    def _compile_ontology_extractors(ontology_spec_raw: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Compile pack-driven ontology node extractors.
        Supported extractor forms:
          - {"node_type":"X","match":["term1","term2",...]}  (substring, case-insensitive)
          - {"node_type":"X","regex":[{"pattern":"...","strength":2}, ...]} (regex)
        Engine remains domain-agnostic: just executes these specs.
        """
        extractors = []
        for ex in (ontology_spec_raw or {}).get("node_extractors", []) or []:
            node_type = ex.get("node_type")
            if not node_type:
                continue
            mode = ex.get("mode", "substring")
            match = ex.get("match") or []
            regex_list = ex.get("regex") or []

            compiled = {"node_type": node_type, "mode": mode, "match": [], "regex": []}

            if isinstance(match, list):
                compiled["match"] = [str(x) for x in match if str(x).strip()]

            if isinstance(regex_list, list):
                reg_compiled = []
                for item in regex_list:
                    pat = item["pattern"]
                    strength = int(item.get("strength", 2))
                    reg_compiled.append(CompiledPattern(regex=re.compile(pat, re.I), strength=strength))
                compiled["regex"] = reg_compiled

            extractors.append(compiled)
        return extractors

    @classmethod
    def load(cls, pack_dir: Path) -> "DomainPack":
        pack_dir = Path(pack_dir).resolve()
        manifest_path = pack_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest.json in {pack_dir}")

        manifest = cls._load_json(manifest_path, default={})
        files = manifest.get("files", {}) or {}

        frames_raw = cls._load_json(pack_dir / files.get("frames", "frames.json"), default=[])
        if not isinstance(frames_raw, list) or not frames_raw:
            raise ValueError("frames.json must be a non-empty list")

        frames = {f["id"]: f for f in frames_raw if "id" in f}

        edges_meta = cls._load_json(pack_dir / files.get("edges", "edges.json"), default={})
        edge_patterns_raw = cls._load_json(pack_dir / files.get("edge_patterns", "edge_patterns.json"), default={})
        infer_rules_raw = cls._load_json(pack_dir / files.get("infer_rules", "infer_rules.json"), default={})
        routing_rules = cls._load_json(pack_dir / files.get("routing_rules", "routing_rules.json"), default=[])
        question_templates = cls._load_json(pack_dir / files.get("question_templates", "question_templates.json"), default={})
        junk_filters = cls._load_json(pack_dir / files.get("junk_filters", "junk_filters.json"), default={})
        scoring = cls._load_json(pack_dir / files.get("scoring", "scoring.json"), default={})

        # NEW: pack-driven ontology spec (optional)
        ontology_spec_raw = cls._load_json(pack_dir / files.get("ontology_spec", "ontology_spec.json"), default={})
        ontology_extractors = cls._compile_ontology_extractors(ontology_spec_raw)

        edge_patterns = cls._compile_pattern_map(edge_patterns_raw)
        infer_rules = cls._compile_pattern_map(infer_rules_raw)

        return cls(
            pack_dir=pack_dir,
            manifest=manifest,
            frames_raw=frames_raw,
            frames=frames,
            edges_meta=edges_meta,
            edge_patterns_raw=edge_patterns_raw,
            infer_rules_raw=infer_rules_raw,
            routing_rules=routing_rules,
            question_templates=question_templates,
            junk_filters=junk_filters,
            scoring=scoring,
            ontology_spec_raw=ontology_spec_raw,
            edge_patterns=edge_patterns,
            infer_rules=infer_rules,
            ontology_extractors=ontology_extractors,
        )

    def edge_types_to_node_types(self, edge_types: List[str]) -> List[str]:
        out = []
        for e in edge_types or []:
            meta = self.edges_meta.get(e, {}) if isinstance(self.edges_meta, dict) else {}
            t = meta.get("node_type")
            if t:
                out.append(t)
        return out

    def extract_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        edges: List[str] = []
        evidence: Dict[str, int] = {}
        for edge, pats in (self.edge_patterns or {}).items():
            best = 0
            for p in pats:
                if p.regex.search(text):
                    best = max(best, p.strength)
            if best > 0:
                edges.append(edge)
                evidence[edge] = best
        return edges, evidence

    def infer_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
        edges: List[str] = []
        evidence: Dict[str, int] = {}
        for edge, pats in (self.infer_rules or {}).items():
            best = 0
            for p in pats:
                if p.regex.search(text):
                    best = max(best, p.strength)
            if best > 0:
                edges.append(edge)
                evidence[edge] = best
        return edges, evidence

    # def route_frame(self, question: str) -> str:
    #     q = (question or "").lower()
    #     for rule in self.routing_rules or []:
    #         fid = rule.get("frame_id")
    #         for kw in (rule.get("any_keywords") or []):
    #             if kw.lower() in q:
    #                 return fid
    #     # default: first frame
    #     return list(self.frames.keys())[0]
    
    def route_frame(self, question: str) -> str:
        q = (question or "").lower()
        
        rules = self.routing_rules or []
        
        # If user accidentally wrapped rules in an extra list: [[...]] -> [...]
        if isinstance(rules, list) and len(rules) == 1 and isinstance(rules[0], list):
            rules = rules[0]
        
        # Walk rules safely
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            fid = rule.get("frame_id")
            kws = rule.get("any_keywords") or []
            if not fid:
                continue
            for kw in kws:
                if kw and str(kw).lower() in q:
                    return fid
        
        # Default: first frame id
        return list(self.frames.keys())[0]

#
# 	@classmethod
# 	def load(cls, pack_dir: Path) -> "DomainPack":
# 		pack_dir = Path(pack_dir).resolve()
# 		manifest_path = pack_dir / "manifest.json"
# 		if not manifest_path.exists():
# 			raise FileNotFoundError(f"Missing manifest.json in {pack_dir}")
#
# 		manifest = json.loads(manifest_path.read_text(errors="ignore"))
# 		files = manifest.get("files", {})
#
# 		frames_raw = cls._load_json(pack_dir / files.get("frames", "frames.json"), default=[])
# 		if not isinstance(frames_raw, list) or not frames_raw:
# 			raise ValueError("frames.json must be a non-empty list")
#
# 		frames = {f["id"]: f for f in frames_raw}
#
# 		entities = cls._load_json(pack_dir / files.get("entities", "entities.json"), default={})
# 		edges_meta = cls._load_json(pack_dir / files.get("edges", "edges.json"), default={})
# 		edge_patterns_raw = cls._load_json(pack_dir / files.get("edge_patterns", "edge_patterns.json"), default={})
# 		infer_rules_raw = cls._load_json(pack_dir / files.get("infer_rules", "infer_rules.json"), default={})
# 		routing_rules = cls._load_json(pack_dir / files.get("routing_rules", "routing_rules.json"), default=[])
# 		question_templates = cls._load_json(pack_dir / files.get("question_templates", "question_templates.json"), default={})
# 		junk_filters = cls._load_json(pack_dir / files.get("junk_filters", "junk_filters.json"), default={})
# 		scoring = cls._load_json(pack_dir / files.get("scoring", "scoring.json"), default={})
#
# 		edge_patterns = cls._compile_map(edge_patterns_raw)
# 		infer_rules = cls._compile_map(infer_rules_raw)
#
# 		return cls(
# 			pack_dir=pack_dir,
# 			manifest=manifest,
# 			frames_raw=frames_raw,
# 			frames=frames,
# 			edges_meta=edges_meta,
# 			edge_patterns_raw=edge_patterns_raw,
# 			infer_rules_raw=infer_rules_raw,
# 			routing_rules=routing_rules,
# 			question_templates=question_templates,
# 			junk_filters=junk_filters,
# 			scoring=scoring,
# 			edge_patterns=edge_patterns,
# 			infer_rules=infer_rules,
# 			entities=entities,
# 		)
#
# 	def edge_types_to_node_types(self, edge_types: List[str]) -> List[str]:
# 		out = []
# 		for e in edge_types or []:
# 			meta = self.edges_meta.get(e, {})
# 			t = meta.get("node_type")
# 			if t:
# 				out.append(t)
# 		return out
#
# 	def extract_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
# 		edges = []
# 		evidence = {}
# 		for edge, pats in self.edge_patterns.items():
# 			best = 0
# 			for p in pats:
# 				if p.regex.search(text):
# 					best = max(best, p.strength)
# 			if best > 0:
# 				edges.append(edge)
# 				evidence[edge] = best
# 		return edges, evidence
#
# 	def infer_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
# 		edges = []
# 		evidence = {}
# 		for edge, pats in self.infer_rules.items():
# 			best = 0
# 			for p in pats:
# 				if p.regex.search(text):
# 					best = max(best, p.strength)
# 			if best > 0:
# 				edges.append(edge)
# 				evidence[edge] = best
# 		return edges, evidence
#
# 	def route_frame(self, question: str) -> str:
# 		q = (question or "").lower()
# 		for rule in self.routing_rules or []:
# 			fid = rule.get("frame_id")
# 			for kw in (rule.get("any_keywords") or []):
# 				if kw.lower() in q:
# 					return fid
# 		# default: choose first frame in file
# 		return list(self.frames.keys())[0]
#
# # from __future__ import annotations
# # import json
# # import re
# # from dataclasses import dataclass
# # from pathlib import Path
# # from typing import Dict, List, Any, Optional, Tuple
# #
# # @dataclass
# # class FrameSpec:
# # 	id: str
# # 	name: str
# # 	required_edges: List[str]
# # 	min_required_covered: int
# # 	allowed_node_types: List[str]
# # 	blocked_node_types: List[str]
# # 	min_steps: int
# # 	max_steps: int
# #
# # @dataclass
# # class CompiledEdgePattern:
# # 	regex: re.Pattern
# # 	strength: int
# #
# # @dataclass
# # class DomainPack:
# # 	root: Path
# # 	manifest: Dict[str, Any]
# #
# # 	# raw data
# # 	seeds: Dict[str, List[str]]
# # 	frames: List[Dict[str, Any]]
# # 	routing_rules: List[Dict[str, Any]]
# # 	edge_patterns_raw: Dict[str, List[Dict[str, Any]]]
# # 	infer_rules_raw: Dict[str, str]
# # 	junk_filters: Dict[str, Any]
# # 	scoring: Dict[str, Any]
# #
# # 	# compiled
# # 	edge_patterns: Dict[str, List[CompiledEdgePattern]]
# # 	infer_rules: Dict[str, re.Pattern]
# # 	frames_by_id: Dict[str, Dict[str, Any]]
# #
# # 	@staticmethod
# # 	def _load_json(p: Path) -> Any:
# # 		return json.loads(p.read_text(errors="ignore"))
# #
# # 	@staticmethod
# # 	def load(pack_dir: Path) -> "DomainPack":
# # 		pack_dir = Path(pack_dir).resolve()
# # 		manifest_path = pack_dir / "manifest.json"
# # 		if not manifest_path.exists():
# # 			raise FileNotFoundError(f"Missing manifest.json in {pack_dir}")
# #
# # 		manifest = DomainPack._load_json(manifest_path)
# #
# # 		def fpath(key: str, default_name: str) -> Path:
# # 			files = manifest.get("files", {})
# # 			name = files.get(key, default_name)
# # 			return pack_dir / name
# #
# # 		seeds = DomainPack._load_json(fpath("seeds", "seeds.json"))
# # 		frames = DomainPack._load_json(fpath("frames", "frames.json"))
# # 		routing_rules = DomainPack._load_json(fpath("routing_rules", "routing_rules.json"))
# # 		edge_patterns_raw = DomainPack._load_json(fpath("edge_patterns", "edge_patterns.json"))
# # 		infer_rules_raw = DomainPack._load_json(fpath("infer_rules", "infer_rules.json"))
# # 		junk_filters = DomainPack._load_json(fpath("junk_filters", "junk_filters.json"))
# # 		scoring = DomainPack._load_json(fpath("scoring", "scoring.json"))
# #
# # 		# compile edge patterns
# # 		edge_patterns: Dict[str, List[CompiledEdgePattern]] = {}
# # 		for edge, arr in edge_patterns_raw.items():
# # 			compiled = []
# # 			for item in arr:
# # 				pat = item["pattern"]
# # 				strength = int(item.get("strength", 2))
# # 				compiled.append(CompiledEdgePattern(regex=re.compile(pat, re.I), strength=strength))
# # 			edge_patterns[edge] = compiled
# #
# # 		# --- infer rules (ALWAYS initialize) ---
# # 		infer_rules: dict = {}  # <-- critical: define it unconditionally
# #
# # 		# infer_rules_path = pack_dir / files.get("infer_rules", "infer_rules.json")
# # 		infer_rules_path = pack_dir / "infer_rules.json"
# #
# # 		infer_rules_raw = {}
# # 		if infer_rules_path.exists():
# # 			infer_rules_raw = json.loads(infer_rules_path.read_text(errors="ignore"))
# #
# # 		# compile (supports list-of-patterns format)
# # 		for edge, arr in (infer_rules_raw or {}).items():
# # 			compiled = []
# # 			# allow old format: "EDGE": "regex"
# # 			if isinstance(arr, str):
# # 				compiled.append(CompiledEdgePattern(regex=re.compile(arr, re.I), strength=2))
# # 			else:
# # 				for item in arr:
# # 					pat = item["pattern"]
# # 					strength = int(item.get("strength", 2))
# # 					compiled.append(CompiledEdgePattern(regex=re.compile(pat, re.I), strength=strength))
# # 			infer_rules[edge] = compiled
# #
# # 		# compile infer rules
# # 		# infer_rules: Dict[str, List[CompiledEdgePattern]] = {}
# # 		# infer_rules: Dict[str, List[CompiledEdgePattern]]
# # 		#
# # 		# for edge, arr in infer_rules_raw.items():
# # 		# 	compiled = []
# # 		# 	# infer_rules[edge] = []
# # 		# 	for item in arr:
# # 		# 		pat = item["pattern"]
# # 		# 		strength = int(item.get("strength", 2))
# # 		# 		compiled.append(
# # 		# 			CompiledEdgePattern(regex=re.compile(pat, re.I), strength=strength)
# # 		# 		)
# # 		# 	infer_rules[edge] = compiled
# #
# # 		# infer_rules: Dict[str, re.Pattern] = {}
# # 		# for edge, pat in infer_rules_raw.items():
# # 		#     infer_rules[edge] = re.compile(pat, re.I)
# #
# # 		# frames_by_id = {fr["id"]: fr for fr in frames}
# # 		frames_by_id = {}
# # 		for fr in frames:
# # 			frames_by_id[fr["id"]] = FrameSpec(**fr)
# #
# # 		dp = DomainPack(
# # 			root=pack_dir,
# # 			manifest=manifest,
# # 			seeds=seeds,
# # 			frames=frames,
# # 			routing_rules=routing_rules,
# # 			edge_patterns_raw=edge_patterns_raw,
# # 			infer_rules_raw=infer_rules_raw,
# # 			junk_filters=junk_filters,
# # 			scoring=scoring,
# # 			edge_patterns=edge_patterns,
# # 			infer_rules=infer_rules,
# # 			frames_by_id=frames_by_id
# # 		)
# # 		dp.validate()
# # 		return dp
# #
# # 	def validate(self) -> None:
# # 		# minimal sanity checks
# # 		# if not isinstance(self.seeds, dict) or not self.seeds:
# # 		# 	raise ValueError("DomainPack: seeds.json must be a non-empty dict")
# # 		if not isinstance(self.seeds, dict):
# # 			self.seeds = {}
# #
# # 		if not isinstance(self.frames, list) or not self.frames:
# # 			raise ValueError("DomainPack: frames.json must be a non-empty list")
# #
# # 		for fr in self.frames:
# # 			for k in ["id", "name", "required_edges", "min_required_covered", "allowed_node_types", "blocked_node_types", "min_steps", "max_steps"]:
# # 				if k not in fr:
# # 					raise ValueError(f"DomainPack: frame missing key '{k}': {fr}")
# #
# # 		if not isinstance(self.edge_patterns_raw, dict):
# # 			raise ValueError("DomainPack: edge_patterns.json must be a dict")
# #
# # 		if not isinstance(self.infer_rules_raw, dict):
# # 			raise ValueError("DomainPack: infer_rules.json must be a dict")
# #
# # 		jf = self.junk_filters or {}
# # 		if "min_token_len" not in jf:
# # 			jf["min_token_len"] = 6
# # 		if "skip_prefixes" not in jf:
# # 			jf["skip_prefixes"] = ["keywords:"]
# #
# # 	def default_frame_id(self) -> str:
# # 		return self.scoring.get("default_frame_id", self.frames[0]["id"])
# #
# # 	def route_frame(self, question: str) -> Dict[str, Any]:
# # 		q = question.lower()
# # 		for rule in self.routing_rules:
# # 			fid = rule.get("frame_id")
# # 			kws = [k.lower() for k in rule.get("any_keywords", [])]
# # 			if fid and any(k in q for k in kws):
# # 				if fid in self.frames_by_id:
# # 					return self.frames_by_id[fid]
# # 		# fallback
# # 		return self.frames_by_id[self.default_frame_id()]
# #
# # 	def extract_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
# # 		edges = []
# # 		evidence = {}
# # 		for edge, pats in self.edge_patterns.items():
# # 			best = 0
# # 			for p in pats:
# # 				if p.regex.search(text):
# # 					best = max(best, p.strength)
# # 			if best > 0:
# # 				edges.append(edge)
# # 				evidence[edge] = best
# # 		return edges, evidence
# #
# # 	def infer_edges_with_strength(self, text: str) -> Tuple[List[str], Dict[str, int]]:
# # 		edges = []
# # 		evidence = {}
# # 		for edge, pats in self.infer_rules.items():
# # 			best = 0
# # 			for p in pats:
# # 				if p.regex.search(text):
# # 					best = max(best, p.strength)
# # 			if best > 0:
# # 				edges.append(edge)
# # 				evidence[edge] = best
# # 		return edges, evidence
# #
# # 	def infer_edges(self, text: str) -> List[str]:
# # 		edges, _ = self.infer_edges_with_strength(text)
# # 		return edges
# #
# # # def infer_edges(self, text: str) -> List[str]:
# # 	# 	out = []
# # 	# 	for edge, pat in self.infer_rules.items():
# # 	# 		if pat.search(text):
# # 	# 			out.append(edge)
# # 	# 	return out
# #
