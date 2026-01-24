# core/config.py

class AppConfig:
    def __init__(self):
        # Chunking
        self.chunk_size = 1200
        self.chunk_overlap = 220

        # Embeddings
        self.hash_dim = 512
        self.mem_dim = 128  # projection dim for router/ebm/memory

        # Retrieval
        self.top_k = 5
        self.alpha_bm25 = 0.45
        self.alpha_cos = 0.25
        self.alpha_cov = 0.30

        # Gates / thresholds
        self.min_intent_cov = 0.18        # qtype concept coverage
        self.min_role_align = 1.0         # target role must dominate
        self.min_schema_coverage = 0.67   # fraction of schema slots filled
        self.max_reretries = 1            # re-retrieve with anchors

        # MoE / EBM
        self.n_experts = 4
        self.expert_hidden = 64
        self.n_candidates = 3

        # Integrity scoring
        self.energy_good_max = 1.25
        self.energy_bad_min = 2.0
        self.router_conf_min = 0.35

        # Memory
        self.mem_steps_max = 180
