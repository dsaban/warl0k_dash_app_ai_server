from dataclasses import dataclass

@dataclass
class DemoConfig:
    # PIM timing tolerance (seconds)
    max_skew_s: float = 2.0

    # Sliding window size for counters (kept simple here)
    window_size: int = 32

    # Nano-gate thresholds (0..1); higher => stricter
    near_threshold: float = 0.62
    far_threshold: float = 0.62

    # CSV DB path
    db_path: str = "db/demo_db.csv"

CFG = DemoConfig()
