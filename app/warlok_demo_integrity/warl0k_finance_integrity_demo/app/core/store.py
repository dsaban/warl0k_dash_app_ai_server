from dataclasses import dataclass
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

@dataclass
class DataStore:
    vendors: pd.DataFrame
    contracts: pd.DataFrame
    invoices: pd.DataFrame
    approvals: pd.DataFrame
    close_tasks: pd.DataFrame
    policies: pd.DataFrame

    @staticmethod
    def load():
        return DataStore(
            vendors=pd.read_csv(DATA_DIR/"vendors.csv"),
            contracts=pd.read_csv(DATA_DIR/"contracts.csv"),
            invoices=pd.read_csv(DATA_DIR/"invoices.csv"),
            approvals=pd.read_csv(DATA_DIR/"approvals.csv"),
            close_tasks=pd.read_csv(DATA_DIR/"close_tasks.csv"),
            policies=pd.read_csv(DATA_DIR/"policies.csv"),
        )

    def save_table(self, name: str, df: pd.DataFrame) -> None:
        (DATA_DIR / f"{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
