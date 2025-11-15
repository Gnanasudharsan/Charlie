# ml_src/data_loader.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import yaml

class DataPaths:
    def __init__(self, cfg_path: str = "ml_configs/paths.yaml"):
        self.cfg_path = cfg_path
        with open(cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.proc_dir = Path(self.cfg["data"]["processed_dir"])
        self.predictions = Path(self.cfg["data"]["predictions"])
        self.vehicles = Path(self.cfg["data"]["vehicles"])
        self.alerts = Path(self.cfg["data"]["alerts"])
        self.required = {
            "predictions": self.predictions,
            "vehicles": self.vehicles,
            "alerts": self.alerts,
        }
        self.min_rows = self.cfg.get("validation", {}).get("min_rows", {})

    def check_exists(self):
        missing = [k for k, p in self.required.items() if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing processed outputs for: {missing}. "
                f"Expected under {self.proc_dir}. Did you run DVC/Airflow? "
                f"Try: `cd Data_Pipeline && dvc pull`."
            )

    def load_all(self) -> dict[str, pd.DataFrame]:
        self.check_exists()
        dfs = {
            "predictions": pd.read_csv(self.predictions),
            "vehicles": pd.read_csv(self.vehicles),
            "alerts": pd.read_csv(self.alerts),
        }
        return dfs

def quick_sanity(dfs: dict[str, pd.DataFrame], min_rows: dict | None = None) -> dict:
    """Lightweight contract checks: non-empty, min rows, no duplicate columns."""
    report = {}
    min_rows = min_rows or {}
    for name, df in dfs.items():
        checks = {}
        checks["shape"] = df.shape
        checks["non_empty"] = df.shape[0] > 0 and df.shape[1] > 0
        if name in min_rows:
            checks["min_rows_ok"] = df.shape[0] >= int(min_rows[name])
        else:
            checks["min_rows_ok"] = True
        checks["duplicate_columns"] = df.columns.duplicated().any()
        checks["null_ratio_by_col"] = (df.isna().mean().round(4)).to_dict()
        report[name] = checks
    return report

if __name__ == "__main__":
    paths = DataPaths()
    dfs = paths.load_all()
    rep = quick_sanity(dfs, paths.min_rows)
    print("Sanity report:")
    for k, v in rep.items():
        print(f"- {k}: {v['shape']}, non_empty={v['non_empty']}, min_rows_ok={v['min_rows_ok']}, dup_cols={v['duplicate_columns']}")