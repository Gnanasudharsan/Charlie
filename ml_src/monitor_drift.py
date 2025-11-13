# ml_src/monitor_drift.py
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd
from pathlib import Path

REF_STATS = "models/reference_stats.json"
CUR_DATA  = "Data_Pipeline/data/processed/predictions.csv"
REPORT    = "reports/drift_report.json"

FEATURES = ["direction_id", "stop_sequence"]  # same as training

def _psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (numeric only)."""
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    # same bin edges computed on reference
    edges = np.linspace(np.nanmin(ref), np.nanmax(ref), bins + 1)
    # avoid degenerate edges
    if not np.isfinite(edges).all() or np.nanmin(ref) == np.nanmax(ref):
        return 0.0
    r_hist, _ = np.histogram(ref, bins=edges)
    c_hist, _ = np.histogram(cur, bins=edges)
    r_pct = np.clip(r_hist / max(r_hist.sum(), 1), 1e-6, 1)
    c_pct = np.clip(c_hist / max(c_hist.sum(), 1), 1e-6, 1)
    return float(np.sum((c_pct - r_pct) * np.log(c_pct / r_pct)))

def _load_reference():
    if not os.path.exists(REF_STATS):
        return None
    with open(REF_STATS) as f:
        return json.load(f)

def main():
    Path("reports").mkdir(exist_ok=True)

    ref = _load_reference()
    if ref is None:
        print("ℹ️ No reference_stats.json found. Skipping drift.")
        return

    if not os.path.exists(CUR_DATA):
        print("ℹ️ No current batch data found. Skipping drift.")
        return

    cur_df = pd.read_csv(CUR_DATA)
    results = {"per_feature_psi": {}, "overall_flag": False}

    for feat in FEATURES:
        if feat not in cur_df.columns or feat not in ref:
            continue
        cur_vals = cur_df[feat].dropna().to_numpy()
        ref_vals = np.array(ref[feat]["values"], dtype=float)
        psi = _psi(ref_vals, cur_vals, bins=10)
        results["per_feature_psi"][feat] = psi

    # simple flag: PSI > 0.25 => major drift, > 0.1 => moderate
    thresh_major, thresh_mod = 0.25, 0.10
    flags = {
        f: ("major" if v >= thresh_major else "moderate" if v >= thresh_mod else "ok")
        for f, v in results["per_feature_psi"].items()
        if np.isfinite(v)
    }
    results["flags"] = flags
    results["overall_flag"] = any(v in ("moderate", "major") for v in flags.values())

    with open(REPORT, "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Drift report written to", REPORT, results)

if __name__ == "__main__":
    main()