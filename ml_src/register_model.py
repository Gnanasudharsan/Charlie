# ml_src/register_model.py
from __future__ import annotations
import os, json, joblib, yaml, time
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path

def _safe_read_yaml(path: str):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return yaml.safe_load(f)

def _infer_signature_example():
    # minimal feature frame used for MLflow signature if data is absent
    return pd.DataFrame({"direction_id": [0], "stop_sequence": [1]})

def main():
    # ---- config & experiment ----
    cfg = _safe_read_yaml("configs/config.yaml")
    tracking_uri = cfg.get("experiment", {}).get("tracking_uri", "file:./mlruns")
    exp_name    = cfg.get("experiment", {}).get("name", "charlie-mbta")
    model_name  = cfg.get("model_registry", {}).get("name", "charlie_mbta_classifier")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    Path("reports").mkdir(exist_ok=True)

    # ---- pick a model artifact to register ----
    candidates = [
        "models/best_model.joblib",
        "models/baseline_logreg.joblib",
        "models/model_lgbm.joblib",
    ]
    model_path = next((p for p in candidates if os.path.exists(p)), None)
    if not model_path:
        print("⚠️ No local model found to register. Skipping.")
        return

    # best effort signature example
    sig_df = _infer_signature_example()

    # ---- log as an MLflow run (if not already logged) ----
    with mlflow.start_run(run_name="register_model"):
        mlflow.log_artifact(model_path, artifact_path="local_model_copy")
        model = joblib.load(model_path)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=sig_df,
        )
        run_id = mlflow.active_run().info.run_id

    # ---- register the model from that run's artifact ----
    logged_model_uri = f"runs:/{run_id}/model"
    reg = mlflow.register_model(model_uri=logged_model_uri, name=model_name)

    # transition to STAGING (best-effort)
    client = mlflow.tracking.MlflowClient()
    # wait a moment for registry backend
    time.sleep(2)
    client.transition_model_version_stage(
        name=model_name,
        version=reg.version,
        stage="Staging",
        archive_existing_versions=False,
    )

    # write a small report
    info = {
        "tracking_uri": tracking_uri,
        "experiment": exp_name,
        "model_name": model_name,
        "registered_version": reg.version,
        "run_id": run_id,
    }
    with open("reports/model_registry_report.json", "w") as f:
        json.dump(info, f, indent=2)
    print("✅ Registered model:", info)

if __name__ == "__main__":
    main()