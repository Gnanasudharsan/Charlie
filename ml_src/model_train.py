# ml_src/model_train.py
from __future__ import annotations
import pandas as pd, numpy as np, yaml, os, joblib, json
import mlflow, mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("train_model")


# -----------------------
# Feature Engineering
# -----------------------
def prepare_data(df: pd.DataFrame):
    # Convert arrival/departure to datetime
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="ignore")

    df = df.dropna(subset=["arrival_time", "departure_time"])
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)

    features = ["direction_id", "stop_sequence"]
    df = df.dropna(subset=features)

    X = df[features]
    y = df["delayed"]
    return X, y


# -----------------------
# Train + Save Model
# -----------------------
def main():
    # Load data via DVC/Airflow output
    paths = DataPaths("ml_configs/paths.yaml")
    try:
        dfs = paths.load_all()
        df_pred = dfs["predictions"]
    except Exception as e:
        logger.warning(f"⚠️ Processed data missing. Skipping training. {e}")
        return

    logger.info(f"Loaded predictions: {df_pred.shape}")

    # Prepare features
    X, y = prepare_data(df_pred)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load experiment config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # -----------------------
    # LightGBM Model
    # -----------------------
    with mlflow.start_run(run_name="baseline_lgbm"):
        model = LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # -----------------------
        # Save local model
        # -----------------------
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model_lgbm.joblib")

        logger.info(f"Metrics: {metrics}")
        logger.info("Saved model to models/model_lgbm.joblib")

        # -----------------------
        # Save reference stats for drift monitoring
        # -----------------------
        ref = {
            "direction_id": {"values": X_train["direction_id"].astype(float).tolist()},
            "stop_sequence": {"values": X_train["stop_sequence"].astype(float).tolist()},
        }

        with open("models/reference_stats.json", "w") as f:
            json.dump(ref, f)
        logger.info("Saved reference_stats.json for drift monitoring.")


if __name__ == "__main__":
    main()