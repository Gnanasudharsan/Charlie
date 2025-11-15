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


# ============================================================
# Feature Engineering
# ============================================================
def prepare_data(df: pd.DataFrame):
    # Convert to datetime
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["arrival_time", "departure_time"])

    # Compute delay
    df["delay_minutes"] = (
        df["departure_time"] - df["arrival_time"]
    ).dt.total_seconds() / 60

    df["delayed"] = (df["delay_minutes"] > 5).astype(int)

    # Features
    features = ["direction_id", "stop_sequence"]
    df = df.dropna(subset=features)

    X = df[features]
    y = df["delayed"]
    return df, X, y


# ============================================================
# Train + Save Model
# ============================================================
def main():
    # Load predictions from processed data
    paths = DataPaths("ml_configs/paths.yaml")
    try:
        dfs = paths.load_all()
        df_pred = dfs["predictions"]
    except Exception as e:
        logger.warning(f"⚠️ Processed data missing. Skipping training. {e}")
        return

    logger.info(f"Loaded predictions: {df_pred.shape}")

    # Prepare engineered dataset
    df_eng, X, y = prepare_data(df_pred)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Load MLflow config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # ============================================================
    # Train LightGBM model
    # ============================================================
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
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        # Log parameters & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Save MLflow model
        mlflow.sklearn.log_model(model, artifact_path="model")

        # ========================================================
        # Save trained model locally
        # ========================================================
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, "models/model_lgbm.joblib")
        logger.info("Saved model_lgbm.joblib")

        logger.info(f"Metrics: {metrics}")

        # ========================================================
        # Save reference distributions for drift monitoring (FIXED)
        # ========================================================
        reference_stats = {}

        for feature in ["direction_id", "stop_sequence", "delay_minutes", "delayed"]:
            if feature in df_eng.columns:
                reference_stats[feature] = (
                    df_eng[feature]
                    .dropna()
                    .astype(float)
                    .tolist()
                )

        # Write updated stats
        with open("models/reference_stats.json", "w") as f:
            json.dump(reference_stats, f, indent=4)

        logger.info("Saved updated reference_stats.json for drift monitoring.")


if __name__ == "__main__":
    main()