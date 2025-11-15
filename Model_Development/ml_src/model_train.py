# Model_Development/ml_src/model_train.py

from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
import os
import joblib
import json
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from lightgbm import LGBMClassifier

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("train_model")


# ============================================================
# FEATURE ENGINEERING (CLEAN + READY FOR ALL SCRIPTS)
# ============================================================
def prepare_data(df: pd.DataFrame):
    df = df.copy()

    # Convert datetime
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Remove invalid rows
    df = df.dropna(subset=["arrival_time", "departure_time"])

    # Compute delay
    df.loc[:, "delay_minutes"] = (
        df["departure_time"] - df["arrival_time"]
    ).dt.total_seconds() / 60

    df.loc[:, "delayed"] = (df["delay_minutes"] > 5).astype(int)

    # Features
    features = ["direction_id", "stop_sequence"]
    df = df.dropna(subset=features)

    X = df[features]
    y = df["delayed"]

    # Return **3 values** for all downstream scripts:
    # df_clean, X, y
    return df, X, y


# ============================================================
# TRAIN MODEL
# ============================================================
def main():
    paths = DataPaths("ml_configs/paths.yaml")

    # -------------------------------
    # Load processed predictions
    # -------------------------------
    try:
        df_pred = paths.load_all()["predictions"]
    except Exception as e:
        logger.error(f"❌ Cannot load processed data. Error: {e}")
        return

    logger.info(f"📥 Loaded predictions: {df_pred.shape}")

    # -------------------------------
    # Feature engineering
    # -------------------------------
    df_clean, X, y = prepare_data(df_pred)

    if len(X) == 0:
        logger.error("❌ No valid rows after feature engineering.")
        return

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -------------------------------
    # MLflow setup
    # -------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    tracking_uri = cfg["experiment"]["tracking_uri"]
    experiment_name = cfg["experiment"]["name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # -------------------------------
    # TRAIN BASELINE MODEL (LGBM)
    # -------------------------------
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

        logger.info(f"📊 Model Metrics: {metrics}")

        # Log to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "baseline_lgbm_model")

        # -------------------------------
        # SAVE MODEL LOCALLY
        # -------------------------------
        model_dir = Path("Model_Development/models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "model_lgbm.joblib"
        joblib.dump(model, model_path)

        logger.info(f"💾 Saved LightGBM model → {model_path}")

        # -------------------------------
        # SAVE REFERENCE DISTRIBUTIONS (FOR DRIFT)
        # -------------------------------
        reference_stats = {}
        drift_features = ["direction_id", "stop_sequence", "delay_minutes", "delayed"]

        for col in drift_features:
            if col in df_clean.columns:
                reference_stats[col] = df_clean[col].dropna().astype(float).tolist()

        ref_path = model_dir / "reference_stats.json"
        with open(ref_path, "w") as f:
            json.dump(reference_stats, f, indent=4)

        logger.info(f"📁 Saved drift reference stats → {ref_path}")


if __name__ == "__main__":
    main()