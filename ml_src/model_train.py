# ml_src/model_train.py
from __future__ import annotations
import pandas as pd, numpy as np, yaml, os, joblib, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from ml_src.data_loader import DataPaths, quick_sanity
from ml_src.utils.logging import get_logger

logger = get_logger("train_model")


def prepare_data(df: pd.DataFrame):
    """Prepare input features and labels for training."""
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["arrival_time", "departure_time"])

    # Compute delay and label
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)

    # Select features
    features = ["direction_id", "stop_sequence"]
    df = df.dropna(subset=features)

    X = df[features]
    y = df["delayed"]
    return X, y


def main():
    """Train and log baseline Logistic Regression model."""
    logger.info("🚀 Starting model training...")

    # Initialize paths
    paths = DataPaths("ml_configs/paths.yaml")

    # Handle missing processed data gracefully (for CI/CD environments)
    try:
        dfs = paths.load_all()
    except FileNotFoundError as e:
        logger.warning("⚠️ No processed data found. Skipping training for CI environment.")
        print(str(e))
        import sys
        sys.exit(0)

    # Load processed predictions dataset
    df_pred = dfs["predictions"]
    logger.info(f"✅ Loaded predictions data: {df_pred.shape}")

    # Prepare features and target
    X, y = prepare_data(df_pred)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"📊 Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # Read MLflow config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # Train and log model
    with mlflow.start_run(run_name="baseline_logreg"):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Evaluation metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        # Log parameters and metrics to MLflow
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Save locally
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/baseline_logreg.joblib")

        # Log outputs
        logger.info(f"🏁 Training completed. Metrics: {metrics}")
        logger.info("💾 Model saved to models/baseline_logreg.joblib")


if __name__ == "__main__":
    main()