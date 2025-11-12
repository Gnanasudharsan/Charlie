from __future__ import annotations
import pandas as pd, numpy as np, yaml, os, joblib, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("train_model")

def prepare_data(df: pd.DataFrame):
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["arrival_time", "departure_time"])
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)
    features = ["direction_id", "stop_sequence"]
    df = df.dropna(subset=features)
    X = df[features]
    y = df["delayed"]
    return X, y

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_prob)
    }
    return metrics

def main():
    # -------------------- Load data --------------------
    paths = DataPaths("ml_configs/paths.yaml")
    dfs = paths.load_all()
    df_pred = dfs["predictions"]
    logger.info(f"Loaded predictions data: {df_pred.shape}")

    X, y = prepare_data(df_pred)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # -------------------- Setup MLflow --------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # -------------------- Define candidate models --------------------
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "LightGBM": LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    }

    results = {}
    best_model_name, best_score = None, -np.inf
    best_model = None

    # -------------------- Train & evaluate each --------------------
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_val, y_val)

            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")

            logger.info(f"{name} metrics: {metrics}")
            results[name] = metrics

            if metrics["roc_auc"] > best_score:
                best_score = metrics["roc_auc"]
                best_model_name = name
                best_model = model

    # -------------------- Save best model --------------------
    os.makedirs("models", exist_ok=True)
    best_model_path = f"models/best_model_{best_model_name.lower()}.joblib"
    joblib.dump(best_model, best_model_path)

    logger.info(f"✅ Best model: {best_model_name} (ROC_AUC={best_score:.4f})")
    logger.info(f"Model saved to {best_model_path}")

    # -------------------- Log comparison summary --------------------
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv("reports/model_comparison.csv", index=True)
    mlflow.log_artifact("reports/model_comparison.csv")

if __name__ == "__main__":
    main()