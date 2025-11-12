# ml_src/bias_analysis.py
from __future__ import annotations
import os
import yaml
import pandas as pd
import numpy as np
import joblib
import mlflow
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ml_src.data_loader import DataPaths
from ml_src.model_train import prepare_data
from ml_src.utils.logging import get_logger
import plotly.express as px

logger = get_logger("bias_analysis")

def main():
    # --- Load data ---
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]
    X, y = prepare_data(df_pred)

    # --- Load model ---
    model_path = "models/best_model_rf.joblib"
    if not os.path.exists(model_path):
        model_path = "models/baseline_logreg.joblib"
    model = joblib.load(model_path)

    # --- Choose slices ---
    if "route_id" in df_pred.columns:
        sensitive_feature = df_pred.loc[X.index, "route_id"]
    elif "direction_id" in df_pred.columns:
        sensitive_feature = df_pred.loc[X.index, "direction_id"]
    else:
        raise ValueError("No route_id or direction_id column found for slicing.")

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # --- Fairlearn MetricFrame ---
    metrics = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "selection_rate": selection_rate,
    }
    frame = MetricFrame(metrics=metrics, y_true=y, y_pred=y_pred, sensitive_features=sensitive_feature)

    # --- Aggregate and group summaries ---
    df_report = frame.by_group
    df_report.reset_index(inplace=True)
    df_report.rename(columns={"index": "group"}, inplace=True)

    logger.info(f"\nBias report summary:\n{df_report.head()}")

    os.makedirs("reports", exist_ok=True)
    report_path = "reports/bias_report.csv"
    df_report.to_csv(report_path, index=False)

    # --- Plot visualization ---
    fig = px.bar(
        df_report,
        x="route_id" if "route_id" in df_report.columns else "direction_id",
        y="accuracy",
        title="Accuracy by Group (Fairness Slice)"
    )
    fig_path = "reports/bias_plot.html"
    fig.write_html(fig_path)

    # --- Log to MLflow ---
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run(run_name="bias_analysis"):
        mlflow.log_artifact(report_path)
        mlflow.log_artifact(fig_path)
        mlflow.log_metric("accuracy_gap", frame.difference(method="between_groups")["accuracy"])
        mlflow.log_metric("selection_rate_gap", frame.difference(method="between_groups")["selection_rate"])
        mlflow.log_param("sensitive_feature", "route_id" if "route_id" in df_pred.columns else "direction_id")

    logger.info("✅ Bias analysis completed and logged to MLflow.")

if __name__ == "__main__":
    main()