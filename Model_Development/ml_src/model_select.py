# Model_Development/ml_src/model_select.py

import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("model_select")


def load_model_safe(path):
    """Load a model safely; return None if missing."""
    if not os.path.exists(path):
        logger.warning(f"⚠️ Model not found: {path}")
        return None
    return joblib.load(path)


def evaluate_model(model, X, y):
    """Evaluate and return key metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_prob),
    }


def main():
    logger.info("🚀 Starting Model Comparison & Final Model Selection...")

    # ----------------------------------------------------------
    # 1️⃣ Load processed data
    # ----------------------------------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    df_eng, X, y = prepare_data(df_pred)

    # ----------------------------------------------------------
    # 2️⃣ Load available models from /models/
    # ----------------------------------------------------------
    model_dir = Path("models")
    if not model_dir.exists():
        raise FileNotFoundError("❌ models/ directory missing.")

    model_candidates = {
        "baseline_lgbm": model_dir / "model_lgbm.joblib",
        "logreg_tuned": model_dir / "logreg_tuned.joblib",
        "final_model_existing": model_dir / "final_model.joblib",
    }

    loaded_models = {
        name: load_model_safe(str(path))
        for name, path in model_candidates.items()
        if load_model_safe(str(path)) is not None
    }

    if not loaded_models:
        raise RuntimeError("❌ No models available for comparison.")

    logger.info(f"📦 Models found: {list(loaded_models.keys())}")

    # ----------------------------------------------------------
    # 3️⃣ Evaluate all models
    # ----------------------------------------------------------
    results = {}
    for name, model in loaded_models.items():
        metrics = evaluate_model(model, X, y)
        results[name] = metrics
        logger.info(f"📊 {name} → {metrics}")

    # ----------------------------------------------------------
    # 4️⃣ Select best model (highest ROC-AUC)
    # ----------------------------------------------------------
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = loaded_models[best_model_name]

    logger.info(f"🏆 Best model selected: {best_model_name}")

    # ----------------------------------------------------------
    # 5️⃣ Save new FINAL model
    # ----------------------------------------------------------
    final_model_path = model_dir / "final_model.joblib"
    joblib.dump(best_model, final_model_path)
    logger.info(f"💾 Final model saved → {final_model_path}")

    # ----------------------------------------------------------
    # 6️⃣ Save JSON comparison report
    # ----------------------------------------------------------
    report_dir = Path("Model_Development/reports")
    report_dir.mkdir(exist_ok=True)

    comparison_json = report_dir / "model_comparison.json"
    with open(comparison_json, "w") as f:
        json.dump(
            {"results": results, "best_model": best_model_name},
            f,
            indent=4,
        )

    logger.info(f"📄 Model comparison report saved → {comparison_json}")

    # ----------------------------------------------------------
    # 7️⃣ Plot comparison
    # ----------------------------------------------------------
    plt.figure(figsize=(8, 5))
    model_names = list(results.keys())
    auc_scores = [results[m]["roc_auc"] for m in model_names]

    plt.bar(model_names, auc_scores)
    plt.title("Model Comparison (ROC-AUC)")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=20)
    plt.tight_layout()

    comparison_plot = report_dir / "model_comparison.png"
    plt.savefig(comparison_plot, dpi=300)
    plt.close()

    logger.info(f"📊 Model comparison plot saved → {comparison_plot}")

    # ----------------------------------------------------------
    # 8️⃣ Log to MLflow
    # ----------------------------------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    try:
        with mlflow.start_run(run_name="model_selection"):

            for name, metrics in results.items():
                mlflow.log_metric(f"{name}_auc", metrics["roc_auc"])
                mlflow.log_metric(f"{name}_accuracy", metrics["accuracy"])
                mlflow.log_metric(f"{name}_f1", metrics["f1"])

            mlflow.log_param("selected_model", best_model_name)

            mlflow.log_artifact(str(comparison_json))
            mlflow.log_artifact(str(comparison_plot))

            mlflow.sklearn.log_model(best_model, "final_model")

        logger.info("📌 Model selection results logged to MLflow")

    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("🎯 Model Selection Completed Successfully!")


if __name__ == "__main__":
    main()