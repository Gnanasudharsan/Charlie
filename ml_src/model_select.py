# ml_src/model_select.py

import os
import json
import joblib
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ml_src.data_loader import DataPaths
from ml_src.model_train import prepare_data
from ml_src.utils.logging import get_logger

logger = get_logger("model_select")


def load_model_safe(path):
    """Load model safely, return None if missing."""
    if not os.path.exists(path):
        logger.warning(f"⚠️ Model not found: {path}")
        return None
    return joblib.load(path)


def evaluate_model(model, X, y):
    """Return metrics for the model."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
    }


def main():
    logger.info("🚀 Starting model comparison & final model selection...")

    # ------------------------------
    # Load data
    # ------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]
    X, y = prepare_data(df_pred)

    # ------------------------------
    # Load models
    # ------------------------------
    models_to_compare = {
        "baseline_lgbm": "models/model_lgbm.joblib",
        "logreg_tuned": "models/logreg_tuned.joblib",
        "baseline_logreg": "models/baseline_logreg.joblib",
    }

    loaded_models = {}
    for name, path in models_to_compare.items():
        model = load_model_safe(path)
        if model:
            loaded_models[name] = model

    if not loaded_models:
        logger.error("❌ No models found to compare.")
        return

    # ------------------------------
    # Evaluate models
    # ------------------------------
    results = {}
    for name, model in loaded_models.items():
        metrics = evaluate_model(model, X, y)
        results[name] = metrics
        logger.info(f"📊 {name} → {metrics}")

    # ------------------------------
    # Select best model (highest AUC)
    # ------------------------------
    best_model_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = loaded_models[best_model_name]
    logger.info(f"🏆 Best Model Selected: {best_model_name}")

    # ------------------------------
    # Save final model
    # ------------------------------
    final_path = "models/final_model.joblib"
    joblib.dump(best_model, final_path)
    logger.info(f"💾 Final model saved to {final_path}")

    # ------------------------------
    # Save comparison report
    # ------------------------------
    os.makedirs("reports", exist_ok=True)
    comparison_path = "reports/model_comparison.json"
    with open(comparison_path, "w") as f:
        json.dump({"results": results, "best_model": best_model_name}, f, indent=4)

    logger.info(f"📑 Model comparison saved to {comparison_path}")

    # ------------------------------
    # Save comparison PLOT
    # ------------------------------
    plt.figure(figsize=(8, 5))
    model_names = list(results.keys())
    auc_scores = [results[m]["roc_auc"] for m in model_names]

    plt.bar(model_names, auc_scores)
    plt.title("Model Comparison (AUC Score)")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=20)

    plot_path = "reports/model_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"📊 Model comparison plot saved to {plot_path}")

    # ------------------------------
    # Log to MLflow
    # ------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run(run_name="model_selection"):
        mlflow.log_metrics({f"{m}_auc": results[m]["roc_auc"] for m in results})
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_artifact(comparison_path)
        mlflow.log_artifact(plot_path)
        mlflow.sklearn.log_model(best_model, "final_model")

    logger.info("✅ Model selection completed and logged to MLflow.")


if __name__ == "__main__":
    main()