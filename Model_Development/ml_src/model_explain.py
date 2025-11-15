# Model_Development/ml_src/model_explain.py

import os
import yaml
import joblib
import shap
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from pathlib import Path

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("model_explain")


def explain_model():

    # -----------------------------------------
    # Load MLflow Config
    # -----------------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # -----------------------------------------
    # Load Processed Predictions
    # -----------------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    df_eng, X, y = prepare_data(df_pred)

    # -----------------------------------------
    # Pick Best Available Model
    # -----------------------------------------
    model_candidates = [
        "models/final_model.joblib",
        "models/model_lgbm.joblib",
        "models/logreg_tuned.joblib",
        "models/best_logreg_tuned.joblib"
    ]

    model_path = next((m for m in model_candidates if os.path.exists(m)), None)
    if not model_path:
        raise FileNotFoundError("❌ No model found in models/ directory.")

    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model → {model_path}")

    # -----------------------------------------
    # Ensure reports directory
    # -----------------------------------------
    reports_dir = Path("Model_Development/reports")
    reports_dir.mkdir(exist_ok=True)

    shap_plot_path = reports_dir / "shap_summary.png"
    shap_csv_path = reports_dir / "shap_importance.csv"
    lime_html_path = reports_dir / "lime_explanation.html"

    # ======================================================
    # 1️⃣ SHAP — Global Feature Importance
    # ======================================================
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary plot saved → {shap_plot_path}")

        # SHAP Importance table
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(shap_csv_path, index=False)
        logger.info(f"SHAP importance CSV saved → {shap_csv_path}")

    except Exception as e:
        logger.warning(f"⚠️ SHAP failed: {e}")
        shap_plot_path = None
        shap_csv_path = None

    # ======================================================
    # 2️⃣ LIME — Local Explanation
    # ======================================================
    try:
        explainer_lime = LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=["OnTime", "Delayed"],
            mode="classification"
        )

        idx = np.random.randint(0, len(X))
        explanation = explainer_lime.explain_instance(
            X.iloc[idx].values,
            model.predict_proba
        )

        explanation.save_to_file(str(lime_html_path))
        logger.info(f"LIME explanation saved → {lime_html_path}")

    except Exception as e:
        logger.warning(f"⚠️ LIME failed: {e}")
        lime_html_path = None

    # ======================================================
    # 3️⃣ Log to MLflow
    # ======================================================
    try:
        with mlflow.start_run(run_name="model_explainability"):
            if shap_plot_path and shap_plot_path.exists():
                mlflow.log_artifact(str(shap_plot_path))
            if shap_csv_path and shap_csv_path.exists():
                mlflow.log_artifact(str(shap_csv_path))
            if lime_html_path and lime_html_path.exists():
                mlflow.log_artifact(str(lime_html_path))

        logger.info("📡 Logged explainability artifacts to MLflow.")

    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("✅ Model Explainability Completed!")


if __name__ == "__main__":
    explain_model()