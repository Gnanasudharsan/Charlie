# Model_Development/ml_src/explainability.py
from __future__ import annotations
import os
import yaml
import joblib
import mlflow
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer

from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("explainability")


def load_best_model() -> str:
    """
    Returns the first available model in priority order.
    Ensures explainability always has a model to analyze.
    """

    candidates = [
        "Model_Development/models/final_model.joblib",
        "Model_Development/models/model_lgbm.joblib",
        "Model_Development/models/best_logreg_tuned.joblib",
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"❌ No trained model found in Model_Development/models/. "
        f"Expected one of: {candidates}"
    )


def main():
    logger.info("🚀 Starting Model Sensitivity Analysis (SHAP + LIME)...")

    # -----------------------------------------------------------
    # Load data
    # -----------------------------------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    X, y = prepare_data(df_pred)

    # -----------------------------------------------------------
    # Load best model
    # -----------------------------------------------------------
    model_path = load_best_model()
    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model from: {model_path}")

    # -----------------------------------------------------------
    # Ensure reports folder exists
    # -----------------------------------------------------------
    os.makedirs("Model_Development/reports", exist_ok=True)

    # -----------------------------------------------------------
    # ⚡ SHAP — Global Explanation
    # -----------------------------------------------------------
    shap_plot_path = "Model_Development/reports/shap_summary.png"
    shap_csv_path = "Model_Development/reports/shap_importance.csv"

    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary saved → {shap_plot_path}")

        # Save SHAP importance ranking
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(shap_csv_path, index=False)
        logger.info(f"SHAP importance CSV saved → {shap_csv_path}")

    except Exception as e:
        logger.warning(f"⚠️ SHAP failed: {e}")

    # -----------------------------------------------------------
    # ⚡ LIME — Local Explanation
    # -----------------------------------------------------------
    try:
        explainer_lime = LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=X.columns.tolist(),
            class_names=["No Delay", "Delay"],
            mode="classification",
        )

        idx = np.random.randint(0, len(X))
        explanation = explainer_lime.explain_instance(
            data_row=X.iloc[idx],
            predict_fn=model.predict_proba
        )

        lime_html_path = "Model_Development/reports/lime_explanation.html"
        explanation.save_to_file(lime_html_path)

        logger.info(f"LIME explanation saved → {lime_html_path}")

    except Exception as e:
        logger.warning(f"⚠️ LIME failed: {e}")

    # -----------------------------------------------------------
    # Log to MLflow
    # -----------------------------------------------------------
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)

        mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
        mlflow.set_experiment(cfg["experiment"]["name"])

        with mlflow.start_run(run_name="model_explainability"):
            if os.path.exists(shap_plot_path): mlflow.log_artifact(shap_plot_path)
            if os.path.exists(shap_csv_path): mlflow.log_artifact(shap_csv_path)
            if os.path.exists(lime_html_path): mlflow.log_artifact(lime_html_path)

        logger.info("📡 Explainability artifacts logged to MLflow.")

    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("🎯 Model Explainability completed successfully!")


if __name__ == "__main__":
    main()