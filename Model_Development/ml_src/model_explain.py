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

    # -------------------------------
    # Load config + MLflow settings
    # -------------------------------
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # -------------------------------
    # Load processed predictions
    # -------------------------------
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    # Prepare data exactly as model expects
    X, y, *_ = prepare_data(df_pred)

    # -------------------------------
    # Pick best available model
    # -------------------------------
    candidate_models = [
        "Model_Development/models/final_model.joblib",
        "Model_Development/models/model_lgbm.joblib",
        "Model_Development/models/best_logreg_tuned.joblib",
    ]

    model_path = next((m for m in candidate_models if os.path.exists(m)), None)
    if not model_path:
        raise FileNotFoundError("❌ No model found in Model_Development/models/")

    model = joblib.load(model_path)
    logger.info(f"Loaded model → {model_path}")

    # -------------------------------
    # Create reports folder
    # -------------------------------
    reports_dir = Path("Model_Development/reports")
    reports_dir.mkdir(exist_ok=True)

    # ==============================================================
    # 1️⃣ SHAP GLOBAL EXPLANATION
    # ==============================================================
    shap_plot_path = reports_dir / "shap_summary.png"
    shap_csv_path = reports_dir / "shap_importance.csv"

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP summary plot saved → {shap_plot_path}")

        # Compute importance values
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

    # ==============================================================
    # 2️⃣ LIME LOCAL EXPLANATION
    # ==============================================================
    lime_path = reports_dir / "lime_explanation.html"

    try:
        explainer_lime = LimeTabularExplainer(
            training_data=X.values,
            feature_names=X.columns.tolist(),
            class_names=["OnTime", "Delayed"],
            mode="classification"
        )

        idx = np.random.randint(0, len(X))
        explanation = explainer_lime.explain_instance(
            data_row=X.iloc[idx].values,
            predict_fn=model.predict_proba
        )

        explanation.save_to_file(str(lime_path))
        logger.info(f"LIME explanation saved → {lime_path}")

    except Exception as e:
        logger.warning(f"⚠️ LIME failed: {e}")
        lime_path = None

    # ==============================================================
    # 3️⃣ Log to MLflow
    # ==============================================================
    try:
        with mlflow.start_run(run_name="model_explainability"):

            if shap_plot_path:
                mlflow.log_artifact(str(shap_plot_path))
            if shap_csv_path:
                mlflow.log_artifact(str(shap_csv_path))
            if lime_path:
                mlflow.log_artifact(str(lime_path))

            logger.info("🎯 Explainability artifacts logged to MLflow.")

    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("✅ Model Explainability Completed!")


if __name__ == "__main__":
    explain_model()