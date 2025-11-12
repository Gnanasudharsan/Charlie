# ml_src/explainability.py
from __future__ import annotations
import os, yaml, joblib, mlflow, shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from ml_src.data_loader import DataPaths
from ml_src.model_train import prepare_data
from ml_src.utils.logging import get_logger

logger = get_logger("explainability")

def main():
    logger.info("🚀 Starting Model Sensitivity Analysis (SHAP + LIME)...")

    # --- Load data ---
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]
    X, y = prepare_data(df_pred)

    # --- Load trained model ---
    model_path = "models/best_model_rf.joblib"
    if not os.path.exists(model_path):
        model_path = "models/baseline_logreg.joblib"
    model = joblib.load(model_path)
    logger.info(f"Loaded model from: {model_path}")

    # --- Create reports folder ---
    os.makedirs("reports", exist_ok=True)

    # -------------------------------------------------------------
    # 🔹 SHAP GLOBAL EXPLANATION
    # -------------------------------------------------------------
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        shap_plot_path = "reports/shap_summary.png"
        plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to {shap_plot_path}")

        # Save mean absolute SHAP values for feature ranking
        shap_importance = pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": np.abs(shap_values.values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False)
        shap_csv_path = "reports/shap_importance.csv"
        shap_importance.to_csv(shap_csv_path, index=False)
        logger.info(f"SHAP importance CSV saved to {shap_csv_path}")
    except Exception as e:
        logger.warning(f"⚠️ SHAP explanation failed: {e}")
        shap_plot_path, shap_csv_path = None, None

    # -------------------------------------------------------------
    # 🔹 LIME LOCAL EXPLANATION
    # -------------------------------------------------------------
    try:
        explainer_lime = LimeTabularExplainer(
            training_data=np.array(X),
            feature_names=X.columns.tolist(),
            class_names=["No Delay", "Delay"],
            mode="classification"
        )

        sample_idx = np.random.randint(0, len(X))
        explanation = explainer_lime.explain_instance(
            data_row=X.iloc[sample_idx],
            predict_fn=model.predict_proba
        )
        lime_html_path = "reports/lime_explanation.html"
        explanation.save_to_file(lime_html_path)
        logger.info(f"LIME explanation saved to {lime_html_path}")
    except Exception as e:
        logger.warning(f"⚠️ LIME explanation failed: {e}")
        lime_html_path = None

    # -------------------------------------------------------------
    # 🔹 Log to MLflow
    # -------------------------------------------------------------
    try:
        with open("configs/config.yaml") as f:
            cfg = yaml.safe_load(f)
        mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
        mlflow.set_experiment(cfg["experiment"]["name"])

        with mlflow.start_run(run_name="model_explainability"):
            if shap_plot_path: mlflow.log_artifact(shap_plot_path)
            if shap_csv_path: mlflow.log_artifact(shap_csv_path)
            if lime_html_path: mlflow.log_artifact(lime_html_path)

        logger.info("✅ Model explainability results logged to MLflow.")
    except Exception as e:
        logger.warning(f"⚠️ MLflow logging failed: {e}")

    logger.info("🎯 Model Sensitivity Analysis completed successfully!")

if __name__ == "__main__":
    main()