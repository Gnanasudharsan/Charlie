# Model_Development/ml_src/bias_analysis.py

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import joblib

from fairlearn.metrics import MetricFrame
from sklearn.metrics import accuracy_score

from Model_Development.ml_src.model_train import prepare_data
from Model_Development.ml_src.data_loader import DataPaths
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("bias_analysis")


def main():

    # ------------------------------------------------
    # Create reports directory
    # ------------------------------------------------
    reports_dir = Path("Model_Development/reports")
    reports_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------
    # Locate model
    # ------------------------------------------------
    model_path_candidates = [
        "Model_Development/models/model_lgbm.joblib",
        "Model_Development/models/final_model.joblib",
        "Model_Development/models/best_logreg_tuned.joblib",
    ]

    model_path = next((p for p in model_path_candidates if os.path.exists(p)), None)

    if not model_path:
        logger.warning("⚠️ No model found. Skipping bias analysis.")
        return

    model = joblib.load(model_path)
    logger.info(f"📦 Loaded model from {model_path}")

    # ------------------------------------------------
    # Load processed predictions data
    # ------------------------------------------------
    try:
        paths = DataPaths("ml_configs/paths.yaml")
        df_pred = paths.load_all()["predictions"]
    except Exception as e:
        logger.warning(f"⚠️ Could not load processed predictions. Skipping bias analysis. Error: {e}")
        return

    # ------------------------------------------------
    # Prepare data
    # ------------------------------------------------
    try:
        X, y = prepare_data(df_pred)
    except Exception as e:
        logger.error(f"❌ prepare_data() failed: {e}")
        return

    if "direction_id" not in X.columns:
        logger.warning("⚠️ direction_id missing — cannot run fairness slicing.")
        return

    sensitive_feature = X["direction_id"]
    y_pred = model.predict(X)

    # ------------------------------------------------
    # Fairness analysis
    # ------------------------------------------------
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score},
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )

    bias_results = {
        "overall_accuracy": [accuracy_score(y, y_pred)],
        "difference_between_groups": [mf.difference()],
        "group_accuracy": [mf.by_group.to_dict()],
    }

    out_csv = reports_dir / "bias_report.csv"
    pd.DataFrame(bias_results).to_csv(out_csv, index=False)

    logger.info(f"📄 Bias report saved to {out_csv}")


if __name__ == "__main__":
    main()