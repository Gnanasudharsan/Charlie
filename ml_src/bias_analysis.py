import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.metrics import MetricFrame, selection_rate, accuracy_score_group_min, accuracy_score_group_max
from sklearn.linear_model import LogisticRegression
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("bias_analysis")

def main():
    # Load processed predictions data
    paths = DataPaths("ml_configs/paths.yaml")
    dfs = paths.load_all()
    df = dfs["predictions"]

    # Choose a sensitive feature
    sensitive_feature = "direction_id"  # example categorical slice
    df = df.dropna(subset=[sensitive_feature, "arrival_time", "departure_time"])
    df["delay_minutes"] = (
        pd.to_datetime(df["departure_time"]) - pd.to_datetime(df["arrival_time"])
    ).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)
    X = df[["stop_sequence"]]
    y = df["delayed"]
    groups = df[sensitive_feature]

    # Baseline model
    base_model = LogisticRegression(max_iter=500)
    base_model.fit(X, y)
    preds = base_model.predict(X)

    # Compute group metrics
    mf = MetricFrame(
        metrics={"accuracy": accuracy_score_group_min},
        y_true=y,
        y_pred=preds,
        sensitive_features=groups
    )
    fairness_df = pd.DataFrame({
        "Group": mf.by_group.index,
        "Accuracy": mf.by_group.values
    })

    # Mitigation via Fairlearn
    mitigator = ExponentiatedGradient(
        estimator=LogisticRegression(max_iter=500),
        constraints=DemographicParity()
    )
    mitigator.fit(X, y, sensitive_features=groups)
    mitigated_preds = mitigator.predict(X)

    # Post-mitigation metrics
    post_mf = MetricFrame(
        metrics={"accuracy": accuracy_score_group_min},
        y_true=y,
        y_pred=mitigated_preds,
        sensitive_features=groups
    )
    fairness_df["Accuracy_After"] = post_mf.by_group.values
    fairness_df["Improvement"] = fairness_df["Accuracy_After"] - fairness_df["Accuracy"]

    os.makedirs("reports", exist_ok=True)
    fairness_df.to_csv("reports/fairness_report.csv", index=False)

    # Plot improvement
    plt.figure(figsize=(7, 4))
    fairness_df.plot(
        x="Group",
        y=["Accuracy", "Accuracy_After"],
        kind="bar",
        title="Group-wise Fairness Before vs After Mitigation"
    )
    plt.tight_layout()
    plt.savefig("reports/fairness_summary.png")
    plt.close()

    logger.info("✅ Fairness mitigation complete. Reports saved under 'reports/'.")

if __name__ == "__main__":
    main()