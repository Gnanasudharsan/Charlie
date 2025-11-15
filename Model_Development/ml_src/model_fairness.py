# ml_src/model_fairness.py
import joblib, yaml, pandas as pd, numpy as np, mlflow, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("model_fairness")

def evaluate_fairness(cfg_path="configs/config.yaml", model_path="models/best_logreg_tuned.joblib"):
    # 1️⃣ Load config + data + model
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]

    df = df_pred.copy()
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["arrival_time", "departure_time"])
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)

    X = df[["direction_id", "stop_sequence"]]
    y_true = df["delayed"]
    model = joblib.load(model_path)
    y_pred = model.predict(X)

    # 2️⃣ Choose slicing feature
    sensitive_feature = df["direction_id"]

    # 3️⃣ Build MetricFrame
    metrics = {
        "accuracy": accuracy_score,
        "recall": recall_score,
        "selection_rate": selection_rate,
    }
    mf = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature)

    logger.info("Fairness metrics by direction_id:")
    logger.info(f"\n{mf.by_group}")

    # 4️⃣ Plot disparities
    plt.figure(figsize=(8, 4))
    mf.by_group.plot(kind="bar")
    plt.title("Fairness Metrics by Direction ID")
    plt.ylabel("Metric Value")
    plt.tight_layout()
    plt.savefig("models/fairness_by_direction.png", dpi=300)
    logger.info("Saved fairness plot → models/fairness_by_direction.png")

    # 5️⃣ Compute bias measure (demographic parity diff)
    dp_diff = demographic_parity_difference(y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_feature)
    logger.info(f"Demographic parity difference: {dp_diff:.4f}")

    # 6️⃣ Log to MLflow
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name="model_fairness"):
        mlflow.log_artifact("models/fairness_by_direction.png")
        mlflow.log_metric("demographic_parity_difference", dp_diff)
        mf.by_group.to_csv("models/fairness_metrics.csv")
        mlflow.log_artifact("models/fairness_metrics.csv")

if __name__ == "__main__":
    evaluate_fairness()