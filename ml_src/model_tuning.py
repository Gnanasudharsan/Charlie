# ml_src/model_tuning.py
import os, yaml, joblib, mlflow, mlflow.sklearn, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("model_tuning")


def prepare_data(df):
    df["arrival_time"] = pd.to_datetime(df["arrival_time"], errors="coerce")
    df["departure_time"] = pd.to_datetime(df["departure_time"], errors="coerce")
    df = df.dropna(subset=["arrival_time", "departure_time"])
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)

    X = df[["direction_id", "stop_sequence"]].dropna()
    y = df["delayed"]
    return X, y


def train_and_log():
    logger.info("🚀 Starting hyperparameter tuning...")

    # Load processed data
    paths = DataPaths("ml_configs/paths.yaml")
    try:
        df_pred = paths.load_all()["predictions"]
    except FileNotFoundError as e:
        logger.warning("⚠️ Processed data not found. Skipping tuning step in CI/CD environment.")
        print(str(e))
        import sys
        sys.exit(0)

    logger.info(f"✅ Loaded dataset for tuning: {df_pred.shape}")

    # Prepare data
    X, y = prepare_data(df_pred)
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    logger.info(f"📈 After SMOTE: {X_res.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # Load MLflow config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    # Hyperparameter tuning
    params = {"C": [0.01, 0.1, 1, 10], "solver": ["lbfgs", "liblinear"]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring="roc_auc")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    logger.info(f"🏆 Best params: {grid.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_val)
    y_prob = best_model.predict_proba(X_val)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_prob),
    }

    # Log with MLflow
    with mlflow.start_run(run_name="logreg_tuned"):
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="tuned_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/logreg_tuned.joblib")
        logger.info("💾 Tuned model saved to models/logreg_tuned.joblib")

    logger.info(f"🎯 Tuning complete. Metrics: {metrics}")


if __name__ == "__main__":
    train_and_log()