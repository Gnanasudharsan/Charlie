# ml_src/model_tuning.py
from __future__ import annotations
import os, yaml, joblib, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("model_tuning")

def prepare_data(df: pd.DataFrame):
    df = df.copy()
    for col in ["arrival_time", "departure_time"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=["arrival_time", "departure_time"])
    df["delay_minutes"] = (df["departure_time"] - df["arrival_time"]).dt.total_seconds() / 60
    df["delayed"] = (df["delay_minutes"] > 5).astype(int)
    features = ["direction_id", "stop_sequence"]
    X = df[features]
    y = df["delayed"]
    return X, y

def train_and_log(cfg_path="configs/config.yaml"):
    # 1️⃣ Load experiment config + data
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    paths = DataPaths("ml_configs/paths.yaml")
    df_pred = paths.load_all()["predictions"]
    X, y = prepare_data(df_pred)
    logger.info(f"Loaded data: {X.shape}, target balance={y.value_counts(normalize=True).to_dict()}")

    # 2️⃣ Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"], stratify=y
    )

        # 3️⃣ Handle imbalance dynamically
    minority_count = y_train.value_counts().min()
    if minority_count < 10:
        logger.warning(f"Minority class too small for SMOTE (count={minority_count}). Skipping oversampling.")
        X_train_res, y_train_res = X_train, y_train
        use_smote = False
    else:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logger.info(f"After SMOTE: {X_train_res.shape}, class ratio={y_train_res.value_counts().to_dict()}")
        use_smote = True

    # 4️⃣ Hyperparameter grid (simplified for small data)
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs", "saga"],
        "class_weight": ["balanced"] if not use_smote else [None]
    }

    # 5️⃣ Configure MLflow
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])

    with mlflow.start_run(run_name="grid_search_logreg"):
        base_model = LogisticRegression(max_iter=1000)
        grid = GridSearchCV(base_model, param_grid, cv=3, scoring="f1", n_jobs=-1, verbose=1)
        grid.fit(X_train_res, y_train_res)

        best_model = grid.best_estimator_
        logger.info(f"Best params: {grid.best_params_}")

        # 6️⃣ Validation
        y_pred = best_model.predict(X_val)
        y_prob = best_model.predict_proba(X_val)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_val, y_prob),
        }

        # 7️⃣ Log metrics & artifacts
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_logreg_tuned.joblib")
        logger.info(f"Metrics: {metrics}")
        logger.info("Best model saved to models/best_logreg_tuned.joblib")

if __name__ == "__main__":
    train_and_log()