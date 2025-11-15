# ml_src/model_explain.py
import joblib, yaml, shap, mlflow, numpy as np, pandas as pd, matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
from ml_src.data_loader import DataPaths
from ml_src.utils.logging import get_logger

logger = get_logger("model_explain")

def explain_model(cfg_path="configs/config.yaml", model_path="models/best_logreg_tuned.joblib"):
    # 1️⃣ Load model & data
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
    y = df["delayed"]

    model = joblib.load(model_path)
    logger.info("Model and data loaded for explainability")

    # 2️⃣ SHAP feature importance
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=300)
    logger.info("Saved SHAP summary plot → models/shap_summary.png")

    # 3️⃣ LIME local explanation
    explainer_lime = LimeTabularExplainer(
        X.values,
        feature_names=X.columns.tolist(),
        class_names=["OnTime", "Delayed"],
        mode="classification"
    )
    i = np.random.randint(0, X.shape[0])
    exp = explainer_lime.explain_instance(X.values[i], model.predict_proba)
    exp.save_to_file("models/lime_explanation.html")
    logger.info("Saved LIME explanation → models/lime_explanation.html")

    # 4️⃣ Log to MLflow
    mlflow.set_tracking_uri(cfg["experiment"]["tracking_uri"])
    mlflow.set_experiment(cfg["experiment"]["name"])
    with mlflow.start_run(run_name="model_explainability"):
        mlflow.log_artifact("models/shap_summary.png")
        mlflow.log_artifact("models/lime_explanation.html")
        mlflow.log_param("explained_instance_index", i)

if __name__ == "__main__":
    explain_model()