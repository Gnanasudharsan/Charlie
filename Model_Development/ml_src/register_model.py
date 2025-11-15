import os
import json
import mlflow
import mlflow.sklearn
from ml_src.utils.logging import get_logger

logger = get_logger("register_model")

MODEL_DIR = "models"
BEST_MODEL_FILE = "final_model.joblib"
MODEL_NAME = "Charlie_MBTA_Model"  # MLflow registry name


def load_best_model_path():
    """Find final selected model."""
    final_path = os.path.join(MODEL_DIR, BEST_MODEL_FILE)

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"❌ No final model found at {final_path}")

    logger.info(f"✅ Final model located: {final_path}")
    return final_path


def register_model_with_mlflow(model_path):
    """Register best model into MLflow Model Registry."""
    mlflow.set_tracking_uri("file:./mlruns")  # local registry
    mlflow.set_experiment("MBTA-Model-Registry")

    with mlflow.start_run(run_name="register_best_model") as run:

        # Load model
        model = mlflow.sklearn.load_model(model_path)

        # Log model
        logger.info("📦 Logging model to MLflow…")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        # Add run details
        mlflow.set_tag("stage", "production")
        mlflow.set_tag("model_type", "classification")
        mlflow.set_tag("selection_method", "performance_based")

        logger.info("✅ Model logged to MLflow")

        # Get model version
        client = mlflow.tracking.MlflowClient()
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["None"])[0].version

        # Transition model to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )

        logger.info(f"🚀 Model registered as PRODUCTION (version {latest_version})")

        return latest_version


def main():
    try:
        model_path = load_best_model_path()
        version = register_model_with_mlflow(model_path)
        logger.info(f"🎉 Model successfully registered at version {version}")

    except Exception as e:
        logger.error(f"❌ Model registry failed: {str(e)}")


if __name__ == "__main__":
    main()