# Model_Development/ml_src/register_model.py

import os
import mlflow
import mlflow.sklearn
from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("register_model")

# Correct folder-safe paths
MODEL_DIR = "Model_Development/models"
BEST_MODEL_FILE = "final_model.joblib"
MODEL_NAME = "Charlie_MBTA_Model"     # MLflow registry name


def load_best_model_path():
    """Locate the final selected model inside Model_Development/models."""
    final_path = os.path.join(MODEL_DIR, BEST_MODEL_FILE)

    if not os.path.exists(final_path):
        raise FileNotFoundError(f"❌ No final model found at {final_path}")

    logger.info(f"✅ Final model located: {final_path}")
    return final_path


def register_model_with_mlflow(model_path):
    """Register best model into MLflow Model Registry."""
    # Correct tracking location for your new structure
    mlflow.set_tracking_uri("file:./Model_Development/mlruns")
    mlflow.set_experiment("MBTA-Model-Registry")

    with mlflow.start_run(run_name="register_best_model") as run:

        # Load model file → correct loading for a raw .joblib file
        import joblib
        model = joblib.load(model_path)

        logger.info("📦 Logging model to MLflow…")

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=MODEL_NAME
        )

        logger.info("✅ Model logged to MLflow")

        # Register & promote to Production
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["None", "Staging", "Production"])

        # Latest version = last entry
        latest_version = versions[-1].version

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