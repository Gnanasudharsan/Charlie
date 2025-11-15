# ml_src/gcp_registry.py

import json
import os
import subprocess
from datetime import datetime
from ml_src.utils.logging import get_logger

logger = get_logger("gcp_registry")

PROJECT = "charlie-478223"
LOCATION = "us-central1"
REPO = "charlie-model-registry"
PACKAGE_NAME = "charlie-mbta-model"


def push_to_gcp(model_path="models/final_model.joblib",
                metadata_path="models/model_metadata.json",
                version=None):

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file missing: {metadata_path}")

    if version is None:
        version = datetime.utcnow().strftime("v%Y%m%d-%H%M%S")

    logger.info(f"📦 Packaging model version={version}")

    # Create package directory
    os.makedirs("gcp_upload", exist_ok=True)

    # Copy files
    subprocess.run(["cp", model_path, "gcp_upload/model.joblib"], check=True)
    subprocess.run(["cp", metadata_path, "gcp_upload/metadata.json"], check=True)

    # TAR the package
    tar_name = f"charlie_model_{version}.tar.gz"
    subprocess.run(
        ["tar", "-czf", tar_name, "-C", "gcp_upload", "."],
        check=True
    )

    logger.info("☁️ Uploading TAR to GCP Artifact Registry...")

    subprocess.run(
        [
            "gcloud", "artifacts", "generic", "upload",
            "--project", PROJECT,
            "--location", LOCATION,
            "--repository", REPO,
            "--package", PACKAGE_NAME,
            "--version", version,
            "--source", tar_name,
        ],
        check=True
    )

    logger.info(f"✅ Successfully uploaded: {PACKAGE_NAME}:{version}")
    return version


if __name__ == "__main__":
    push_to_gcp()