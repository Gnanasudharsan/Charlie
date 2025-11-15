# Model_Development/ml_src/gcp_registry.py

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("gcp_registry")

# ------------------------------
# GCP CONFIGURATION
# ------------------------------
PROJECT = "charlie-478223"
LOCATION = "us-central1"
REPO = "charlie-model-registry"
PACKAGE_NAME = "charlie-mbta-model"


def push_to_gcp(
    model_path="Model_Development/models/final_model.joblib",
    metadata_path="Model_Development/models/model_metadata.json",
    version=None,
):
    """
    Uploads a model + metadata to Google Cloud Artifact Registry.
    Automatically generates metadata if missing.
    """

    # ------------------------------
    # Validate model exists
    # ------------------------------
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file missing: {model_path}")

    # ------------------------------
    # Auto-generate metadata if missing
    # ------------------------------
    if not os.path.exists(metadata_path):
        logger.warning("⚠️ No metadata found — creating model_metadata.json automatically.")
        metadata = {
            "model_name": Path(model_path).name,
            "created_at": datetime.utcnow().isoformat(),
            "version": version or "auto",
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

    # Reload metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # ------------------------------
    # Versioning
    # ------------------------------
    if version is None:
        version = datetime.utcnow().strftime("v%Y%m%d-%H%M%S")

    metadata["version"] = version

    # Update metadata file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"📦 Preparing upload → version={version}")

    # ------------------------------
    # Prepare upload folder
    # ------------------------------
    upload_dir = Path("gcp_upload")
    upload_dir.mkdir(exist_ok=True)

    subprocess.run(["cp", model_path, f"{upload_dir}/model.joblib"], check=True)
    subprocess.run(["cp", metadata_path, f"{upload_dir}/metadata.json"], check=True)

    # ------------------------------
    # TAR the package
    # ------------------------------
    tar_name = f"charlie_model_{version}.tar.gz"
    subprocess.run(
        ["tar", "-czf", tar_name, "-C", str(upload_dir), "."],
        check=True
    )

    logger.info("☁️ Uploading TAR to GCP Artifact Registry...")

    # ------------------------------
    # Upload to GCP Artifact Registry
    # ------------------------------
    cmd = [
        "gcloud", "artifacts", "generic", "upload",
        "--project", PROJECT,
        "--location", LOCATION,
        "--repository", REPO,
        "--package", PACKAGE_NAME,
        "--version", version,
        "--source", tar_name,
    ]

    subprocess.run(cmd, check=True)

    logger.info(f"✅ Successfully uploaded to Artifact Registry → {PACKAGE_NAME}:{version}")
    return version


if __name__ == "__main__":
    push_to_gcp()