# Model_Development/ml_src/gcp_registry.py

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from Model_Development.ml_src.utils.logging import get_logger

logger = get_logger("gcp_registry")


# ---------------------------------------------------
# GCP CONFIGURATION
# ---------------------------------------------------
PROJECT = "charlie-478223"
LOCATION = "us-central1"
REPO = "charlie-model-registry"
PACKAGE_NAME = "charlie-mbta-model"


def push_to_gcp(
    model_path="models/final_model.joblib",
    metadata_path="models/model_metadata.json",
    version=None,
):
    """
    Uploads a model + metadata to Google Cloud Artifact Registry.
    Now compatible with GitHub Actions and local runs.
    """

    # ---------------------------------------------------
    # Validate model exists
    # ---------------------------------------------------
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model not found: {model_path}")

    # ---------------------------------------------------
    # Auto-create metadata if missing
    # ---------------------------------------------------
    if not metadata_path.exists():
        logger.warning("⚠️ Metadata file missing — creating model_metadata.json automatically.")
        metadata = {
            "model_name": model_path.name,
            "created_at": datetime.utcnow().isoformat(),
            "version": version or "auto",
        }
        metadata_path.write_text(json.dumps(metadata, indent=4))
    else:
        metadata = json.loads(metadata_path.read_text())

    # ---------------------------------------------------
    # Generate version
    # ---------------------------------------------------
    if version is None:
        version = datetime.utcnow().strftime("v%Y%m%d-%H%M%S")

    metadata["version"] = version
    metadata_path.write_text(json.dumps(metadata, indent=4))

    logger.info(f"📦 Packaging model → version={version}")

    # ---------------------------------------------------
    # Create staging directory
    # ---------------------------------------------------
    upload_dir = Path("gcp_upload")
    upload_dir.mkdir(exist_ok=True)

    # Copy files
    subprocess.run(["cp", str(model_path), f"{upload_dir}/model.joblib"], check=True)
    subprocess.run(["cp", str(metadata_path), f"{upload_dir}/metadata.json"], check=True)

    # ---------------------------------------------------
    # Create TAR package
    # ---------------------------------------------------
    tar_name = f"charlie_model_{version}.tar.gz"
    subprocess.run(
        ["tar", "-czf", tar_name, "-C", str(upload_dir), "."],
        check=True,
    )

    logger.info("☁️ Uploading to Google Cloud Artifact Registry...")

    # ---------------------------------------------------
    # Upload TAR to GCP Artifact Registry
    # ---------------------------------------------------
    cmd = [
        "gcloud",
        "artifacts",
        "generic",
        "upload",
        "--project",
        PROJECT,
        "--location",
        LOCATION,
        "--repository",
        REPO,
        "--package",
        PACKAGE_NAME,
        "--version",
        version,
        "--source",
        tar_name,
    ]

    subprocess.run(cmd, check=True)

    logger.info(
        f"✅ Successfully uploaded model package → {PACKAGE_NAME}:{version}"
    )
    return version


if __name__ == "__main__":
    push_to_gcp()