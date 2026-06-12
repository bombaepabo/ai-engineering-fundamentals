# app/services/model_downloader.py
import os
import logging
from google.cloud import storage
from app.config import settings

logger = logging.getLogger("app.api")


def download_models_from_gcs():
    """
    Downloads ML model files from Google Cloud Storage if they are missing locally.
    """
    model_paths = {
        "category_pipeline.joblib": "models/category_pipeline.joblib",
        "priority_pipeline.joblib": "models/priority_pipeline.joblib",
        "sentiment_pipeline.joblib": "models/sentiment_pipeline.joblib",
    }

    # 1. Check if all models already exist locally (common for local development)
    missing_models = [name for name, path in model_paths.items() if not os.path.exists(path)]
    if not missing_models:
        logger.info("All ML model files are present locally. Skipping GCS download.")
        return

    # 2. Check if a bucket name is configured in the environment
    bucket_name = settings.model_bucket_name
    if not bucket_name:
        logger.warning(
            "Some ML model files are missing locally, but MODEL_BUCKET_NAME is not set. "
            "Classifier will start in mock mode."
        )
        return

    logger.info(f"Downloading missing models {missing_models} from GCS bucket: {bucket_name}...")
    try:
        # 3. Initialize GCS client. In GCP Cloud Run, it automatically detects
        # the container's service account credentials (Application Default Credentials).
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

        for name in missing_models:
            blob = bucket.blob(name)
            local_path = model_paths[name]
            logger.info(f"Downloading {name} -> {local_path}...")
            blob.download_to_filename(local_path)

        logger.info("Successfully downloaded all missing models from Google Cloud Storage!")
    except Exception as e:
        logger.error(
            f"Failed to download models from GCS bucket '{bucket_name}': {e}. "
            "Falling back to mock mode."
        )
