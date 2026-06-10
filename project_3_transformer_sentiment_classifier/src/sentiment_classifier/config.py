from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "train.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "tfidf_logistic_regression.joblib"

REPORTS_DIR = PROJECT_ROOT / "reports"

TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_MODEL_DIR = MODELS_DIR / "distilbert_sentiment"
