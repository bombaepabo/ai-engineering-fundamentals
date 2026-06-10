from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT /"data"/ "SVHN_single_grey1.h5"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "cnn_model.pth"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 5
RANDOM_STATE = 42