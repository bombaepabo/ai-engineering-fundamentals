import json

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_prediction.config import CONFUSION_MATRIX_PATH, FIGURES_DIR, METRICS_PATH, REPORTS_DIR


def calculate_metrics(y_true, y_pred, y_prob) -> dict:
    """Calculate classification metrics for churn prediction."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }


def print_evaluation_report(y_true, y_pred, metrics: dict) -> None:
    """Print model metrics and classification report."""
    print("Model evaluation")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")

    print("\nClassification report:")
    print(classification_report(y_true, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))


def save_metrics(metrics: dict) -> None:
    """Save metrics to a JSON report file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"\nMetrics saved to {METRICS_PATH}")


def save_confusion_matrix(y_true, y_pred) -> None:
    """Save a confusion matrix image."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No churn", "Churn"],
        cmap="Blues",
        values_format="d",
    )

    display.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()

    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
