import joblib

from churn_prediction.config import MODEL_PATH
from churn_prediction.features import create_train_test_data
from churn_prediction.reporting import (
    calculate_metrics,
    print_evaluation_report,
    save_confusion_matrix,
    save_metrics,
)


def evaluate_model() -> None:
    """Load the trained model and evaluate it on the test set."""
    _, X_test, _, y_test = create_train_test_data()

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(y_test, y_pred, y_prob)

    print_evaluation_report(y_test, y_pred, metrics)
    save_metrics(metrics)
    save_confusion_matrix(y_test, y_pred)


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()
