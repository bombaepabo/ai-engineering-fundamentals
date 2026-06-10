import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sentiment_classifier.config import MODEL_PATH
from sentiment_classifier.features import create_train_test_data


def evaluate_model() -> None:
    """Evaluate saved sentiment model."""
    _, X_test, _, y_test = create_train_test_data()

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def main() -> None:
    evaluate_model()


if __name__ == "__main__":
    main()