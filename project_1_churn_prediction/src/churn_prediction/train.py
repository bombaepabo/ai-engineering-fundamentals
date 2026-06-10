import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from churn_prediction.config import FEATURE_COLUMNS_PATH, MODEL_PATH
from churn_prediction.features import create_train_test_data


def train_model() -> None:
    """Train the selected Random Forest churn model."""
    X_train, X_test, y_train, y_test = create_train_test_data()

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model: Tuned Random Forest")
    print(f"Accuracy: {accuracy:.4f}")
    print()
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(X_train.columns.tolist(), FEATURE_COLUMNS_PATH)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
