import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from sentiment_classifier.config import MODEL_PATH
from sentiment_classifier.features import create_train_test_data


def train_model() -> None:
    """Train TF-IDF + Logistic Regression baseline model."""
    X_train, X_test, y_train, y_test = create_train_test_data()

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=10000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()