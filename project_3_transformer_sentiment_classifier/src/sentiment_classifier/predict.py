import joblib

from sentiment_classifier.config import MODEL_PATH


def predict_sentiment(text: str) -> None:
    """Predict sentiment for one text."""
    model = joblib.load(MODEL_PATH)

    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    confidence = probabilities.max()

    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")


def main() -> None:
    predict_sentiment("whatever.")


if __name__ == "__main__":
    main()