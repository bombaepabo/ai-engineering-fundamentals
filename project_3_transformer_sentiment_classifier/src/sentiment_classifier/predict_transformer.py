import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from sentiment_classifier.config import TRANSFORMER_MODEL_DIR
from sentiment_classifier.features import ID_TO_LABEL


def get_device() -> torch.device:
    """Use GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_transformer_model():
    """Load fine-tuned tokenizer and transformer model."""
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_MODEL_DIR)

    return tokenizer, model


def predict_sentiment(text: str) -> None:
    """Predict sentiment with the fine-tuned transformer."""
    device = get_device()
    print(f"Using device: {device}")

    tokenizer, model = load_transformer_model()
    model = model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)

    predicted_id = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, predicted_id].item()
    prediction = ID_TO_LABEL[predicted_id]

    print(f"Text: {text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")


def main() -> None:
    predict_sentiment("you are unsurprisingly dogshit?")


if __name__ == "__main__":
    main()