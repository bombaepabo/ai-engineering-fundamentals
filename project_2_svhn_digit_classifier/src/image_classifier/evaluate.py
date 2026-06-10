import torch
from sklearn.metrics import classification_report, confusion_matrix

from image_classifier.config import MODEL_PATH
from image_classifier.data import create_dataloaders
from image_classifier.model import CNNClassifier
from image_classifier.train import evaluate, get_device


def evaluate_saved_model() -> None:
    """Evaluate saved CNN model on the test set."""
    device = get_device()
    print(f"Using device: {device}")

    _, _, test_loader = create_dataloaders()

    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    criterion = torch.nn.CrossEntropyLoss()

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    all_predictions = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(all_labels, all_predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(all_labels, all_predictions))


def main() -> None:
    evaluate_saved_model()


if __name__ == "__main__":
    main()