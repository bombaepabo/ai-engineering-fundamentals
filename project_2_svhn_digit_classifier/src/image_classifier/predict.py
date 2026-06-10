import matplotlib.pyplot as plt
import torch

from image_classifier.config import FIGURES_DIR, MODEL_PATH
from image_classifier.data import load_arrays
from image_classifier.model import CNNClassifier
from image_classifier.train import get_device


def predict_one(index: int = 0) -> None:
    """Predict one test image and save a visualization."""
    device = get_device()
    print(f"Using device: {device}")

    _, _, _, _, X_test, y_test = load_arrays()

    image = torch.tensor(X_test[index], dtype=torch.float32)
    label = int(y_test[index])

    # CNN expects channels, height, width.
    image_for_model = image.unsqueeze(0)

    # Model expects batch, channels, height, width.
    image_for_model = image_for_model.unsqueeze(0).to(device)

    model = CNNClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(image_for_model)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = probabilities.argmax(dim=1).item()
        confidence = probabilities[0, prediction].item()

    print(f"True label: {label}")
    print(f"Predicted label: {prediction}")
    print(f"Confidence: {confidence:.2%}")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(image.numpy(), cmap="gray", interpolation="nearest")
    plt.title(
        f"True: {label} | Predicted: {prediction} | Confidence: {confidence:.2%}"
    )
    plt.axis("off")
    plt.savefig(FIGURES_DIR / "sample_prediction.png", dpi=150)
    plt.close()

    print(f"Prediction figure saved to {FIGURES_DIR / 'sample_prediction.png'}")


def main() -> None:
    predict_one(index=2)


if __name__ == "__main__":
    main()