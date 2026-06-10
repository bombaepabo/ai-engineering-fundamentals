import torch
from torch import nn
from torch.optim import Adam

from image_classifier.config import EPOCHS, LEARNING_RATE, MODEL_PATH
from image_classifier.data import create_dataloaders
from image_classifier.model import CNNClassifier


def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device) -> tuple[float, float]:
    """Train the model for one epoch."""
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def evaluate(model, data_loader, criterion, device) -> tuple[float, float]:
    """Evaluate the model on validation or test data."""
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def train_model() -> None:
    """Train CNN model and save it."""
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader, _ = create_dataloaders()

    model = CNNClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )
        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"| train_loss={train_loss:.4f} "
            f"| train_acc={train_accuracy:.4f} "
            f"| val_loss={val_loss:.4f} "
            f"| val_acc={val_accuracy:.4f}"
        )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()