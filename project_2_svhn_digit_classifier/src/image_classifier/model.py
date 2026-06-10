import torch
from torch import nn

class CNNClassifier(nn.Module):
    """Small CNN for SVHN grayscale digit classification."""

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

def main() -> None:
    model = CNNClassifier()

    dummy_batch = torch.randn(64, 1, 32, 32)
    outputs = model(dummy_batch)

    print(model)
    print("Input shape:", dummy_batch.shape)
    print("Output shape:", outputs.shape)


if __name__ == "__main__":
    main()