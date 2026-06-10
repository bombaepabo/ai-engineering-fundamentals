import h5py
import torch
from torch.utils.data import DataLoader, Dataset

from image_classifier.config import BATCH_SIZE, DATASET_PATH


class SVHNDataset(Dataset):
    """PyTorch dataset for the SVHN grayscale H5 data."""

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = torch.tensor(self.images[index], dtype=torch.float32)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # Add channel dimension: 32x32 -> 1x32x32
        image = image.unsqueeze(0)

        return image, label


def load_arrays():
    """Load train, validation, and test arrays from the H5 dataset."""
    with h5py.File(DATASET_PATH, "r") as dataset:
        X_train = dataset["X_train"][:]
        y_train = dataset["y_train"][:]
        X_val = dataset["X_val"][:]
        y_val = dataset["y_val"][:]
        X_test = dataset["X_test"][:]
        y_test = dataset["y_test"][:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def create_dataloaders():
    """Create PyTorch DataLoaders for train, validation, and test data."""
    X_train, y_train, X_val, y_val, X_test, y_test = load_arrays()

    train_dataset = SVHNDataset(X_train, y_train)
    val_dataset = SVHNDataset(X_val, y_val)
    test_dataset = SVHNDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def inspect_dataset() -> None:
    """Inspect the SVHN H5 dataset structure."""
    with h5py.File(DATASET_PATH, "r") as dataset:
        print("Dataset keys:")
        print(list(dataset.keys()))

        for key in dataset.keys():
            data = dataset[key]
            print(f"{key}: shape={data.shape}, dtype={data.dtype}")


def main() -> None:
    inspect_dataset()

    train_loader, val_loader, test_loader = create_dataloaders()

    images, labels = next(iter(train_loader))

    print("\nOne training batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels example:", labels[:10])


if __name__ == "__main__":
    main()