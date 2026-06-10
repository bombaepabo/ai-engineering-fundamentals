# SVHN Grayscale Digit Classifier 🔢

This project implements a Convolutional Neural Network (CNN) in PyTorch to classify single-digit grayscale images from the Street View House Numbers (SVHN) dataset.

## CNN Architecture
The model uses 3 convolutional layers followed by max-pooling for feature extraction, and a fully connected classifier with Dropout regularization to prevent overfitting:

```text
Input (1x32x32) 
  ├──► Conv2D (32 filters, 3x3) ──► ReLU ──► MaxPool2D ──► (32x16x16)
  ├──► Conv2D (64 filters, 3x3) ──► ReLU ──► MaxPool2D ──► (64x8x8)
  ├──► Conv2D (128 filters, 3x3) ──► ReLU ──► MaxPool2D ──► (128x4x4)
  └──► Fully Connected Classifier:
         ├──► Flatten (2048 units)
         ├──► Linear (128 units) ──► ReLU ──► Dropout (0.3)
         └──► Linear Output (10 digit classes)
```

## How to Run

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Train the CNN**:
   Loads the SVHN training partition, transforms images to grayscale tensor format, and trains the CNN model:
   ```bash
   uv run python -m image_classifier.train
   ```

3. **Evaluate the Model**:
   Evaluates validation test accuracy and classification statistics:
   ```bash
   uv run python -m image_classifier.evaluate
   ```

4. **Predict**:
   Run inference on local image samples:
   ```bash
   uv run python -m image_classifier.predict --image-path path/to/digit.png
   ```
