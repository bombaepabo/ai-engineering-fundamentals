# Transformer Sentiment Classifier 🎭

An NLP classification project comparing a classical machine learning baseline with a fine-tuned Hugging Face **DistilBERT** transformer model to classify text reviews into positive, neutral, or negative sentiments.

## Models Compared
1. **Baseline**: TF-IDF text representation combined with a Logistic Regression classifier (Local/Fast).
2. **Transformer**: DistilBERT (`distilbert-base-uncased`) fine-tuned on the reviews dataset using Hugging Face's `Trainer` API.

## How to Run

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Train Baseline Model**:
   ```bash
   uv run python -m sentiment_classifier.train_baseline
   ```

3. **Fine-tune DistilBERT Transformer**:
   ```bash
   uv run python -m sentiment_classifier.train_transformer
   ```

4. **Predict**:
   ```bash
   uv run python -m sentiment_classifier.predict_transformer --text "The product worked perfectly at first, but broke after two days."
   ```
