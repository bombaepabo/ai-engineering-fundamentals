import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sentiment_classifier.config import (
    TRANSFORMER_MODEL_DIR,
    TRANSFORMER_MODEL_NAME,
)
from sentiment_classifier.features import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    create_transformer_train_test_data,
)

def create_huggingface_datasets():
    """Create Hugging Face datasets from reusable project split logic."""
    train_data, test_data = create_transformer_train_test_data()

    return Dataset.from_pandas(train_data), Dataset.from_pandas(test_data)

def tokenize_dataset(dataset, tokenizer):
    """Convert text into token IDs for the transformer."""
    return dataset.map(
        lambda batch: tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        ),
        batched=True,
    )

def compute_metrics(eval_pred):
    """Compute transformer evaluation metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }

def train_transformer() -> None:
    """Fine-tune DistilBERT for sentiment classification."""
    train_dataset, test_dataset = create_huggingface_datasets()

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL_NAME,
        num_labels=3,
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(TRANSFORMER_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_dir=str(TRANSFORMER_MODEL_DIR / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    results = trainer.evaluate()
    print(results)

    trainer.save_model(str(TRANSFORMER_MODEL_DIR))
    tokenizer.save_pretrained(str(TRANSFORMER_MODEL_DIR))

    print(f"Transformer model saved to {TRANSFORMER_MODEL_DIR}")


def main() -> None:
    train_transformer()


if __name__ == "__main__":
    main()
