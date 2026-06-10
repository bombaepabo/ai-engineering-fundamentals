from sklearn.model_selection import train_test_split

from sentiment_classifier.data import load_data

LABEL_TO_ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


def create_train_test_data():
    """Create train/test split for sentiment classification."""
    data = load_data()

    X = data["text"]
    y = data["sentiment"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

def create_transformer_train_test_data():
    """Create train/test split with numeric labels for transformer training."""
    data = load_data()

    data["label"] = data["sentiment"].map(LABEL_TO_ID)
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    return train_test_split(
        data[["text", "label"]],
        test_size=0.2,
        random_state=42,
        stratify=data["label"],
    )