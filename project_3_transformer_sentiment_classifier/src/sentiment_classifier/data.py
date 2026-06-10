import pandas as pd 
from sentiment_classifier.config import DATA_PATH

def load_data():
    """Load tweet sentiment dataset."""
    data = pd.read_csv(DATA_PATH, encoding="latin1")

    data = data[["text", "sentiment"]].copy()
    data = data.dropna(subset=["text", "sentiment"])

    data["text"] = data["text"].astype(str)
    data["sentiment"] = data["sentiment"].astype(str)

    return data

def main():
    data = load_data()

    print("First rows:")
    print(data.head())

    print("\nShape:")
    print(data.shape)

    print("\nSentiment distribution:")
    print(data["sentiment"].value_counts())

if __name__ == "__main__":
    main()