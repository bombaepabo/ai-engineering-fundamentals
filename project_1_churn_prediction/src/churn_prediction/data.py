import pandas as pd
from churn_prediction.config import DATA_PATH
def load_data():
    """Load the customer churn dataset from the specified path."""
    try:
        data = pd.read_csv(DATA_PATH)
        print(f"Data loaded successfully from {DATA_PATH}")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {DATA_PATH} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None

def inspect_data(data):
    print("\nFirst 5 rows of the dataset:")
    print(data.head())

    print("\nShape")
    print(data.shape)

    print("\nColumns")
    print(data.columns.tolist())

    print("\nData types")
    print(data.dtypes)

    print("\nMissing values")
    print(data.isnull().sum())

    print("\nChurn distribution:")
    print(data["Churn"].value_counts())

def main():
    """Main function to load and display the dataset."""
    data = load_data()
    if data is not None:
        inspect_data(data)

if __name__ == "__main__":
    main()