import pandas as pd 
from sklearn.model_selection import train_test_split
from churn_prediction.data import load_data



def prepare_features(data):
    data = data.copy()

    data = data.drop(columns=["customerID"])

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

    data["TotalCharges"] = data["TotalCharges"].fillna(data["TotalCharges"].median())

    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    X = pd.get_dummies(X, drop_first=True)
    
    return X, y


def create_train_test_data(test_size=0.2, random_state=42):
    data = load_data()
    if data is None:
       raise FileNotFoundError("Could not load dataset.")
    X,y = prepare_features(data)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def main():
    X_train, X_test, y_train, y_test = create_train_test_data()
    print("Train and test sets created successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print("\nFeature columns:")
    print(X_train.columns.tolist()[:20])

    print("\nTarget distribution in training set:")
    print(y_train.value_counts(normalize=True))

if __name__ == "__main__":
    main()