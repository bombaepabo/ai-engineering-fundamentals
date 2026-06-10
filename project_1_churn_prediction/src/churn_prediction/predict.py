import joblib
import pandas as pd

from churn_prediction.config import FEATURE_COLUMNS_PATH, MODEL_PATH


def prepare_customer_features(customer: dict) -> pd.DataFrame:
    """Prepare one customer for prediction."""
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    customer_df = pd.DataFrame([customer])

    # Remove customerID because it was removed during training.
    customer_df = customer_df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric, same as training.
    customer_df["TotalCharges"] = pd.to_numeric(
        customer_df["TotalCharges"],
        errors="coerce",
    )

    # Convert categorical features into dummy columns.
    customer_df = pd.get_dummies(customer_df, drop_first=True)

    # Match exactly the training feature columns.
    customer_df = customer_df.reindex(columns=feature_columns, fill_value=0)

    return customer_df


def predict_customer(customer: dict) -> None:
    """Predict churn for one customer."""
    model = joblib.load(MODEL_PATH)

    customer_features = prepare_customer_features(customer)

    prediction = model.predict(customer_features)[0]
    probability = model.predict_proba(customer_features)[0][1]

    label = "Yes" if prediction == 1 else "No"

    print(f"Churn prediction: {label}")
    print(f"Churn probability: {probability:.2%}")


def main() -> None:
    customer = {
        "customerID": "NEW-CUSTOMER",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.85,
        "TotalCharges": 1080.50,
    }

    predict_customer(customer)


if __name__ == "__main__":
    main()