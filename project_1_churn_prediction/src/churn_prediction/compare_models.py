import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_prediction.features import create_train_test_data


def get_models() -> dict:
    """Return candidate models for comparison."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000)),
            ]
        ),
        "Logistic Regression Balanced": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42,
        ),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test) -> dict:
    """Train one model and return evaluation metrics."""
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def compare_models() -> pd.DataFrame:
    """Train and compare multiple churn prediction models."""
    X_train, X_test, y_train, y_test = create_train_test_data()

    results = []

    for model_name, model in get_models().items():
        print(f"Training {model_name}...")

        metrics = evaluate_model(model, X_train, X_test, y_train, y_test)

        results.append(
            {
                "model": model_name,
                **metrics,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    return results_df


def main() -> None:
    results = compare_models()

    print("\nModel comparison:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
