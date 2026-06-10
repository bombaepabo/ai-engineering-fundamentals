import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from churn_prediction.features import create_train_test_data


def get_search_configs() -> dict:
    """Return model and parameter grid configurations."""
    return {
        "Logistic Regression": {
            "model": Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000)),
                ]
            ),
            "params": {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 5],
                "class_weight": [None, "balanced"],
            },
        },
    }


def calculate_test_metrics(model, X_test, y_test) -> dict:
    """Calculate test metrics for a fitted model."""
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


def run_grid_search(scoring: str = "f1") -> pd.DataFrame:
    """Run GridSearchCV for candidate models."""
    X_train, X_test, y_train, y_test = create_train_test_data()

    results = []

    for model_name, config in get_search_configs().items():
        print(f"Running GridSearchCV for {model_name}...")

        search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring=scoring,
            cv=5,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        test_metrics = calculate_test_metrics(search.best_estimator_, X_test, y_test)

        results.append(
            {
                "model": model_name,
                "best_cv_score": search.best_score_,
                "best_params": search.best_params_,
                **test_metrics,
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1", ascending=False)

    return results_df


def main() -> None:
    results = run_grid_search(scoring="f1")

    print("\nGrid search results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
