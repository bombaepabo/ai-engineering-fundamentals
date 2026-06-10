# Churn Prediction: Telco Customer Churn

## Project Proposal

Customer churn is one of the most important problems for subscription-based businesses. When customers cancel their service, the company loses recurring revenue and may also spend more money trying to replace those customers with new ones.

This project uses machine learning to predict whether a telecom customer is likely to churn based on customer account information, service usage, billing details, and contract type.

The goal is not only to build a model with good accuracy, but also to understand which customer patterns may indicate churn risk. This kind of project is useful for AI engineering because it covers the full beginner machine learning workflow: loading data, preparing features, training a model, evaluating results, saving the model, and making predictions on new customer data.

## Problem Statement

The business problem is:

> Can we predict whether a customer will leave the company before they actually churn?

If the company can identify high-risk customers earlier, it can take action such as:

- offering discounts or retention plans
- improving customer support
- recommending better contract options
- prioritizing outreach to customers most likely to leave

This is a binary classification problem:

- `0` = customer did not churn
- `1` = customer churned

## Dataset

This project uses the Telco Customer Churn dataset.

The dataset contains customer information such as:

- gender
- senior citizen status
- partner/dependent status
- tenure
- phone service
- internet service
- online security and backup services
- contract type
- paperless billing
- payment method
- monthly charges
- total charges
- churn status

The dataset is stored at:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

## Project Structure

```text
project 1/
  data/
    WA_Fn-UseC_-Telco-Customer-Churn.csv
  models/
    logistic_regression_model.joblib
    feature_columns.joblib
  src/
    churn_prediction/
      __init__.py
      config.py
      data.py
      features.py
      train.py
      evaluate.py
      predict.py
  pyproject.toml
  uv.lock
  README.md
```

## Approach

The project follows a simple machine learning pipeline:

1. Load the raw customer churn dataset.
2. Inspect the dataset shape, columns, missing values, and churn distribution.
3. Clean and prepare the data for machine learning.
4. Convert categorical columns into numeric features.
5. Split the data into training and testing sets.
6. Train a tuned Random Forest model.
7. Evaluate the model using classification metrics.
8. Save the trained model and feature columns.
9. Use the saved model to predict churn for a new customer.

## Feature Preparation

The raw dataset is not directly ready for machine learning. The project prepares the data by:

- removing `customerID` because it is only an identifier
- converting `TotalCharges` from text to numeric values
- filling missing `TotalCharges` values with the median
- converting `Churn` from `Yes`/`No` to `1`/`0`
- encoding categorical features with one-hot encoding
- splitting the dataset into training and testing data

## Model

The first baseline model was Logistic Regression. After model comparison and hyperparameter tuning, the selected model is a balanced Random Forest.

Logistic Regression was useful as a baseline because:

- the target is binary
- it trains quickly
- it is easier to understand than more complex models
- it gives a strong starting point for comparison

GridSearchCV selected a balanced Random Forest because it improved churn recall and F1-score compared with the first baseline.

The selected Random Forest uses:

```text
n_estimators=200
max_depth=10
min_samples_leaf=5
class_weight=balanced
```

Later, this project can be improved further by comparing the selected model with Gradient Boosting, XGBoost, or LightGBM.

## Evaluation

The model is evaluated using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

For this problem, accuracy alone is not enough. The most important class is usually `Churn = 1`, because the business wants to find customers who are likely to leave.

Recall for churn is especially important because a low recall means the model is missing customers who actually churned.

Initial Logistic Regression baseline:

```text
Model: Logistic Regression
Accuracy: about 82%
Churn recall: about 60%
Churn F1-score: about 64%
```

Tuned Random Forest result:

```text
Model: Balanced Random Forest
Accuracy: about 79%
Churn recall: about 80%
Churn F1-score: about 67%
ROC-AUC: about 87%
PR-AUC: about 70%
```

The tuned model has slightly lower accuracy than the baseline, but it finds many more customers who actually churn. For churn prediction, this can be a better business tradeoff because missing at-risk customers may be more expensive than contacting some customers who would not have churned.

## How To Run

Install dependencies and sync the environment:

```bash
uv sync
```

Inspect the dataset:

```bash
uv run python -m churn_prediction.data
```

Prepare features and check train/test shapes:

```bash
uv run python -m churn_prediction.features
```

Train the model:

```bash
uv run python -m churn_prediction.train
```

Evaluate the saved model:

```bash
uv run python -m churn_prediction.evaluate
```

Predict churn for a sample customer:

```bash
uv run python -m churn_prediction.predict
```

## Expected Output

After training, the project saves:

```text
models/random_forest_model.joblib
models/feature_columns.joblib
```

The prediction script prints an output like:

```text
Churn prediction: Yes
Churn probability: 71.32%
```

## Why This Project Matters

This project is a practical introduction to AI engineering because it connects a real business problem with a working machine learning system.

It teaches how to:

- structure a Python ML project
- manage dependencies with `uv`
- prepare real-world tabular data
- train and evaluate a classification model
- save model artifacts
- reuse a trained model for prediction

## Future Improvements

Possible next improvements:

- compare Logistic Regression with Random Forest
- tune model hyperparameters
- improve churn recall
- add visualizations for churn patterns
- add feature importance analysis
- create a FastAPI endpoint for predictions
- add tests for data preparation and prediction
- track experiments and metrics
