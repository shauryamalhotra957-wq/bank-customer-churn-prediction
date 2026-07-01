 # Bank Customer Churn Prediction

Machine learning project that predicts whether a bank customer will stay with the bank or churn.

## Target

`Exited = 0` means the customer continued with the bank.

`Exited = 1` means the customer left the bank.

## Dataset

The project works with the classic `Churn_Modelling.csv` dataset used in Kaggle bank churn notebooks.

Expected columns:

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary
- Exited

## Models

The training script compares multiple models and automatically selects the best one using ROC-AUC, F1 score, average dice score, and accuracy.

Models included:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- HistGradientBoosting
- XGBoost if installed
- LightGBM if installed

## Metrics

The script prints:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Confusion matrix
- Dice score for continued customers
- Dice score for churned customers
- Average dice score

## Run on Kaggle

Add the dataset as notebook input, then run:

```bash
python bank_churn_model.py
```

If the CSV is not found, the script tries to download a fallback copy using KaggleHub.

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Put `Churn_Modelling.csv` in the project folder, then run:

```bash
python bank_churn_model.py
```

## Output files

The script creates:

```text
outputs/bank_churn_model.joblib
outputs/model_leaderboard.csv
outputs/test_predictions.csv
```

## Predict one customer

```python
from bank_churn_model import predict_customer

customer = {
    "CreditScore": 650,
    "Geography": "Spain",
    "Gender": "Male",
    "Age": 42,
    "Tenure": 6,
    "Balance": 80000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 90000.0
}

print(predict_customer(customer))
```
