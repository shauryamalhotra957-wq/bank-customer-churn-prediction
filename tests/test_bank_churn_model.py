import numpy as np
import pandas as pd
import pytest

from bank_churn_model import load_data, make_features


def churn_frame():
    return pd.DataFrame(
        [
            {
                "CreditScore": 650,
                "Geography": "Spain",
                "Gender": "Male",
                "Age": 42,
                "Tenure": 0,
                "Balance": 80000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 0.0,
                "Exited": 0,
            },
            {
                "CreditScore": 580,
                "Geography": "France",
                "Gender": "Female",
                "Age": 55,
                "Tenure": 4,
                "Balance": 120000.0,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 0,
                "EstimatedSalary": 64000.0,
                "Exited": 1,
            },
        ]
    )


def test_make_features_handles_zero_divisors_without_infinity():
    features, target, _, _ = make_features(churn_frame())

    assert target.tolist() == [0, 1]
    assert "BalanceSalaryRatio" in features
    assert not np.isinf(features.select_dtypes(include=[np.number]).to_numpy()).any()
    assert pd.isna(features.loc[0, "AgeTenureRatio"])


def test_load_data_uses_the_explicit_dataset_path(tmp_path):
    dataset = tmp_path / "chosen.csv"
    churn_frame().to_csv(dataset, index=False)

    loaded, selected = load_data(dataset, allow_download=False)

    assert selected == str(dataset)
    assert loaded["Exited"].tolist() == [0, 1]


def test_load_data_reports_a_missing_explicit_dataset(tmp_path):
    missing = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        load_data(missing, allow_download=False)
