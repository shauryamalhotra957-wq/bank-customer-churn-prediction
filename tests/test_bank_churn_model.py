import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from bank_churn_model import best_threshold, dice_from_cm, make_features, predict_customer, select_csv


class ChurnModelUnitTests(unittest.TestCase):
    def test_dice_from_confusion_matrix(self):
        cm = np.array([[80, 10], [5, 25]])
        d0, d1, avg = dice_from_cm(cm)

        self.assertAlmostEqual(d0, 160 / 175)
        self.assertAlmostEqual(d1, 50 / 65)
        self.assertAlmostEqual(avg, (d0 + d1) / 2)

    def test_best_threshold_can_separate_simple_probabilities(self):
        y_true = np.array([0, 0, 1, 1])
        probs = np.array([0.05, 0.40, 0.60, 0.95])

        threshold = best_threshold(y_true, probs)
        preds = (probs >= threshold).astype(int)

        np.testing.assert_array_equal(preds, y_true)
        self.assertGreater(threshold, 0.40)
        self.assertLessEqual(threshold, 0.60)

    def test_make_features_handles_zero_denominators_without_infinity(self):
        df = pd.DataFrame(
            [
                {
                    "CreditScore": 650,
                    "Geography": "Spain",
                    "Gender": "Male",
                    "Age": 42,
                    "Tenure": 0,
                    "Balance": 80000.0,
                    "NumOfProducts": 0,
                    "HasCrCard": 1,
                    "IsActiveMember": 1,
                    "EstimatedSalary": 0.0,
                    "Exited": 1,
                }
            ]
        )

        X, y, _, _ = make_features(df)

        self.assertEqual(y.tolist(), [1])
        self.assertIn("BalanceSalaryRatio", X.columns)
        self.assertIn("AgeTenureRatio", X.columns)
        self.assertIn("BalancePerProduct", X.columns)
        self.assertTrue(pd.isna(X.loc[0, "BalanceSalaryRatio"]))
        self.assertTrue(pd.isna(X.loc[0, "AgeTenureRatio"]))
        self.assertTrue(pd.isna(X.loc[0, "BalancePerProduct"]))

    def test_predict_customer_reports_missing_required_columns(self):
        with patch("bank_churn_model.joblib.load", return_value={}):
            with self.assertRaisesRegex(ValueError, "Missing required prediction columns"):
                predict_customer({"Age": 42})

    def test_select_csv_prefers_expected_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            unrelated = tmp_path / "other.csv"
            expected = tmp_path / "churn.csv"

            pd.DataFrame({"a": [1], "b": [2]}).to_csv(unrelated, index=False)
            pd.DataFrame(
                {
                    "CreditScore": [650],
                    "Geography": ["Spain"],
                    "Gender": ["Male"],
                    "Age": [42],
                    "Tenure": [6],
                    "Balance": [80000.0],
                    "NumOfProducts": [2],
                    "HasCrCard": [1],
                    "IsActiveMember": [1],
                    "EstimatedSalary": [90000.0],
                    "Exited": [0],
                }
            ).to_csv(expected, index=False)

            self.assertEqual(select_csv([str(unrelated), str(expected)]), str(expected))


if __name__ == "__main__":
    unittest.main()
