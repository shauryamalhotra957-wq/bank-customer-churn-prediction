from __future__ import annotations

import unittest

import pandas as pd

from bank_churn_model import make_features


def churn_frame(targets):
    size = len(targets)
    return pd.DataFrame(
        {
            "CreditScore": [650] * size,
            "Geography": ["France"] * size,
            "Gender": ["Female"] * size,
            "Age": list(range(30, 30 + size)),
            "Tenure": [5] * size,
            "Balance": [75_000.0] * size,
            "NumOfProducts": [2] * size,
            "HasCrCard": [1] * size,
            "IsActiveMember": [1] * size,
            "EstimatedSalary": [90_000.0] * size,
            "Exited": targets,
        }
    )


class TargetValidationTests(unittest.TestCase):
    def test_accepts_both_binary_target_classes(self):
        _, target, _, _ = make_features(churn_frame([0, 1]))

        self.assertEqual(set(target), {0, 1})

    def test_rejects_non_binary_target_values(self):
        with self.assertRaisesRegex(ValueError, "binary classes"):
            make_features(churn_frame([0, 2]))

    def test_rejects_single_class_target(self):
        with self.assertRaisesRegex(ValueError, "both binary classes"):
            make_features(churn_frame([0, 0]))

    def test_rejects_missing_or_non_numeric_target(self):
        with self.assertRaisesRegex(ValueError, "only binary values"):
            make_features(churn_frame([0, "unknown"]))


if __name__ == "__main__":
    unittest.main()
