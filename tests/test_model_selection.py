from __future__ import annotations

import inspect
import unittest

from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from bank_churn_model import compare_candidate_models


class ModelSelectionTests(unittest.TestCase):
    def test_candidate_ranking_uses_validation_metrics_only(self) -> None:
        X, y = make_classification(
            n_samples=160,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y
        )
        models = {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
            "Dummy": DummyClassifier(strategy="prior")
        }

        leaderboard, thresholds = compare_candidate_models(
            models,
            StandardScaler(),
            X_train,
            y_train,
            X_valid,
            y_valid
        )

        self.assertEqual(set(leaderboard["model"]), set(models))
        self.assertEqual(set(thresholds), set(models))
        self.assertIn("validation_roc_auc", leaderboard.columns)
        self.assertNotIn("test_roc_auc", leaderboard.columns)
        self.assertGreaterEqual(
            leaderboard.loc[0, "validation_roc_auc"],
            leaderboard.loc[1, "validation_roc_auc"]
        )

    def test_model_comparison_has_no_test_split_input(self) -> None:
        parameters = inspect.signature(compare_candidate_models).parameters

        self.assertNotIn("X_test", parameters)
        self.assertNotIn("y_test", parameters)


if __name__ == "__main__":
    unittest.main()
