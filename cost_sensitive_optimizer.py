"""
Cost-Sensitive Decision Threshold Optimizer for Churn Prevention.
Calculates the optimal classification cutoff threshold that maximizes expected
net business return given intervention incentives and churn acquisition replacement costs.
"""
from typing import Dict, List, Tuple
import numpy as np

class CostSensitiveThresholdOptimizer:
    def __init__(
        self,
        retention_campaign_cost: float = 50.0,
        customer_lifetime_value: float = 400.0,
        retention_success_rate: float = 0.45
    ):
        self.campaign_cost = retention_campaign_cost
        self.clv = customer_lifetime_value
        self.success_rate = retention_success_rate

    def calculate_net_profit(self, y_true: List[int], y_prob: List[float], threshold: float) -> float:
        """
        Computes expected profit:
        For every predicted positive: we incur campaign_cost.
        If true positive: we have a success_rate chance of saving customer_lifetime_value.
        """
        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)
        y_pred = (y_prob_arr >= threshold).astype(int)

        tp = np.sum((y_pred == 1) & (y_true_arr == 1))
        fp = np.sum((y_pred == 1) & (y_true_arr == 0))

        cost = (tp + fp) * self.campaign_cost
        saved_revenue = tp * self.success_rate * self.clv
        return round(float(saved_revenue - cost), 2)

    def find_optimal_threshold(self, y_true: List[int], y_prob: List[float], steps: int = 100) -> Tuple[float, float]:
        best_threshold = 0.5
        max_profit = float('-inf')

        for t in np.linspace(0.05, 0.95, steps):
            profit = self.calculate_net_profit(y_true, y_prob, t)
            if profit > max_profit:
                max_profit = profit
                best_threshold = round(float(t), 3)

        return best_threshold, max_profit
