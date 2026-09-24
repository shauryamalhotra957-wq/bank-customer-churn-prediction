import pytest
from cost_sensitive_optimizer import CostSensitiveThresholdOptimizer

def test_profit_calculation():
    optimizer = CostSensitiveThresholdOptimizer(
        retention_campaign_cost=50.0,
        customer_lifetime_value=400.0,
        retention_success_rate=0.5
    )
    # Saved per TP: 0.5 * 400 = 200. Cost per intervention: 50. Net gain per TP = 150. Loss per FP = 50.
    y_true = [1, 1, 0, 0]
    y_prob = [0.8, 0.7, 0.6, 0.2]
    
    # threshold = 0.65 -> flags idx 0, 1 (2 TPs, 0 FPs) -> net profit = 2 * 150 = 300
    profit = optimizer.calculate_net_profit(y_true, y_prob, threshold=0.65)
    assert profit == 300.0

def test_optimal_threshold_search():
    optimizer = CostSensitiveThresholdOptimizer(
        retention_campaign_cost=50.0,
        customer_lifetime_value=400.0,
        retention_success_rate=0.5
    )
    y_true = [1, 1, 0, 0, 1]
    y_prob = [0.9, 0.8, 0.4, 0.3, 0.75]
    best_t, max_p = optimizer.find_optimal_threshold(y_true, y_prob)
    assert best_t > 0.4
    assert max_p > 0
