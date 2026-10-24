import torch

from global_weather.continuous.feedback import (
    OnlineLearningPolicy, candidate_decision, masked_error_metrics,
)


def test_masked_error_metrics_ignore_missing_values():
    prediction = torch.tensor([1.0, 3.0, 1000.0])
    target = torch.tensor([2.0, 1.0, float("nan")])
    mask = torch.tensor([True, True, False])
    result = masked_error_metrics(prediction, target, mask)
    assert result["count"] == 2
    assert abs(result["bias"] - .5) < 1e-6


def test_candidate_requires_primary_improvement_and_guardrails():
    policy = OnlineLearningPolicy(min_new_samples=1, max_new_samples=2,
                                  min_relative_improvement=.01,
                                  max_relative_guardrail_degradation=.02)
    good = candidate_decision(
        {"rmse": 1.0, "mae": .8, "abs_bias": .2},
        {"rmse": .98, "mae": .79, "abs_bias": .19}, policy)
    assert good["accepted"]
    bad = candidate_decision(
        {"rmse": 1.0, "mae": .8, "abs_bias": .2},
        {"rmse": .98, "mae": .9, "abs_bias": .19}, policy)
    assert not bad["accepted"]
