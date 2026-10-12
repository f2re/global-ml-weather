"""Regression checks for horizon-dependent chronological leakage."""
import pytest

from global_weather.unified_observation_training import validate_forecast_windows


def test_legacy_72h_window_cannot_supervise_240h():
    rows = [{'split': 'train', 'issue': '2021-12-25T00:00:00Z'}]
    validate_forecast_windows(rows, list(range(3, 73, 3)))
    with pytest.raises(ValueError, match='crosses train'):
        validate_forecast_windows(rows, list(range(3, 241, 3)))


@pytest.mark.parametrize('split,issue', [
    ('validation', '2022-06-25T00:00:00Z'),
    ('test', '2022-12-25T00:00:00Z'),
    ('validation', '2022-01-01T06:00:00Z'),
])
def test_all_split_boundaries_are_enforced(split, issue):
    with pytest.raises(ValueError, match='crosses'):
        validate_forecast_windows([{'split': split, 'issue': issue}], [240])


def test_contained_240h_window():
    validate_forecast_windows([{'split': 'train', 'issue': '2021-12-01T00:00:00Z'}], [240])
