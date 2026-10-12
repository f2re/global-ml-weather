"""Causality and chronological windows for the joint observation adapter."""
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from global_weather.joint_observation_data import (
    BOUNDARIES, JointObservationDataset, START, common_windows, _validate_issue,
)
from global_weather.observations import utc


@pytest.mark.parametrize('horizon', [72, 240])
def test_windows_preserve_full_declared_horizon_and_profile_guards(horizon):
    rows = common_windows(horizon)
    assert {row['split'] for row in rows} == set(BOUNDARIES)
    for row in rows:
        begin, end = BOUNDARIES[row['split']]
        issue = utc(row['issue'])
        assert issue - timedelta(hours=54) >= begin
        assert issue + timedelta(hours=horizon + 42) < end
        assert _validate_issue(issue, horizon)[1] == row['split']


@pytest.mark.parametrize('horizon', [0, 1, 73, 241, True])
def test_invalid_horizon_rejected(horizon):
    with pytest.raises(ValueError):
        common_windows(horizon)


def test_old_short_window_cannot_be_reused_at_longer_horizon():
    _, end = BOUNDARIES['train']
    issue = end - timedelta(hours=140)
    _validate_issue(issue, 72)
    with pytest.raises(ValueError, match='crosses'):
        _validate_issue(issue, 240)


def _adapter(profile):
    dataset = JointObservationDataset.__new__(JointObservationDataset)
    dataset.horizon_hours = 72
    dataset.lead_hours = list(range(3, 73, 3))
    values = np.ones((200, 1, 6), dtype=np.float32)
    mask = np.ones_like(values, dtype=bool)
    available = np.zeros_like(values)
    dataset.surface = SimpleNamespace(values=values, mask=mask, available=available)
    dataset.profile = profile
    return dataset


def test_forecast_only_reads_past_and_masks_late_surface_measurement():
    issue = START + timedelta(hours=60)
    calls = []

    class Profile:
        def records(self, begin, end, **kwargs):
            calls.append((begin, end, kwargs))
            assert end <= issue
            assert kwargs['issue'] == issue
            return []

    dataset = _adapter(Profile())
    dataset.surface.available[60, 0, 0] = 61 * 3600
    dataset.surface.values[61:] = np.nan
    values, mask, profiles = dataset.forecast_history(issue)
    assert values.shape == (12, 1, 6)
    assert values[-1, 0, 0] == 0
    assert not mask[-1, 0, 0]
    assert np.isfinite(values).all()
    assert profiles == []
    assert calls == [(issue - timedelta(hours=12), issue, {'issue': issue})]


def test_profile_scalar_actual_pressure_time_and_identity_preserved():
    issue = START + timedelta(hours=60)
    measurement = {'pressure_pa': 84231.0, 'variable': 'temperature',
                   'observation_id': 'real-scalar-id',
                   'observed_at': (issue + timedelta(minutes=17)).isoformat()}

    class Profile:
        def records(self, begin, end, **kwargs):
            return [] if kwargs.get('issue') else [measurement]

    dataset = _adapter(Profile())
    x, xm, y, ym, profile_inputs, profile_targets = dataset.sample(issue)
    assert x.shape == xm.shape == (12, 1, 6)
    assert y.shape == ym.shape == (24, 1, 6)
    assert profile_inputs == []
    assert profile_targets == [measurement]
    assert profile_targets[0] is measurement


def test_forecast_refuses_fractional_hour_instead_of_rounding():
    dataset = _adapter(None)
    with pytest.raises(ValueError, match='integer-hour'):
        dataset.forecast_history(START + timedelta(hours=60, minutes=17))
