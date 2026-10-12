"""Causality and immutable batch publication, not weather skill."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from global_weather.joint_observation_inference import selected_issues, validate_history


def test_issue_selection_never_samples_targets():
    issue = datetime(2024, 8, 4, tzinfo=timezone.utc)
    class PastOnly:
        def issues(self, split, limit):
            assert (split, limit) == ('validation', 3)
            return [issue + timedelta(days=1), issue]
        def sample(self, *args):
            raise AssertionError('Inference queried target-bearing sample.')
    assert selected_issues(PastOnly()) == [issue, issue + timedelta(days=1)]
    assert selected_issues(PastOnly(), issues=[issue.isoformat()]) == [issue]
    with pytest.raises(ValueError, match='distinct'):
        selected_issues(PastOnly(), issues=[issue.isoformat()] * 2)
    with pytest.raises(ValueError, match='whole'):
        selected_issues(PastOnly(), issues=[(issue + timedelta(seconds=1)).isoformat()])


@pytest.mark.parametrize('observed_delta,available_delta', [(1, 0), (0, 1), (-12, 0), (-13, 0)])
def test_late_future_and_outside_history_profiles_rejected(observed_delta, available_delta):
    issue = datetime(2024, 8, 4, tzinfo=timezone.utc)
    values = np.zeros((12, 2, 6), dtype=np.float32)
    mask = np.ones(values.shape, dtype=bool)
    row = {'observed_at': (issue + timedelta(hours=observed_delta)).isoformat(),
           'available_at': (issue + timedelta(hours=available_delta)).isoformat()}
    with pytest.raises(ValueError, match='Future, late'):
        validate_history(values, mask, [row], issue)


def test_masked_nan_input_and_unregistered_profile_are_allowed():
    issue = datetime(2024, 8, 4, tzinfo=timezone.utc)
    values = np.ones((12, 2, 6), dtype=np.float32)
    mask = np.ones(values.shape, dtype=bool)
    values[0, 0, 0] = np.nan
    mask[0, 0, 0] = False
    row = {'observed_at': issue.isoformat(), 'available_at': issue.isoformat()}
    validate_history(values, mask, [row], issue)
    mask[0, 0, 0] = True
    with pytest.raises(ValueError, match='nonfinite'):
        validate_history(values, mask, [row], issue)


def test_issue_bundle_binds_grib_inputs_and_prevents_overwrite(tmp_path, monkeypatch):
    pytest.importorskip('eccodes', reason='Required in GRIB acceptance environment')
    import json
    from types import SimpleNamespace
    import torch
    import global_weather.joint_observation_inference as module
    import global_weather.profile_physics as physics
    from global_weather.grid import build_grid
    from global_weather.grib_export import sha256
    issue = datetime(2024, 8, 4, tzinfo=timezone.utc)
    grid = build_grid(0)
    values = np.ones((12, 1, 6), dtype=np.float32)
    mask = np.ones(values.shape, dtype=bool)
    profiles = [{'observed_at': issue.isoformat(), 'available_at': issue.isoformat()}]
    class PastOnly:
        surface = SimpleNamespace(latlon=np.array([[45., 0.]]))
        fingerprint = {'input_sha256': 'a'*64}
        def forecast_history(self, when, limit):
            assert when == issue
            return values, mask, profiles
        def sample(self, *args):
            raise AssertionError('Target window opened.')
    prediction = torch.zeros((grid.n_cells, 37, 6))
    prediction[..., 0] = 280
    prediction[..., 1] = .003
    prediction[..., 4] = 100000
    prediction[..., 5] = float('nan')
    valid = torch.isfinite(prediction)
    frames = [SimpleNamespace(profiles=prediction, profile_variable_mask=valid,
                              native_physical=torch.tensor([[280., 270., 2., 3., 95000., 101325.]]),
                              lead_hours=lead, valid_time=issue+timedelta(hours=lead)) for lead in (0, 3)]
    monkeypatch.setattr(module, 'timed_forecast', lambda *args, **kwargs:
                        (frames, {'median_seconds': .1}))
    monkeypatch.setattr(physics, 'screen_humidity', lambda rows, policy: (rows, {}))
    model = SimpleNamespace(grid=grid, physical_policy=object())
    state = {'epoch': 2, 'config': {}, 'identity': {},
             'trained_variable_mask': np.ones((37, 5), dtype=bool).tolist()}
    output = tmp_path / 'issue'
    result = module.save_issue(PastOnly(), model, state, 'a'*64, issue, output,
                               horizon_hours=3, spacing=90,
                               static_fields={'elevation_m': np.zeros(grid.n_cells), 'static_sha256': 'b'*64})
    assert result['messages'] == 2*37*6+2
    manifest = json.loads((output / 'forecast.json').read_text())
    inventory = json.loads((output / 'forecast.grib2.json').read_text())
    assert manifest['future_targets_read'] is False
    assert inventory['forecast_metadata_sha256'] == sha256(output / 'forecast.json')
    assert manifest['input_provenance']['surface_input_sha256'] == sha256(output / 'inputs.npz')
    assert json.loads((output / 'profile_inputs.json').read_text()) == profiles
    assert manifest['diagnostic_physics']['omega_supervised'] is False
    with pytest.raises(FileExistsError):
        module.save_issue(PastOnly(), model, state, 'a'*64, issue, output)
