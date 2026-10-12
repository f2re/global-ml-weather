"""Causality and checkpoint compatibility tests (synthetic software evidence)."""
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.observation_data import START
from global_weather.station_climatology import StationClimatology
from global_weather.unified_observation_inference import (
    benchmark, causal_history, checked_hash, load_model, sha256,
)
from global_weather.unified_observation_model import UnifiedObservationModel


def fixture_dataset():
    # Sentinel future values expose any accidental target-window access.
    values = np.ones((30, 2, 6), dtype=np.float32)
    values[12:] = np.nan
    return SimpleNamespace(values=values, mask=np.ones_like(values, dtype=bool),
                           available=np.zeros_like(values), latlon=np.array([[0., 0.], [30., 60.]]))


def test_causal_history_does_not_access_future_and_masks_late_observations():
    dataset = fixture_dataset()
    dataset.available[11, 0, 0] = 12 * 3600
    dataset.values[11, 0, 0] = np.nan
    clim = StationClimatology(np.zeros((12, 2, 6)), np.ones(6))
    x, mask, issue = causal_history(dataset, clim, (START + timedelta(hours=11)).isoformat())
    assert issue == START + timedelta(hours=11)
    assert x.shape == (12, 2, 6)
    assert not mask[11, 0, 0] and x[11, 0, 0] == 0
    assert np.isfinite(x).all()


def test_issue_requires_complete_history_and_exact_archive_hour():
    dataset = fixture_dataset()
    clim = StationClimatology(np.zeros((12, 2, 6)), np.ones(6))
    for when in [START, START + timedelta(hours=11, minutes=1)]:
        with pytest.raises(ValueError):
            causal_history(dataset, clim, when.isoformat())


def checkpoint_fixture(tmp_path):
    coords = np.array([[0., 0.], [30., 60.]])
    model = UnifiedObservationModel(build_grid(0), coords, np.ones(6), np.ones(5),
                                    hidden=16, lead_hours=[3, 6])
    path = tmp_path / 'state.pt'
    torch.save({'model': model.state_dict(), 'epoch': 1, 'lead_hours': [3, 6],
                'config': {'mesh_level': 0, 'hidden': 16}}, path)
    return path, coords


def test_checkpoint_loader_rejects_wrong_hash_geometry_norms_and_horizon(tmp_path):
    path, coords = checkpoint_fixture(tmp_path)
    with pytest.raises(ValueError, match='hash mismatch'):
        checked_hash(path, '0' * 64)
    for geom, std, horizon in [
        (coords[::-1].copy(), np.ones(6), 6),
        (coords, np.full(6, 2.), 6),
        (coords, np.ones(6), 72),
    ]:
        with pytest.raises(ValueError):
            load_model(path, sha256(path), geom, std, horizon, torch.device('cpu'))


def test_checkpoint_prefix_rollout_and_benchmark_are_finite(tmp_path):
    path, coords = checkpoint_fixture(tmp_path)
    model, state = load_model(path, sha256(path), coords, np.ones(6), 3, torch.device('cpu'))
    prediction, measured = benchmark(model, np.ones((12, 2, 6), dtype=np.float32),
                                    np.ones((12, 2, 6), dtype=bool), START,
                                    torch.device('cpu'), warmups=0, repeats=2)
    assert prediction.lead_hours == [3]
    assert torch.isfinite(prediction.surface_anomalies).all()
    assert len(measured['seconds']) == 2 and min(measured['seconds']) > 0
    assert measured['parameters_total'] == sum(p.numel() for p in model.parameters())
    assert measured['cuda_peak_allocated_bytes'] is None
    assert state['epoch'] == 1


def test_checkpoint_rejects_nonfinite_sparse_graph_without_densifying(tmp_path):
    path, coords = checkpoint_fixture(tmp_path)
    state = torch.load(path, weights_only=True)
    adjacency = state['model']['graph.adjacency'].coalesce()
    values = adjacency.values().clone()
    values[0] = float('nan')
    state['model']['graph.adjacency'] = torch.sparse_coo_tensor(
        adjacency.indices(), values, adjacency.shape,
    )
    torch.save(state, path)
    with pytest.raises(ValueError, match='Nonfinite checkpoint'):
        load_model(path, sha256(path), coords, np.ones(6), 3, torch.device('cpu'))
