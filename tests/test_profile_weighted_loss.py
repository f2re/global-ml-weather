"""Tests for vertical pressure weighting, humidity thresholding, and hydrostatic balance."""
from datetime import timedelta
import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid
from global_weather.profile_graphcast_normalization import GraphCastNormalization, _reference
from global_weather.profile_model_v2 import PressureProfileModel
from global_weather.profile_training import START, VARIABLES
from global_weather.profile_training_v2 import (
    MIN_HUMIDITY_PRESSURE_PA,
    objective,
    vertical_pressure_weights,
)
from global_weather.vertical import (
    PRESSURE_HPA,
    PROFILE_UNITS,
    hydrostatic_residual,
)


def normalization_fixture():
    return GraphCastNormalization({
        **_reference(),
        'source_identity': {
            key: '0' * 64
            for key in ('database_sha256', 'dataset_manifest_sha256', 'source_sha256', 'admission_sha256')
        },
    })


def test_vertical_pressure_weights_properties():
    weights = vertical_pressure_weights()
    assert len(weights) == len(PRESSURE_HPA) == 37
    assert np.all(weights > 0)
    assert pytest.approx(float(weights.mean()), abs=1e-6) == 1.0
    assert pytest.approx(float(weights.sum()), abs=1e-6) == 37.0
    # Tropospheric layer (700 hPa) has substantially greater atmospheric mass than 1 hPa layer
    idx_700 = PRESSURE_HPA.index(700)
    idx_1 = PRESSURE_HPA.index(1)
    assert weights[idx_700] > weights[idx_1] * 10


def test_objective_masks_stratospheric_humidity_noise():
    torch.manual_seed(29)
    norms = normalization_fixture()
    grid = build_pyramid(0)[0]
    model = PressureProfileModel(grid, norms, 8)
    issue = START + timedelta(days=3)

    # Synthetic targets: normal tropospheric values plus extreme stratospheric humidity noise (50 g/kg at 50 hPa)
    records = [
        dict(
            variable=name,
            value=[270.0, 0.002, 4.0, -3.0, 30000.0][i],
            units=PROFILE_UNITS[i],
            latitude=10.0,
            longitude=20.0,
            pressure_pa=70000.0,
            observed_at=issue.isoformat(),
            available_at=issue.isoformat(),
            profile_id='fixture',
        )
        for i, name in enumerate(VARIABLES)
    ]
    # Extreme stratospheric humidity artifact (e.g. 50 g/kg at 50 hPa, which would give ~1e11 squared error)
    corrupted_strat_q = dict(
        variable='specific_humidity',
        value=0.050,
        units='kg kg-1',
        latitude=10.0,
        longitude=20.0,
        pressure_pa=5000.0,  # 50 hPa (< 300 hPa)
        observed_at=issue.isoformat(),
        available_at=issue.isoformat(),
        profile_id='fixture',
    )
    targets = [dict(row, observed_at=(issue + timedelta(hours=12)).isoformat()) for row in records]
    targets_with_corrupted = targets + [corrupted_strat_q]

    frames = model(records, issue)

    # With default threshold (30 000 Pa = 300 hPa), corrupted stratospheric q is masked
    loss_masked, counts_masked = objective(model, frames, targets_with_corrupted)
    assert counts_masked[1] == 1  # Only the tropospheric q record at 700 hPa is counted
    assert loss_masked.item() < 10.0  # Normalized loss remains physical (order of 1.0)

    # Without threshold (min_humidity_pressure_pa=0), the corrupted record explodes the loss
    loss_unmasked, counts_unmasked = objective(model, frames, targets_with_corrupted, min_humidity_pressure_pa=0.0)
    assert counts_unmasked[1] == 2
    assert loss_unmasked.item() > 1000.0  # Explodes due to 1/std^2 in stratosphere

    # Verify gradients flow with finite nonzero values
    loss_masked.backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert (model.head.weight.grad.abs().sum(dim=1) > 0).all()


def test_hydrostatic_residual_soft_loss():
    torch.manual_seed(29)
    norms = normalization_fixture()
    grid = build_pyramid(0)[0]
    model = PressureProfileModel(grid, norms, 8)
    issue = START + timedelta(days=3)

    records = [
        dict(
            variable=name,
            value=[270.0, 0.001, 4.0, -3.0, 30000.0][i],
            units=PROFILE_UNITS[i],
            latitude=10.0,
            longitude=20.0,
            pressure_pa=70000.0,
            observed_at=issue.isoformat(),
            available_at=issue.isoformat(),
            profile_id='fixture',
        )
        for i, name in enumerate(VARIABLES)
    ]
    frames = model(records, issue)
    profiles = frames[0].profiles[..., :5]

    residual = hydrostatic_residual(profiles, model.pressure_pa)
    assert residual.shape == (grid.n_cells, 36)
    assert torch.isfinite(residual).all()

    hydro_loss = (residual / 1000.0).square().mean()
    assert hydro_loss.item() >= 0.0
    hydro_loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()
