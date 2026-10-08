"""Synthetic software checks; these do not establish station forecast accuracy."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid, latlon
from global_weather.observation_model import (
    LEAD_HOURS, NearestSphericalOperator, StationObservationModel,
    exact_lead_index, station_loss,
)


def model_fixture():
    grids = build_pyramid(0)
    coordinates = latlon(grids[0].xyz[:3])
    mean = [280., 270., 1., 2., 90000., 101000.]
    std = [12., 10., 5., 6., 3000., 2000.]
    return StationObservationModel(grids, coordinates, mean, std, hidden=8)


def test_nearest_spherical_operator_constant_and_cell_analytic_cases():
    grid = build_pyramid(0)[0]
    coordinates = latlon(grid.xyz)
    operator = NearestSphericalOperator(grid, coordinates)
    field = torch.arange(grid.n_cells, dtype=torch.float32)[:, None]
    assert torch.equal(operator(field), field)
    assert torch.equal(operator(torch.ones_like(field) * 17), torch.ones_like(field) * 17)
    dateline = NearestSphericalOperator(grid, [[0., 180.], [0., -180.]])
    assert int(dateline.cells[0]) == int(dateline.cells[1])
    with pytest.raises(ValueError, match="global grid"):
        operator(field[:-1])


def test_exact_time_operator_rejects_rounding_and_future_out_of_horizon():
    issue = datetime(2021, 1, 1, tzinfo=timezone.utc)
    assert exact_lead_index(issue, issue + timedelta(hours=72)) == 23
    assert exact_lead_index(issue, issue + timedelta(hours=3)) == 0
    for hours in (0, 3.0001, 73, -3):
        with pytest.raises(ValueError, match="exactly"):
            exact_lead_index(issue, issue + timedelta(hours=hours))


def test_full_sphere_37_layers_and_explicit_untrained_profile_mask():
    model = model_fixture()
    history = model.train_mean.expand(12, 3, 6).clone()
    mask = torch.ones_like(history, dtype=torch.bool)
    forecast = model(history, mask)
    assert forecast.native_normalized.shape == (24, 3, 6)
    assert forecast.lead_hours == LEAD_HOURS
    assert forecast.profile_diagnostics.shape == (12, 37, 6)
    assert not forecast.profile_target_mask.any()
    assert forecast.scientific_acceptance is False
    assert "no physical units" in forecast.profile_units
    assert tuple(model.pressure_hpa.tolist())[-1] == 1


def test_missing_nan_measurements_never_enter_state_or_loss():
    model = model_fixture()
    history = model.train_mean.expand(12, 3, 6).clone()
    mask = torch.ones_like(history, dtype=torch.bool)
    mask[:, 1, 2] = False
    history[:, 1, 2] = float("nan")
    forecast = model(history, mask)
    assert torch.isfinite(forecast.native_normalized).all()
    targets = model.train_mean.expand(24, 3, 6).clone()
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    target_mask[..., 5] = False
    targets[..., 5] = float("nan")
    loss = station_loss(forecast.native_normalized, targets, target_mask,
                        model.train_mean, model.train_std)
    assert torch.isfinite(loss)


def test_station_gradient_audit_is_honest_about_unsupervised_profiles():
    torch.manual_seed(7)
    model = model_fixture()
    history = model.train_mean + torch.randn(12, 3, 6) * model.train_std
    forecast = model(history, torch.ones_like(history, dtype=torch.bool))
    target = model.train_mean + torch.ones(24, 3, 6) * model.train_std
    loss = station_loss(forecast.native_normalized, target, torch.ones_like(target, dtype=torch.bool),
                        model.train_mean, model.train_std)
    loss.backward()
    supported = {name: parameter for name, parameter in model.named_parameters()
                 if not name.startswith("profile_head.")}
    for name, parameter in supported.items():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    assert model.profile_head.weight.grad is None
    assert model.profile_head.bias.grad is None
    assert not forecast.scientific_acceptance


def test_station_loss_excludes_missing_values_and_weights_variables_equally():
    prediction = torch.zeros(2, 2, 6, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., 0] = 2
    target[0, 0, 1] = 1
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    mask[..., 0] = True
    mask[0, 0, 1] = True
    loss = station_loss(prediction, target, mask, torch.zeros(6), torch.ones(6))
    assert float(loss.detach()) == pytest.approx((1.5 + .5) / 2)
    loss.backward()
    assert not prediction.grad[..., 2:].any()
    with pytest.raises(ValueError, match="finite admitted"):
        station_loss(prediction, target, torch.zeros_like(mask), torch.zeros(6), torch.ones(6))


def test_no_invented_statistics_or_silent_history_truncation():
    grids = build_pyramid(0)
    for bad_std in ([1] * 5, [1, 1, 0, 1, 1, 1], [float("nan")] * 6):
        with pytest.raises(ValueError, match="train-only"):
            StationObservationModel(grids, [[0, 0]], [0] * 6, bad_std, hidden=8)
    model = model_fixture()
    with pytest.raises(ValueError, match="12, stations, 6"):
        model(torch.zeros(13, 3, 6), torch.ones(13, 3, 6, dtype=torch.bool))
    history = model.train_mean.expand(12, 3, 6).clone()
    history[0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        model(history, torch.ones_like(history, dtype=torch.bool))


def test_new_model_has_no_empirical_upper_air_or_reanalysis_buffers():
    model = model_fixture()
    names = set(dict(model.named_buffers()))
    assert "train_mean" in names and "train_std" in names
    assert not any("profile_mean" in name or "profile_std" in name for name in names)
    assert np.array_equal(model.train_mean.numpy(), [280., 270., 1., 2., 90000., 101000.])
