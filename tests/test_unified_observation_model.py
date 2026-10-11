"""Tests for the joint surface and upper-air observation model."""
import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.unified_observation_model import (
    UnifiedObservationModel,
    unified_loss,
    LEAD_HOURS,
    UnifiedForecast,
)


def fixture_setup(n_stations: int = 16, hidden: int = 32):
    grid = build_grid(1)
    np.random.seed(42)
    station_latlon = np.column_stack([
        np.random.uniform(-80, 80, n_stations),
        np.random.uniform(-180, 180, n_stations),
    ])
    surf_scales = torch.tensor([16.0, 14.0, 4.0, 4.0, 5000.0, 1200.0], dtype=torch.float32)
    prof_scales = torch.tensor([15.0, 0.005, 10.0, 10.0, 50000.0], dtype=torch.float32)
    clim_mean = np.random.randn(12, grid.xyz.shape[0], 37, 5).astype(np.float32)

    model = UnifiedObservationModel(
        grid=grid,
        station_latlon=station_latlon,
        surface_scales=surf_scales,
        profile_scales=prof_scales,
        hidden=hidden,
        climatology_grid_mean=clim_mean,
    )
    return model, grid, station_latlon


def test_unified_model_forward():
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]

    surf_anom = torch.randn(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    prof_anom = torch.randn(12, n_cells, 37, 5)
    prof_mask = torch.ones(12, n_cells, 37, 5, dtype=torch.bool)

    forecast = model(surf_anom, surf_mask, month_idx=0, profile_anomalies=prof_anom, profile_mask=prof_mask)

    assert isinstance(forecast, UnifiedForecast)
    assert forecast.surface_anomalies.shape == (24, n_stations, 6)
    assert forecast.profile_anomalies.shape == (24, n_cells, 37, 5)
    assert forecast.lead_hours == LEAD_HOURS
    assert not forecast.scientific_acceptance


def test_unified_model_gradient_flow():
    torch.manual_seed(42)
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]

    surf_anom = torch.randn(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    prof_anom = torch.randn(12, n_cells, 37, 5)
    prof_mask = torch.ones(12, n_cells, 37, 5, dtype=torch.bool)

    forecast = model(surf_anom, surf_mask, month_idx=5, profile_anomalies=prof_anom, profile_mask=prof_mask)

    target_surf = torch.randn(24, n_stations, 6)
    mask_surf = torch.ones(24, n_stations, 6, dtype=torch.bool)
    target_prof = torch.randn(24, n_cells, 37, 5)
    mask_prof = torch.ones(24, n_cells, 37, 5, dtype=torch.bool)

    loss = unified_loss(forecast, target_surf, mask_surf, target_prof, mask_prof)
    assert torch.isfinite(loss)
    loss.backward()

    # Check that ALL trainable parameters have nonzero gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Nonfinite gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_unified_loss_masking():
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]

    surf_anom = torch.zeros(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)

    forecast = model(surf_anom, surf_mask, month_idx=0)

    # Valid targets
    target_surf = torch.randn(24, n_stations, 6)
    mask_surf = torch.zeros(24, n_stations, 6, dtype=torch.bool)
    mask_surf[:, :5, :2] = True # Partial observations

    loss = unified_loss(forecast, target_surf, mask_surf)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0
