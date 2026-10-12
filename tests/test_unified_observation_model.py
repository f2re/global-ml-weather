"""Tests for the joint surface, upper-air, satellite, and DEM observation model."""
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

    # Synthetic terrain features for unit tests
    t_elev = np.random.uniform(-100, 3000, grid.xyz.shape[0]).astype(np.float32)
    t_land = np.random.uniform(0, 1, grid.xyz.shape[0]).astype(np.float32)
    t_slope = np.random.uniform(0, 0.01, grid.xyz.shape[0]).astype(np.float32)
    st_elev = np.random.uniform(0, 2000, n_stations).astype(np.float32)

    model = UnifiedObservationModel(
        grid=grid,
        station_latlon=station_latlon,
        surface_scales=surf_scales,
        profile_scales=prof_scales,
        hidden=hidden,
        climatology_grid_mean=clim_mean,
        terrain_elevation_m=t_elev,
        terrain_land_fraction=t_land,
        terrain_slope=t_slope,
        station_elevation_m=st_elev,
    )
    return model, grid, station_latlon


def test_unified_model_forward():
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]
    n_leads = len(LEAD_HOURS)

    surf_anom = torch.randn(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    prof_anom = torch.randn(12, n_cells, 37, 5)
    prof_mask = torch.ones(12, n_cells, 37, 5, dtype=torch.bool)
    sat_anom = torch.randn(12, n_cells, 1)
    sat_mask = torch.ones(12, n_cells, 1, dtype=torch.bool)

    forecast = model(
        surf_anom,
        surf_mask,
        month_idx=0,
        profile_anomalies=prof_anom,
        profile_mask=prof_mask,
        satellite_anomalies=sat_anom,
        satellite_mask=sat_mask,
    )

    assert isinstance(forecast, UnifiedForecast)
    assert forecast.surface_anomalies.shape == (n_leads, n_stations, 6)
    assert forecast.profile_anomalies.shape == (n_leads, n_cells, 37, 5)
    assert forecast.lead_hours == LEAD_HOURS
    assert len(forecast.lead_hours) == 80  # 10 days / 240 hours at 3h intervals
    assert not forecast.scientific_acceptance


def test_unified_model_gradient_flow():
    torch.manual_seed(42)
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]
    n_leads = len(LEAD_HOURS)

    surf_anom = torch.randn(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)
    prof_anom = torch.randn(12, n_cells, 37, 5)
    prof_mask = torch.ones(12, n_cells, 37, 5, dtype=torch.bool)
    sat_anom = torch.randn(12, n_cells, 1)
    sat_mask = torch.ones(12, n_cells, 1, dtype=torch.bool)

    forecast = model(
        surf_anom,
        surf_mask,
        month_idx=5,
        profile_anomalies=prof_anom,
        profile_mask=prof_mask,
        satellite_anomalies=sat_anom,
        satellite_mask=sat_mask,
    )

    target_surf = torch.randn(n_leads, n_stations, 6)
    mask_surf = torch.ones(n_leads, n_stations, 6, dtype=torch.bool)
    target_prof = torch.randn(n_leads, n_cells, 37, 5)
    mask_prof = torch.ones(n_leads, n_cells, 37, 5, dtype=torch.bool)

    loss = unified_loss(forecast, target_surf, mask_surf, target_prof, mask_prof)
    assert torch.isfinite(loss)
    loss.backward()

    # Check that ALL trainable parameters have nonzero finite gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Nonfinite gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def test_unified_loss_masking():
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_leads = len(LEAD_HOURS)

    surf_anom = torch.zeros(12, n_stations, 6)
    surf_mask = torch.ones(12, n_stations, 6, dtype=torch.bool)

    forecast = model(surf_anom, surf_mask, month_idx=0)

    # Partial observation target
    target_surf = torch.randn(n_leads, n_stations, 6)
    mask_surf = torch.zeros(n_leads, n_stations, 6, dtype=torch.bool)
    mask_surf[:24, :5, :2] = True

    loss = unified_loss(forecast, target_surf, mask_surf)
    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_unified_model_batched_forward_and_backward():
    torch.manual_seed(42)
    model, grid, station_latlon = fixture_setup()
    n_stations = len(station_latlon)
    n_cells = grid.xyz.shape[0]
    n_leads = len(LEAD_HOURS)
    B = 4

    surf_anom = torch.randn(B, 12, n_stations, 6)
    surf_mask = torch.ones(B, 12, n_stations, 6, dtype=torch.bool)
    prof_anom = torch.randn(B, 12, n_cells, 37, 5)
    prof_mask = torch.ones(B, 12, n_cells, 37, 5, dtype=torch.bool)
    sat_anom = torch.randn(B, 12, n_cells, 1)
    sat_mask = torch.ones(B, 12, n_cells, 1, dtype=torch.bool)

    forecast = model(
        surf_anom,
        surf_mask,
        month_idx=[0, 3, 6, 9],
        profile_anomalies=prof_anom,
        profile_mask=prof_mask,
        satellite_anomalies=sat_anom,
        satellite_mask=sat_mask,
        issue_hour=[0, 6, 12, 18],
    )

    assert isinstance(forecast, UnifiedForecast)
    assert forecast.surface_anomalies.shape == (B, n_leads, n_stations, 6)
    assert forecast.profile_anomalies.shape == (B, n_leads, n_cells, 37, 5)

    target_surf = torch.randn(B, n_leads, n_stations, 6)
    mask_surf = torch.ones(B, n_leads, n_stations, 6, dtype=torch.bool)
    target_prof = torch.randn(B, n_leads, n_cells, 37, 5)
    mask_prof = torch.ones(B, n_leads, n_cells, 37, 5, dtype=torch.bool)

    loss = unified_loss(forecast, target_surf, mask_surf, target_prof, mask_prof)
    assert torch.isfinite(loss)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert torch.isfinite(param.grad).all(), f"Nonfinite gradient for {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"
