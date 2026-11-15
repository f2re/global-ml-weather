"""Unit tests for Stage S2: Joint unfreezing of weather forecast core and satellite operators."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.station_climatology import StationClimatology
from global_weather.station_anomaly_model import StationAnomalyModel
from global_weather.satellite_radiometric_training import generate_satellite_batch
from global_weather.satellite_joint_stage2_training import (
    JointSatelliteStage2Model,
    load_s2_model,
)


def create_synthetic_base_model():
    grid = build_grid(0)
    n_stations = 12
    station_latlon = np.zeros((n_stations, 2), dtype=np.float32)
    scales = np.ones(6, dtype=np.float32)
    base_model = StationAnomalyModel([grid], station_latlon, scales, hidden=16)
    return grid, base_model


def test_joint_model_unfreezing():
    grid, base = create_synthetic_base_model()
    model = JointSatelliteStage2Model(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
    )

    # In Stage S2, both base and satellite parameters must be trainable
    base_trainable = [p for p in model.base_model.parameters() if p.requires_grad]
    sat_trainable = [p for n, p in model.named_parameters() if not n.startswith('base_model.') and p.requires_grad]

    assert len(base_trainable) == sum(1 for _ in model.base_model.parameters())
    assert len(sat_trainable) > 0


def test_joint_gradient_flow_to_both_forecast_and_satellite_branches():
    grid, base = create_synthetic_base_model()
    model = JointSatelliteStage2Model(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
    )

    device = torch.device('cpu')
    n_stations = 12
    tx_anom = torch.zeros(12, n_stations, 6, device=device)
    tx_mask = torch.ones(12, n_stations, 6, dtype=torch.bool, device=device)
    ty_anom = torch.zeros(24, n_stations, 6, device=device)
    ty_mask = torch.ones(24, n_stations, 6, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)

    sat_inputs = generate_satellite_batch(
        grid.n_cells,
        3,
        device,
        base_profiles=base_profiles,
    )

    total_loss, metrics = model(tx_anom, tx_mask, ty_anom, ty_mask, sat_inputs)
    assert torch.isfinite(total_loss)
    assert metrics['loss_surface'] >= 0
    assert metrics['loss_sat'] > 0
    assert metrics['loss_hydro'] >= 0

    total_loss.backward()

    # Base forecast parameters must receive nonzero gradients
    base_grads = [p.grad for p in model.base_model.parameters() if p.grad is not None]
    assert len(base_grads) > 0
    assert sum(float(g.abs().sum()) for g in base_grads) > 0

    # Satellite parameters must also receive nonzero gradients
    assert model.gsics_slope.grad is not None and model.gsics_slope.grad.abs().sum() > 0
    assert model.gsics_offset.grad is not None and model.gsics_offset.grad.abs().sum() > 0
    assert model.msugs_proj.weight.grad is not None and model.msugs_proj.weight.grad.abs().sum() > 0

    film_grads = [p.grad for p in model.film_block.parameters() if p.grad is not None]
    assert film_grads and sum(float(g.abs().sum()) for g in film_grads) > 0


def test_real_s1_and_s0_checkpoints_loading_s2():
    base_ckpt = Path("outputs/extended_2023_2026/training/epoch-0081/state.pt")
    sat_ckpt = Path("outputs/s1_satellite_radiometric_warmup/training/epoch-0007/state.pt")
    clim_path = Path("outputs/extended_2023_2026/training/station_climatology.npz")

    if not base_ckpt.is_file() or not sat_ckpt.is_file() or not clim_path.is_file():
        pytest.skip("Required checkpoints not found in local workspace")

    clim = StationClimatology.load(clim_path)
    device = torch.device('cpu')
    model = load_s2_model(
        base_ckpt,
        sat_ckpt,
        clim,
        mesh_level=4,
        hidden=32,
        config={'hydrostatic_loss_weight': 1.0, 'sat_assimilation_loss_weight': 0.25},
        device=device,
    )

    assert model.base_model.xyz.shape[0] == 2562
    assert all(p.requires_grad for p in model.base_model.parameters())
    assert model.gsics_slope.requires_grad
