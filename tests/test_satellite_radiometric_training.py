"""Unit tests for Stage S1: Russian satellite radiometric warmup."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.station_climatology import StationClimatology
from global_weather.station_anomaly_model import StationAnomalyModel
from global_weather.satellite_radiometric_training import (
    SatelliteRadiometricModel,
    generate_satellite_batch,
    load_base_model,
)


def create_synthetic_base_model():
    grid = build_grid(0)
    n_stations = 12
    station_latlon = np.zeros((n_stations, 2), dtype=np.float32)
    scales = np.ones(6, dtype=np.float32)
    base_model = StationAnomalyModel([grid], station_latlon, scales, hidden=16)
    return grid, base_model


def test_base_model_freezing():
    _, base = create_synthetic_base_model()
    for p in base.parameters():
        p.requires_grad = False
    base.eval()

    sat_model = SatelliteRadiometricModel(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
    )

    # Base parameters must have requires_grad = False
    for p in sat_model.base_model.parameters():
        assert not p.requires_grad

    # Trainable satellite parameters must have requires_grad = True
    trainable = [p for p in sat_model.parameters() if p.requires_grad]
    assert len(trainable) > 0
    assert sat_model.gsics_slope.requires_grad
    assert sat_model.gsics_offset.requires_grad


def test_satellite_model_gradient_flow():
    grid, base = create_synthetic_base_model()
    sat_model = SatelliteRadiometricModel(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
    )

    device = torch.device('cpu')
    tx_anom = torch.zeros(12, 12, 6, device=device)
    tx_mask = torch.ones(12, 12, 6, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, base_profiles = sat_model.base_model.analysis_state(tx_anom, tx_mask)

    sat_inputs = generate_satellite_batch(
        grid.n_cells,
        3,
        device,
        model=sat_model,
        base_profiles=base_profiles,
    )

    total_loss, metrics = sat_model(tx_anom, tx_mask, sat_inputs)
    assert torch.isfinite(total_loss)
    assert metrics['loss_sat_total'] > 0

    total_loss.backward()

    # Base model parameters must have grad is None
    for name, p in sat_model.base_model.named_parameters():
        assert p.grad is None, f"Frozen parameter {name} received gradient!"

    # Satellite modules must receive nonzero gradients
    assert sat_model.gsics_slope.grad is not None and sat_model.gsics_slope.grad.abs().sum() > 0
    assert sat_model.gsics_offset.grad is not None and sat_model.gsics_offset.grad.abs().sum() > 0
    assert sat_model.msugs_proj.weight.grad is not None and sat_model.msugs_proj.weight.grad.abs().sum() > 0

    film_grads = [p.grad for p in sat_model.film_block.parameters() if p.grad is not None]
    assert film_grads and all(torch.isfinite(g).all() for g in film_grads)
    assert sum(float(g.abs().sum()) for g in film_grads) > 0

    ikfs2_grads = [p.grad for p in sat_model.ikfs2_operator.forward_rt_surrogate.parameters() if p.grad is not None]
    assert ikfs2_grads and sum(float(g.abs().sum()) for g in ikfs2_grads) > 0

    mtvza_grads = [p.grad for p in sat_model.mtvza_operator.allsky_rt_surrogate.parameters() if p.grad is not None]
    assert mtvza_grads and sum(float(g.abs().sum()) for g in mtvza_grads) > 0

    varbc_grads = [p.grad for p in sat_model.ikfs2_operator.varbc.parameters() if p.grad is not None]
    assert varbc_grads and sum(float(g.abs().sum()) for g in varbc_grads) > 0


def test_real_checkpoint_s1_coupling():
    ckpt_path = Path("outputs/extended_2023_2026/training/epoch-0081/state.pt")
    clim_path = Path("outputs/extended_2023_2026/training/station_climatology.npz")
    if not ckpt_path.is_file() or not clim_path.is_file():
        pytest.skip("Real checkpoint or climatology not present in workspace")

    clim = StationClimatology.load(clim_path)
    device = torch.device('cpu')
    base_model = load_base_model(ckpt_path, clim, mesh_level=4, hidden=32, device=device)

    sat_model = SatelliteRadiometricModel(
        base_model,
        n_sats=3,
        hidden_dim=32,
        num_ikfs2=180,
        num_mtvza=29,
    ).to(device)

    # Base parameters must be completely frozen
    for p in sat_model.base_model.parameters():
        assert not p.requires_grad

    assert sat_model.base_model.xyz.shape[0] == 2562  # Mesh-4 grid cells

