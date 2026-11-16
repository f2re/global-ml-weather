"""Unit tests for Stage S3: All-sky microwave assimilation and Arctic sea ice conditioning."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from global_weather.grid import build_grid
from global_weather.station_climatology import StationClimatology
from global_weather.station_anomaly_model import StationAnomalyModel
from global_weather.satellite_allsky_stage3_training import (
    JointSatelliteStage3Model,
    generate_satellite_stage3_batch,
    load_s3_model,
)


def create_synthetic_base_model():
    grid = build_grid(0)
    n_stations = 12
    station_latlon = np.zeros((n_stations, 2), dtype=np.float32)
    scales = np.ones(6, dtype=np.float32)
    base_model = StationAnomalyModel([grid], station_latlon, scales, hidden=16)
    return grid, base_model


def test_stage3_model_unfreezing():
    grid, base = create_synthetic_base_model()
    model = JointSatelliteStage3Model(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
        sat_assimilation_loss_weight=0.5,
    )

    # In Stage S3, both forecast core and satellite parameters must be trainable
    base_trainable = [p for p in model.base_model.parameters() if p.requires_grad]
    sat_trainable = [p for n, p in model.named_parameters() if not n.startswith('base_model.') and p.requires_grad]

    assert len(base_trainable) == sum(1 for _ in model.base_model.parameters())
    assert len(sat_trainable) > 0


def test_stage3_allsky_and_arctic_ice_gradient_flow():
    grid, base = create_synthetic_base_model()
    model = JointSatelliteStage3Model(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
        sat_assimilation_loss_weight=0.5,
    )

    device = torch.device('cpu')
    n_stations = 12
    tx_anom = torch.zeros(12, n_stations, 6, device=device)
    tx_mask = torch.ones(12, n_stations, 6, dtype=torch.bool, device=device)
    ty_anom = torch.zeros(24, n_stations, 6, device=device)
    ty_mask = torch.ones(24, n_stations, 6, dtype=torch.bool, device=device)

    with torch.no_grad():
        _, base_profiles = model.base_model.analysis_state(tx_anom, tx_mask)

    sat_inputs = generate_satellite_stage3_batch(
        grid.n_cells,
        3,
        device,
        model=model,
        base_profiles=base_profiles,
        xyz_nodes=model.base_model.xyz,
    )

    total_loss, metrics = model(tx_anom, tx_mask, ty_anom, ty_mask, sat_inputs)
    assert torch.isfinite(total_loss)
    assert metrics['loss_surface'] >= 0
    assert metrics['loss_sat'] > 0
    assert metrics['loss_hydro'] >= 0
    assert metrics['loss_mtvza'] > 0
    assert metrics['c_sym_mean'] >= 0.0
    assert metrics['sea_ice_conditioned_count'] >= 0

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

    # MTVZA operator allsky surrogate parameters must receive gradients
    mtvza_grads = [p.grad for p in model.mtvza_operator.parameters() if p.grad is not None]
    assert mtvza_grads and sum(float(g.abs().sum()) for g in mtvza_grads) > 0


def test_arctic_ice_conditioning_masks_surface_channels():
    grid, base = create_synthetic_base_model()
    model = JointSatelliteStage3Model(
        base,
        n_sats=3,
        hidden_dim=16,
        num_ikfs2=180,
        num_mtvza=29,
    )
    device = torch.device('cpu')
    n_obs = 10
    obs_tb = torch.full((n_obs, 29), 240.0, device=device)
    obs_tb[:, 6] = 210.0
    obs_tb[:, 7] = 145.0
    m_prof = torch.zeros(n_obs, 37 * 3, device=device)
    varbc_p = torch.zeros(n_obs, 7, device=device)

    # 1. Clear ocean point (sea_ice = 0.0, land = 0.0)
    surf_ocean = torch.zeros(n_obs, 4, device=device)
    out_ocean = model.mtvza_operator(obs_tb, m_prof, surf_ocean, varbc_p)
    # Surface sensitive channels should be admitted over open ocean
    assert out_ocean['admissible_mask'][:, :12].sum() > 0

    # 2. Arctic ice point (sea_ice = 0.9, land = 0.0)
    surf_ice = torch.zeros(n_obs, 4, device=device)
    surf_ice[:, 3] = 0.9  # sea ice fraction > 0.05
    out_ice = model.mtvza_operator(obs_tb, m_prof, surf_ice, varbc_p)
    # Surface sensitive channels must be masked over ice
    assert (out_ice['admissible_mask'][:, :12] == 0).all()
    # Sounding channels (52-57 GHz and 183 GHz, channels 12:29) must remain admitted
    assert (out_ice['admissible_mask'][:, 12:] == 1).all()


def test_real_s2_checkpoint_loading_s3():
    s2_ckpt_path = Path("outputs/s2_joint_clearsky_and_amv/training/epoch-0001/state.pt")
    clim_path = Path("outputs/extended_2023_2026/training/station_climatology.npz")

    if not s2_ckpt_path.is_file() or not clim_path.is_file():
        pytest.skip("Required S2 checkpoint or climatology not found in local workspace")

    clim = StationClimatology.load(clim_path)
    cfg = {
        'schema': 'satellite-stage-s3-config-1',
        'stage_id': 'S3_allsky_microwave_and_arctic_ice',
        'mesh_level': 4,
        'hidden': 32,
    }
    model = load_s3_model(
        s2_ckpt_path,
        clim,
        mesh_level=4,
        hidden=32,
        config=cfg,
        device=torch.device('cpu'),
    )
    assert isinstance(model, JointSatelliteStage3Model)
    assert model.sat_weight == 0.5
    assert sum(1 for p in model.parameters() if p.requires_grad) == sum(1 for _ in model.parameters())
