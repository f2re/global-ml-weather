"""Scientific admission of irregular observation equivalents; remote execution."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from global_weather.analysis.observation_operator import (
    observation_space_loss, predict_observations,
    IKFS2CloudClearedOperator, MeteorMTVZAAllSkyOperator, VariationalBiasCorrection,
)
from global_weather.losses import SatelliteAssimilationLoss
from global_weather.grid import build_grid, latlon
from global_weather.vertical import PRESSURE_HPA

ISSUE = datetime(2021, 1, 1, tzinfo=timezone.utc)
PRESSURE = np.asarray(PRESSURE_HPA) * 100.


def frame(grid, *, hours=0):
    profiles = torch.full((grid.n_cells, 37, 6), 260. + hours * 2)
    profiles[..., 5] = float("nan")
    profiles.requires_grad_(True)
    variable_mask = torch.ones_like(profiles, dtype=torch.bool)
    variable_mask[..., 5] = False
    return SimpleNamespace(
        profiles=profiles, surface=torch.full((grid.n_cells, 8), 280., requires_grad=True),
        profile_mask=torch.ones((grid.n_cells, 37), dtype=torch.bool),
        profile_variable_mask=variable_mask,
        surface_mask=torch.ones((grid.n_cells, 8), dtype=torch.bool),
        valid_time=ISSUE + timedelta(hours=hours),
    )


def observation(grid, *, variable="temperature", units="K", **changes):
    latitude, longitude = latlon(grid.xyz)[0]
    result = dict(source="radiosonde", variable=variable, value=263., units=units,
                  pressure_pa=99000., latitude=float(latitude), longitude=float(longitude),
                  observed_at=ISSUE.isoformat(), available_at=ISSUE.isoformat(), valid=True)
    result.update(changes)
    return result


@pytest.mark.parametrize("variable,units", [
    ("t2m", "K"), ("td2m", "K"), ("u10", "m s-1"), ("v10", "m s-1"),
    ("surface_pressure", "Pa"), ("mslp", "Pa"),
])
def test_real_station_unknown_reference_cannot_supervise_physical_surface(variable, units):
    grid = build_grid(0)
    target = observation(grid, variable=variable, units=units, source="station",
                         provider="NOAA_GHCNh", pressure_pa=None,
                         height_reference="nominal_surface_height_requires_station_metadata")
    accepted, rejected = predict_observations([frame(grid)], grid, PRESSURE, [target])
    assert not accepted
    assert rejected == {"surface_reference_not_admitted": 1}


def test_real_provider_cannot_bypass_height_gate_by_synthetic_label():
    grid = build_grid(0)
    target = observation(grid, variable="t2m", source="station", provider="NOAA_GHCNh",
                         data_kind="synthetic", pressure_pa=None)
    accepted, rejected = predict_observations([frame(grid)], grid, PRESSURE, [target])
    assert not accepted and rejected == {"surface_reference_not_admitted": 1}


def test_explicit_synthetic_fixture_still_checks_interpolation():
    grid = build_grid(0)
    target = observation(grid, variable="t2m", source="synthetic_fixture",
                         data_kind="synthetic", pressure_pa=None)
    accepted, rejected = predict_observations([frame(grid)], grid, PRESSURE, [target])
    assert not rejected
    assert accepted[0].predicted == pytest.approx(280.)


def test_confirmed_height_needs_matching_model_terrain():
    grid = build_grid(0)
    forecast = frame(grid)
    target = observation(grid, variable="t2m", source="station", pressure_pa=None,
                         instrument_height_verified=True, instrument_height_m=2.,
                         height_reference="height_above_ground", elevation_m=150.)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not accepted and rejected == {"surface_reference_not_admitted": 1}
    forecast.surface_elevation_m = torch.full((grid.n_cells,), 150.)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not rejected and len(accepted) == 1
    forecast.surface_elevation_m.fill_(160.)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not accepted and rejected == {"surface_reference_not_admitted": 1}


def test_pressure_sensor_reference_must_match_lower_boundary():
    grid = build_grid(0)
    forecast = frame(grid)
    forecast.surface_elevation_m = torch.full((grid.n_cells,), 150.)
    target = observation(grid, variable="surface_pressure", units="Pa", source="station",
                         pressure_pa=None, instrument_height_verified=True,
                         height_reference="model_lower_boundary", pressure_reference_elevation_m=150.)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not rejected and len(accepted) == 1
    target["pressure_reference_elevation_m"] = 151.5
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not accepted and rejected == {"surface_reference_not_admitted": 1}


def test_unsupported_omega_nan_does_not_poison_temperature_loss():
    grid = build_grid(0)
    forecast = frame(grid)
    targets = [observation(grid), observation(grid, variable="omega", units="Pa s-1", value=0.)]
    loss, report = observation_space_loss([forecast], grid, PRESSURE, targets)
    assert report["accepted"] == 1
    assert report["rejected"] == {"outside_supported_forecast": 1}
    assert float(loss) == pytest.approx(9.)
    loss.backward()
    assert torch.isfinite(forecast.profiles.grad).all()
    assert (forecast.profiles.grad[..., 5] == 0).all()
    assert forecast.profiles.grad[..., 0].abs().sum() > 0


def test_both_pressure_brackets_required_even_near_exact_level():
    grid = build_grid(0)
    forecast = frame(grid)
    forecast.profile_variable_mask[:, 1, 0] = False
    with torch.no_grad():
        forecast.profiles[:, 1, 0] = float("nan")
    target = observation(grid, pressure_pa=99999.)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not accepted and rejected == {"outside_supported_forecast": 1}
    target["pressure_pa"] = 100000.
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [target])
    assert not rejected and accepted[0].predicted == pytest.approx(260.)


def test_optional_mask_does_not_change_legacy_temperature_support():
    grid = build_grid(0)
    forecast = frame(grid)
    del forecast.profile_variable_mask
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [observation(grid)])
    assert not rejected and accepted[0].predicted == pytest.approx(260.)


def test_malformed_profile_variable_mask_is_rejected():
    grid = build_grid(0)
    forecast = frame(grid)
    forecast.profile_variable_mask = torch.ones((grid.n_cells, 37), dtype=torch.bool)
    accepted, rejected = predict_observations([forecast], grid, PRESSURE, [observation(grid)])
    assert not accepted and rejected == {"invalid_profile_variable_mask": 1}


def test_interpolation_requires_supported_pressure_and_both_times():
    grid = build_grid(0)
    left, right = frame(grid), frame(grid, hours=3)
    target = observation(grid, observed_at=(ISSUE + timedelta(hours=1, minutes=30)).isoformat())
    accepted, rejected = predict_observations([left, right], grid, PRESSURE, [target])
    assert not rejected and accepted[0].predicted == pytest.approx(263.)
    right.profile_variable_mask[..., 0] = False
    accepted, rejected = predict_observations([left, right], grid, PRESSURE, [target])
    assert not accepted and rejected == {"outside_supported_forecast": 1}
    target["observed_at"] = ISSUE.isoformat()
    target["pressure_pa"] = 100001.
    accepted, rejected = predict_observations([left, right], grid, PRESSURE, [target])
    assert not accepted and rejected == {"outside_supported_forecast": 1}


def test_ikfs2_operator_cloud_clearing_and_pc_projection():
    op = IKFS2CloudClearedOperator(num_selected_channels=180, num_pcs=32, num_model_levels=37)
    n_obs = 4
    obs_spectra = torch.full((n_obs, 180), 250.0)
    model_t = torch.full((n_obs, 37), 240.0)
    model_q = torch.full((n_obs, 37), -5.0)
    model_t_skin = torch.full((n_obs, 1), 280.0)
    mu_sat = torch.full((n_obs, 1), 0.8)
    varbc_pred = torch.zeros((n_obs, 7))

    cld_frac = torch.tensor([0.01, 0.50, 0.0, 0.9])
    cth_p = torch.tensor([800.0, 300.0, 900.0, 200.0])

    out = op(obs_spectra, model_t, model_q, model_t_skin, mu_sat, varbc_pred, cld_frac, cth_p)
    assert "pc_innovation" in out and out["pc_innovation"].shape == (n_obs, 32)
    assert "channel_mask" in out and out["channel_mask"].shape == (n_obs, 180)
    assert torch.allclose(out["channel_mask"][0], torch.ones(180))
    assert out["channel_mask"][1, -1] < 0.01


def test_mtvza_allsky_operator_c_sym_and_surface_mask():
    op = MeteorMTVZAAllSkyOperator(num_channels=29, num_model_levels=37)
    n_obs = 3
    obs_tb = torch.full((n_obs, 29), 240.0)
    obs_tb[:, 6] = 220.0
    obs_tb[:, 7] = 160.0
    model_profiles = torch.zeros((n_obs, 37 * 3))
    surface_state = torch.tensor([
        [285.0, 7.0, 0.0, 0.0],
        [285.0, 5.0, 1.0, 0.0],
        [250.0, 10.0, 0.0, 0.8],
    ])
    varbc_pred = torch.zeros((n_obs, 7))

    out = op(obs_tb, model_profiles, surface_state, varbc_pred)
    assert out["normalized_innovation"].shape == (n_obs, 29)
    assert out["c_sym"].shape == (n_obs,)
    assert out["admissible_mask"][0, :12].sum() == 12
    assert out["admissible_mask"][1, :12].sum() == 0
    assert out["admissible_mask"][2, :12].sum() == 0


def test_satellite_assimilation_loss_robust_qc():
    loss_fn = SatelliteAssimilationLoss(huber_delta=1.5, gross_error_sigma=3.5, varbc_reg_weight=1e-3)
    mtvza_out = {
        "normalized_innovation": torch.tensor([[1.0, 2.0, 10.0]]),
        "admissible_mask": torch.tensor([[1.0, 1.0, 1.0]]),
    }
    ikfs2_out = {
        "masked_channel_innovation": torch.tensor([[0.5, -0.8]]),
        "channel_mask": torch.tensor([[1.0, 1.0]]),
    }
    amv_uv_pred = torch.tensor([[10.0, 5.0]])
    amv_uv_obs = torch.tensor([[12.0, 6.0]])
    amv_sigma = torch.tensor([[2.0]])
    amv_qi = torch.tensor([[0.90]])

    varbc_param = torch.nn.Parameter(torch.tensor([0.1, 0.2]))
    varbc_prior = torch.tensor([0.0, 0.0])

    loss_dict = loss_fn(
        mtvza_out, ikfs2_out,
        amv_uv_pred, amv_uv_obs, amv_sigma, amv_qi,
        [varbc_param], [varbc_prior],
    )
    assert "loss_sat_total" in loss_dict
    assert loss_dict["loss_sat_total"] > 0
    assert torch.isfinite(loss_dict["loss_sat_total"])
    loss_dict["loss_sat_total"].backward()
    assert varbc_param.grad is not None and torch.isfinite(varbc_param.grad).all()
