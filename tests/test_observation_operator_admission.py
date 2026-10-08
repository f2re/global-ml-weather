"""Scientific admission of irregular observation equivalents; remote execution."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from global_weather.analysis.observation_operator import (
    observation_space_loss, predict_observations,
)
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
