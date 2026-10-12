import numpy as np
import pytest

from global_weather.grid import build_grid, latlon, EARTH_RADIUS_M
from global_weather.joint_forecast_physics import (
    apply_forecast_physics, integrate_pressure_velocity, spherical_wind_divergence,
)


def test_constant_divergence_pressure_integration_sign_units():
    p = np.array([100000., 50000., 100.])
    divergence = np.full((2, 3), 1e-5)
    omega, mask = integrate_pressure_velocity(divergence, np.ones_like(divergence, bool), p)
    np.testing.assert_allclose(omega, np.broadcast_to(-1e-5 * (p - 100), (2, 3)))
    assert mask.all()
    assert omega[0, 0] == pytest.approx(-.999)


def test_missing_divergence_blocks_entire_downward_path():
    div = np.array([[1e-5, np.nan, 1e-5, 1e-5]])
    omega, mask = integrate_pressure_velocity(div, np.ones_like(div, bool), [100000., 70000., 10000., 100.])
    assert mask.tolist() == [[False, False, True, True]]
    assert np.isnan(omega[0, :2]).all()


def test_shared_flux_has_closed_sphere_area_sum():
    grid = build_grid(1)
    rng = np.random.default_rng(41)
    u, v = rng.normal(size=(2, grid.n_cells, 3)), rng.normal(size=(2, grid.n_cells, 3))
    divergence, mask = spherical_wind_divergence(u, v, np.ones_like(u, bool), grid)
    assert mask.all()
    closure = (divergence * grid.areas_m2[:, None]).sum(axis=-2)
    np.testing.assert_allclose(closure, 0, atol=1e-7)


def test_solid_body_rotation_divergence_small_relative_to_rotation_rate():
    grid = build_grid(2)
    rotation_rate = 7.292115e-5
    latitude = np.deg2rad(latlon(grid.xyz)[:, 0])
    u = (rotation_rate * EARTH_RADIUS_M * np.cos(latitude))[:, None]
    v = np.zeros_like(u)
    divergence, mask = spherical_wind_divergence(u, v, np.ones_like(u, bool), grid)
    assert mask.all()
    # The pair-midpoint reconstruction is approximate on unequal Voronoi cells.
    # This tests local truncation, separately from exact global flux closure.
    rms = np.sqrt(np.average(divergence[:, 0] ** 2, weights=grid.areas_m2))
    assert rms < .03 * rotation_rate


def test_missing_wind_blocks_cell_and_neighbors_without_zero_fill():
    grid = build_grid(0)
    u = np.zeros((grid.n_cells, 3))
    u[0, 1] = np.nan
    divergence, mask = spherical_wind_divergence(u, np.zeros_like(u), np.ones_like(u, bool), grid)
    neighbors = grid.edges[1, grid.edges[0] == 0]
    assert not mask[np.r_[0, neighbors], 1].any()
    assert np.isnan(divergence[np.r_[0, neighbors], 1]).all()


def test_geometric_issuance_mask_preserves_target_mask_and_diagnostic_origin():
    grid = build_grid(0)
    profiles = np.zeros((1, grid.n_cells, 37, 6))
    profiles[..., 0] = 270
    profiles[..., 4] = 50000
    mask = np.ones_like(profiles, bool)
    output, issued, metadata = apply_forecast_physics(profiles, mask, grid, np.zeros(grid.n_cells))
    assert mask.all()
    assert issued.all()
    np.testing.assert_allclose(output[..., 5], 0)
    assert metadata['omega_supervised'] is False
    assert metadata['target_mask_modified'] is False
    profiles[..., 0, 4] = -100
    output, issued, _ = apply_forecast_physics(profiles, mask, grid, np.zeros(grid.n_cells))
    assert not issued[..., 0, :].any()
    assert np.isnan(output[..., 0, :]).all()
