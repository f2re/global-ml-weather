"""R11 products: barycentric regridding on 37 levels, below-ground masks and GRIB2 round trip."""
import json

import numpy as np
import pytest

from global_weather.assimilation.mesh import LatentMesh
from global_weather.assimilation.products import regrid_lead
from global_weather.grib_export import bases, regular_grid
from global_weather.vertical import PRESSURE_HPA

P = np.asarray(PRESSURE_HPA, float) * 100.


def setup(level=3, spacing=10.0):
    mesh = LatentMesh(level, multimesh=False)
    lat, lon, target = regular_grid(spacing)
    glat, glon = np.meshgrid(lat, lon, indexing='ij')
    interp = mesh.interpolation(glat.ravel(), np.where(glon.ravel() > 180, glon.ravel() - 360, glon.ravel()))
    return mesh, target, interp


def node_fields(mesh, ps=101000.):
    n = mesh.n_nodes
    profiles = np.zeros((n, 37, 6), np.float32)
    profiles[..., 0] = 250.
    profiles[..., 4] = 9.80665 * 1000.
    axis = np.array([0., 0., 1.])
    velocity = 20. * np.cross(axis, mesh.xyz)          # solid-body rotation, tangent everywhere
    east, north = bases(mesh.xyz)
    profiles[..., 2] = (velocity * east).sum(-1)[:, None]
    profiles[..., 3] = (velocity * north).sum(-1)[:, None]
    surface = np.full(n, ps)
    mask = P[None, :] <= surface[:, None]
    profiles[~mask] = np.nan
    return profiles, mask, surface


def test_constant_fields_are_exact_and_solid_body_wind_is_accurate():
    mesh, target, interp = setup()
    profiles, mask, surface = node_fields(mesh)
    out, valid, sfc = regrid_lead(profiles, mask, surface, surface, mesh.xyz, interp.nodes, interp.weights, target, P)
    assert valid.all() and np.allclose(out[..., 0], 250.) and np.allclose(sfc[:, 0], 101000.)
    east, north = bases(target)
    expected = 20. * np.cross([0., 0., 1.], target)
    error_u = np.abs(out[:, 0, 2] - (expected * east).sum(-1)).max()
    error_v = np.abs(out[:, 0, 3] - (expected * north).sum(-1)).max()
    assert error_u < 0.2 and error_v < 0.2      # O(h^2) on a ~960 km mesh, 20 m/s field


def test_points_below_the_interpolated_surface_are_masked_not_extrapolated():
    mesh, target, interp = setup()
    profiles, mask, surface = node_fields(mesh, ps=86000.)
    out, valid, sfc = regrid_lead(profiles, mask, surface, surface, mesh.xyz, interp.nodes, interp.weights, target, P)
    below = P > 86000.
    assert not valid[:, below].any() and valid[:, ~below].all()
    assert np.isnan(out[:, below]).all() and np.isfinite(out[:, ~below]).all()


def test_mixed_triangle_renormalises_over_nodes_above_ground():
    mesh, target, interp = setup()
    profiles, mask, surface = node_fields(mesh)
    high = np.zeros(mesh.n_nodes, bool)
    high[interp.nodes[0, 0]] = True                     # one node of the first triangle is a mountain
    surface[high] = 60000.
    mask = P[None, :] <= surface[:, None]
    profiles[~mask] = np.nan
    out, valid, sfc = regrid_lead(profiles, mask, surface, surface, mesh.xyz, interp.nodes, interp.weights, target, P)
    level = int(np.flatnonzero(P == 70000.)[0])
    ps0 = sfc[0, 0]
    assert valid[0, level] == (70000. <= ps0)
    if valid[0, level]:
        assert out[0, level, 0] == pytest.approx(250.)


def test_grib2_round_trip_of_an_r11_forecast(tmp_path):
    pytest.importorskip('eccodes')
    from global_weather.assimilation.forecast import forecast, load_frozen
    from global_weather.assimilation.products import decode_check, export_grib
    from global_weather.assimilation.selftest import SELFTEST_CONFIG
    from global_weather.assimilation.training import train
    train(SELFTEST_CONFIG, tmp_path / 'run', synthetic=True, device='cpu', max_steps=1)
    model, experiment, provenance = load_frozen(tmp_path / 'run', synthetic=True)
    forecast(model, experiment, '2021-03-01T12:00:00Z', tmp_path / 'fc', hours=6, members=1, provenance=provenance)
    report = export_grib(tmp_path / 'fc', tmp_path / 'fc.grib2', spacing=10.0)
    assert len(report['messages']) == 3 * (37 * 6 + 2)
    assert json.loads((tmp_path / 'fc.grib2.json').read_text())['grib_sha256'] == report['grib_sha256']
    check = decode_check(tmp_path / 'fc.grib2', tmp_path / 'fc', spacing=10.0, points=300)
    assert check['messages'] == check['expected_messages'] and check['mask_mismatches'] == 0
    assert check['max_abs_error']['temperature'] < 1e-3 and check['max_abs_error']['geopotential'] < 1e-1
    with pytest.raises(FileExistsError):
        export_grib(tmp_path / 'fc', tmp_path / 'fc.grib2', spacing=10.0)
