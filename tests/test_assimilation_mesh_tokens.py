"""R11 multimesh geometry, interpolation, tokens, errors and admitted stores."""
from datetime import datetime, timezone
import json

import numpy as np
import pytest
import torch

from global_weather.assimilation.mesh import LatentMesh, divergence, gradient, multimesh_edges, torch_buffers
from global_weather.assimilation.sources import SyntheticSource, TokenStore, prepare_store, seconds
from global_weather.assimilation.tokens import (PLATFORMS, VARIABLES, ErrorModel, Normalizer, TokenBatch,
                                                vertical_slots, wind_directions)
from global_weather.grid import EARTH_RADIUS_M, build_grid, latlon, unit_xyz
from global_weather.joint_forecast_physics import spherical_wind_divergence


def test_multimesh_contains_every_refinement_level_on_finest_nodes():
    edges, level = multimesh_edges(3)
    assert set(np.unique(level)) == {0, 1, 2, 3}
    for coarse in range(4):
        expected = build_grid(coarse).edges
        assert expected.shape[1] == np.sum(level == coarse)
    assert len(np.unique(edges.T, axis=0)) == edges.shape[1]


def test_one_processor_layer_reach_is_global_from_coarse_vertex_and_large_from_any_node():
    mesh = LatentMesh(4)
    assert mesh.reach_km(1) > 7000
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    n = mesh.n_nodes
    adjacency = csr_matrix((np.ones(len(mesh.senders)), (mesh.senders, mesh.receivers)), shape=(n, n))
    hops = shortest_path(adjacency, unweighted=True, indices=[n - 1, n // 2], directed=True)
    for row, node in zip(hops, (n - 1, n // 2)):
        angle = np.arccos(np.clip(mesh.xyz[row <= 4] @ mesh.xyz[node], -1, 1))
        assert EARTH_RADIUS_M * angle.max() / 1000 > 1620


def test_barycentric_interpolation_is_exact_for_linear_cartesian_fields():
    mesh = LatentMesh(3)
    rng = np.random.default_rng(1)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, 200)))
    lon = rng.uniform(-180, 180, 200)
    lat[:3], lon[:3] = [89.999, -89.999, 0.], [179.999, -180., 0.]
    weights = mesh.interpolation(lat, lon)
    assert np.allclose(weights.weights.sum(1), 1) and (weights.weights >= 0).all()
    points = unit_xyz(lat, lon)
    # A linear field on the sphere is exact up to the gnomonic projection factor.
    field = mesh.xyz @ np.array([0.3, -0.2, 0.9])
    approx = (field[weights.nodes] * weights.weights).sum(1)
    projected = points @ np.array([0.3, -0.2, 0.9])
    assert np.max(np.abs(approx - projected)) < 0.02


def test_torch_divergence_matches_numpy_reference_and_gradient_is_consistent():
    mesh = LatentMesh(3)
    buffers = torch_buffers(mesh, dtype=torch.float64)
    ll = np.deg2rad(latlon(mesh.xyz))
    lat, lon = ll[:, 0], ll[:, 1]
    east = np.stack((-np.sin(lon), np.cos(lon), 0 * lon), -1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), -1)
    rng = np.random.default_rng(2)
    u, v = rng.normal(size=(2, mesh.n_nodes, 2))
    wind = torch.tensor(u[..., None] * east[:, None] + v[..., None] * north[:, None])
    result = divergence(wind, buffers['xyz'], buffers['dual_pairs'], buffers['dual_midpoint'], buffers['dual_normal'],
                        buffers['dual_length'], buffers['areas']).numpy()
    reference, _ = spherical_wind_divergence(u, v, np.ones_like(u, bool), mesh.grid)
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-18)
    constant = gradient(torch.ones(mesh.n_nodes, dtype=torch.float64), buffers['xyz'], buffers['dual_pairs'],
                        buffers['dual_normal'], buffers['dual_length'], buffers['areas'])
    assert float(constant.abs().max()) == 0
    field = gradient(torch.tensor(np.sin(lat)), buffers['xyz'], buffers['dual_pairs'], buffers['dual_normal'],
                     buffers['dual_length'], buffers['areas']).numpy()
    exact = (np.cos(lat) / EARTH_RADIUS_M)[:, None] * north
    assert np.linalg.norm(field - exact) / np.linalg.norm(exact) < 0.05


def _batch(**overrides):
    base = dict(platform=np.array([PLATFORMS.index('aircraft')], np.int16),
                variable=np.array([VARIABLES.index('u')], np.int16), channel=np.array([-1], np.int32),
                group=np.array([1], np.int64), identity=np.array([2], np.int64), observed=np.array([0.]),
                available=np.array([10.]), latitude=np.array([45.]), longitude=np.array([10.]),
                pressure_pa=np.array([25000.]), height_m=np.array([np.nan]), value=np.array([30.]),
                error=np.array([np.nan]), quality=np.ones(1, np.float32))
    base.update(overrides)
    return TokenBatch(**base)


def test_token_admission_refuses_ambiguous_or_acausal_measurements():
    _batch().validate()
    with pytest.raises(ValueError, match='not admitted'):
        _batch(variable=np.array([VARIABLES.index('geopotential')], np.int16)).validate()
    with pytest.raises(ValueError, match='before'):
        _batch(available=np.array([-1.])).validate()
    with pytest.raises(ValueError, match='measured pressure'):
        _batch(pressure_pa=np.array([np.nan])).validate()
    duplicate = TokenBatch.concatenate([_batch(), _batch()])
    with pytest.raises(ValueError, match='twice'):
        duplicate.validate()
    direction = wind_directions(_batch())
    np.testing.assert_allclose(direction[0], [-np.sin(np.radians(10)), np.cos(np.radians(10)), 0.], atol=1e-12)
    slots = vertical_slots(_batch())
    assert slots.shape == (1, 11) and slots[0, :8].sum() == pytest.approx(1)


def test_error_model_tables_relative_and_passport_errors():
    model = ErrorModel(json.loads(open('configs/observation_errors_r11.json').read()))
    sigma = model.sigma(_batch(pressure_pa=np.array([50000.])))
    assert sigma[0] == pytest.approx(2.5)
    q = _batch(platform=np.array([PLATFORMS.index('radiosonde')], np.int16),
               variable=np.array([VARIABLES.index('specific_humidity')], np.int16), value=np.array([1e-3]))
    assert model.sigma(q)[0] == pytest.approx(1.5e-4)
    radiance = _batch(platform=np.array([PLATFORMS.index('satellite_radiance')], np.int16),
                      variable=np.array([VARIABLES.index('brightness_temperature')], np.int16),
                      pressure_pa=np.array([np.nan]), value=np.array([250.]))
    with pytest.raises(ValueError, match='Missing observation error'):
        model.sigma(radiance)
    from dataclasses import replace
    assert model.sigma(replace(radiance, error=np.array([0.7])))[0] == 0.7


def test_normalizer_uses_pinned_profile_norms_and_refuses_unknown_channels():
    mean = np.tile([250., 1e-3, 5., 0., 5e4], (37, 1))
    std = np.tile([10., 1e-3, 10., 10., 1e4], (37, 1))
    normalizer = Normalizer(mean, std, {})
    value, usable = normalizer.normalise(_batch(value=np.array([25.])), [])
    assert usable.all() and value[0] == pytest.approx(2.0)
    radiance = _batch(platform=np.array([PLATFORMS.index('satellite_radiance')], np.int16),
                      variable=np.array([VARIABLES.index('brightness_temperature')], np.int16),
                      channel=np.array([0], np.int32), pressure_pa=np.array([np.nan]))
    assert not normalizer.normalise(radiance, ['MTVZA:1'])[1].any()


def _records(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_store_admits_aircraft_and_refuses_unpassported_satellite(tmp_path):
    bounds = {'train': ('2021-01-01T00:00:00+00:00', '2022-01-01T00:00:00+00:00'),
              'validation': ('2022-01-01T00:00:00+00:00', '2022-07-01T00:00:00+00:00')}
    aircraft = [{'observation_id': f'a{i}', 'group_id': 'flight-7', 'variable': 'temperature', 'units': 'K',
                 'value': 230. + i, 'pressure_pa': 25000. + 100 * i, 'latitude': 50., 'longitude': 10. + i,
                 'observed_at': f'2021-03-01T0{i}:15:00Z', 'available_at': f'2021-03-01T0{i}:45:00Z'} for i in range(4)]
    _records(tmp_path / 'aircraft.jsonl', aircraft)
    manifest = prepare_store(tmp_path / 'aircraft.jsonl', 'aircraft', tmp_path / 'store', train_end=bounds['train'][1],
                             split_bounds=bounds)
    assert manifest['counts']['admitted'] == 4 and 'aircraft|temperature|' in manifest['statistics']
    store = TokenStore(tmp_path / 'store')
    issue = seconds('2021-03-01T03:00:00Z')
    causal = store.window(issue - 12 * 3600, issue, split='train', available_before=issue)
    assert len(causal) == 3  # 00:15, 01:15 and 02:15 (available 02:45); 03:15 is in the future
    assert np.all(causal.available <= issue)
    late = store.window(issue - 12 * 3600, issue, split='train', available_before=issue - 20 * 60)
    assert len(late) == 2  # the 02:15 report is not yet available 20 minutes before issue
    satellite = [{'observation_id': 's1', 'granule_id': 'g', 'variable': 'brightness_temperature', 'units': 'K',
                  'value': 250., 'latitude': 0., 'longitude': 0., 'instrument': 'MTVZA', 'channel': 3,
                  'observed_at': '2021-03-01T00:00:00Z', 'available_at': '2021-03-01T01:00:00Z'}]
    _records(tmp_path / 'sat.jsonl', satellite)
    with pytest.raises(ValueError, match='radiometric passport'):
        prepare_store(tmp_path / 'sat.jsonl', 'satellite_radiance', tmp_path / 'sat', train_end=bounds['train'][1],
                      split_bounds=bounds)


def test_synthetic_source_is_labelled_and_causal():
    source = SyntheticSource(0, stations=5, soundings=2)
    issue = seconds(datetime(2021, 3, 1, 12, tzinfo=timezone.utc))
    batch = source.window(issue - 12 * 3600, issue, split='train', available_before=issue).validate()
    assert source.data_kind == 'synthetic' and np.all(batch.available <= issue) and len(batch)


def test_area_static_aggregates_mean_fraction_and_subgrid_spread():
    from global_weather.assimilation.static import aggregate
    coarse, fine = build_grid(2), build_grid(4)
    height = np.where(fine.xyz[:, 2] > 0, 1000., 0.)
    land = fine.xyz[:, 2] > 0
    fields = aggregate(coarse, fine.xyz, height, land)
    assert fields['samples_per_cell'].min() >= 8
    north = coarse.xyz[:, 2] > 0.3
    assert np.allclose(fields['elevation_m'][north], 1000.) and np.allclose(fields['land_fraction'][north], 1.)
    assert np.allclose(fields['orography_std_m'][north], 0.)
    equator = np.abs(coarse.xyz[:, 2]) < 0.05
    assert np.all((fields['land_fraction'][equator] > 0) & (fields['land_fraction'][equator] < 1))
    assert np.all(fields['orography_std_m'][equator] > 0)


def test_r11_dual_geometry_matches_spherical_voronoi_without_polygons():
    """R11 builds its exact Voronoi dual from face circumcentres, so it needs no materialised polygons."""
    from global_weather.assimilation.geometry import dual_geometry
    from global_weather.grid import build_grid
    with_polygons = build_grid(3)
    without_polygons = build_grid(3, exact_voronoi_max_cells=12)
    assert with_polygons.has_polygon_geometry and not without_polygons.has_polygon_geometry
    np.testing.assert_allclose(dual_geometry(without_polygons)['areas_m2'], with_polygons.areas_m2, rtol=1e-11)
    assert dual_geometry(with_polygons) is dual_geometry(with_polygons)
    assert not hasattr(with_polygons, '_joint_flux_geometry')
