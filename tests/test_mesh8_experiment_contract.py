"""Mesh-8 qualification contracts; no meteorological skill claim."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from global_weather.grid import (
    EARTH_RADIUS_M,
    MESH8_FINGERPRINT_QUANTIZATION,
    _base_icosahedron,
    _directed_edges,
    _mesh8_geometry_fingerprint,
    _vector_refine_once,
    build_grid,
    grid_cell_count,
    nominal_area_scale_km,
)
from global_weather.joint_observation_training import MESH8_ROLE, _config
from global_weather.profile_hydrostatic_model import HydrostaticFlowProfileModel
from global_weather.profile_training import START
from test_profile_seasonal_model import normalization


ROOT = Path(__file__).resolve().parents[1]


def mesh8_config():
    return json.loads((ROOT / 'configs' / 'joint_observation_training.json').read_text())


def test_canonical_joint_experiment_is_pinned_to_mesh8_qualification():
    cfg = _config(mesh8_config())
    assert cfg['experiment_id'] == 'joint-mesh8-qualification-r10'
    assert cfg['experiment_role'] == MESH8_ROLE
    assert cfg['scientific_acceptance'] is False
    assert cfg['mesh_level'] == 8
    assert grid_cell_count(cfg['mesh_level']) == 655_362
    assert nominal_area_scale_km(8) == pytest.approx(27.89796, rel=1e-5)
    assert cfg['rollout_hours'] == 3
    assert cfg['hidden'] == 4
    assert cfg['epochs'] == 1
    assert cfg['edge_chunk_size'] == 32768


def test_mesh8_qualification_rejects_unreviewed_resource_escalation():
    cfg = mesh8_config()
    for patch in (
        {'rollout_hours': 6},
        {'hidden': 8},
        {'epochs': 2},
        {'edge_chunk_size': None},
        {'experiment_role': 'science'},
    ):
        candidate = dict(cfg)
        candidate.update(patch)
        with pytest.raises(ValueError):
            _config(candidate)


def test_historical_mesh2_config_remains_loadable():
    historical = json.loads((ROOT / 'configs' / 'joint_observation_training_mesh2_pilot.json').read_text())
    cfg = _config(historical)
    assert cfg['mesh_level'] == 2
    assert cfg['rollout_hours'] == 72
    assert cfg['edge_chunk_size'] is None


def test_vector_refinement_preserves_icosahedral_topology_on_small_fixture():
    xyz, faces = _base_icosahedron()
    refined_xyz, refined_faces = _vector_refine_once(xyz, faces)
    assert refined_xyz.shape == (42, 3)
    assert refined_faces.shape == (80, 3)
    edges = _directed_edges(refined_faces)
    degree = np.bincount(edges[0], minlength=len(refined_xyz))
    assert np.count_nonzero(degree == 5) == 12
    assert np.isin(degree, [5, 6]).all()


def test_mesh8_fingerprint_ignores_harmless_sub_quantum_float_drift():
    # Identity must not depend on sub-metre normalisation drift between NumPy
    # builds. A displacement larger than the declared quantisation must still
    # change the geometry identity.
    xyz = np.array([[.12345671, .2, .3], [-.4, .5, .6]], dtype=np.float64)
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    baseline = _mesh8_geometry_fingerprint(xyz, edges)
    tiny = xyz.copy(); tiny[0, 0] += .1/MESH8_FINGERPRINT_QUANTIZATION
    assert _mesh8_geometry_fingerprint(tiny, edges) == baseline
    shifted = xyz.copy(); shifted[0, 0] += 2/MESH8_FINGERPRINT_QUANTIZATION
    assert _mesh8_geometry_fingerprint(shifted, edges) != baseline


def test_high_resolution_area_algorithm_closes_sphere_without_voronoi_polygons():
    # Force the mesh-8 area path on a tiny grid so regular CI does not allocate
    # the real 655k-cell geometry. The real level-8 build runs in its own job.
    grid = build_grid(1, exact_voronoi_max_cells=12)
    assert grid.area_method == 'one_third_spherical_triangle_control_volume'
    assert grid.has_polygon_geometry is False
    assert grid.vertices.shape == (0, 3)
    assert grid.regions == []
    assert grid.areas_m2.sum() == pytest.approx(4*np.pi*EARTH_RADIUS_M**2, rel=1e-9)
    assert (grid.areas_m2 > 0).all()


def test_chunked_flow_exchange_matches_historical_full_edge_result():
    torch.manual_seed(19)
    grid = build_grid(0)
    norms = normalization()
    full = HydrostaticFlowProfileModel(grid, norms, hidden=4)
    chunked = HydrostaticFlowProfileModel(grid, norms, hidden=4)
    chunked.load_state_dict(full.state_dict())
    chunked.edge_chunk_size = 7
    pressure = full.pressure_encoder(full.log_pressure) + full.levels
    state = full._initial_state(START, pressure)
    frame = full._decode(state, START, 0)
    expected = full._flow_neighbours(state, frame)
    actual = chunked._flow_neighbours(state, frame)
    assert torch.allclose(actual, expected, rtol=2e-6, atol=2e-7)
