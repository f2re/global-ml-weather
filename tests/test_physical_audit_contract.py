"""Regression contracts introduced by the 2026-10-12 physical audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from global_weather.grid import grid_cell_count, nominal_area_scale_km


ROOT = Path(__file__).resolve().parents[1]


def test_mesh2_is_coarse_global_grid_not_20km():
    assert grid_cell_count(2) == 162
    assert nominal_area_scale_km(2) == pytest.approx(1774.42, rel=1e-3)
    assert nominal_area_scale_km(2) > 1000.0


def test_20_to_30km_requires_level8_scale_in_this_hierarchy():
    assert grid_cell_count(8) == 655_362
    assert nominal_area_scale_km(8) == pytest.approx(27.90, rel=1e-3)
    assert 20.0 < nominal_area_scale_km(8) < 30.0


def test_legacy_unified_config_is_surface_only_72h_control():
    cfg = json.loads((ROOT / 'configs' / 'unified_observation_training.json').read_text())
    assert cfg['experiment_role'] == 'legacy_surface_control'
    assert cfg['scientific_acceptance'] is False
    assert cfg['horizon_hours'] == 72
    assert cfg['lead_step_hours'] == 3
    assert cfg['surface_weight'] == 1.0
    assert cfg['profile_weight'] == 0.0
    assert cfg['supervised_branches'] == ['surface']
    assert set(cfg['unsupervised_branches']) == {'profile', 'satellite'}
    assert cfg['mesh_level'] == 2
    assert cfg['mesh_cells'] == grid_cell_count(2)
    assert cfg['nominal_area_scale_km'] == pytest.approx(nominal_area_scale_km(2), rel=1e-3)
    assert cfg['claimed_resolution_km'] is None
