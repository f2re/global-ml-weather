import json

import numpy as np
import pytest

from global_weather.grid import build_grid
from global_weather.joint_static import land_at_points, land_aware_height, load_static, terrain_slope
from global_weather.observation_training import digest


def test_land_holes_and_negative_land_preserved_without_bathymetry():
    polygon = {'type': 'Polygon', 'coordinates': [
        [[-10, -10], [10, -10], [10, 10], [-10, 10], [-10, -10]],
        [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]],
    ]}
    land = land_at_points(polygon, [5, 0, 30], [5, 0, 30])
    assert land.tolist() == [True, False, False]
    np.testing.assert_array_equal(land_aware_height([-30, -100, -2000], land), [-30, 0, 0])


def test_dateline_polygon_does_not_cover_greenwich():
    polygon = {'type': 'Polygon', 'coordinates': [
        [[170, -10], [-170, -10], [-170, 10], [170, 10], [170, -10]],
    ]}
    assert land_at_points(polygon, [0, 0, 0], [179, -179, 0]).tolist() == [True, True, False]


def test_constant_terrain_has_zero_slope():
    grid = build_grid(0)
    np.testing.assert_allclose(terrain_slope(grid, np.full(grid.n_cells, -10.)), 0)


def test_load_static_validates_hash_and_geometry(tmp_path):
    grid = build_grid(0)
    np.savez(tmp_path / 'static.npz', elevation_m=np.zeros(grid.n_cells, np.float32),
             land_mask=np.zeros(grid.n_cells, bool), slope=np.zeros(grid.n_cells, np.float32), xyz=grid.xyz)
    metadata = {'schema': 'joint-static-terrain-1', 'grid_fingerprint': grid.fingerprint,
                'artifact_sha256': digest(tmp_path / 'static.npz')}
    (tmp_path / 'manifest.json').write_text(json.dumps(metadata))
    static = load_static(tmp_path, grid)
    assert static['static_sha256'] == metadata['artifact_sha256']
    with (tmp_path / 'static.npz').open('ab') as file:
        file.write(b'changed')
    with pytest.raises(ValueError, match='hash differs'):
        load_static(tmp_path, grid)
