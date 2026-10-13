#!/usr/bin/env python3
"""Build the real 655362-cell mesh and exercise one full graph exchange on CPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from global_weather.grid import (
    EARTH_RADIUS_M,
    MESH8_FINGERPRINT_QUANTIZATION,
    MESH8_FINGERPRINT_SCHEMA,
    build_mesh8_grid,
    grid_cell_count,
    nominal_area_scale_km,
)
from global_weather.model import GraphOps


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run() -> dict:
    started = time.perf_counter()
    grid = build_mesh8_grid()
    geometry_seconds = time.perf_counter() - started
    degree = np.bincount(grid.edges[0], minlength=grid.n_cells)
    expected_area = 4*np.pi*EARTH_RADIUS_M**2
    if grid.level != 8 or grid.n_cells != grid_cell_count(8):
        raise RuntimeError('Mesh-8 cell identity differs.')
    if grid.faces.shape != (1_310_720, 3):
        raise RuntimeError('Mesh-8 face count differs.')
    if grid.edges.shape != (2, 3_932_160):
        raise RuntimeError('Mesh-8 directed edge count differs.')
    if np.count_nonzero(degree == 5) != 12 or not np.isin(degree, [5, 6]).all():
        raise RuntimeError('Mesh-8 degree topology differs.')
    if not np.isclose(grid.areas_m2.sum(), expected_area, rtol=1e-9):
        raise RuntimeError('Mesh-8 global area closure failed.')
    if grid.has_polygon_geometry or grid.area_method != 'one_third_spherical_triangle_control_volume':
        raise RuntimeError('Mesh-8 high-resolution area policy differs.')

    graph_started = time.perf_counter()
    graph = GraphOps(grid)
    scalar = torch.arange(grid.n_cells, dtype=torch.float32).unsqueeze(-1)
    neighbours = graph.neighbours(scalar)
    graph_seconds = time.perf_counter() - graph_started
    if neighbours.shape != scalar.shape or not torch.isfinite(neighbours).all():
        raise RuntimeError('Mesh-8 sparse neighbour exchange failed.')

    located = np.asarray(grid.locate([90., 0., -90., 59.95], [0., 180., 0., 30.32]))
    if located.shape != (4,) or np.any((located < 0) | (located >= grid.n_cells)):
        raise RuntimeError('Mesh-8 location operator failed.')

    return {
        'schema': 'mesh8-geometry-smoke-2',
        'mesh_level': 8,
        'cells': grid.n_cells,
        'faces': int(len(grid.faces)),
        'directed_edges': int(grid.edges.shape[1]),
        'grid_fingerprint': grid.fingerprint,
        'fingerprint_schema': MESH8_FINGERPRINT_SCHEMA.rstrip(b'\0').decode('ascii'),
        'fingerprint_coordinate_quantization': MESH8_FINGERPRINT_QUANTIZATION,
        'fingerprint_coordinate_tolerance_on_unit_sphere': 1.0/MESH8_FINGERPRINT_QUANTIZATION,
        'five_neighbour_cells': int(np.count_nonzero(degree == 5)),
        'six_neighbour_cells': int(np.count_nonzero(degree == 6)),
        'nominal_area_scale_km': grid.nominal_area_scale_km,
        'nominal_formula_scale_km': nominal_area_scale_km(8),
        'area_method': grid.area_method,
        'polygon_geometry': grid.has_polygon_geometry,
        'relative_area_closure_error': float(abs(grid.areas_m2.sum()-expected_area)/expected_area),
        'geometry_seconds': geometry_seconds,
        'graph_exchange_seconds': graph_seconds,
        'peak_rss_bytes': peak_rss_bytes(),
        'graph_output_mean': float(neighbours.mean()),
        'scientific_acceptance': False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    report = run()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
