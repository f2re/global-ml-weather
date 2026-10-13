#!/usr/bin/env python3
"""Exercise the omega continuity diagnostic on the real 655362-cell mesh.

Q5 of R10 calls this diagnostic on mesh8. The historical implementation read
SphericalVoronoi polygons that mesh8 does not materialise and failed with
IndexError. This smoke builds the face-circumcentre dual, checks closure, and
compares divergence with the analytic potential flow chi = R V0 sin(lat).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from global_weather.grid import EARTH_RADIUS_M, build_mesh8_grid, latlon
from global_weather.joint_forecast_physics import apply_forecast_physics, dual_geometry, spherical_wind_divergence


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == 'darwin' else value * 1024)


def run(levels: int = 4) -> dict:
    started = time.perf_counter()
    grid = build_mesh8_grid()
    geometry = dual_geometry(grid)
    dual_seconds = time.perf_counter() - started
    closure = abs(geometry['areas_m2'].sum() / (4 * np.pi * EARTH_RADIUS_M**2) - 1)
    latitude = np.deg2rad(latlon(grid.xyz)[:, 0])
    v = np.repeat((10. * np.cos(latitude))[None, :, None], levels, axis=2)
    u = np.zeros_like(v)
    divergence_started = time.perf_counter()
    divergence, mask = spherical_wind_divergence(u, v, np.ones_like(u, bool), grid)
    divergence_seconds = time.perf_counter() - divergence_started
    exact = -20. * np.sin(latitude) / EARTH_RADIUS_M
    weights = geometry['areas_m2']
    relative_l2 = float(np.sqrt(np.sum(weights * (divergence[0, :, 0] - exact) ** 2) / np.sum(weights * exact**2)))
    if not mask.all() or relative_l2 > 1e-3 or closure > 1e-9:
        raise RuntimeError('Mesh-8 omega diagnostic failed accuracy or closure checks.')
    profiles = np.zeros((1, grid.n_cells, 37, 6), dtype=np.float32)
    profiles[..., 0] = 250.
    profiles[..., 4] = np.linspace(110., 480000., 37, dtype=np.float32)
    profile_mask = np.ones_like(profiles, bool)
    physics_started = time.perf_counter()
    _, issued, metadata = apply_forecast_physics(profiles, profile_mask, grid, np.zeros(grid.n_cells))
    physics_seconds = time.perf_counter() - physics_started
    if not issued[..., 5].all():
        raise RuntimeError('Mesh-8 omega issuance support is incomplete for a resting atmosphere.')
    return {'schema': 'mesh8-omega-smoke-1', 'cells': grid.n_cells,
            'undirected_edges': int(len(geometry['pairs'])), 'dual_area_closure_relative_error': float(closure),
            'potential_flow_relative_l2': relative_l2, 'levels_in_divergence_check': levels,
            'dual_geometry_seconds': dual_seconds, 'divergence_seconds': divergence_seconds,
            'full_physics_seconds_one_lead_37_levels': physics_seconds,
            'omega_discretization': metadata['omega_discretization'],
            'peak_rss_bytes': peak_rss_bytes(), 'scientific_acceptance': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--levels', type=int, default=4)
    args = parser.parse_args(argv)
    result = run(args.levels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
