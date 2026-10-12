"""Diagnostic pressure velocity from finite-volume wind divergence.

This is a continuity diagnostic, not a directly observed or learned omega.
The imposed upper boundary is omega(100 Pa)=0. No mass correction or lower
boundary condition is imposed; incomplete wind paths remain missing.
"""
from __future__ import annotations

import numpy as np

from .grid import EARTH_RADIUS_M, latlon
from .vertical import PRESSURE_HPA

G0 = 9.80665


def _transport(vector, origin, destination):
    """Shortest-geodesic parallel transport of a tangent Cartesian vector."""
    return vector - (np.sum(vector * destination, axis=-1) /
                     (1 + np.sum(origin * destination, axis=-1)))[..., None] * (origin + destination)


def _edge_geometry(grid):
    cached = getattr(grid, '_joint_flux_geometry', None)
    if cached is not None:
        return cached
    pairs = grid.edges[:, grid.edges[0] < grid.edges[1]].T
    midpoints, normals, lengths = [], [], []
    for left, right in pairs:
        shared = sorted(set(grid.regions[left]) & set(grid.regions[right]))
        if len(shared) != 2:
            raise ValueError('A closed Voronoi edge requires exactly two shared vertices.')
        va, vb = grid.vertices[shared]
        midpoint = va + vb
        midpoint /= np.linalg.norm(midpoint)
        normal = grid.xyz[right] - grid.xyz[left]
        normal /= np.linalg.norm(normal)
        midpoints.append(midpoint)
        normals.append(normal)
        lengths.append(EARTH_RADIUS_M * np.arccos(np.clip(np.dot(va, vb), -1, 1)))
    cached = pairs, np.asarray(midpoints), np.asarray(normals), np.asarray(lengths)
    grid._joint_flux_geometry = cached
    return cached


def spherical_wind_divergence(u, v, mask, grid):
    """Return divergence [s^-1] and support for arrays (...,cell,pressure).

    A shared edge has one midpoint flux, added with opposite signs to the two
    cells. Every adjacent wind value must be supported for a cell divergence.
    The cell-centre interpolation is second-order geometry-dependent and does
    not assert exact local divergence for every rotational field.
    """
    u, v, mask = np.asarray(u, float), np.asarray(v, float), np.asarray(mask, bool)
    if u.shape != v.shape or mask.shape != u.shape or u.ndim < 2 or u.shape[-2] != grid.n_cells:
        raise ValueError('Wind arrays require matching (...,cell,pressure) shapes.')
    valid = mask & np.isfinite(u) & np.isfinite(v)
    ll = np.deg2rad(latlon(grid.xyz))
    lat, lon = ll[:, 0], ll[:, 1]
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), axis=-1)
    velocity = (np.where(valid, u, 0)[..., None] * east[:, None, :]
                + np.where(valid, v, 0)[..., None] * north[:, None, :])
    pairs, midpoint, normal, length = _edge_geometry(grid)
    sums = np.zeros_like(u)
    supported = valid.copy()
    for edge, (left, right) in enumerate(pairs):
        vl = _transport(velocity[..., left, :, :], grid.xyz[left], midpoint[edge])
        vr = _transport(velocity[..., right, :, :], grid.xyz[right], midpoint[edge])
        edge_valid = valid[..., left, :] & valid[..., right, :]
        flux = np.sum(.5 * (vl + vr) * normal[edge], axis=-1) * length[edge]
        flux = np.where(edge_valid, flux, 0)
        sums[..., left, :] += flux
        sums[..., right, :] -= flux
        supported[..., left, :] &= edge_valid
        supported[..., right, :] &= edge_valid
    divergence = sums / grid.areas_m2[:, None]
    return np.where(supported, divergence, np.nan), supported


def integrate_pressure_velocity(divergence, mask, pressure_pa):
    """Integrate d(omega)/dp=-div(V) from the top; positive p points downward.

    Input descending pressures may be arbitrary. Trapezoidal integration of
    constant positive divergence gives negative omega with units Pa s^-1.
    One missing level invalidates every lower level on that column path.
    """
    div, mask, pressure = np.asarray(divergence, float), np.asarray(mask, bool), np.asarray(pressure_pa, float)
    if (div.shape != mask.shape or pressure.ndim != 1 or div.shape[-1] != len(pressure)
            or len(pressure) < 2 or not np.isfinite(pressure).all() or np.any(pressure <= 0)
            or np.any(np.diff(pressure) >= 0)):
        raise ValueError('Need matching divergence/mask and descending positive pressures.')
    support = mask & np.isfinite(div)
    omega = np.full_like(div, np.nan)
    output_mask = np.zeros_like(mask)
    output_mask[..., -1] = support[..., -1]
    omega[..., -1] = np.where(output_mask[..., -1], 0, np.nan)
    for level in range(len(pressure) - 2, -1, -1):
        output_mask[..., level] = output_mask[..., level + 1] & support[..., level]
        value = omega[..., level + 1] - .5 * (div[..., level + 1] + div[..., level]) * (pressure[level] - pressure[level + 1])
        omega[..., level] = np.where(output_mask[..., level], value, np.nan)
    return omega, output_mask


def apply_forecast_physics(profiles, profile_mask, grid, elevation_m):
    """Return physical profiles, geometric issuance masks, diagnostic metadata."""
    profiles, mask = np.array(profiles, dtype=float, copy=True), np.array(profile_mask, bool, copy=True)
    elevation = np.asarray(elevation_m, float)
    if (profiles.shape != mask.shape or profiles.shape[-3:] != (grid.n_cells, 37, 6)
            or elevation.shape != (grid.n_cells,) or not np.isfinite(elevation).all()):
        raise ValueError('Need profiles (...,cell,37,6), masks and finite terrain metres.')
    mask &= np.isfinite(profiles)
    # This is a product geometry mask, never a training target mask.
    above_terrain = mask[..., 4] & (profiles[..., 4] >= G0 * elevation[:, None])
    wind_support = mask[..., 2] & mask[..., 3] & above_terrain
    divergence, divergence_mask = spherical_wind_divergence(profiles[..., 2], profiles[..., 3], wind_support, grid)
    omega, omega_mask = integrate_pressure_velocity(divergence, divergence_mask, np.asarray(PRESSURE_HPA) * 100.)
    profiles[..., 5] = omega
    mask[..., 5] = omega_mask
    mask &= above_terrain[..., None]
    profiles = np.where(mask, profiles, np.nan)
    metadata = {
        'omega_origin': 'diagnostic_pressure_coordinate_continuity',
        'omega_units': 'Pa s-1', 'omega_supervised': False,
        'omega_equation': 'div_h(V) + d(omega)/dp = 0',
        'omega_boundary': 'omega(p=100 Pa)=0 imposed; no lower boundary or mass correction',
        'omega_discretization': 'shared spherical Voronoi edge midpoint flux; parallel-transported pair winds; vertical trapezoid',
        'omega_support': 'both wind components and geometric support at every adjacent cell and top-to-level path',
        'issuance_mask': 'predicted geopotential >= g0 * independent land-aware point DEM height',
        'target_mask_modified': False,
        'limitations': ['diagnostic omega is not direct observation supervision',
                        'finite-volume interpolation has truncation error',
                        'imposed top boundary and absent lower mass correction may bias omega',
                        'geopotential/geometric DEM datum approximation'],
    }
    return profiles.astype(np.float32), mask, metadata
