"""Diagnostic pressure velocity from finite-volume wind divergence.

This is a continuity diagnostic, not a directly observed or learned omega.
The imposed upper boundary is omega(100 Pa)=0. No mass correction or lower
boundary condition is imposed; incomplete wind paths remain missing.

The dual (Voronoi) geometry is derived from the Delaunay faces of the mesh:
the two endpoints of the dual edge are the spherical circumcentres of the two
faces sharing a primal edge. This is the same geometry as SphericalVoronoi on
historical meshes, but it does not require materialised polygons and therefore
also works on mesh8. Flux accumulation is vectorised through a sparse signed
incidence matrix, processed in bounded edge chunks.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .grid import EARTH_RADIUS_M, latlon
from .vertical import PRESSURE_HPA

G0 = 9.80665
DEFAULT_EDGE_CHUNK = 262_144


def _transport(vector, origin, destination):
    """Shortest-geodesic parallel transport of a tangent Cartesian vector."""
    return vector - (np.sum(vector * destination, axis=-1) /
                     (1 + np.sum(origin * destination, axis=-1)))[..., None] * (origin + destination)


def _circumcentres(xyz, faces):
    """Unit circumcentres of spherical Delaunay faces, oriented outward."""
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    flip = np.einsum('ij,ij->i', normal, a + b + c) < 0
    normal[flip] *= -1
    return normal


def _spherical_triangle_area(a, b, c):
    """Unit-sphere area of triangles (a,b,c), Van Oosterom-Strackee formula."""
    numerator = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    denominator = 1.0 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    return 2.0 * np.arctan2(numerator, denominator)


def dual_geometry(grid):
    """Return undirected pairs, dual-edge midpoint, normal, length and dual areas.

    pairs[e] = (left, right) with left < right. normal points left -> right.
    Areas are the exact spherical Voronoi areas built from the same circumcentres,
    so divergence and the dual edges use one consistent control volume.
    """
    cached = getattr(grid, '_dual_flux_geometry', None)
    if cached is not None:
        return cached
    faces = np.asarray(grid.faces, dtype=np.int64)
    xyz = np.asarray(grid.xyz, dtype=np.float64)
    raw = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    raw.sort(axis=1)
    owner = np.tile(np.arange(len(faces)), 3)
    pairs, inverse, counts = np.unique(raw, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    if not np.all(counts == 2):
        raise ValueError('Every primal edge must be shared by exactly two faces.')
    order = np.argsort(inverse, kind='stable')
    face_pair = owner[order].reshape(-1, 2)
    centres = _circumcentres(xyz, faces)
    va, vb = centres[face_pair[:, 0]], centres[face_pair[:, 1]]
    midpoint = va + vb
    midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)
    left, right = pairs[:, 0], pairs[:, 1]
    normal = xyz[right] - xyz[left]
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    length = EARTH_RADIUS_M * np.arccos(np.clip(np.einsum('ij,ij->i', va, vb), -1, 1))
    # Each dual edge closes two Voronoi sub-triangles: (left, va, vb) and (right, va, vb).
    sub = np.concatenate((_spherical_triangle_area(xyz[left], va, vb),
                          _spherical_triangle_area(xyz[right], va, vb)))
    areas = np.bincount(np.concatenate((left, right)), weights=sub, minlength=len(xyz)) * EARTH_RADIUS_M**2
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise RuntimeError('Nonpositive dual control volume.')
    if not np.isclose(areas.sum(), 4 * np.pi * EARTH_RADIUS_M**2, rtol=1e-9):
        raise RuntimeError('Dual control volumes do not close the sphere.')
    incidence = sparse.csr_matrix(
        (np.r_[np.ones(len(pairs)), -np.ones(len(pairs))],
         (np.r_[left, right], np.r_[np.arange(len(pairs)), np.arange(len(pairs))])),
        shape=(len(xyz), len(pairs)))
    cached = {'pairs': pairs, 'midpoint': midpoint, 'normal': normal, 'length': length,
              'areas_m2': areas, 'incidence': incidence, 'abs_incidence': abs(incidence).tocsr()}
    grid._dual_flux_geometry = cached
    return cached


def _edge_geometry(grid):
    """Historical interface: (pairs, midpoints, normals, lengths)."""
    geometry = dual_geometry(grid)
    return geometry['pairs'], geometry['midpoint'], geometry['normal'], geometry['length']


def spherical_wind_divergence(u, v, mask, grid, *, edge_chunk=DEFAULT_EDGE_CHUNK):
    """Return divergence [s^-1] and support for arrays (...,cell,pressure).

    A shared edge has one midpoint flux, added with opposite signs to the two
    cells. Every adjacent wind value must be supported for a cell divergence.
    The cell-centre interpolation is second-order geometry-dependent and does
    not assert exact local divergence for every rotational field.
    """
    u, v, mask = np.asarray(u, float), np.asarray(v, float), np.asarray(mask, bool)
    if u.shape != v.shape or mask.shape != u.shape or u.ndim < 2 or u.shape[-2] != grid.n_cells:
        raise ValueError('Wind arrays require matching (...,cell,pressure) shapes.')
    if type(edge_chunk) is not int or edge_chunk < 1:
        raise ValueError('edge_chunk must be a positive integer.')
    geometry = dual_geometry(grid)
    pairs, midpoint, normal, length = (geometry[key] for key in ('pairs', 'midpoint', 'normal', 'length'))
    lead_shape, levels = u.shape[:-2], u.shape[-1]
    valid = mask & np.isfinite(u) & np.isfinite(v)
    # Cell-major layout (cell, lead*level) for sparse accumulation.
    uu = np.moveaxis(np.where(valid, u, 0).reshape(-1, grid.n_cells, levels), 1, 0).reshape(grid.n_cells, -1)
    vv = np.moveaxis(np.where(valid, v, 0).reshape(-1, grid.n_cells, levels), 1, 0).reshape(grid.n_cells, -1)
    ok = np.moveaxis(valid.reshape(-1, grid.n_cells, levels), 1, 0).reshape(grid.n_cells, -1)
    ll = np.deg2rad(latlon(grid.xyz))
    lat, lon = ll[:, 0], ll[:, 1]
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), axis=-1)
    flux = np.zeros((len(pairs), uu.shape[1]))
    edge_valid = np.zeros((len(pairs), uu.shape[1]), dtype=bool)
    for start in range(0, len(pairs), edge_chunk):
        stop = min(start + edge_chunk, len(pairs))
        left, right = pairs[start:stop, 0], pairs[start:stop, 1]
        mid = midpoint[start:stop][:, None, :]
        vel_l = uu[left][..., None] * east[left][:, None, :] + vv[left][..., None] * north[left][:, None, :]
        vel_r = uu[right][..., None] * east[right][:, None, :] + vv[right][..., None] * north[right][:, None, :]
        vl = _transport(vel_l, grid.xyz[left][:, None, :], mid)
        vr = _transport(vel_r, grid.xyz[right][:, None, :], mid)
        both = ok[left] & ok[right]
        value = np.sum(.5 * (vl + vr) * normal[start:stop][:, None, :], axis=-1) * length[start:stop][:, None]
        flux[start:stop] = np.where(both, value, 0)
        edge_valid[start:stop] = both
    sums = np.asarray(geometry['incidence'] @ flux)
    invalid = np.asarray(geometry['abs_incidence'] @ (~edge_valid).astype(np.float64)) > 0
    supported = ok & ~invalid
    divergence = np.where(supported, sums / geometry['areas_m2'][:, None], np.nan)

    def restore(array):
        return np.moveaxis(array.reshape(grid.n_cells, -1, levels), 0, 1).reshape(*lead_shape, grid.n_cells, levels)

    return restore(divergence), restore(supported)


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
        'omega_discretization': ('spherical Voronoi dual edges from Delaunay face circumcentres; '
                                 'midpoint flux of parallel-transported pair winds; vertical trapezoid'),
        'omega_support': 'both wind components and geometric support at every adjacent cell and top-to-level path',
        'issuance_mask': 'predicted geopotential >= g0 * independent land-aware point DEM height',
        'target_mask_modified': False,
        'limitations': ['diagnostic omega is not direct observation supervision',
                        'finite-volume interpolation has truncation error',
                        'imposed top boundary and absent lower mass correction may bias omega',
                        'geopotential/geometric DEM datum approximation'],
    }
    return profiles.astype(np.float32), mask, metadata
