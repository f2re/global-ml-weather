"""Exact spherical Voronoi dual of the R11 latent mesh, built from Delaunay faces.

The two endpoints of a dual edge are the spherical circumcentres of the two
faces that share the primal edge. This reproduces SphericalVoronoi on the
historical meshes without materialised polygons. Dual areas are built from the
same circumcentres, so divergence, gradient and control volumes use one
consistent geometry and the areas close the sphere to round-off.

The R9/R10 diagnostic in ``joint_forecast_physics`` uses barycentric control
volumes on mesh8 because its stored mesh8 areas are barycentric. R11 keeps its
own geometry here, so the two experiments never share a mutable cache.
"""
from __future__ import annotations

import numpy as np

from ..grid import EARTH_RADIUS_M

CACHE_ATTRIBUTE = '_r11_voronoi_dual'


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

    pairs[e] = (left, right) with left < right; normal points left -> right.
    Areas are the exact spherical Voronoi areas built from the same circumcentres.
    """
    cached = getattr(grid, CACHE_ATTRIBUTE, None)
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
    cached = {'pairs': pairs, 'midpoint': midpoint, 'normal': normal, 'length': length,
              'areas_m2': areas}
    setattr(grid, CACHE_ATTRIBUTE, cached)
    return cached


__all__ = ['dual_geometry']
