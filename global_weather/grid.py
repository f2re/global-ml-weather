"""Icosahedral Delaunay mesh and its spherical Voronoi (hex/pent) dual.

This is NOT H3. The native hierarchy has 10*4**level+2 cells. Pooling groups
are nearest-coarse-centre groups, not exact nested polygon intersections.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import hashlib
import numpy as np
from scipy.spatial import ConvexHull, SphericalVoronoi, cKDTree

EARTH_RADIUS_M = 6_371_008.8


def unit_xyz(lat_deg, lon_deg):
    lat, lon = np.deg2rad(lat_deg), np.deg2rad(lon_deg)
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), axis=-1)


def latlon(xyz):
    return np.rad2deg(np.stack((np.arcsin(np.clip(xyz[..., 2], -1, 1)),
                                np.arctan2(xyz[..., 1], xyz[..., 0])), axis=-1))


@dataclass
class SphereGrid:
    level: int
    xyz: np.ndarray
    faces: np.ndarray
    edges: np.ndarray  # [2,E], directed source -> destination
    areas_m2: np.ndarray
    vertices: np.ndarray
    regions: list[list[int]]

    @property
    def n_cells(self):
        return len(self.xyz)

    @cached_property
    def tree(self):
        return cKDTree(self.xyz)

    @cached_property
    def fingerprint(self):
        h = hashlib.sha256()
        h.update(np.round(self.xyz, 12).astype('<f8').tobytes())
        h.update(self.edges.astype('<i8').tobytes())
        return h.hexdigest()

    def locate(self, lat_deg, lon_deg):
        """Nearest centre identifies a Voronoi cell, including poles/dateline."""
        if not np.all(np.isfinite(lat_deg)) or not np.all(np.isfinite(lon_deg)):
            raise ValueError('Coordinates must be finite.')
        if np.any(np.abs(np.asarray(lat_deg)) > 90):
            raise ValueError('Latitude outside [-90,90].')
        return self.tree.query(unit_xyz(lat_deg, lon_deg))[1]

    def region_mask(self, south=-90., north=90., west=-180., east=180.):
        """Cell-centre selection, not fractional overlap; supports date-line crossing."""
        if not (-90 <= south <= north <= 90 and -180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError('Invalid geographic bounds.')
        ll = latlon(self.xyz)
        lon_ok = ((ll[:, 1] >= west) & (ll[:, 1] <= east)) if west <= east else (
            (ll[:, 1] >= west) | (ll[:, 1] <= east))
        return (ll[:, 0] >= south) & (ll[:, 0] <= north) & lon_ok

    def save(self, path):
        offsets = np.r_[0, np.cumsum([len(x) for x in self.regions])]
        np.savez_compressed(path, level=self.level, xyz=self.xyz, faces=self.faces,
                            edge_index=self.edges, area_m2=self.areas_m2,
                            polygon_vertices=self.vertices,
                            polygon_offsets=offsets, polygon_indices=np.concatenate(self.regions),
                            fingerprint=self.fingerprint)


def build_grid(level=1, *, max_cells=200_000):
    if not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError('level must be an integer in [0,9].')
    n = 10 * 4**level + 2
    if n > max_cells:
        raise ValueError(f'{n} cells exceeds explicit build limit {max_cells}.')
    phi = (1 + np.sqrt(5.)) / 2
    points = []
    for a in (-1., 1.):
        for b in (-phi, phi):
            points.extend(((0., a, b), (a, b, 0.), (b, 0., a)))
    xyz = np.unique(np.array(points), axis=0)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    faces = np.sort(ConvexHull(xyz).simplices, axis=1)
    faces = faces[np.lexsort(faces.T[::-1])]
    for _ in range(level):
        points = list(xyz)
        midpoints = {}
        def midpoint(i, j):
            key = tuple(sorted((int(i), int(j))))
            if key not in midpoints:
                p = (xyz[i] + xyz[j]) / 2
                midpoints[key] = len(points)
                points.append(p / np.linalg.norm(p))
            return midpoints[key]
        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        xyz = np.asarray(points)
        faces = np.asarray(refined, dtype=np.int64)
    pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    edges = np.concatenate((pairs, pairs[:, ::-1])).T
    voronoi = SphericalVoronoi(xyz)
    voronoi.sort_vertices_of_regions()
    areas = voronoi.calculate_areas() * EARTH_RADIUS_M**2
    degrees = np.bincount(edges[0], minlength=len(xyz))
    if np.count_nonzero(degrees == 5) != 12 or not np.isin(degrees, [5, 6]).all():
        raise RuntimeError('Unexpected mesh topology.')
    if not np.isclose(areas.sum(), 4*np.pi*EARTH_RADIUS_M**2, rtol=1e-9):
        raise RuntimeError('Spherical area closure failed.')
    return SphereGrid(level, xyz, faces, edges, areas, voronoi.vertices, voronoi.regions)


def build_pyramid(level=1):
    return [build_grid(i) for i in range(level, -1, -1)]
