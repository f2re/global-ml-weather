"""Icosahedral Delaunay mesh and its spherical dual geometry.

This is NOT H3. The native hierarchy has 10*4**level+2 cells. Pooling groups
are nearest-coarse-centre groups, not exact nested polygon intersections.

Levels 0..7 retain the historical construction order and exact
:class:`scipy.spatial.SphericalVoronoi` areas so existing fingerprints and
checkpoints do not change. Mesh level 8 uses vectorised refinement and
vertex-centred one-third spherical-triangle control-volume areas. The latter
close globally and are suitable for conservative weighting, but they are not
exact Voronoi polygons and are labelled accordingly.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import hashlib
import numpy as np
from scipy.spatial import ConvexHull, SphericalVoronoi, cKDTree

EARTH_RADIUS_M = 6_371_008.8
MESH8_LEVEL = 8
MESH8_CELLS = 655_362
# Preserve every mesh that the pre-mesh8 implementation admitted by default.
EXACT_VORONOI_MAX_CELLS = 200_000
# Level 8 is the supported high-resolution research ceiling by default.
DEFAULT_MAX_CELLS = MESH8_CELLS


def grid_cell_count(level: int) -> int:
    """Return the exact number of cells in this icosahedral hierarchy."""
    if not isinstance(level, int) or not 0 <= level <= 9:
        raise ValueError('level must be an integer in [0,9].')
    return 10 * 4**level + 2


def nominal_area_scale_km(level: int) -> float:
    """Return sqrt(global mean cell area) in km, not resolved wavelength.

    This diagnostic exists to prevent a mesh level from being mislabeled as a
    meteorological resolution. Actual resolvable scales are generally larger
    and must be measured experimentally.
    """
    mean_area_m2 = 4 * np.pi * EARTH_RADIUS_M**2 / grid_cell_count(level)
    return float(np.sqrt(mean_area_m2) / 1000.0)


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
    area_method: str = 'spherical_voronoi'
    has_polygon_geometry: bool = True

    @property
    def n_cells(self):
        return len(self.xyz)

    @property
    def nominal_area_scale_km(self):
        """sqrt(mean cell area) in km; a transparent geometry diagnostic only."""
        return float(np.sqrt(np.mean(self.areas_m2)) / 1000.0)

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
        """Nearest centre identifies a model cell, including poles/dateline."""
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
        lengths = [len(x) for x in self.regions]
        offsets = np.r_[0, np.cumsum(lengths)]
        polygon_indices = (np.concatenate(self.regions).astype(np.int64, copy=False)
                           if self.regions else np.empty(0, dtype=np.int64))
        np.savez_compressed(path, level=self.level, xyz=self.xyz, faces=self.faces,
                            edge_index=self.edges, area_m2=self.areas_m2,
                            polygon_vertices=self.vertices,
                            polygon_offsets=offsets, polygon_indices=polygon_indices,
                            fingerprint=self.fingerprint,
                            nominal_area_scale_km=self.nominal_area_scale_km,
                            area_method=self.area_method,
                            has_polygon_geometry=self.has_polygon_geometry)


def _base_icosahedron():
    phi = (1 + np.sqrt(5.)) / 2
    points = []
    for a in (-1., 1.):
        for b in (-phi, phi):
            points.extend(((0., a, b), (a, b, 0.), (b, 0., a)))
    xyz = np.unique(np.array(points), axis=0)
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)
    faces = np.sort(ConvexHull(xyz).simplices, axis=1)
    faces = faces[np.lexsort(faces.T[::-1])]
    return xyz, faces.astype(np.int64, copy=False)


def _legacy_refine(xyz, faces, steps):
    """Historical refinement order used by existing mesh 0..7 fingerprints."""
    for _ in range(steps):
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
    return xyz, faces


def _vector_refine_once(xyz, faces):
    """Memory-bounded deterministic refinement used only for mesh level 8."""
    f = len(faces)
    raw = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64, copy=False)
    raw.sort(axis=1)
    pairs, inverse = np.unique(raw, axis=0, return_inverse=True)
    midpoint_xyz = xyz[pairs[:, 0]] + xyz[pairs[:, 1]]
    midpoint_xyz /= np.linalg.norm(midpoint_xyz, axis=1, keepdims=True)
    base = len(xyz)
    mids = base + inverse
    ab, bc, ca = mids[:f], mids[f:2*f], mids[2*f:]
    refined = np.empty((4*f, 3), dtype=np.int64)
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    refined[0::4] = np.column_stack((a, ab, ca))
    refined[1::4] = np.column_stack((b, bc, ab))
    refined[2::4] = np.column_stack((c, ca, bc))
    refined[3::4] = np.column_stack((ab, bc, ca))
    return np.concatenate((xyz, midpoint_xyz), axis=0), refined


def _topology(level):
    xyz, faces = _base_icosahedron()
    # The old implementation admitted levels through 7, so preserve their
    # exact construction order. Only the final 7->8 refinement is vectorised.
    legacy_level = min(level, 7)
    xyz, faces = _legacy_refine(xyz, faces, legacy_level)
    for _ in range(legacy_level, level):
        xyz, faces = _vector_refine_once(xyz, faces)
    return xyz, faces


def _directed_edges(faces):
    pairs = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])).astype(np.int64, copy=False)
    pairs.sort(axis=1)
    pairs = np.unique(pairs, axis=0)
    return np.concatenate((pairs, pairs[:, ::-1])).T


def _triangle_control_areas(xyz, faces):
    """Vertex-centred 1/3 spherical-triangle areas with exact global closure."""
    a, b, c = xyz[faces[:, 0]], xyz[faces[:, 1]], xyz[faces[:, 2]]
    numerator = np.abs(np.einsum('ij,ij->i', a, np.cross(b, c)))
    denominator = 1.0 + np.einsum('ij,ij->i', a, b) + np.einsum('ij,ij->i', b, c) + np.einsum('ij,ij->i', c, a)
    triangle = 2.0 * np.arctan2(numerator, denominator) * EARTH_RADIUS_M**2
    areas = np.zeros(len(xyz), dtype=np.float64)
    np.add.at(areas, faces.reshape(-1), np.repeat(triangle / 3.0, 3))
    return areas


def build_grid(level=1, *, max_cells=DEFAULT_MAX_CELLS, exact_voronoi_max_cells=EXACT_VORONOI_MAX_CELLS):
    n = grid_cell_count(level)
    if type(max_cells) is not int or max_cells < 12 or n > max_cells:
        raise ValueError(f'{n} cells exceeds explicit build limit {max_cells}.')
    if type(exact_voronoi_max_cells) is not int or exact_voronoi_max_cells < 12:
        raise ValueError('exact_voronoi_max_cells must be an integer >= 12.')
    xyz, faces = _topology(level)
    if len(xyz) != n:
        raise RuntimeError('Icosahedral refinement produced an unexpected cell count.')
    edges = _directed_edges(faces)
    degrees = np.bincount(edges[0], minlength=len(xyz))
    if np.count_nonzero(degrees == 5) != 12 or not np.isin(degrees, [5, 6]).all():
        raise RuntimeError('Unexpected mesh topology.')

    if n <= exact_voronoi_max_cells:
        voronoi = SphericalVoronoi(xyz)
        voronoi.sort_vertices_of_regions()
        areas = voronoi.calculate_areas() * EARTH_RADIUS_M**2
        vertices = voronoi.vertices
        regions = voronoi.regions
        area_method = 'spherical_voronoi'
        has_polygons = True
    else:
        areas = _triangle_control_areas(xyz, faces)
        vertices = np.empty((0, 3), dtype=np.float64)
        regions = []
        area_method = 'one_third_spherical_triangle_control_volume'
        has_polygons = False

    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise RuntimeError('Nonpositive or nonfinite spherical cell area.')
    if not np.isclose(areas.sum(), 4*np.pi*EARTH_RADIUS_M**2, rtol=1e-9):
        raise RuntimeError('Spherical area closure failed.')
    return SphereGrid(level, xyz, faces, edges, areas, vertices, regions,
                      area_method=area_method, has_polygon_geometry=has_polygons)


def build_pyramid(level=1):
    return [build_grid(i) for i in range(level, -1, -1)]
