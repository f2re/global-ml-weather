"""Offline, hashed point terrain with an independent GeoJSON land mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from .grid import EARTH_RADIUS_M, build_grid, build_mesh8_grid, latlon
from .observation_training import digest, save
from .terrain import dem_samples, local_file

MAX_DEM_BYTES = 900_000_000
MAX_LAND_BYTES = 10_000_000


def _inside_ring(longitude, latitude, ring):
    vertices = np.asarray(ring, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] < 2 or len(vertices) < 4
            or not np.isfinite(vertices[:, :2]).all()):
        raise ValueError('Invalid GeoJSON polygon ring.')
    # Unwrap actual date-line crossings; shift query longitude onto this ring.
    x = np.rad2deg(np.unwrap(np.deg2rad(vertices[:, 0])))
    y = vertices[:, 1]
    px = longitude + 360 * np.round((np.mean(x) - longitude) / 360)
    inside = np.zeros(np.shape(longitude), dtype=bool)
    for index in range(len(x) - 1):
        x0, x1, y0, y1 = x[index], x[index + 1], y[index], y[index + 1]
        crossing = (y0 > latitude) != (y1 > latitude)
        if y1 != y0:
            boundary_x = x0 + (latitude - y0) * (x1 - x0) / (y1 - y0)
            inside ^= crossing & (px < boundary_x)
    return inside


def land_at_points(geojson, latitude, longitude):
    """Point inclusion, including polygon holes; never derived from DEM sign."""
    latitude, longitude = np.asarray(latitude, float), np.asarray(longitude, float)
    if (latitude.shape != longitude.shape or not np.isfinite(latitude + longitude).all()
            or np.any(np.abs(latitude) > 90) or np.any(np.abs(longitude) > 180)):
        raise ValueError('Invalid geographic point coordinates.')
    if geojson.get('type') == 'FeatureCollection':
        geometries = [feature.get('geometry') for feature in geojson['features']]
    elif geojson.get('type') == 'Feature':
        geometries = [geojson.get('geometry')]
    else:
        geometries = [geojson]
    land = np.zeros(latitude.shape, bool)
    for geometry in geometries:
        if not isinstance(geometry, dict) or geometry.get('type') not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Independent land source must contain Polygon/MultiPolygon geometries.')
        polygons = [geometry['coordinates']] if geometry['type'] == 'Polygon' else geometry['coordinates']
        for polygon in polygons:
            if not polygon:
                raise ValueError('Empty land polygon.')
            inside = _inside_ring(longitude, latitude, polygon[0])
            for hole in polygon[1:]:
                inside &= ~_inside_ring(longitude, latitude, hole)
            land |= inside
    return land


def land_aware_height(dem_elevation_m, land_mask):
    elevation, land = np.asarray(dem_elevation_m, float), np.asarray(land_mask, bool)
    if elevation.shape != land.shape or not np.isfinite(elevation).all():
        raise ValueError('Need finite elevation and matching independent land mask.')
    return np.where(land, elevation, 0).astype(np.float32)


def terrain_slope(grid, elevation):
    elevation = np.asarray(elevation, float)
    source, target = grid.edges
    distance = EARTH_RADIUS_M * np.arccos(np.clip(np.sum(grid.xyz[source] * grid.xyz[target], axis=-1), -1, 1))
    edge_slope = np.abs(elevation[target] - elevation[source]) / distance
    total = np.zeros(grid.n_cells)
    count = np.zeros(grid.n_cells)
    np.add.at(total, source, edge_slope)
    np.add.at(count, source, 1)
    return (total / count).astype(np.float32)


def prepare_static(dem_path, dem_manifest, land_geojson, land_sha256, output, *, mesh_level):
    """Prepare a new immutable local artifact without network operations."""
    dem_path, manifest_path, land_path = map(local_file, (dem_path, dem_manifest, land_geojson))
    if dem_path.stat().st_size > MAX_DEM_BYTES or land_path.stat().st_size > MAX_LAND_BYTES:
        raise ValueError('Static source exceeds declared size budget.')
    if (not isinstance(land_sha256, str) or len(land_sha256) != 64
            or digest(land_path) != land_sha256):
        raise ValueError('Independent land GeoJSON hash differs.')
    dem_metadata = json.loads(manifest_path.read_text())
    grid = build_mesh8_grid() if mesh_level == 8 else build_grid(mesh_level)
    ll = latlon(grid.xyz)
    land = land_at_points(json.loads(land_path.read_text()), ll[:, 0], ll[:, 1])
    sampled = dem_samples(dem_path, dem_metadata, ll[:, 0], ll[:, 1])
    elevation = land_aware_height(sampled, land)
    source_identity = {'dem_sha256': dem_metadata['sha256'], 'dem_manifest_sha256': digest(manifest_path),
                       'land_sha256': land_sha256, 'grid_fingerprint': grid.fingerprint}
    output = Path(output)
    if output.exists():
        existing = load_static(output, grid)
        if existing['manifest']['sources'] != source_identity:
            raise ValueError('Static inputs changed; use a new output ID.')
        return existing['manifest']
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.joint-static-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'static'
        stage.mkdir()
        np.savez_compressed(stage / 'static.npz', elevation_m=elevation, land_mask=land,
                            slope=terrain_slope(grid, elevation), xyz=grid.xyz)
        manifest = {'schema': 'joint-static-terrain-1', 'sources': source_identity,
                    'grid_fingerprint': grid.fingerprint, 'mesh_level': mesh_level,
                    'artifact_sha256': digest(stage / 'static.npz'),
                    'units': {'elevation_m': 'm', 'land_mask': '1', 'slope': '1'},
                    'operator': 'nearest DEM pixel at model cell centre; GeoJSON point-in-polygon land support',
                    'ocean_atmospheric_boundary_m': 0, 'negative_land_preserved': True,
                    'area_average': False, 'land_fraction': False,
                    'source_paths': {'dem': str(dem_path), 'dem_manifest': str(manifest_path), 'land': str(land_path)},
                    'limitations': ['coastline is independent coarse vector point classification',
                                    'point elevations are not cell area means',
                                    'DEM/geopotential datum approximation'],
                    'retention': 'retain with experiment and checkpoint; no automatic expiry'}
        if (digest(land_path) != land_sha256
                or digest(manifest_path) != source_identity['dem_manifest_sha256']
                or digest(dem_path) != source_identity['dem_sha256']):
            raise ValueError('Static source changed during preparation.')
        save(stage / 'manifest.json', manifest)
        stage.rename(output)
    return manifest


def load_static(path, grid):
    root = Path(path)
    if root.is_symlink() or (root / 'manifest.json').is_symlink() or (root / 'static.npz').is_symlink():
        raise ValueError('Static artifact symlinks are forbidden.')
    manifest = json.loads((root / 'manifest.json').read_text())
    if (manifest.get('schema') != 'joint-static-terrain-1'
            or manifest.get('grid_fingerprint') != grid.fingerprint
            or digest(root / 'static.npz') != manifest['artifact_sha256']):
        raise ValueError('Static geometry or hash differs.')
    with np.load(root / 'static.npz', allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in ('elevation_m', 'land_mask', 'slope', 'xyz')}
    if (values['elevation_m'].shape != (grid.n_cells,) or values['land_mask'].shape != (grid.n_cells,)
            or values['slope'].shape != (grid.n_cells,) or values['land_mask'].dtype != bool
            or values['xyz'].shape != grid.xyz.shape or not np.array_equal(values['xyz'], grid.xyz)
            or not np.isfinite(values['elevation_m']).all() or not np.isfinite(values['slope']).all()
            or np.any(values['slope'] < 0) or np.any(values['elevation_m'][~values['land_mask']] != 0)):
        raise ValueError('Invalid land-aware static arrays.')
    values.update(manifest=manifest, static_sha256=manifest['artifact_sha256'], manifest_sha256=digest(root / 'manifest.json'))
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem', required=True)
    parser.add_argument('--dem-manifest', required=True)
    parser.add_argument('--land-geojson', required=True)
    parser.add_argument('--land-sha256', required=True)
    parser.add_argument('--mesh-level', type=int, required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    prepare_static(args.dem, args.dem_manifest, args.land_geojson, args.land_sha256,
                   args.output, mesh_level=args.mesh_level)


if __name__ == '__main__':
    main()
