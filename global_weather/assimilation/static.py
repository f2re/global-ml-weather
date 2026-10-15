"""Area-mean terrain on the latent mesh from the verified DEM and land polygons.

A latent cell of mesh 5 spans ~240 km. A point sample of the DEM at its
centre is not the height of the atmosphere's lower boundary there. This
module samples the DEM and the independent land polygons at the centres of
a mesh `subsample_levels` finer (64 points per cell for 3 levels), assigns
every fine point to its nearest latent node and stores
* elevation_m: area mean of the land-aware height (ocean contributes 0 m),
* land_fraction: share of fine points on land,
* orography_std_m: sub-cell standard deviation of the land-aware height,
* slope: mean absolute edge gradient of the area-mean height.
Point-sampled joint static artifacts are accepted only as a labelled fallback.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial import cKDTree

from ..grid import build_grid, build_mesh8_grid, latlon
from ..joint_static import land_at_points, land_aware_height, terrain_slope
from ..observation_training import digest, save

SCHEMA = 'latent-area-static-1'


def aggregate(grid, fine_xyz: np.ndarray, fine_height: np.ndarray, fine_land: np.ndarray) -> dict:
    owner = cKDTree(grid.xyz).query(fine_xyz)[1]
    count = np.bincount(owner, minlength=grid.n_cells)
    if np.any(count == 0):
        raise ValueError('Every latent cell needs at least one sub-cell sample.')
    mean = np.bincount(owner, weights=fine_height, minlength=grid.n_cells) / count
    second = np.bincount(owner, weights=fine_height ** 2, minlength=grid.n_cells) / count
    land = np.bincount(owner, weights=fine_land.astype(float), minlength=grid.n_cells) / count
    spread = np.sqrt(np.maximum(second - mean ** 2, 0))
    return {'elevation_m': mean.astype(np.float32), 'land_fraction': land.astype(np.float32),
            'orography_std_m': spread.astype(np.float32), 'slope': terrain_slope(grid, mean),
            'samples_per_cell': count.astype(np.int32)}


def prepare_area_static(dem_path, dem_manifest, land_geojson, land_sha256, output, *, mesh_level: int,
                        subsample_levels: int = 3) -> dict:
    from ..terrain import dem_samples, local_file
    if type(mesh_level) is not int or not 1 <= mesh_level <= 7 or subsample_levels not in (2, 3):
        raise ValueError('Latent mesh 1..7 with 2 or 3 subsample levels.')
    dem_path, manifest_path, land_path = map(local_file, (dem_path, dem_manifest, land_geojson))
    if digest(land_path) != land_sha256:
        raise ValueError('Independent land GeoJSON hash differs.')
    grid = build_grid(mesh_level)
    fine_level = mesh_level + subsample_levels
    fine = build_mesh8_grid() if fine_level == 8 else build_grid(fine_level, max_cells=3_000_000)
    ll = latlon(fine.xyz)
    land = land_at_points(json.loads(land_path.read_text()), ll[:, 0], ll[:, 1])
    height = land_aware_height(dem_samples(dem_path, json.loads(manifest_path.read_text()), ll[:, 0], ll[:, 1]), land)
    fields = aggregate(grid, fine.xyz, height.astype(float), land)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Static output exists; use a new ID.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.latent-static-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'static'
        stage.mkdir()
        np.savez_compressed(stage / 'static.npz', xyz=grid.xyz, **fields)
        manifest = {'schema': SCHEMA, 'mesh_level': mesh_level, 'grid_fingerprint': grid.fingerprint,
                    'subsample_mesh_level': fine_level, 'artifact_sha256': digest(stage / 'static.npz'),
                    'sources': {'dem_sha256': digest(dem_path), 'dem_manifest_sha256': digest(manifest_path),
                                'land_sha256': land_sha256},
                    'operator': 'mean of land-aware DEM point samples at finer-mesh centres per nearest latent node',
                    'units': {'elevation_m': 'm', 'land_fraction': '1', 'orography_std_m': 'm', 'slope': '1'},
                    'ocean_atmospheric_boundary_m': 0, 'negative_land_preserved': True,
                    'limitations': ['finite sub-sampling, not an exact polygon integral',
                                    'DEM/geopotential datum approximation']}
        save(stage / 'manifest.json', manifest)
        stage.rename(output)
    return manifest


def load_latent_static(path, grid) -> tuple[dict, dict]:
    root = Path(path)
    if root.is_symlink():
        raise ValueError('Static artifact symlinks are forbidden.')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest.get('grid_fingerprint') != grid.fingerprint or digest(root / 'static.npz') != manifest.get('artifact_sha256'):
        raise ValueError('Static artifact belongs to another grid or changed.')
    with np.load(root / 'static.npz', allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if not np.array_equal(data['xyz'], grid.xyz):
        raise ValueError('Static node coordinates differ from the latent mesh.')
    if manifest.get('schema') == SCHEMA:
        fields = {key: data[key].astype(float) for key in ('elevation_m', 'land_fraction', 'orography_std_m', 'slope')}
        kind = 'area_mean'
    elif manifest.get('schema') == 'joint-static-terrain-1':
        fields = {'elevation_m': data['elevation_m'].astype(float), 'land_fraction': data['land_mask'].astype(float),
                  'orography_std_m': np.zeros(grid.n_cells), 'slope': data['slope'].astype(float)}
        kind = 'point_sample_fallback'
    else:
        raise ValueError('Unknown static artifact schema.')
    for value in fields.values():
        if value.shape != (grid.n_cells,) or not np.isfinite(value).all():
            raise ValueError('Static field shape or finiteness differs.')
    return fields, {'kind': kind, 'schema': manifest['schema'], 'artifact_sha256': manifest['artifact_sha256'],
                    'manifest_sha256': digest(root / 'manifest.json')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dem', required=True); parser.add_argument('--dem-manifest', required=True)
    parser.add_argument('--land-geojson', required=True); parser.add_argument('--land-sha256', required=True)
    parser.add_argument('--mesh-level', type=int, required=True); parser.add_argument('--subsample-levels', type=int, default=3)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    manifest = prepare_area_static(args.dem, args.dem_manifest, args.land_geojson, args.land_sha256, args.output,
                                   mesh_level=args.mesh_level, subsample_levels=args.subsample_levels)
    print(json.dumps({'schema': manifest['schema'], 'artifact_sha256': manifest['artifact_sha256']}))


if __name__ == '__main__':
    main()
