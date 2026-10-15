"""R11 product output: 37 pressure levels on a regular latitude-longitude grid and GRIB2.

The latent mesh 5 has a 241 km node spacing. Products are interpolated in the
spherical triangle that contains each grid point (barycentric weights, as in
the observation operator); this does not add resolution. The default 1 degree
grid is finer than the model, so neighbouring points are correlated.

* scalars (T, q, Phi, omega): barycentric over the nodes that are above ground
  at that level, weights renormalised;
* wind: node ENU components are turned into global Cartesian vectors,
  interpolated, then projected onto the target point's east and north;
* below ground: a point is masked at pressure p when p exceeds the
  interpolated surface pressure; no value is extrapolated below the surface;
* surface pressure and MSLP: barycentric interpolation.

GRIB2 uses WMO discipline 0 codes (T 0/0/0, q 0/1/0, u 0/2/2, v 0/2/3, Phi 0/3/4,
omega 0/2/8 on isobaric surfaces; surface pressure 0/3/0 at the ground, MSLP
0/3/1 at mean sea level), a bitmap for masked points and simple packing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from ..grib_export import MISSING_CENTRE, PROFILE_CODES, bases, regular_grid, sha256
from ..vertical import PRESSURE_HPA, PROFILE_UNITS, PROFILE_VARIABLES
from .mesh import LatentMesh

SCHEMA = 'physical-assimilation-grib2-1'
MISSING = 9999.0
SURFACE_CODES = (('surface_pressure', 'Pa', 0, 1), ('mslp', 'Pa', 1, 101))  # (name, units, parameter, fixed surface)


def _mesh_for(manifest: dict, xyz: np.ndarray) -> LatentMesh:
    n = len(xyz)
    level = manifest.get('mesh_level')
    if level is None:
        level = int(round(np.log((n - 2) / 10) / np.log(4)))
    mesh = LatentMesh(int(level), multimesh=False)
    if mesh.n_nodes != n or not np.allclose(mesh.xyz, xyz, atol=1e-6):
        raise ValueError('Forecast nodes do not match the latent mesh geometry.')
    if manifest.get('grid_fingerprint') not in (None, mesh.grid.fingerprint):
        raise ValueError('Forecast grid fingerprint differs from the rebuilt mesh.')
    return mesh


def regrid_lead(profiles, mask, surface_pressure, mslp, node_xyz, nodes, weights, target_xyz, pressure_pa):
    """One lead: node profiles [N,37,6] -> grid [M,37,6] with mask [M,37]; surfaces [M,2]."""
    w = weights[:, :, None]                                    # [M,3,1]
    ps = (surface_pressure[nodes] * weights).sum(1)            # [M]
    sea = (mslp[nodes] * weights).sum(1)
    valid_nodes = mask[nodes]                                  # [M,3,37]
    vw = w * valid_nodes                                       # [M,3,37]
    total = vw.sum(1)                                          # [M,37]
    above = (pressure_pa[None, :] <= ps[:, None]) & (total > 0)
    safe = np.where(total > 0, total, 1.0)
    values = np.nan_to_num(profiles[nodes])                    # [M,3,37,6]
    out = np.full((len(target_xyz), len(pressure_pa), 6), np.nan)
    for k in (0, 1, 4, 5):
        out[..., k] = (vw * values[..., k]).sum(1) / safe
    east_n, north_n = bases(node_xyz)
    vector = (values[..., 2:3] * east_n[nodes][:, :, None, :] + values[..., 3:4] * north_n[nodes][:, :, None, :])
    vector = (vw[..., None] * vector).sum(1) / safe[..., None]  # [M,37,3]
    east_t, north_t = bases(target_xyz)
    out[..., 2] = (vector * east_t[:, None, :]).sum(-1)
    out[..., 3] = (vector * north_t[:, None, :]).sum(-1)
    out[~above] = np.nan
    return out, above, np.stack((ps, sea), -1)


def _keys(origin, lead, lat, lon, spacing, centre, subcentre):
    return {'centre': centre, 'subCentre': subcentre, 'tablesVersion': 4, 'localTablesVersion': 0,
            'productionStatusOfProcessedData': 2, 'typeOfProcessedData': 2, 'significanceOfReferenceTime': 1,
            'year': origin.year, 'month': origin.month, 'day': origin.day, 'hour': origin.hour,
            'minute': origin.minute, 'second': origin.second, 'discipline': 0, 'productDefinitionTemplateNumber': 0,
            'typeOfGeneratingProcess': 2, 'generatingProcessIdentifier': 255, 'backgroundProcess': 255,
            'hoursAfterDataCutoff': 65535, 'minutesAfterDataCutoff': 255, 'indicatorOfUnitOfTimeRange': 1,
            'forecastTime': int(lead), 'shapeOfTheEarth': 1, 'scaleFactorOfRadiusOfSphericalEarth': 1,
            'scaledValueOfRadiusOfSphericalEarth': 63710088, 'Ni': len(lon), 'Nj': len(lat),
            'latitudeOfFirstGridPointInDegrees': float(lat[0]), 'latitudeOfLastGridPointInDegrees': float(lat[-1]),
            'longitudeOfFirstGridPointInDegrees': float(lon[0]), 'longitudeOfLastGridPointInDegrees': float(lon[-1]),
            'iDirectionIncrementInDegrees': spacing, 'jDirectionIncrementInDegrees': spacing,
            'iScansNegatively': 0, 'jScansPositively': 0, 'jPointsAreConsecutive': 0, 'alternativeRowScanning': 0,
            'uvRelativeToGrid': 0, 'packingType': 'grid_simple', 'bitsPerValue': 24, 'bitmapPresent': 1,
            'missingValue': MISSING}


def _write(ec, stream, sample, keys, values, valid, missing_surface=False):
    message = ec.codes_grib_new_from_samples(sample)
    try:
        for key, value in keys.items():
            ec.codes_set(message, key, value)
        if missing_surface:
            ec.codes_set_missing(message, 'scaleFactorOfFirstFixedSurface')
            ec.codes_set_missing(message, 'scaledValueOfFirstFixedSurface')
        if np.any(valid & (values == MISSING)):
            raise ValueError('A valid value collides with the GRIB missing-value sentinel.')
        ec.codes_set_values(message, np.where(valid, values, MISSING))
        ec.codes_write(message, stream)
    finally:
        ec.codes_release(message)


def export_grib(forecast_dir, output, *, spacing: float = 1.0, member: int = 0,
                centre: int = MISSING_CENTRE, subcentre: int = 0) -> dict:
    try:
        import eccodes as ec
    except ImportError as error:
        raise RuntimeError('GRIB2 export requires the grib optional dependency (ecCodes).') from error
    forecast_dir, output = Path(forecast_dir), Path(output)
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():
        raise FileExistsError('Preserve existing GRIB2 artifacts; choose a new output path.')
    manifest_path, archive = forecast_dir / 'forecast.json', forecast_dir / 'forecast.npz'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'physical-assimilation-forecast-1':
        raise ValueError('Only R11 forecasts are exported here.')
    archive_hash, manifest_hash = sha256(archive), sha256(manifest_path)
    if manifest.get('forecast_sha256') != archive_hash:
        raise ValueError('Forecast archive SHA256 differs from its manifest.')
    if manifest.get('profile_variables') != list(PROFILE_VARIABLES) or manifest.get('profile_units') != list(PROFILE_UNITS):
        raise ValueError('Profile variable order or units differ.')
    origin = datetime.fromisoformat(manifest['issue_time'].replace('Z', '+00:00')).astimezone(timezone.utc)
    with np.load(archive, allow_pickle=False) as data:
        if not 0 <= member < data['profiles'].shape[0]:
            raise ValueError('Member outside the forecast ensemble.')
        profiles, mask = data['profiles'][member], data['profile_mask'][member]
        surface_pressure, mslp = data['surface_pressure'][member], data['mslp'][member]
        node_xyz, leads = data['xyz'], data['lead_hours']
        pressure_pa = data['pressure_pa']
    if not np.array_equal(pressure_pa, np.asarray(PRESSURE_HPA, float) * 100.):
        raise ValueError('Exactly the canonical 37 pressure levels are required.')
    mesh = _mesh_for(manifest, node_xyz)
    lat, lon, target_xyz = regular_grid(spacing)
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing='ij')
    interpolation = mesh.interpolation(grid_lat.ravel(), np.where(grid_lon.ravel() > 180, grid_lon.ravel() - 360,
                                                                  grid_lon.ravel()))
    nodes, weights = interpolation.nodes, interpolation.weights
    inventory = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.r11-grib-', dir=output.parent) as staging:
        staged = Path(staging) / 'forecast.grib2'
        with staged.open('wb') as stream:
            for t, lead in enumerate(leads):
                fields, valid, surface = regrid_lead(profiles[t], mask[t], surface_pressure[t], mslp[t], node_xyz,
                                                     nodes, weights, target_xyz, pressure_pa)
                base = _keys(origin, lead, lat, lon, spacing, centre, subcentre)
                valid_time = (origin + timedelta(hours=int(lead))).isoformat()
                for level, pressure in enumerate(pressure_pa):
                    for variable, (category, parameter) in enumerate(PROFILE_CODES):
                        keys = {**base, 'parameterCategory': category, 'parameterNumber': parameter,
                                'typeOfFirstFixedSurface': 100, 'scaleFactorOfFirstFixedSurface': 0,
                                'scaledValueOfFirstFixedSurface': int(pressure), 'typeOfSecondFixedSurface': 255}
                        _write(ec, stream, 'regular_ll_pl_grib2', keys, fields[:, level, variable], valid[:, level])
                        inventory.append({'message': len(inventory) + 1, 'variable': PROFILE_VARIABLES[variable],
                                          'units': PROFILE_UNITS[variable], 'parameter_category': category,
                                          'parameter_number': parameter, 'pressure_pa': int(pressure),
                                          'lead_hours': int(lead), 'valid_time': valid_time,
                                          'valid_points': int(valid[:, level].sum())})
                for index, (name, units, parameter, fixed) in enumerate(SURFACE_CODES):
                    keys = {**base, 'parameterCategory': 3, 'parameterNumber': parameter,
                            'typeOfFirstFixedSurface': fixed, 'typeOfSecondFixedSurface': 255}
                    _write(ec, stream, 'regular_ll_sfc_grib2', keys, surface[:, index],
                           np.isfinite(surface[:, index]), missing_surface=True)
                    inventory.append({'message': len(inventory) + 1, 'variable': name, 'units': units,
                                      'parameter_category': 3, 'parameter_number': parameter,
                                      'fixed_surface_type': fixed, 'lead_hours': int(lead), 'valid_time': valid_time,
                                      'valid_points': int(np.isfinite(surface[:, index]).sum())})
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(archive) != archive_hash or sha256(manifest_path) != manifest_hash:
            raise ValueError('Source forecast changed during export.')
        report = {'schema': SCHEMA, 'scientific_acceptance': False, 'issue_time': origin.isoformat(),
                  'member': member, 'centre': centre, 'subcentre': subcentre,
                  'grib_sha256': sha256(staged), 'forecast_npz_sha256': archive_hash,
                  'forecast_manifest_sha256': manifest_hash, 'checkpoint_sha256': manifest.get('checkpoint_sha256'),
                  'eccodes_version': ec.codes_get_api_version(), 'data_kind': manifest.get('data_kind'),
                  'grid': {'template': 0, 'spacing_degrees': spacing, 'Ni': len(lon), 'Nj': len(lat),
                           'sampling': 'barycentric in the latent spherical triangle; Cartesian wind vectors',
                           'source_node_spacing_km': float(mesh.node_spacing_m / 1000), 'conservative': False},
                  'mask_policy': 'pressure greater than the interpolated surface pressure is masked (bitmap)',
                  'near_surface': 'station-native T/Td are not encoded: instrument heights are unknown',
                  'messages': inventory}
        staged_json = Path(staging) / 'inventory.json'
        staged_json.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.link(staged_json, sidecar)
        try:
            os.link(staged, output)
        except BaseException:
            sidecar.unlink()
            raise
    return report


def decode_check(grib_path, forecast_dir, *, member: int = 0, spacing: float = 1.0, points: int = 2000,
                 seed: int = 0) -> dict:
    """Decode every message with ecCodes and compare sampled points with a separate interpolation.

    The sampled points get their own triangle search and weights, but the same
    regridding formula. The check therefore covers GRIB encoding, the bitmap,
    grid ordering and the point search; it is not an independent method.
    """
    import eccodes as ec
    manifest = json.loads((Path(forecast_dir) / 'forecast.json').read_text())
    with np.load(Path(forecast_dir) / 'forecast.npz', allow_pickle=False) as data:
        profiles, mask = data['profiles'][member], data['profile_mask'][member]
        surface_pressure, node_xyz, leads = data['surface_pressure'][member], data['xyz'], data['lead_hours']
    mesh = _mesh_for(manifest, node_xyz)
    lat, lon, target_xyz = regular_grid(spacing)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(target_xyz), min(points, len(target_xyz)), replace=False)
    grid_lat, grid_lon = np.meshgrid(lat, lon, indexing='ij')
    sample_lat, sample_lon = grid_lat.ravel()[chosen], grid_lon.ravel()[chosen]
    interpolation = mesh.interpolation(sample_lat, np.where(sample_lon > 180, sample_lon - 360, sample_lon))
    codes = {code: k for k, code in enumerate(PROFILE_CODES)}
    pressure_pa = np.asarray(PRESSURE_HPA, float) * 100.
    worst, messages, masked_mismatch = {}, 0, 0
    with open(grib_path, 'rb') as stream:
        expected_cache = {}
        while True:
            message = ec.codes_grib_new_from_file(stream)
            if message is None:
                break
            try:
                messages += 1
                category, parameter = ec.codes_get(message, 'parameterCategory'), ec.codes_get(message, 'parameterNumber')
                lead = ec.codes_get(message, 'forecastTime')
                if ec.codes_get(message, 'typeOfFirstFixedSurface', ktype=int) != 100:
                    continue
                values = ec.codes_get_values(message)[chosen]
                level = int(np.flatnonzero(pressure_pa == ec.codes_get(message, 'level') * 100.)[0])
                t = int(np.flatnonzero(leads == lead)[0])
                if t not in expected_cache:
                    expected_cache = {t: regrid_lead(profiles[t], mask[t], surface_pressure[t], surface_pressure[t],
                                                     node_xyz, interpolation.nodes, interpolation.weights,
                                                     target_xyz[chosen], pressure_pa)}
                fields, valid, _ = expected_cache[t]
                k = codes[(category, parameter)]
                decoded_valid = values != MISSING
                masked_mismatch += int((decoded_valid != valid[:, level]).sum())
                both = decoded_valid & valid[:, level]
                if both.any():
                    error = float(np.abs(values[both] - fields[both, level, k]).max())
                    worst[PROFILE_VARIABLES[k]] = max(worst.get(PROFILE_VARIABLES[k], 0.0), error)
            finally:
                ec.codes_release(message)
    return {'messages': messages, 'expected_messages': len(leads) * (37 * 6 + 2), 'points_checked': len(chosen),
            'mask_mismatches': masked_mismatch, 'max_abs_error': worst}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast', required=True, help='R11 forecast directory (forecast.npz + forecast.json).')
    parser.add_argument('--output', required=True)
    parser.add_argument('--spacing-degrees', type=float, default=1.0)
    parser.add_argument('--member', type=int, default=0)
    parser.add_argument('--centre', type=int, default=MISSING_CENTRE)
    parser.add_argument('--subcentre', type=int, default=0)
    parser.add_argument('--check', action='store_true', help='Decode the written file and compare with the source.')
    args = parser.parse_args(argv)
    report = export_grib(args.forecast, args.output, spacing=args.spacing_degrees, member=args.member,
                         centre=args.centre, subcentre=args.subcentre)
    result = {'messages': len(report['messages']), 'grib_sha256': report['grib_sha256']}
    if args.check:
        result['check'] = decode_check(args.output, args.forecast, member=args.member, spacing=args.spacing_degrees)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
