"""Immutable research GRIB2 export of physical frozen-profile forecasts.

The regular product grid is a nearest-cell sample of the spherical model grid;
it does not increase model resolution or conserve area integrals.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import tempfile
import os

import numpy as np
from scipy.spatial import cKDTree

from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS

# WMO discipline 0, category, parameter number, fixed surface type, level (SI).
PROFILE_CODES = ((0, 0), (1, 0), (2, 2), (2, 3), (3, 4), (2, 8))
MISSING_CENTRE = 65535  # Two-byte all-bits-one missing originating centre.


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream)


def _digest(stream) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(chunk)
    return digest.hexdigest()


def regular_grid(spacing: float):
    if not np.isfinite(spacing) or not 0.25 <= spacing <= 90:
        raise ValueError('Product grid spacing must be between 0.25 and 90 degrees.')
    ni, nj_intervals = round(360 / spacing), round(180 / spacing)
    if not np.isclose(ni * spacing, 360) or not np.isclose(nj_intervals * spacing, 180):
        raise ValueError('Grid spacing must divide both 180 and 360 degrees.')
    # Cell centres avoid duplicated poles and the ambiguous ENU basis at a pole.
    lat = 90 - spacing / 2 - np.arange(nj_intervals) * spacing
    lon = np.arange(ni) * spacing
    longitude, latitude = np.meshgrid(lon, lat)
    phi, lam = np.deg2rad(latitude.ravel()), np.deg2rad(longitude.ravel())
    xyz = np.column_stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)))
    return lat, lon, xyz


def bases(xyz):
    lam = np.arctan2(xyz[:, 1], xyz[:, 0])
    phi = np.arcsin(np.clip(xyz[:, 2], -1, 1))
    east = np.column_stack((-np.sin(lam), np.cos(lam), np.zeros_like(lam)))
    north = np.column_stack((-np.sin(phi)*np.cos(lam), -np.sin(phi)*np.sin(lam), np.cos(phi)))
    return east, north


def wind_rotation(source_xyz, target_xyz):
    """Nearest-cell tangent vectors parallel-transported along the short arc."""
    # Rodrigues rotation maps the source radial vector onto the target radial.
    cross = np.cross(source_xyz, target_xyz)
    cosine = (source_xyz * target_xyz).sum(axis=-1)
    if np.any(cosine <= -1 + 1e-10):
        raise ValueError('Antipodal wind transport is undefined.')
    east, north = bases(source_xyz)
    def rotate(vector):
        return vector + np.cross(cross, vector) + np.cross(cross, np.cross(cross, vector)) / (1 + cosine[:, None])
    target_east, target_north = bases(target_xyz)
    transported_east, transported_north = rotate(east), rotate(north)
    return ((transported_east * target_east).sum(-1), (transported_north * target_east).sum(-1),
            (transported_east * target_north).sum(-1), (transported_north * target_north).sum(-1))


def validate_forecast(arrays, metadata):
    if metadata.get('schema') != 'measured-profile-frozen-forecast-1':
        raise ValueError('Only explicit frozen physical profile schema is supported.')
    if metadata.get('profile_variables') != list(PROFILE_VARIABLES) or metadata.get('profile_units') != list(PROFILE_UNITS):
        raise ValueError('Profile variable order or physical units differ.')
    if metadata.get('wind_basis') != 'local_enu_vector':
        raise ValueError('Explicit local ENU wind basis is required.')
    origin = datetime.fromisoformat(metadata['issue_time'].replace('Z', '+00:00'))
    if origin.tzinfo is None or origin.microsecond:
        raise ValueError('Issue time requires timezone and whole seconds.')
    origin = origin.astimezone(timezone.utc)
    leads = np.asarray(metadata['lead_hours'])
    maximum_lead = 240 if metadata.get('source_forecast_schema') == 'unified-research-forecast-1' else 72
    if leads.ndim != 1 or len(leads) == 0 or not np.isfinite(leads).all() or not np.equal(leads, leads.astype(int)).all() or (leads < 0).any() or (leads > maximum_lead).any() or (np.diff(leads) <= 0).any():
        raise ValueError(f'Forecast leads must be increasing integer hours in [0,{maximum_lead}].')
    expected_times = [(origin + timedelta(hours=int(lead))).isoformat() for lead in leads]
    parsed_times = [datetime.fromisoformat(item.replace('Z', '+00:00')) for item in metadata['valid_times']]
    if any(value.tzinfo is None for value in parsed_times):
        raise ValueError('Valid times require explicit timezones.')
    actual_times = [value.astimezone(timezone.utc).isoformat() for value in parsed_times]
    if actual_times != expected_times:
        raise ValueError('Valid times do not match issue and forecast leads.')
    xyz = arrays['xyz']
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or not np.allclose(np.linalg.norm(xyz, axis=1), 1, atol=1e-6):
        raise ValueError('Source coordinates must be finite unit spherical vectors.')
    shape = (len(leads), len(xyz), len(PRESSURE_HPA), len(PROFILE_VARIABLES))
    if arrays['profiles'].shape != shape or arrays['profile_variable_mask'].shape != shape:
        raise ValueError('Profile values and independent variable masks have inconsistent shapes.')
    if arrays['profile_variable_mask'].dtype != np.bool_:
        raise ValueError('Profile variable mask must be boolean.')
    if not np.array_equal(arrays['pressure_pa'], np.asarray(PRESSURE_HPA) * 100):
        raise ValueError('Exactly the canonical 37 pressure levels are required in Pa.')
    if not np.isfinite(arrays['profiles'][arrays['profile_variable_mask']]).all():
        raise ValueError('A valid profile value is nonfinite.')
    return origin, leads.astype(int)


def canonical_profiles(data, metadata):
    """Adapt explicitly named physical forecasts; diagnostics remain masked."""
    if metadata.get('schema') == 'joint-observation-research-forecast-1':
        canonical = {**metadata, 'schema': 'measured-profile-frozen-forecast-1',
                     'source_forecast_schema': metadata['schema']}
        if not np.array_equal(data['lead_hours'], np.asarray(metadata['lead_hours'])):
            raise ValueError('Joint archive leads differ from metadata.')
        return {name: data[name] for name in ('profiles', 'profile_variable_mask', 'pressure_pa', 'xyz')}, canonical
    if metadata.get('schema') == 'measured-profile-frozen-forecast-1':
        return {name: data[name] for name in ('profiles', 'profile_variable_mask', 'pressure_pa', 'xyz')}, metadata
    if metadata.get('schema') != 'unified-research-forecast-1':
        raise ValueError('Unsupported forecast schema.')
    if metadata.get('profile_variables') != list(PROFILE_VARIABLES[:5]) or metadata.get('profile_units') != list(PROFILE_UNITS[:5]):
        raise ValueError('Unified profile variable order or physical units differ.')
    if metadata.get('wind_basis') != 'east_north_local':
        raise ValueError('Explicit local ENU wind basis is required.')
    values, mask, latlon = data['profile'], data['profile_issue_mask'], data['grid_latlon']
    if mask.dtype != np.bool_ or values.shape != mask.shape or values.shape[-1] != 5:
        raise ValueError('Unified profile values/masks differ.')
    if latlon.ndim != 2 or latlon.shape[1] != 2 or not np.isfinite(latlon).all() or (np.abs(latlon[:, 0]) > 90).any():
        raise ValueError('Unified grid latitude/longitude is invalid.')
    if not np.array_equal(data['lead_hours'], np.asarray(metadata['lead_hours'])):
        raise ValueError('Unified archive leads differ from metadata.')
    phi, lam = np.deg2rad(latlon[:, 0]), np.deg2rad(latlon[:, 1])
    xyz = np.column_stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)))
    arrays = {'profiles': np.concatenate((values, np.full(values.shape[:-1]+(1,), np.nan)), axis=-1),
              'profile_variable_mask': np.concatenate((mask, np.zeros(mask.shape[:-1]+(1,), dtype=bool)), axis=-1),
              'pressure_pa': data['pressure_hpa']*100, 'xyz': xyz}
    canonical = {**metadata, 'schema': 'measured-profile-frozen-forecast-1',
                 'source_forecast_schema': 'unified-research-forecast-1',
                 'profile_variables': list(PROFILE_VARIABLES), 'profile_units': list(PROFILE_UNITS),
                 'wind_basis': 'local_enu_vector'}
    return arrays, canonical


def export_forecast(source: str | Path, output: str | Path, *, spacing: float = 5,
                    centre: int = MISSING_CENTRE, subcentre: int = 0,
                    station_radius_km: float = 50) -> dict:
    """Write concatenated GRIB2 messages and mandatory inventory/provenance JSON."""
    try:
        import eccodes as ec
    except ImportError as error:
        raise RuntimeError('GRIB2 export requires the grib optional dependency and ecCodes library.') from error
    if not 0 <= centre <= 65535 or not 0 <= subcentre <= 65535:
        raise ValueError('Centre/subcentre must fit their WMO two-byte fields.')
    if not np.isfinite(station_radius_km) or not 0 < station_radius_km <= 500:
        raise ValueError('Station support radius must be in (0,500] km.')
    source, output = Path(source), Path(output)
    sidecar = output.with_suffix(output.suffix + '.json')
    if output.exists() or sidecar.exists() or output.is_symlink() or sidecar.is_symlink():
        raise FileExistsError('Preserve existing GRIB2 artifacts; choose a new output path.')
    if any(parent.is_symlink() for parent in output.parents):
        raise ValueError('Output parent symlinks are forbidden.')
    metadata_path, archive = source / 'forecast.json', source / 'forecast.npz'
    metadata_hash, archive_hash = sha256(metadata_path), sha256(archive)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('artifact_sha256', metadata.get('forecast_npz_sha256')) != archive_hash:
        raise ValueError('Frozen forecast archive SHA256 differs.')
    with np.load(archive, allow_pickle=False) as data:
        arrays, canonical = canonical_profiles(data, metadata)
        station_arrays = None
        if metadata.get('schema') in ('unified-research-forecast-1', 'joint-observation-research-forecast-1'):
            from .observation_data import NATIVE_VARIABLES, UNITS
            if metadata.get('surface_variables') != list(NATIVE_VARIABLES) or metadata.get('surface_units') != list(UNITS):
                raise ValueError('Native station variable order or units differ.')
            station_arrays = {name: data[name] for name in ('surface', 'surface_issue_mask', 'station_latlon')}
    origin, leads = validate_forecast(arrays, canonical)
    lat, lon, target_xyz = regular_grid(spacing)
    _, indices = cKDTree(arrays['xyz']).query(target_xyz, workers=1)
    rotation = wind_rotation(arrays['xyz'][indices], target_xyz)
    if station_arrays is not None:
        coordinates = station_arrays['station_latlon']
        surface, surface_mask = station_arrays['surface'], station_arrays['surface_issue_mask']
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all() or (np.abs(coordinates[:, 0]) > 90).any() or len(coordinates) == 0:
            raise ValueError('Station coordinates are invalid.')
        if surface.shape != (len(leads), len(coordinates), 6) or surface_mask.shape != surface.shape or surface_mask.dtype != np.bool_:
            raise ValueError('Native surface values and independent masks differ.')
        if not np.isfinite(surface[surface_mask]).all() or (surface[..., 4:][surface_mask[..., 4:]] <= 0).any():
            raise ValueError('Valid station pressure must be finite and positive.')
        phi, lam = np.deg2rad(coordinates[:, 0]), np.deg2rad(coordinates[:, 1])
        station_xyz = np.column_stack((np.cos(phi)*np.cos(lam), np.cos(phi)*np.sin(lam), np.sin(phi)))
        distance, station_indices = cKDTree(station_xyz).query(target_xyz, workers=1)
        distance_km = 2*np.arcsin(np.clip(distance/2, 0, 1))*6371.0088
        station_support = distance_km <= station_radius_km
    inventory: list[dict] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.grib-export-', dir=output.parent) as staging:
        staged = Path(staging) / 'forecast.grib2'
        with staged.open('wb') as stream:
            for t, lead in enumerate(leads):
                for level, pressure in enumerate(arrays['pressure_pa']):
                    fields = arrays['profiles'][t, indices, level].copy()
                    masks = arrays['profile_variable_mask'][t, indices, level].copy()
                    u, v = fields[:, 2].copy(), fields[:, 3].copy()
                    fields[:, 2] = rotation[0]*u + rotation[1]*v
                    fields[:, 3] = rotation[2]*u + rotation[3]*v
                    masks[:, 2] = masks[:, 3] = masks[:, 2] & masks[:, 3]
                    for variable, (category, parameter) in enumerate(PROFILE_CODES):
                        message = ec.codes_grib_new_from_samples('regular_ll_pl_grib2')
                        try:
                            keys = {'centre': centre, 'subCentre': subcentre, 'tablesVersion': 4,
                                    'localTablesVersion': 0, 'productionStatusOfProcessedData': 2,
                                    'typeOfProcessedData': 2, 'significanceOfReferenceTime': 1,
                                    'year': origin.year, 'month': origin.month, 'day': origin.day,
                                    'hour': origin.hour, 'minute': origin.minute, 'second': origin.second,
                                    'discipline': 0, 'productDefinitionTemplateNumber': 0,
                                    'typeOfGeneratingProcess': 2, 'generatingProcessIdentifier': 255,
                                    'backgroundProcess': 255, 'hoursAfterDataCutoff': 65535, 'minutesAfterDataCutoff': 255,
                                    'indicatorOfUnitOfTimeRange': 1, 'forecastTime': int(lead),
                                    'parameterCategory': category, 'parameterNumber': parameter,
                                    'typeOfFirstFixedSurface': 100, 'scaleFactorOfFirstFixedSurface': 0,
                                    'scaledValueOfFirstFixedSurface': int(pressure),
                                    'typeOfSecondFixedSurface': 255,
                                    'shapeOfTheEarth': 1, 'scaleFactorOfRadiusOfSphericalEarth': 1,
                                    'scaledValueOfRadiusOfSphericalEarth': 63710088,
                                    'Ni': len(lon), 'Nj': len(lat),
                                    'latitudeOfFirstGridPointInDegrees': float(lat[0]),
                                    'latitudeOfLastGridPointInDegrees': float(lat[-1]),
                                    'longitudeOfFirstGridPointInDegrees': float(lon[0]),
                                    'longitudeOfLastGridPointInDegrees': float(lon[-1]),
                                    'iDirectionIncrementInDegrees': spacing, 'jDirectionIncrementInDegrees': spacing,
                                    'iScansNegatively': 0, 'jScansPositively': 0, 'jPointsAreConsecutive': 0,
                                    'alternativeRowScanning': 0, 'uvRelativeToGrid': 0,
                                    'packingType': 'grid_simple', 'bitsPerValue': 24, 'bitmapPresent': 1,
                                    'missingValue': 9999.0}
                            for key, value in keys.items():
                                ec.codes_set(message, key, value)
                            values = np.where(masks[:, variable], fields[:, variable], 9999.0)
                            if np.any(masks[:, variable] & (fields[:, variable] == 9999.0)):
                                raise ValueError('Valid value collides with GRIB missing-value sentinel.')
                            ec.codes_set_values(message, values)
                            ec.codes_write(message, stream)
                            inventory.append({'message': len(inventory)+1, 'variable': PROFILE_VARIABLES[variable],
                                              'units': PROFILE_UNITS[variable], 'discipline': 0,
                                              'parameter_category': category, 'parameter_number': parameter,
                                              'pressure_pa': int(pressure), 'lead_hours': int(lead),
                                              'field_origin': metadata.get('field_origins', {}).get(PROFILE_VARIABLES[variable], 'model_or_source'),
                                              'valid_time': (origin+timedelta(hours=int(lead))).isoformat(),
                                              'valid_points': int(masks[:, variable].sum())})
                        finally:
                            ec.codes_release(message)
            if station_arrays is not None:
                for t, lead in enumerate(leads):
                    for variable, parameter, fixed_surface in ((5, 1, 101),):
                        message = ec.codes_grib_new_from_samples('regular_ll_sfc_grib2')
                        try:
                            surface_keys = {**keys, 'forecastTime': int(lead), 'parameterCategory': 3,
                                            'parameterNumber': parameter, 'typeOfFirstFixedSurface': fixed_surface}
                            surface_keys.pop('scaleFactorOfFirstFixedSurface')
                            surface_keys.pop('scaledValueOfFirstFixedSurface')
                            for key, value in surface_keys.items():
                                ec.codes_set(message, key, value)
                            ec.codes_set_missing(message, 'scaleFactorOfFirstFixedSurface')
                            ec.codes_set_missing(message, 'scaledValueOfFirstFixedSurface')
                            mask = surface_mask[t, station_indices, variable] & station_support
                            values = surface[t, station_indices, variable]
                            if np.any(mask & (values == 9999.0)):
                                raise ValueError('Valid pressure collides with GRIB missing sentinel.')
                            ec.codes_set_values(message, np.where(mask, values, 9999.0))
                            ec.codes_write(message, stream)
                            inventory.append({'message': len(inventory)+1,
                                              'variable': metadata['surface_variables'][variable], 'units': 'Pa',
                                              'discipline': 0, 'parameter_category': 3, 'parameter_number': parameter,
                                              'fixed_surface_type': fixed_surface, 'lead_hours': int(lead),
                                              'valid_time': (origin+timedelta(hours=int(lead))).isoformat(),
                                              'valid_points': int(mask.sum()), 'station_support_radius_km': station_radius_km})
                        finally:
                            ec.codes_release(message)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256(archive) != archive_hash or sha256(metadata_path) != metadata_hash:
            raise ValueError('Source forecast changed during export.')
        report = {'schema': 'global-weather-research-grib2-1', 'scientific_acceptance': False,
                  'issue_time': origin.isoformat(), 'centre': centre, 'subcentre': subcentre,
                  'originating_centre_status': 'missing/unassigned' if centre == MISSING_CENTRE else 'operator-supplied',
                  'grib_sha256': sha256(staged), 'forecast_npz_sha256': archive_hash,
                  'forecast_metadata_sha256': metadata_hash, 'checkpoint': metadata.get('checkpoint'),
                  'training_identity': metadata.get('identity'), 'eccodes_version': ec.codes_get_api_version(),
                  'checkpoint_sha256': metadata.get('checkpoint_sha256'),
                  'checkpoint_epoch': metadata.get('checkpoint_epoch'),
                  'station_climatology_sha256': metadata.get('station_climatology_sha256'),
                  'source_revision': metadata.get('source_revision'),
                  'normalization_sha256': metadata.get('normalization_sha256'),
                  'input_provenance': metadata.get('input_provenance'),
                  'static_provenance': metadata.get('static_provenance'),
                  'profile_training_support': metadata.get('profile_training_support'),
                  'below_ground_mask_status': metadata.get('below_ground_mask_status'),
                  'diagnostic_physics': metadata.get('diagnostic_physics'),
                  'field_origins': metadata.get('field_origins'),
                  'source_schema': metadata['schema'], 'source_limitations': metadata.get('limitations', []),
                  'grid': {'template': 0, 'spacing_degrees': spacing, 'Ni': len(lon), 'Nj': len(lat),
                           'sampling': 'nearest spherical cell; scalar point samples; short-arc parallel transport of ENU wind',
                           'conservative': False, 'earth_radius_m': 6371008.8},
                  'mask_policy': 'source issuance variable mask; both components required for wind transport',
                  'surface_export': ('MSLP only; sparse nearest station within support radius; station pressure omitted '
                                     'because destination ground elevation differs; temperature and wind omitted because sensor heights are unknown') if station_arrays is not None
                                     else 'not encoded: frozen schema lacks explicit surface variable/unit/accumulation metadata',
                  'station_support_radius_km': station_radius_km if station_arrays is not None else None,
                  'messages': inventory}
        staged_json = Path(staging) / 'inventory.json'
        staged_json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        # Hard links provide atomic no-overwrite publication on this filesystem.
        os.link(staged_json, sidecar)
        try:
            os.link(staged, output)
        except BaseException:
            sidecar.unlink()
            raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecast', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--spacing-degrees', default=5., type=float)
    parser.add_argument('--centre', default=MISSING_CENTRE, type=int)
    parser.add_argument('--subcentre', default=0, type=int)
    parser.add_argument('--station-radius-km', default=50., type=float)
    args = parser.parse_args(argv)
    result = export_forecast(args.forecast, args.output, spacing=args.spacing_degrees,
                             centre=args.centre, subcentre=args.subcentre, station_radius_km=args.station_radius_km)
    print(json.dumps({'messages': len(result['messages']), 'grib_sha256': result['grib_sha256']}))


if __name__ == '__main__':
    main()
