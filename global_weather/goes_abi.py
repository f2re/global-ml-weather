"""Offline converter for operational GOES-18 ABI L1b RadC band 13.

The NOAA GOES-R PUG specifies the per-file Planck coefficients and sweep-x
fixed-grid navigation. The NOAA NODD public bucket's S3 object Last-Modified
is the retrospective archive availability used here. Local receipt latency is
unknown. This converter performs no network operations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from pathlib import Path
import re

import numpy as np
import torch

from .grid import EARTH_RADIUS_M
from .multimodal.contracts import Sequence, fingerprint, hash_value, utc
from .multimodal.io import load_sensors, load_sequence, read_json, regular, resolve, save_sequence, sha256, write_json


SOURCE = 'goes_abi'
PLATFORM = 'GOES-18'
CHANNEL = '13'
CALIBRATION_ID = 'noaa-goes18-abi-l1b-radc-c13-planck-1'
BUCKET = 'https://noaa-goes18.s3.amazonaws.com/'
AVAILABILITY_BASIS = 'noaa_nodd_s3_object_last_modified'
GEOMETRY_REFERENCE = 'NOAA ABI fixed grid; spherical area of native pixel corners; footprint sqrt(area)'
RADIANCE_UNITS = 'mW m-2 sr-1 (cm-1)-1'
PLANCK_FIELDS = ('planck_fk1', 'planck_fk2', 'planck_bc1', 'planck_bc2')
NATIVE_SHAPE = (1500, 2500)
PROJECTION = (6378137.0, 6356752.31414, 35786023.0, -137.0)
MAX_RAW_BYTES = 5 * 1024 * 1024
MAX_PIXELS = 262144
KEY = re.compile(r'ABI-L1b-RadC/(\d{4})/(\d{3})/(\d{2})/OR_ABI-L1b-RadC-M6C13_G18_'
                 r's(\d{14})_e(\d{14})_c(\d{14})\.nc\Z')


def _sensor(sensor):
    identity = (sensor.source, sensor.platform, sensor.kind, sensor.data_kind)
    channels = [(c.id, c.quantity, c.units, c.calibration_id) for c in sensor.channels]
    if (identity != (SOURCE, PLATFORM, 'imager', 'real')
            or channels != [(CHANNEL, 'brightness_temperature', 'K', CALIBRATION_ID)]):
        raise ValueError('Expected the physical GOES-18 ABI RadC C13 sensor registry entry.')


def _time(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d{1,6})?(?:Z|\+00:00)', value):
        raise ValueError('ABI receipt and scan times require explicit UTC.')
    # Python 3.10 requires three or six fractional digits for fromisoformat.
    return utc(re.sub(r'\.(\d{1,5})(?=Z|\+00:00)', lambda m: '.' + m[1].ljust(6, '0'), value))


def _key_times(key):
    match = KEY.fullmatch(key) if isinstance(key, str) else None
    if match is None:
        raise ValueError('Expected an operational NOAA GOES-18 RadC C13 object key.')
    stamps = tuple(datetime.strptime(value[:13], '%Y%j%H%M%S').replace(
        tzinfo=timezone.utc, microsecond=int(value[-1]) * 100_000) for value in match.groups()[3:])
    start, end, created = stamps
    if (start.strftime('%Y/%j/%H') != '/'.join(match.groups()[:3])
            or not 1 <= (end - start).total_seconds() <= 300
            or not 0 <= (created - end).total_seconds() <= 1200):
        raise ValueError('ABI key scan chronology differs from the official object prefix.')
    return stamps


def _file(root, ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('A relative native ABI path and SHA256 are required.')
    hash_value(ref['sha256'])
    rel = Path(ref['path'])
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('ABI file path leaves its data directory.')
    path = regular(root / rel)
    if not 1 <= path.stat().st_size <= MAX_RAW_BYTES:
        raise ValueError('Invalid ABI native file size.')
    return path


def _object(root, item):
    if not isinstance(item, dict) or set(item) != {'raw', 'receipt'}:
        raise ValueError('ABI object requires a native file and an archive receipt.')
    path = _file(root, item['raw'])
    receipt = read_json(resolve(root, item['receipt']))
    keys = {'schema', 'key', 'url', 'content_length', 'etag', 'sha256', 'archive_object_last_modified',
            'http_last_modified', 'observed_at'}
    if not isinstance(receipt, dict) or set(receipt) != keys or receipt['schema'] != 'noaa-nodd-s3-object-1':
        raise ValueError('Invalid NOAA object receipt schema.')
    start, end, created = _key_times(receipt['key'])
    written = _time(receipt['archive_object_last_modified'])
    last_modified = parsedate_to_datetime(receipt['http_last_modified'])
    if (receipt['url'] != BUCKET + receipt['key'] or not isinstance(receipt['etag'], str)
            or not re.fullmatch(r'[0-9a-f]{32}', receipt['etag'])
            or type(receipt['content_length']) is not int or receipt['content_length'] != path.stat().st_size
            or receipt['sha256'] != item['raw']['sha256'] or written != last_modified
            or not 0 <= (written - created).total_seconds() <= 1200 or written > _time(receipt['observed_at'])):
        raise ValueError('NOAA receipt disagrees with the native ABI object or its publication chronology.')
    frame = {'id': fingerprint({'source': SOURCE, 'platform': PLATFORM, 'channel': CHANNEL,
                                'scan_start': start.isoformat(), 'scan_end': end.isoformat()}),
             'scan_start': start.timestamp(), 'scan_end': end.timestamp(), 'available_at': written.timestamp()}
    return path, receipt, (start, end, created), frame


def _crop(crop):
    if not isinstance(crop, list) or len(crop) != 4 or any(type(value) is not int for value in crop):
        raise ValueError('Crop must contain four integer half-open native grid indices.')
    r0, r1, c0, c1 = crop
    shape = r1 - r0, c1 - c0
    if (not 0 <= r0 < r1 <= NATIVE_SHAPE[0] or not 0 <= c0 < c1 <= NATIVE_SHAPE[1]
            or min(shape) < 8 or shape[0] * shape[1] > MAX_PIXELS):
        raise ValueError('Crop lies outside the supported fixed-grid size.')
    return slice(r0, r1), slice(c0, c1), shape


def _planck(rad, coefficients):
    fk1, fk2, bc1, bc2 = (coefficients[field] for field in PLANCK_FIELDS)
    if not np.isfinite([fk1, fk2, bc1, bc2]).all() or min(fk1, fk2, bc2) <= 0:
        raise ValueError('Invalid native ABI Planck coefficients.')
    return (fk2 / np.log1p(fk1 / rad) - bc1) / bc2


def _latlon(x, y):
    """GOES-R PUG sweep-x navigation and ellipsoid-normal view zenith."""
    major, minor, height, lon0 = PROJECTION
    H = major + height
    ratio = (major / minor) ** 2
    x, y = np.meshgrid(x, y)
    cx, sx, cy, sy = np.cos(x), np.sin(x), np.cos(y), np.sin(y)
    a = sx**2 + cx**2 * (cy**2 + ratio * sy**2)
    b = -2 * H * cx * cy
    discriminant = b**2 - 4 * a * (H**2 - major**2)
    if (discriminant <= 0).any():
        raise ValueError('Chosen ABI crop intersects off-Earth pixels.')
    ray = (-b - np.sqrt(discriminant)) / (2 * a)
    rx, ry, rz = ray * cx * cy, -ray * sx, ray * cx * sy
    X = H - rx
    lat = np.arctan2(ratio * rz, np.hypot(X, ry))
    lon = (lon0 - np.degrees(np.arctan2(ry, X)) + 180) % 360 - 180
    normal = np.stack((X, -ry, ratio * rz), axis=-1)
    look = np.stack((rx, ry, -rz), axis=-1)
    cosine = (normal * look).sum(-1) / (np.linalg.norm(normal, axis=-1) * ray)
    zenith = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return np.degrees(lat), lon, zenith


def _triangle_area(a, b, c):
    cross = np.cross(b - a, c - a)
    numerator = np.abs((a * cross).sum(-1))
    denominator = 1 + (a * b + b * c + c * a).sum(-1)
    return 2 * np.arctan2(numerator, denominator) * EARTH_RADIUS_M ** 2


def _geometry(x, y):
    lat, lon, view = _latlon(x, y)
    def edges(axis):
        delta = np.diff(axis)
        return np.r_[axis[0] - delta[0] / 2, (axis[1:] + axis[:-1]) / 2, axis[-1] + delta[-1] / 2]
    verts_lat, verts_lon, _ = _latlon(edges(x), edges(y))
    phi, lam = np.radians(verts_lat), np.radians(verts_lon)
    p = np.stack((np.cos(phi) * np.cos(lam), np.cos(phi) * np.sin(lam), np.sin(phi)), -1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]
    area = _triangle_area(a, b, c) + _triangle_area(a, c, d)
    return lat, lon, view, area, np.sqrt(area) / 1000


def _read(path, receipt, stamps, rows, columns):
    from netCDF4 import Dataset

    with path.open('rb') as stream:
        payload = stream.read(MAX_RAW_BYTES + 1)
    if (len(payload) != receipt['content_length'] or hashlib.sha256(payload).hexdigest() != receipt['sha256'] or
            hashlib.md5(payload, usedforsecurity=False).hexdigest() != receipt['etag']):
        raise ValueError('ABI content differs from its verified SHA256 or single-part ETag.')
    with Dataset('abi', memory=payload) as ds:
        names = ('time_coverage_start', 'time_coverage_end', 'date_created')
        if (tuple(_time(getattr(ds, name)) for name in names) != stamps
                or getattr(ds, 'dataset_name', '') != receipt['key'].rsplit('/', 1)[-1]
                or getattr(ds, 'platform_ID', '') != 'G18' or getattr(ds, 'scene_id', '') != 'CONUS'
                or getattr(ds, 'instrument_ID', '') != 'FM3'
                or getattr(ds, 'production_environment', '') != 'OE' or getattr(ds, 'production_data_source', '') != 'Realtime'
                or getattr(ds, 'timeline_id', '') != 'ABI Mode 6'
                or ds['band_id'].size != 1 or int(ds['band_id'][...]) != 13):
            raise ValueError('NetCDF product identity and scan times differ from the official object.')
        rad, flags = ds['Rad'], ds['DQF']
        if (rad.shape != NATIVE_SHAPE or flags.shape != NATIVE_SHAPE or rad.dimensions != ('y', 'x')
                or flags.dimensions != rad.dimensions or rad.dtype.kind not in 'iu' or flags.dtype.kind not in 'iu'
                or rad.units != RADIANCE_UNITS or not 0 < float(rad.scale_factor)
                or not np.isfinite([rad.scale_factor, rad.add_offset]).all()
                or rad.grid_mapping != 'goes_imager_projection' or not np.array_equal(flags.flag_values, [0, 1, 2, 3, 4])
                or str(flags.flag_meanings).split()[0] != 'good_pixel_qf'):
            raise ValueError('Native ABI grid, packed radiance units or DQF interpretation are incompatible.')
        projection = ds['goes_imager_projection']
        actual = tuple(float(getattr(projection, name)) for name in ('semi_major_axis', 'semi_minor_axis',
                        'perspective_point_height', 'longitude_of_projection_origin'))
        if (actual != PROJECTION or projection.grid_mapping_name != 'geostationary'
                or projection.sweep_angle_axis != 'x' or float(projection.latitude_of_projection_origin) != 0):
            raise ValueError('Native GOES-West fixed-grid geometry changed.')
        axes = []
        for axis, index, size, start, delta in (
                ('x', columns, 2500, -0.069972, 0.000056), ('y', rows, 1500, 0.128212, -0.000056)):
            coord = ds[axis]
            if coord.shape != (size,) or coord.units != 'rad':
                raise ValueError('ABI fixed grid needs native one-dimensional scan angles in radians.')
            values = np.asarray(coord[index], dtype=np.float64)
            expected = start + np.arange(index.start, index.stop) * delta
            if not np.isfinite(values).all() or not np.allclose(values, expected, atol=2e-8, rtol=0):
                raise ValueError('ABI fixed-grid axes differ from the GOES-West nominal 2 km grid.')
            axes.append(values)
        coefficients = {}
        for field in PLANCK_FIELDS:
            variable = ds[field]
            if variable.ndim != 0:
                raise ValueError('Native ABI calibration coefficient is not scalar.')
            coefficients[field] = float(variable[...])
        radiance = rad[rows, columns]  # netCDF4 applies native packing scale and offset once.
        dqf = flags[rows, columns]
    radiance_data, dqf_data = np.ma.getdata(radiance), np.ma.getdata(dqf)
    mask = (~np.ma.getmaskarray(radiance) & ~np.ma.getmaskarray(dqf) & (dqf_data == 0)
            & np.isfinite(radiance_data) & (radiance_data > 0))
    brightness = np.zeros(radiance.shape, dtype=np.float32)
    brightness[mask] = _planck(radiance_data[mask].astype(np.float64), coefficients)
    if not mask.any() or not np.isfinite(brightness[mask]).all() or (brightness[mask] <= 0).any():
        raise ValueError('No finite positive native ABI brightness temperatures survived DQF.')
    return axes, brightness, mask, coefficients


def _report_geometry(crop, shape):
    return {'crop': crop, 'shape': list(shape), 'reference': GEOMETRY_REFERENCE}


def _assemble(sensor, axes, brightness, valid, frames, source_sha):
    lat, lon, view, area, footprint = _geometry(*axes)
    shape = lat.shape

    def repeated(array):
        return torch.tensor(np.broadcast_to(array, (len(frames), *shape)).copy(), dtype=torch.float32)

    def times(name):
        return torch.tensor([frame[name] for frame in frames], dtype=torch.float64)

    return Sequence(
        sensor.id, torch.as_tensor(np.stack(brightness)[:, None]), torch.as_tensor(np.stack(valid)[:, None]),
        times('scan_end'), times('available_at'), torch.tensor(lat, dtype=torch.float32),
        torch.tensor(lon, dtype=torch.float32), repeated(view), repeated(np.full(shape, np.nan)),
        repeated(footprint), torch.tensor(area, dtype=torch.float32), tuple(f['id'] for f in frames), (CHANNEL,),
        fingerprint({'projection': PROJECTION, 'x': axes[0].tolist(), 'y': axes[1].tolist()}), GEOMETRY_REFERENCE,
        source_sha, sensor.measurement_signature, 'real', scan_start_unix=times('scan_start'),
    ).validate(sensor)


def _decode(objects, rows, columns, root, *, verify_report):
    axes = None
    frames, brightness, valid, evidence = [], [], [], []
    for item in objects:
        expected = {'raw', 'receipt', 'planck_coefficients'} if verify_report else {'raw', 'receipt'}
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError('Unexpected fields in the native ABI frame description.')
        source = {key: item[key] for key in ('raw', 'receipt')}
        path, receipt, stamps, frame = _object(root, source)
        coordinates, values, mask, coefficients = _read(path, receipt, stamps, rows, columns)
        if axes is None:
            axes = coordinates
        if not all(np.array_equal(a, b) for a, b in zip(axes, coordinates)):
            raise ValueError('Native fixed-grid coordinates changed across ABI frames.')
        if verify_report and coefficients != item['planck_coefficients']:
            raise ValueError('ABI passport calibration differs from the native NetCDF.')
        frames.append(frame)
        brightness.append(values)
        valid.append(mask)
        evidence.append({**source, 'planck_coefficients': coefficients})
    return axes, brightness, valid, frames, evidence


def _compare_sequences(actual, native):
    for key in ('sensor_id', 'frame_ids', 'channel_ids', 'grid_id', 'geometry_reference', 'data_kind',
                'link_reference', 'link_grid_fingerprint'):
        if getattr(actual, key) != getattr(native, key):
            raise ValueError(f'ABI sequence {key} differs from native evidence.')
    if any(getattr(actual, key) is not None for key in ('link_pixel', 'link_cell', 'link_weight')):
        raise ValueError('ABI imager sequences cannot contain microwave link arrays.')
    for key in ('values', 'valid', 'observed_unix', 'available_unix', 'scan_start_unix', 'latitude',
                'longitude', 'view_zenith_deg', 'footprint_km', 'area_m2', 'solar_zenith_deg'):
        a, b = getattr(actual, key), getattr(native, key)
        if a is None or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f'ABI sequence {key} differs from native evidence.')
        equal = torch.allclose(a, b, rtol=0, atol=0, equal_nan=True) if a.is_floating_point() else torch.equal(a, b)
        if not equal:
            raise ValueError(f'ABI sequence {key} differs from native evidence.')


def validate_station_passport(report_path, sensor, expected_sha, *, verify_raw=False):
    """Check a prepared provenance report against local raw/evidence identities."""
    _sensor(sensor)
    path = regular(report_path)
    report = read_json(path)
    fixed = {'schema': 'station-goes-abi-sequence-1', 'status': 'sequence_prepared', 'sensor': sensor.id,
             'data_kind': 'real', 'meteorologically_validated': False, 'sha256': hash_value(expected_sha),
             'availability_basis': AVAILABILITY_BASIS, 'source': SOURCE, 'platform': PLATFORM,
             'channel': CHANNEL, 'calibration_id': CALIBRATION_ID, 'radiance_units': RADIANCE_UNITS,
             'quality': 'native DQF=0 (good_pixel_qf)'}
    if not isinstance(report, dict) or any(report.get(key) != val for key, val in fixed.items()):
        raise ValueError('Invalid physical NOAA preparation report.')
    rows, columns, shape = _crop(report['geometry']['crop'])
    if report['geometry'] != _report_geometry(report['geometry']['crop'], shape):
        raise ValueError('Invalid ABI geometry passport.')
    n = report['frames']
    if (type(n) is not int or not 1 <= n <= 48 or n * shape[0] * shape[1] > 8000000
            or report['shape'] != [n, 1, *shape] or len(report['objects']) != n):
        raise ValueError('Invalid ABI preparation report shape.')
    frames = []
    for item in report['objects']:
        if not isinstance(item, dict) or set(item) != {'raw', 'receipt', 'planck_coefficients'}:
            raise ValueError('Unexpected fields in the native ABI frame passport.')
        _, _, _, frame = _object(path.parent, {k: item[k] for k in ('raw', 'receipt')})
        if not isinstance(item['planck_coefficients'], dict) or set(item['planck_coefficients']) != set(PLANCK_FIELDS):
            raise ValueError('Incomplete native Planck coefficients in ABI passport.')
        _planck(np.array([50.]), item['planck_coefficients'])
        frames.append(frame)
    if (frames != report['frame_index'] or len(set(frame['id'] for frame in frames)) != n
            or any(a['scan_end'] >= b['scan_end'] for a, b in zip(frames, frames[1:]))):
        raise ValueError('ABI frame identities or chronology differ from verified native receipts.')
    if verify_raw:
        axes, brightness, valid, source_frames, _ = _decode(report['objects'], rows, columns, path.parent,
                                                           verify_report=True)
        native = _assemble(sensor, axes, brightness, valid, source_frames, expected_sha)
        actual = load_sequence(path.with_suffix('.npz'), sensor)
        if actual.source_sha256 != expected_sha or source_frames != frames:
            raise ValueError('ABI NPZ SHA256 or source receipts differ from the immutable passport.')
        _compare_sequences(actual, native)
    return report


def prepare(manifest_path, registry_path, sensor_id, output):
    manifest_path = regular(manifest_path)
    output = Path(output).absolute()
    report_path = output.with_suffix('.json')
    if output.parent != manifest_path.parent or output.suffix != '.npz':
        raise ValueError('Place the new ABI sequence NPZ next to the native manifest.')
    if output.exists() or output.is_symlink() or report_path.exists() or report_path.is_symlink():
        raise FileExistsError('The ABI sequence or report already exists.')
    digest = sha256(manifest_path)
    plan = read_json(manifest_path)
    if (not isinstance(plan, dict) or set(plan) != {'schema', 'crop', 'objects'}
            or plan['schema'] != 'station-goes-abi-native-1'):
        raise ValueError('Invalid native ABI preparation manifest.')
    sensor = next((s for s in load_sensors(registry_path) if s.id == sensor_id), None)
    if sensor is None:
        raise ValueError('Unknown ABI sensor id.')
    _sensor(sensor)
    rows, columns, shape = _crop(plan['crop'])
    objects = plan['objects']
    if not isinstance(objects, list) or not 1 <= len(objects) <= 48 or len(objects) * shape[0] * shape[1] > 8000000:
        raise ValueError('ABI sequence exceeds the allowed number of data values.')
    axes, brightness, valid, frames, evidence = _decode(objects, rows, columns, manifest_path.parent,
                                                        verify_report=False)
    seq = _assemble(sensor, axes, brightness, valid, frames, digest)
    if sha256(manifest_path) != digest:
        raise ValueError('ABI native preparation manifest changed during processing.')
    save_sequence(output, seq)
    report = {'schema': 'station-goes-abi-sequence-1', 'status': 'sequence_prepared', 'sensor': sensor.id,
              'data_kind': 'real', 'meteorologically_validated': False, 'sha256': sha256(output), 'frames': len(frames),
              'shape': list(seq.values.shape), 'frame_index': frames, 'objects': evidence, 'source': SOURCE,
              'platform': PLATFORM, 'channel': CHANNEL, 'calibration_id': CALIBRATION_ID, 'radiance_units': RADIANCE_UNITS,
              'quality': 'native DQF=0 (good_pixel_qf)', 'geometry': _report_geometry(plan['crop'], shape),
              'availability_basis': AVAILABILITY_BASIS, 'archive_receipt_limitation': 'local operational latency unknown'}
    write_json(report_path, report)
    validate_station_passport(report_path, sensor, report['sha256'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('manifest', 'registry', 'sensor', 'output'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args(argv)
    print(prepare(args.manifest, args.registry, args.sensor, args.output))


if __name__ == '__main__':
    main()
