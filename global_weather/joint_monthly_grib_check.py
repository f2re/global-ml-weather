"""Independent full-month ecCodes decoding gate; no weather skill claim."""
from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

# International WMO codes, independent of the exporter inventory.
FIELDS = {(0, 0): ('temperature', 'K'), (1, 0): ('specific_humidity', 'kg kg-1'),
          (2, 2): ('u', 'm s-1'), (2, 3): ('v', 'm s-1'),
          (3, 4): ('geopotential', 'm2 s-2'), (2, 8): ('omega', 'Pa s-1')}
PRESSURES = (1000,975,950,925,900,875,850,825,800,775,750,700,650,600,550,
             500,450,400,350,300,250,225,200,175,150,125,100,70,50,30,20,
             10,7,5,3,2,1)
LEADS = tuple(range(0, 73, 3))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def expected_dates(month, hour):
    year, number = map(int, month.split('-'))
    return [datetime(year, number, day, hour, tzinfo=timezone.utc)
            for day in range(1, calendar.monthrange(year, number)[1]+1)]


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError('Issue timestamps must explicitly use UTC.')
    return parsed


def local_file(path):
    path = Path(path)
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents) or not path.is_file():
        raise ValueError('GRIB evidence requires a regular file without symlink parents.')
    return path


def unit_key(unit):
    return ''.join(unit.split()).replace('**', '').replace('^', '')


def decode_issue(ec, path, issue, spacing):
    if not math.isfinite(spacing) or not .25 <= spacing <= 90:
        raise ValueError('Invalid regular global product spacing.')
    ni, nj = round(360/spacing), round(180/spacing)
    if not math.isclose(ni*spacing, 360) or not math.isclose(nj*spacing, 180):
        raise ValueError('Invalid regular global product spacing.')
    expected = {(category, number, 100, pressure*100, lead)
                for category, number in FIELDS for pressure in PRESSURES for lead in LEADS}
    expected |= {(3, 1, 101, None, lead) for lead in LEADS}
    seen = set()
    counts = {name: {'messages': 0, 'valid_points': 0, 'missing_points': 0} for name, _ in FIELDS.values()}
    counts['mslp'] = {'messages': 0, 'valid_points': 0, 'missing_points': 0}
    get = ec.codes_get_long
    with path.open('rb') as stream:
        while (handle := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                if get(handle, 'edition') != 2 or get(handle, 'discipline') != 0:
                    raise ValueError('GRIB edition or discipline differs.')
                if get(handle, 'productDefinitionTemplateNumber') != 0 or get(handle, 'indicatorOfUnitOfTimeRange') != 1:
                    raise ValueError('GRIB must use instantaneous forecast template and hours.')
                reference = datetime(*(get(handle, key) for key in ('year','month','day','hour','minute','second')), tzinfo=timezone.utc)
                if reference != issue or get(handle, 'significanceOfReferenceTime') != 1:
                    raise ValueError('GRIB reference timestamp differs from monthly issue.')
                lead = get(handle, 'forecastTime')
                valid = issue + timedelta(hours=lead)
                if get(handle, 'validityDate') != int(valid.strftime('%Y%m%d')) or get(handle, 'validityTime') != int(valid.strftime('%H%M')):
                    raise ValueError('GRIB validity timestamp differs.')
                category, number = get(handle, 'parameterCategory'), get(handle, 'parameterNumber')
                surface = get(handle, 'typeOfFirstFixedSurface')
                if surface == 100:
                    pressure = get(handle, 'scaledValueOfFirstFixedSurface')*10.**(-get(handle, 'scaleFactorOfFirstFixedSurface'))
                    name, units = FIELDS.get((category, number), (None, None))
                elif surface == 101 and (category, number) == (3, 1):
                    pressure, name, units = None, 'mslp', 'Pa'
                else:
                    raise ValueError('Unexpected WMO parameter or fixed surface.')
                identity = (category, number, surface, pressure, lead)
                if identity not in expected or identity in seen:
                    raise ValueError('Unexpected or duplicate GRIB message inventory.')
                if unit_key(ec.codes_get(handle, 'units')) != unit_key(units):
                    raise ValueError('Decoded GRIB physical units differ.')
                if (get(handle, 'gridDefinitionTemplateNumber') != 0 or ec.codes_get(handle, 'gridType') != 'regular_ll'
                        or get(handle, 'Ni') != ni or get(handle, 'Nj') != nj
                        or get(handle, 'numberOfPoints') != ni*nj):
                    raise ValueError('GRIB regular global grid dimensions differ.')
                for key, value in (('latitudeOfFirstGridPointInDegrees', 90-spacing/2),
                                   ('latitudeOfLastGridPointInDegrees', -90+spacing/2),
                                   ('longitudeOfFirstGridPointInDegrees', 0),
                                   ('longitudeOfLastGridPointInDegrees', 360-spacing),
                                   ('iDirectionIncrementInDegrees', spacing), ('jDirectionIncrementInDegrees', spacing)):
                    if not math.isclose(ec.codes_get_double(handle, key), value, abs_tol=1e-6):
                        raise ValueError('GRIB grid geometry differs.')
                if any(get(handle, key) != 0 for key in ('iScansNegatively','jScansPositively','jPointsAreConsecutive','alternativeRowScanning','uvRelativeToGrid')):
                    raise ValueError('GRIB scan direction or wind basis differs.')
                if get(handle, 'bitmapPresent') != 1:
                    raise ValueError('Explicit independent GRIB bitmap is required.')
                bitmap = np.asarray(ec.codes_get_array(handle, 'bitmap'))
                values = np.asarray(ec.codes_get_values(handle))
                if bitmap.shape != (ni*nj,) or values.shape != bitmap.shape or not np.isin(bitmap, [0, 1]).all():
                    raise ValueError('Decoded bitmap or data shape differs.')
                missing = int((bitmap == 0).sum())
                if get(handle, 'numberOfMissing') != missing or not np.isfinite(values[bitmap == 1]).all():
                    raise ValueError('Decoded valid values or missing count differs.')
                counts[name]['messages'] += 1
                counts[name]['valid_points'] += int((bitmap == 1).sum())
                counts[name]['missing_points'] += missing
                seen.add(identity)
            finally:
                ec.codes_release(handle)
    if seen != expected:
        raise ValueError('Incomplete GRIB message inventory.')
    return {'issue_time': issue.isoformat(), 'messages': len(seen), 'fields': counts, 'Ni': ni, 'Nj': nj}


def check_month(forecasts, output):
    """Decode every message; bind before/after hashes and publish a new report."""
    root, output = Path(forecasts), Path(output)
    if output.exists() or output.is_symlink() or any(parent.is_symlink() for parent in output.parents):
        raise FileExistsError('Choose a new immutable check report path.')
    manifest_path = local_file(root/'month.json')
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get('schema') != 'joint-monthly-verification-1' or manifest.get('stage') != 'forecast'
            or manifest.get('status') != 'complete' or manifest.get('future_targets_read') is not False
            or manifest.get('horizon_hours') != 72 or manifest.get('lead_hours') != list(LEADS)):
        raise ValueError('A complete causal 72-hour monthly forecast is required.')
    dates = expected_dates(manifest['month'], manifest['issue_hour'])
    products = manifest.get('products')
    if not isinstance(products, list) or [timestamp(row['issue_time']) for row in products] != dates:
        raise ValueError('Forecast products do not cover the complete ordered month.')
    bound = []
    for descriptor in products:
        directory = descriptor.get('directory')
        if not isinstance(directory, str) or Path(directory).name != directory or directory in ('.', '..'):
            raise ValueError('Invalid forecast descriptor directory.')
        issue_path = local_file(root/directory/'issue.json')
        if json.loads(issue_path.read_text()) != descriptor:
            raise ValueError('Issue descriptor differs from monthly manifest.')
        path = local_file(root/directory/'product'/'forecast.grib2')
        if digest(path) != descriptor.get('grib_sha256'):
            raise ValueError('Forecast GRIB SHA256 differs from descriptor.')
        bound.append((path, descriptor['grib_sha256'], issue_path, digest(issue_path)))
    # Import only after the full-month and provenance gates have succeeded.
    import eccodes as ec
    spacing = (manifest.get('config') or {}).get('spacing_degrees', 5.)
    started = datetime.now(timezone.utc).isoformat()
    days = []
    for issue, (path, expected_hash, descriptor_path, descriptor_hash) in zip(dates, bound):
        result = decode_issue(ec, path, issue, spacing)
        if digest(path) != expected_hash or digest(descriptor_path) != descriptor_hash:
            raise ValueError('GRIB or issue descriptor changed during decoding.')
        result.update(grib_sha256=expected_hash, descriptor_sha256=descriptor_hash)
        days.append(result)
    if digest(manifest_path) != manifest_hash:
        raise ValueError('Monthly manifest changed during decoding.')
    for path, expected_hash, descriptor_path, descriptor_hash in bound:
        if digest(path) != expected_hash or digest(descriptor_path) != descriptor_hash:
            raise ValueError('Monthly artifact changed before report publication.')
    report = {'schema': 'joint-monthly-independent-grib-check-1', 'status': 'complete',
              'month': manifest['month'], 'forecast_manifest_sha256': manifest_hash,
              'forecast_manifest_path': str(manifest_path.resolve()),
              'checkpoint_sha256': manifest.get('checkpoint_sha256'), 'source_revision': manifest.get('source_revision'),
              'decoder_version': ec.codes_get_api_version(), 'checked_issues': len(days),
              'checked_messages': sum(day['messages'] for day in days), 'days': days,
              'started_utc': started, 'finished_utc': datetime.now(timezone.utc).isoformat(),
              'scientific_acceptance': False, 'weather_accuracy_verified': False,
              'scope': 'all-message independent structural decoding, WMO metadata, finite supported values and hash binding'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forecasts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    report = check_month(args.forecasts, args.output)
    print(json.dumps({'status': report['status'], 'checked_messages': report['checked_messages']}))


if __name__ == '__main__':
    main()
