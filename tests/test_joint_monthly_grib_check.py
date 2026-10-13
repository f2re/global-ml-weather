"""Independent decoding checks using deterministic synthetic ecCodes messages."""
from datetime import datetime, timezone
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from global_weather import joint_monthly_grib_check as check


def monthly_fixture(tmp_path, monkeypatch):
    root = tmp_path/'forecasts'
    product = root/'20220901T060000Z'/'product'
    product.mkdir(parents=True)
    grib = product/'forecast.grib2'
    grib.write_bytes(b'synthetic fixture; decoder mocked')
    issue = datetime(2022, 9, 1, 6, tzinfo=timezone.utc)
    descriptor = {'directory': '20220901T060000Z', 'issue_time': issue.isoformat(), 'grib_sha256': check.digest(grib)}
    (product/'forecast.json').write_text(json.dumps({'issue_time': issue.isoformat()}))
    inventory = {'grid': {'spacing_degrees': 90}, 'grib_sha256': descriptor['grib_sha256'],
                 'forecast_metadata_sha256': check.digest(product/'forecast.json')}
    (product/'forecast.grib2.json').write_text(json.dumps(inventory))
    (product/'completion.json').write_text(json.dumps({'grib_sha256': descriptor['grib_sha256'],
        'grib_inventory_sha256': check.digest(product/'forecast.grib2.json')}))
    (product.parent/'issue.json').write_text(json.dumps(descriptor))
    manifest = {'schema': 'joint-monthly-verification-1', 'stage': 'forecast', 'status': 'complete',
                'future_targets_read': False, 'horizon_hours': 72, 'lead_hours': list(check.LEADS),
                'month': '2022-09', 'issue_hour': 6, 'products': [descriptor],
                'config': {'spacing_degrees': 5}}
    (root/'month.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(check, 'expected_dates', lambda month, hour: [issue])
    return root, grib, manifest


def fake_decoder(monkeypatch, *, duplicate=False, nonfinite=False, corrupt_key=None):
    messages = []
    for category, number in check.FIELDS:
        for pressure in check.PRESSURES:
            for lead in check.LEADS:
                messages.append((category, number, 100, pressure*100, lead))
    messages.extend((3, 1, 101, None, lead) for lead in check.LEADS)
    if duplicate:
        messages[-1] = messages[0]
    iterator = iter(messages)
    def get(handle, key):
        category, number, surface, pressure, lead = handle
        from datetime import timedelta
        valid = datetime(2022, 9, 1, 6) + timedelta(hours=lead)
        keys = {'edition': 2, 'discipline': 0, 'productDefinitionTemplateNumber': 0,
                'centre': 65535, 'subCentre': 0, 'tablesVersion': 4, 'localTablesVersion': 0,
                'productionStatusOfProcessedData': 2, 'typeOfGeneratingProcess': 2,
                'typeOfSecondFixedSurface': 255, 'bitMapIndicator': 0,
                'indicatorOfUnitOfTimeRange': 1, 'year': 2022, 'month': 9, 'day': 1,
                'hour': 6, 'minute': 0, 'second': 0, 'significanceOfReferenceTime': 1,
                'forecastTime': lead, 'validityDate': int(valid.strftime('%Y%m%d')),
                'validityTime': int(valid.strftime('%H%M')), 'parameterCategory': category,
                'parameterNumber': number, 'typeOfFirstFixedSurface': surface,
                'scaledValueOfFirstFixedSurface': pressure, 'scaleFactorOfFirstFixedSurface': 0,
                'gridDefinitionTemplateNumber': 0, 'gridType': 'regular_ll', 'Ni': 4, 'Nj': 2,
                'numberOfPoints': 8, 'latitudeOfFirstGridPointInDegrees': 45.,
                'latitudeOfLastGridPointInDegrees': -45., 'longitudeOfFirstGridPointInDegrees': 0.,
                'longitudeOfLastGridPointInDegrees': 270., 'iDirectionIncrementInDegrees': 90.,
                'jDirectionIncrementInDegrees': 90., 'bitmapPresent': 1, 'numberOfMissing': 1,
                'units': 'Pa' if surface == 101 else check.FIELDS[(category, number)][1]}
        keys.update({name: 0 for name in ('iScansNegatively','jScansPositively','jPointsAreConsecutive','alternativeRowScanning','uvRelativeToGrid')})
        if corrupt_key:
            keys[corrupt_key] = 254 if corrupt_key == 'bitMapIndicator' else 98
        return keys[key]
    def values(handle):
        # Missing values can be NaN; only bitmap-supported values must be finite.
        result = np.array([np.nan, 1, 2, 3, 4, 5, 6, 7.])
        if nonfinite:
            result[1] = np.nan
        return result
    ec = SimpleNamespace(codes_grib_new_from_file=lambda stream: next(iterator, None),
                         codes_get_long=get, codes_get=get, codes_get_double=get,
                         codes_get_array=lambda handle, key: np.array([0, 1, 1, 1, 1, 1, 1, 1]),
                         codes_get_values=values, codes_release=lambda handle: None,
                         codes_get_api_version=lambda: 'synthetic-decoder-fixture')
    monkeypatch.setitem(sys.modules, 'eccodes', ec)


def test_complete_inventory_and_bitmap_supported_values(tmp_path, monkeypatch):
    root, _, _ = monthly_fixture(tmp_path, monkeypatch)
    fake_decoder(monkeypatch)
    result = check.check_month(root, tmp_path/'check.json')
    assert result['checked_messages'] == 5575
    assert result['checked_issues'] == 1
    assert result['days'][0]['fields']['temperature'] == {'messages': 925, 'valid_points': 6475, 'missing_points': 925}
    assert result['days'][0]['fields']['mslp']['messages'] == 25
    assert result['days'][0]['Ni'] == 4  # Bound per-product spacing overrides stale monthly API defaults.
    assert result['weather_accuracy_verified'] is False
    with pytest.raises(FileExistsError):
        check.check_month(root, tmp_path/'check.json')


def test_tampered_sha_and_incomplete_month_rejected_before_decode(tmp_path, monkeypatch):
    root, grib, manifest = monthly_fixture(tmp_path, monkeypatch)
    grib.write_bytes(b'tampered bytes')
    with pytest.raises(ValueError, match='SHA256'):
        check.check_month(root, tmp_path/'bad.json')
    manifest['products'] = []
    (root/'month.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete ordered month'):
        check.check_month(root, tmp_path/'bad.json')
    manifest['status'] = 'running'
    (root/'month.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='complete causal'):
        check.check_month(root, tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


@pytest.mark.parametrize('key', ['centre', 'subCentre', 'tablesVersion', 'localTablesVersion',
    'productionStatusOfProcessedData', 'typeOfGeneratingProcess', 'typeOfSecondFixedSurface', 'bitMapIndicator'])
def test_research_provenance_and_bitmap_reuse_corruption(tmp_path, monkeypatch, key):
    root, _, _ = monthly_fixture(tmp_path, monkeypatch)
    fake_decoder(monkeypatch, corrupt_key=key)
    with pytest.raises(ValueError, match='provenance key|independent GRIB bitmap'):
        check.check_month(root, tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()


def test_product_grid_inventory_hash_tampering(tmp_path, monkeypatch):
    root, grib, _ = monthly_fixture(tmp_path, monkeypatch)
    path = grib.with_suffix('.grib2.json')
    inventory = json.loads(path.read_text())
    inventory['grid']['spacing_degrees'] = 10
    path.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match='inventory provenance SHA256'):
        check.check_month(root, tmp_path/'bad.json')


@pytest.mark.parametrize('failure', ['duplicate', 'nonfinite'])
def test_decoder_rejects_corrupt_inventory_or_valid_values(tmp_path, monkeypatch, failure):
    root, _, _ = monthly_fixture(tmp_path, monkeypatch)
    fake_decoder(monkeypatch, **{failure: True})
    with pytest.raises(ValueError, match='duplicate' if failure == 'duplicate' else 'valid values'):
        check.check_month(root, tmp_path/'bad.json')
    assert not (tmp_path/'bad.json').exists()
