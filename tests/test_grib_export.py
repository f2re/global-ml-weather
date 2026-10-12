"""GRIB2 structural round trips; synthetic fields do not verify weather skill."""
from datetime import datetime, timezone
import hashlib
import json

import numpy as np
import pytest

from global_weather.grib_export import (MISSING_CENTRE, export_forecast, regular_grid,
                                        validate_forecast, wind_rotation)
from global_weather.vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS


def forecast_fixture(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    lat, lon, xyz = regular_grid(90)
    shape = (2, len(xyz), 37, 6)
    values = np.zeros(shape)
    values[..., 0] = 280
    values[..., 1] = .003
    values[..., 2] = 12
    values[..., 3] = -3
    values[..., 4] = 90000
    values[..., 5] = -.2
    mask = np.ones(shape, dtype=bool)
    mask[0, 0, 0, 0] = False
    values[0, 0, 0, 0] = np.nan
    mask[..., 5] = False
    values[..., 5] = np.nan
    arrays = {'profiles': values, 'profile_variable_mask': mask,
              'pressure_pa': np.array(PRESSURE_HPA)*100, 'xyz': xyz}
    np.savez_compressed(source / 'forecast.npz', **arrays)
    metadata = {'schema': 'measured-profile-frozen-forecast-1',
                'issue_time': '2026-10-12T06:15:30+00:00', 'lead_hours': [0, 72],
                'valid_times': ['2026-10-12T06:15:30+00:00', '2026-10-15T06:15:30+00:00'],
                'profile_variables': list(PROFILE_VARIABLES), 'profile_units': list(PROFILE_UNITS),
                'wind_basis': 'local_enu_vector',
                'artifact_sha256': hashlib.sha256((source / 'forecast.npz').read_bytes()).hexdigest()}
    (source / 'forecast.json').write_text(json.dumps(metadata))
    return source, arrays, metadata


def test_regular_grid_and_transport():
    lat, lon, xyz = regular_grid(90)
    assert list(lat) == [45, -45]
    assert list(lon) == [0, 90, 180, 270]
    rotation = wind_rotation(xyz, xyz)
    np.testing.assert_allclose(rotation[0], 1)
    np.testing.assert_allclose(rotation[1], 0, atol=1e-15)
    np.testing.assert_allclose(rotation[2], 0, atol=1e-15)
    np.testing.assert_allclose(rotation[3], 1)
    # Along the equator parallel transport preserves east/north components.
    rotation = wind_rotation(np.array([[1., 0, 0]]), np.array([[0., 1, 0]]))
    np.testing.assert_allclose(rotation, [[1], [0], [0], [1]], atol=1e-15)
    with pytest.raises(ValueError, match='Antipodal'):
        wind_rotation(np.array([[1., 0, 0]]), np.array([[-1., 0, 0]]))
    with pytest.raises(ValueError, match='divide'):
        regular_grid(7)


def test_metadata_rejects_units_times_and_valid_nan(tmp_path):
    _, arrays, metadata = forecast_fixture(tmp_path)
    origin, leads = validate_forecast(arrays, metadata)
    assert origin == datetime(2026, 10, 12, 6, 15, 30, tzinfo=timezone.utc)
    metadata['valid_times'][1] = '2026-10-15T07:15:30+00:00'
    with pytest.raises(ValueError, match='Valid times'):
        validate_forecast(arrays, metadata)
    metadata['valid_times'][1] = '2026-10-15T06:15:30+00:00'
    arrays['profile_variable_mask'][0, 0, 0, 0] = True
    with pytest.raises(ValueError, match='nonfinite'):
        validate_forecast(arrays, metadata)


def test_grib2_roundtrip_and_no_overwrite(tmp_path):
    ec = pytest.importorskip('eccodes', reason='Optional GRIB integration requires grib extra; required in GRIB acceptance')
    source, _, _ = forecast_fixture(tmp_path)
    output = tmp_path / 'forecast.grib2'
    report = export_forecast(source, output, spacing=90)
    assert len(report['messages']) == 2*37*6
    assert report['scientific_acceptance'] is False
    assert report['grib_sha256'] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert report['messages'][5]['valid_points'] == 0
    count = 0
    with output.open('rb') as stream:
        while (message := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                expected = report['messages'][count]
                assert ec.codes_get(message, 'edition') == 2
                assert ec.codes_get(message, 'centre') == MISSING_CENTRE
                assert ec.codes_get(message, 'productionStatusOfProcessedData') == 2
                assert ec.codes_get(message, 'scaledValueOfFirstFixedSurface') == expected['pressure_pa']
                assert ec.codes_get(message, 'parameterCategory') == expected['parameter_category']
                assert ec.codes_get(message, 'parameterNumber') == expected['parameter_number']
                assert ec.codes_get(message, 'forecastTime') == expected['lead_hours']
                assert ec.codes_get(message, 'second') == 30
                assert ec.codes_get(message, 'numberOfMissing') == 8 - expected['valid_points']
                assert ec.codes_get(message, 'gridType') == 'regular_ll'
                assert ec.codes_get(message, 'uvRelativeToGrid') == 0
                if count == 0:
                    values = ec.codes_get_values(message)
                    np.testing.assert_allclose(values[1:], 280)
                    np.testing.assert_allclose(ec.codes_get_array(message, 'latitudes'), [45]*4+[-45]*4)
                count += 1
            finally:
                ec.codes_release(message)
    assert count == len(report['messages'])
    with pytest.raises(FileExistsError):
        export_forecast(source, output, spacing=90)
    # Inputs remain immutable and the archive checksum is checked before coding.
    metadata = json.loads((source / 'forecast.json').read_text())
    metadata['artifact_sha256'] = '0'*64
    (source / 'forecast.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='SHA256'):
        export_forecast(source, tmp_path / 'second.grib2', spacing=90)


def test_unified_sparse_pressure_export(tmp_path):
    ec = pytest.importorskip('eccodes', reason='Required in GRIB acceptance environment')
    from global_weather.observation_data import NATIVE_VARIABLES, UNITS
    source, arrays, old = forecast_fixture(tmp_path)
    # First regular target point is at 45 N, 0 E; only it has 50 km support.
    profiles = arrays['profiles'][..., :5]
    lat = np.rad2deg(np.arcsin(arrays['xyz'][:, 2]))
    lon = np.rad2deg(np.arctan2(arrays['xyz'][:, 1], arrays['xyz'][:, 0]))
    np.savez_compressed(source / 'forecast.npz', profile=profiles,
                        profile_issue_mask=np.zeros(profiles.shape, dtype=bool),
                        grid_latlon=np.column_stack((lat, lon)),
                        pressure_hpa=np.array(PRESSURE_HPA), lead_hours=np.array([0, 72]),
                        surface=np.broadcast_to([280, 275, 2, 3, 95000, 101325], (2, 1, 6)),
                        surface_issue_mask=np.ones((2, 1, 6), dtype=bool), station_latlon=np.array([[45., 0.]]))
    metadata = {**old, 'schema': 'unified-research-forecast-1',
                'profile_variables': list(PROFILE_VARIABLES[:5]), 'profile_units': list(PROFILE_UNITS[:5]),
                'surface_variables': list(NATIVE_VARIABLES), 'surface_units': list(UNITS),
                'wind_basis': 'east_north_local',
                'forecast_npz_sha256': hashlib.sha256((source/'forecast.npz').read_bytes()).hexdigest()}
    metadata.pop('artifact_sha256')
    (source/'forecast.json').write_text(json.dumps(metadata))
    report = export_forecast(source, tmp_path/'unified.grib2', spacing=90)
    assert len(report['messages']) == 2*37*6+2
    assert all(item['valid_points'] == 0 for item in report['messages'][:-2])
    assert all(item['valid_points'] == 1 for item in report['messages'][-2:])
    assert report['messages'][-2]['variable'] == 'station_mean_sea_level_pressure'
    assert report['messages'][-1]['fixed_surface_type'] == 101
    decoded = []
    with (tmp_path/'unified.grib2').open('rb') as stream:
        while (message := ec.codes_grib_new_from_file(stream)) is not None:
            try:
                if ec.codes_get(message, 'typeOfFirstFixedSurface') in (1, 101):
                    decoded.append((ec.codes_get(message, 'parameterNumber'), ec.codes_get_values(message)[0],
                                    ec.codes_get(message, 'numberOfMissing')))
            finally:
                ec.codes_release(message)
    assert decoded == [(1, 101325, 7), (1, 101325, 7)]
