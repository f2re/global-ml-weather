from datetime import datetime, timezone
from email.utils import format_datetime
import hashlib
import json

from netCDF4 import Dataset
import numpy as np
import pytest

from global_weather.goes_abi import (
    BUCKET, CALIBRATION_ID, GEOMETRY_REFERENCE, PLANCK_FIELDS, PROJECTION, RADIANCE_UNITS,
    _geometry, _key_times, _planck, prepare, validate_station_passport,
)
from global_weather.multimodal.contracts import Channel, Sensor
from global_weather.multimodal.io import load_sequence, save_sensors, sha256


OBJECT = ('ABI-L1b-RadC/2024/265/12/'
          'OR_ABI-L1b-RadC-M6C13_G18_s20242651201177_e20242651203562_c20242651204053.nc')
CROP = [500, 508, 1900, 1908]
COEFFICIENTS = dict(zip(PLANCK_FIELDS, (10803.3, 1392.74, 0.0755, 0.99975)))


def _ref(path):
    return {'path': path.name, 'sha256': sha256(path)}


def _setup(tmp_path, *, band=13):
    sensor = Sensor('goes18_c13', 'goes_abi', 'GOES-18', 'imager',
                    (Channel('13', 'brightness_temperature', 'K', None, None, CALIBRATION_ID),),
                    None, None, 'real')
    registry = tmp_path / 'registry.json'
    save_sensors(registry, (sensor,))
    native = tmp_path / 'native.nc'
    with Dataset(native, 'w', format='NETCDF4') as ds:
        ds.createDimension('x', 2500)
        ds.createDimension('y', 1500)
        ds.setncatts(dict(dataset_name=OBJECT.rsplit('/', 1)[1], platform_ID='G18', scene_id='CONUS', instrument_ID='FM3',
                         production_environment='OE', production_data_source='Realtime', timeline_id='ABI Mode 6',
                         time_coverage_start='2024-09-21T12:01:17.7Z', time_coverage_end='2024-09-21T12:03:56.2Z',
                         date_created='2024-09-21T12:04:05.3Z'))
        ds.createVariable('band_id', 'i2').assignValue(band)
        x = ds.createVariable('x', 'f8', ('x',))
        y = ds.createVariable('y', 'f8', ('y',))
        x.units = y.units = 'rad'
        x[:] = -0.069972 + np.arange(2500) * 0.000056
        y[:] = 0.128212 - np.arange(1500) * 0.000056
        var = ds.createVariable('goes_imager_projection', 'i2')
        var.setncatts(dict(zip(('semi_major_axis', 'semi_minor_axis', 'perspective_point_height',
                               'longitude_of_projection_origin'), PROJECTION)))
        var.setncatts(dict(grid_mapping_name='geostationary', latitude_of_projection_origin=0.0,
                          sweep_angle_axis='x'))
        for key, value in COEFFICIENTS.items():
            ds.createVariable(key, 'f8').assignValue(value)
        rad = ds.createVariable('Rad', 'i2', ('y', 'x'), fill_value=4095, zlib=True, chunksizes=(250, 250))
        rad.setncatts(dict(units=RADIANCE_UNITS, scale_factor=0.05, add_offset=-1.0,
                          grid_mapping='goes_imager_projection'))
        rad.set_auto_maskandscale(False)
        dqf = ds.createVariable('DQF', 'i1', ('y', 'x'), fill_value=-1, zlib=True, chunksizes=(250, 250))
        dqf.setncatts(dict(flag_values=np.arange(5, dtype=np.int8), flag_meanings=(
            'good_pixel_qf conditionally_usable_pixel_qf out_of_range_pixel_qf no_value_pixel_qf '
            'focal_plane_temperature_threshold_exceeded_qf')))
        raw = np.full((8, 8), 1000, dtype=np.int16)
        raw[0, 1] = 4095
        flags = np.zeros((8, 8), dtype=np.int8)
        flags[0, 0] = 1
        rad[CROP[0]:CROP[1], CROP[2]:CROP[3]] = raw
        dqf[CROP[0]:CROP[1], CROP[2]:CROP[3]] = flags

    written = datetime(2024, 9, 21, 12, 4, 25, tzinfo=timezone.utc)
    receipt = tmp_path / 'receipt.json'
    receipt.write_text(json.dumps(dict(schema='noaa-nodd-s3-object-1', key=OBJECT, url=BUCKET + OBJECT,
                                        content_length=native.stat().st_size, sha256=sha256(native),
                                        etag=hashlib.md5(native.read_bytes(), usedforsecurity=False).hexdigest(),
                                        archive_object_last_modified=written.isoformat(),
                                        http_last_modified=format_datetime(written, usegmt=True),
                                        observed_at='2026-11-22T12:00:00Z')))
    manifest = tmp_path / 'native.json'
    manifest.write_text(json.dumps(dict(schema='station-goes-abi-native-1', crop=CROP,
                                         objects=[dict(raw=_ref(native), receipt=_ref(receipt))])))
    return sensor, registry, native, receipt, manifest


def test_prepare_netCDF_with_native_packing_DQF_and_confirmed_archive_clock(tmp_path):
    sensor, registry, native, _, manifest = _setup(tmp_path)
    output = tmp_path / 'sequence.npz'
    report = prepare(manifest, registry, sensor.id, output)
    assert validate_station_passport(output.with_suffix('.json'), sensor, sha256(output), verify_raw=True) == report
    seq = load_sequence(output, sensor)
    assert seq.values.shape == (1, 1, 8, 8)
    assert seq.valid.sum() == 62
    assert not seq.valid[0, 0, 0, :2].any()
    assert float(seq.values[0, 0, 0, 2]) == pytest.approx(_planck(np.array([49.]), COEFFICIENTS)[0])
    assert seq.geometry_reference == GEOMETRY_REFERENCE
    assert seq.scan_start_unix[0].item() == _key_times(OBJECT)[0].timestamp()
    assert seq.causal('2024-09-21T12:04:24Z', sensor) is None
    assert seq.causal('2024-09-22T00:00:00Z', sensor) is not None
    with pytest.raises(FileExistsError):
        prepare(manifest, registry, sensor.id, output)

    before = native.read_bytes()
    native.write_bytes(bytes([before[0] ^ 1]) + before[1:])
    with pytest.raises(ValueError, match='SHA256'):
        validate_station_passport(output.with_suffix('.json'), sensor, sha256(output), verify_raw=True)


def test_reject_substituted_band_and_receipt_origin(tmp_path):
    sensor, registry, _, receipt, manifest = _setup(tmp_path, band=14)
    with pytest.raises(ValueError, match='product identity'):
        prepare(manifest, registry, sensor.id, tmp_path / 'wrong.npz')
    evidence = json.loads(receipt.read_text())
    evidence['url'] = 'https://example.com/' + OBJECT
    receipt.write_text(json.dumps(evidence))
    data = json.loads(manifest.read_text())
    data['objects'][0]['receipt'] = _ref(receipt)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='receipt disagrees'):
        prepare(manifest, registry, sensor.id, tmp_path / 'wrong.npz')


def test_rehashing_modified_abi_pixels_does_not_replace_native_validation(tmp_path):
    sensor, registry, _, _, manifest = _setup(tmp_path)
    output = tmp_path / 'sequence.npz'
    report = prepare(manifest, registry, sensor.id, output)
    with np.load(output, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['values'][0, 0, 1, 1] += 2.
    np.savez_compressed(output, **arrays)
    report['sha256'] = sha256(output)
    path = output.with_suffix('.json')
    path.write_text(json.dumps(report))
    assert validate_station_passport(path, sensor, report['sha256'], verify_raw=False) == report
    with pytest.raises(ValueError, match='values differs from native evidence'):
        validate_station_passport(path, sensor, report['sha256'], verify_raw=True)


def test_fixed_grid_nadir_geometry_and_positive_pixel_areas():
    scan = np.array([-0.000056, 0, 0.000056])
    lat, lon, zenith, area, footprint = _geometry(scan, scan)
    assert (lat[1, 1], lon[1, 1], zenith[1, 1]) == pytest.approx((0, -137, 0))
    assert np.all(np.diff(lon, axis=1) > 0)
    assert np.all(np.diff(lat, axis=0) > 0)
    assert np.all(area > 0)
    assert footprint[1, 1] == pytest.approx(2, rel=0.01)
