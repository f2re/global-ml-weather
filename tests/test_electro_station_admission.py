"""Synthetic contract fixtures for Elektro-L. No provider scene is represented."""
import json
import hashlib

import numpy as np
import pytest

from global_weather.connectors.ecosystem import STAC_TIMESTAMPS, electro_l_native
from global_weather.connectors.raster_bridge import read_capsule
from global_weather.multimodal.contracts import Channel, Sensor
from global_weather.multimodal.io import reference, save_sensors, write_json
from global_weather.multimodal.prepare import prepare_electro_capsule, sequence_from_capsules


def _native_fixture(root, minute, fault=None, window=None, *, archive=False):
    rasterio = pytest.importorskip('rasterio')
    root.mkdir()
    time = f'2026-11-01T10:{minute}:00Z'
    size = 12 if window is not None else 8
    profile = dict(driver='GTiff', width=size, height=size, count=1, crs='EPSG:4326',
                   transform=rasterio.transform.from_origin(20, 60, .1, .1))
    raster = root / 'channel.tif'
    counts = np.arange(size * size, dtype='uint16').reshape(size, size) + 2700
    with rasterio.open(raster, 'w', dtype='uint16', **profile) as out:
        out.write(counts, 1)
    quality = root / 'quality.tif'
    flags = np.zeros((size, size), dtype='uint8')
    flags[0, 1] = 2
    flags[2, 4] = 2
    with rasterio.open(quality, 'w', dtype='uint8', **profile) as out:
        out.write(flags, 1)
    band = {'unit': 'K', 'scale': .1, 'offset': -10.}
    if fault == 'missing_scale':
        del band['scale']
    asset = root / 'asset.json'
    scene_id = 'fictional-scene-' + minute
    uri = ('https://s3.gptl.ru/fixture/channel.tif' if archive else 'https://example.invalid/channel.tif')
    asset_id = hashlib.sha256(f'{scene_id}\n{uri}'.encode()).hexdigest()[:32]
    write_json(asset, dict(id=asset_id, item_id=scene_id,
               filename=raster.name, platform='EL3', time=time, time_assumed=fault == 'assumed_time',
               uri=uri, level='L2IR', category='channel',
               channel=9, raster_bands=[band]))
    producer = root / 'producer.json'
    producer_band = band | {'scale': .2} if fault == 'producer_scale' else band
    producer_properties = {'platform': 'EL3', 'datetime': time, 'processing:level': 'L2IR'}
    if fault == 'producer_timing':
        producer_properties['end_datetime'] = '2026-11-01T10:19:00Z'
    producer_assets = {'ch9': {'href': uri, 'raster:bands': [producer_band]}}
    if fault == 'producer_channel':
        producer_assets['ch9']['channel'] = 8
    quality_evidence = {'good_flag': 1 if fault == 'wrong_flag' else 0, 'reference': 'synthetic QC fixture'}
    if fault == 'producer_quality':
        quality_evidence['asset_key'] = 'qa'
        producer_assets['qa'] = {'roles': ['quality'], 'file:checksum': '0' * 64}
    item = {'type': 'Feature', 'id': scene_id, 'properties': producer_properties, 'assets': producer_assets}
    if archive:
        item['stac_extensions'] = [STAC_TIMESTAMPS]
        producer_properties.update(start_datetime='2026-11-01T10:05:00Z', end_datetime='2026-11-01T10:20:00Z',
                                   published='2026-11-02T00:01:00Z')
        producer_assets['ch9'].update(href=uri + '?versionId=raster-version',
                                      published='2026-11-01T10:22:00Z',
                                      **{'file:checksum': reference(root, raster)['sha256']})
        quality_evidence['asset_key'] = 'qa'
        quality_sha = reference(root, quality)['sha256']
        producer_assets['qa'] = {'roles': ['quality'], 'file:checksum': quality_sha,
                                'href': 'https://s3.gptl.ru/fixture/quality.tif?versionId=qc-version',
                                'published': '2026-11-01T10:27:00Z'}
        if fault == 'rolling_address':
            producer_assets['ch9']['href'] = uri
        if fault == 'missing_archive_checksum':
            del producer_assets['qa']['file:checksum']
        if fault == 'late_asset_ready':
            producer_assets['qa']['ready_at'] = '2026-11-01T10:29:00Z'
        if fault == 'changed_archive_asset':
            producer_assets['ch9']['updated'] = '2026-11-01T10:23:00Z'
        if fault == 'unpublished_archive_asset':
            producer_assets['qa']['unpublished'] = '2026-11-01T10:28:00Z'
        if fault == 'only_metadata_published':
            del producer_assets['ch9']['published']
        if fault == 'same_version':
            producer_assets['qa']['href'] = producer_assets['ch9']['href']
    write_json(producer, item)
    download = root / 'channel.tif.download.json'
    write_json(download, dict(asset_id=asset_id, size=raster.stat().st_size,
               sha256=reference(root, raster)['sha256'],
               time='2026-11-03T10:30:00Z' if archive else '2026-11-01T10:30:00Z'))
    geometry = root / 'geometry.json'
    write_json(geometry, dict(reference='synthetic geometry fixture', view_zenith_deg=20, footprint_km=4))
    files = {role: reference(root, path) for role, path in dict(
        producer_item=producer, asset=asset, raster=raster, download=download, quality=quality, geometry=geometry).items()}
    if fault == 'missing_quality':
        del files['quality']
    manifest = root / 'native.json'
    review = dict(schema='electro-l-native-l2ir-1', files=files,
        calibration={'revision': 'fixture-revision', 'reference': 'synthetic calibration fixture', 'asset_key': 'ch9'},
        quality=quality_evidence,
        timing={'scan_start': '2026-11-01T10:05:00Z', 'scan_end': '2026-11-01T10:20:00Z',
                'provider_ready_at': '2026-11-01T10:10:00Z' if fault == 'early_provider' else '2026-11-01T10:21:00Z',
                'reference': 'synthetic timing fixture'})
    if archive:
        review['timing']['availability_basis'] = 'provider_asset_publication'
    if window is not None:
        review['window'] = window
    write_json(manifest, review)
    return manifest


def _sequence_plan(root, capsule, metadata, native):
    manifest = reference(root, capsule / 'manifest.json')
    sensor = Sensor('electro-fixture', 'electro_l', 'EL3', 'imager', (
        Channel('9', 'brightness_temperature', 'K', None, None, metadata['calibration_reference']),
    ), None, None, 'real')
    registry = root / 'registry.json'
    save_sensors(registry, [sensor])
    identifiers = {key: metadata[key] for key in ('source', 'platform', 'native_scene_id')}
    identifiers['capsule_manifest_sha256'] = manifest['sha256']
    evidence = {}
    for kind, fields in (
        ('origin', ('upstream', 'instrument', 'channel_id', 'calibration_reference', 'calibration_status')),
        ('geometry', ('geometry_reference',)),
        ('quality', ('quality_support',)),
        ('timing', ('observed_at', 'scan_start', 'scan_end', 'available_at')),
    ):
        path = root / f'{kind}.json'
        report = identifiers | {key: metadata[key] for key in fields} | {'schema': f'multimodal-ir-{kind}-1'}
        if kind == 'origin':
            report['native_manifest'] = reference(root, native)
        write_json(path, report)
        evidence[kind] = reference(root, path)
    admission = root / 'admission.json'
    write_json(admission, {'schema': 'multimodal-ir-admission-1', 'capsules': {manifest['sha256']: {
        'scan_start': metadata['scan_start'], 'scan_end': metadata['scan_end'],
        'available_at': metadata['available_at'], 'evidence': evidence,
    }}})
    plan = root / 'plan.json'
    write_json(plan, {'schema': 'multimodal-capsule-plan-1', 'sensor': sensor.id, 'data_kind': 'real',
        'sensor_registry': reference(root, registry), 'declarations': {'real_admission': reference(root, admission)},
        'frames': [{'channels': {'9': {'manifest': manifest, 'pixels': reference(root, capsule / 'pixels.npz')}}}]})
    return plan, sensor


def test_electro_kelvin_qc_stable_calibration_and_conservative_ready_time(tmp_path):
    native = _native_fixture(tmp_path / 'native-one', '12')
    second = _native_fixture(tmp_path / 'native-two', '13')
    out = tmp_path / 'capsule'
    metadata = prepare_electro_capsule(native, out)
    second_spec, _, _, _ = electro_l_native(second)
    assert metadata['calibration_reference'] == second_spec['calibration_reference']
    assert metadata['native_asset_sha256'] != second_spec['native_asset_sha256']
    assert metadata['provider_ready_at'] == '2026-11-01T10:21:00Z'
    assert metadata['available_at'] == '2026-11-01T10:30:00Z'
    plan, sensor = _sequence_plan(tmp_path, out, metadata, native)
    seq = sequence_from_capsules(plan)
    assert seq.values[0, 0, 0, 0] == 260.
    assert seq.valid[0, 0, :2, :2].tolist() == [[True, False], [True, True]]
    assert seq.causal('2026-11-01T10:29:59Z', sensor) is None
    assert seq.causal('2026-11-01T10:30:00Z', sensor) is not None


def test_electro_archive_uses_the_last_exact_asset_publication(tmp_path):
    native = _native_fixture(tmp_path / 'historical', '12', archive=True)
    out = tmp_path / 'capsule'
    metadata = prepare_electro_capsule(native, out)
    assert metadata['local_downloaded_at'] == '2026-11-03T10:30:00Z'
    assert metadata['raster_published_at'] == '2026-11-01T10:22:00Z'
    assert metadata['available_at'] == metadata['quality_published_at'] == '2026-11-01T10:27:00Z'
    plan, sensor = _sequence_plan(tmp_path, out, metadata, native)
    seq = sequence_from_capsules(plan)
    assert seq.causal('2026-11-01T10:26:59Z', sensor) is None
    assert seq.causal('2026-11-01T10:27:00Z', sensor) is not None


@pytest.mark.parametrize('fault', [
    'rolling_address', 'missing_archive_checksum', 'late_asset_ready', 'changed_archive_asset',
    'unpublished_archive_asset', 'only_metadata_published', 'same_version',
])
def test_electro_archive_rejects_ambiguous_version_or_availability(tmp_path, fault):
    native = _native_fixture(tmp_path / 'archive', '12', fault, archive=True)
    with pytest.raises(ValueError):
        prepare_electro_capsule(native, tmp_path / 'capsule')


def test_electro_archive_binds_pixels_before_import(tmp_path, monkeypatch):
    from global_weather.connectors import raster_bridge

    native = _native_fixture(tmp_path / 'archive', '12', archive=True)
    import_raster = raster_bridge.import_raster

    def replace_quality_then_import(spec, out, **kwargs):
        with spec['quality'].open('ab') as changed:
            changed.write(b'x')
        return import_raster(spec, out, **kwargs)

    monkeypatch.setattr(raster_bridge, 'import_raster', replace_quality_then_import)
    out = tmp_path / 'capsule'
    with pytest.raises(ValueError, match='published archive version'):
        prepare_electro_capsule(native, out)
    assert not out.exists()


@pytest.mark.parametrize('fault', [
    'missing_scale', 'assumed_time', 'missing_quality', 'wrong_flag', 'early_provider', 'producer_scale',
    'producer_channel', 'producer_timing', 'producer_quality',
])
def test_electro_blocks_unproven_physical_inputs(tmp_path, fault):
    native = _native_fixture(tmp_path / 'native', '12', fault)
    with pytest.raises(ValueError):
        prepare_electro_capsule(native, tmp_path / 'capsule')


def test_electro_separates_metadata_validation_from_pixel_reads(tmp_path):
    native = _native_fixture(tmp_path / 'native', '12')
    out = tmp_path / 'capsule'
    metadata = prepare_electro_capsule(native, out)
    plan, _ = _sequence_plan(tmp_path, out, metadata, native)
    quality = native.parent / 'quality.tif'
    content = quality.read_bytes()
    quality.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    sequence_from_capsules(plan, verify_native=False)
    with pytest.raises(ValueError, match='SHA256 нативного файла'):
        sequence_from_capsules(plan)


def test_electro_rejects_self_consistent_fabricated_capsule(tmp_path):
    native = _native_fixture(tmp_path / 'native', '12')
    out = tmp_path / 'capsule'
    prepare_electro_capsule(native, out)
    metadata, arrays = read_capsule(out)
    arrays['values'][0, 0] += 4
    np.savez_compressed(out / 'pixels.npz', **arrays)
    metadata['arrays_sha256'] = reference(out, out / 'pixels.npz')['sha256']
    (out / 'manifest.json').write_text(json.dumps(metadata), encoding='utf-8')
    plan, _ = _sequence_plan(tmp_path, out, metadata, native)
    with pytest.raises(ValueError, match='не воспроизводятся'):
        sequence_from_capsules(plan)


def test_electro_applies_the_same_native_window_to_pixels_flags_and_coordinates(tmp_path):
    window = dict(row_offset=1, column_offset=2, height=8, width=8)
    native = _native_fixture(tmp_path / 'native', '12', window=window)
    out = tmp_path / 'capsule'
    metadata = prepare_electro_capsule(native, out)
    assert metadata['native_window'] == window
    plan, _ = _sequence_plan(tmp_path, out, metadata, native)
    seq = sequence_from_capsules(plan)
    assert tuple(seq.values.shape) == (1, 1, 8, 8)
    assert float(seq.values[0, 0, 0, 0]) == pytest.approx(261.4)
    assert float(seq.latitude[0, 0]) == pytest.approx(59.85)
    assert float(seq.longitude[0, 0]) == pytest.approx(20.25)
    assert not seq.valid[0, 0, 1, 2]


@pytest.mark.parametrize(('patch', 'message'), [
    ({'width': 32769}, 'выходит за пределы'),
    ({'row_offset': True}, 'целочисленные'),
    ({'column_offset': 5}, 'exceeds the verified dimensions'),
])
def test_electro_rejects_invalid_native_window(tmp_path, patch, message):
    window = dict(row_offset=1, column_offset=2, height=8, width=8) | patch
    native = _native_fixture(tmp_path / 'native', '12', window=window)
    with pytest.raises(ValueError, match=message):
        prepare_electro_capsule(native, tmp_path / 'capsule')
