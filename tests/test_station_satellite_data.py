"""Analytical fixtures for the Arctic input contract, never proof of live admission."""
import hashlib
import json

import pytest
import torch

from global_weather.multimodal.contracts import Channel, Sensor, Sequence, utc
from global_weather.multimodal.io import read_json, reference, save_sensors, save_sequence, sha256, write_json
from global_weather import station_satellite_data as bridge


ISSUES = ('2024-06-01T12:00:00Z', '2025-06-01T12:00:00Z', '2026-06-01T12:00:00Z')


def _fixture(root, *, duplicate=False, future=False, empty=False, electro=False):
    dataset = root / 'stations'
    write_json(dataset / 'dataset.json', {
        'schema': 'native-station-dataset-1', 'data_kind': 'real', 'campaign': 'extended_2023_2026',
        'samples': [dict(issue=issue, split=split) for issue, split in zip(ISSUES, ('train', 'validation', 'test'))],
    })
    source, platform, channel = ('electro_l', 'EL3', '10') if electro else ('arktika_m', 'ARCM1', '8')
    sensor = Sensor('fixture-ir', source, platform, 'imager',
                    (Channel(channel, 'brightness_temperature', 'K', None, None, 'fixture'),), None, None, 'real')
    registry = root / 'sensors.json'
    save_sensors(registry, (sensor,))
    write_json(root / 'admission.json', {'schema': 'multimodal-ir-admission-1', 'capsules': {'fixture': {}}})
    write_json(root / 'plan.json', {'schema': 'multimodal-capsule-plan-1', 'data_kind': 'real', 'sensor': sensor.id,
                                   'sensor_registry': reference(root, registry),
                                   'frames': [{}], 'declarations': {'real_admission': reference(root, root / 'admission.json')}})
    mapping = {}
    for index, issue in enumerate(ISSUES):
        t = utc(issue).timestamp()
        spatial = torch.ones((8, 8))
        pixels = 240 + torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) + index * 110
        valid = torch.ones((1, 1, 8, 8), dtype=torch.bool)
        valid[..., 0, 0] = False
        if empty and not index:
            valid[:] = False
        seen = t + 1 if future and not index else t - 1200
        frame_id = hashlib.sha256(str(0 if duplicate else index).encode()).hexdigest()
        sequence = Sequence(
            sensor_id=sensor.id, values=pixels, valid=valid, observed_unix=torch.tensor([t-1800], dtype=torch.float64),
            available_unix=torch.tensor([seen], dtype=torch.float64), latitude=60 * spatial, longitude=35 * spatial,
            view_zenith_deg=(30 * spatial)[None], solar_zenith_deg=(45 * spatial)[None], footprint_km=(5 * spatial)[None],
            area_m2=spatial, frame_ids=(frame_id,), channel_ids=(channel,), grid_id='fixture-grid', geometry_reference='fixture-only',
            source_sha256='1' * 64, sensor_signature=sensor.measurement_signature, data_kind='real',
            scan_start_unix=torch.tensor([t-3600], dtype=torch.float64),
        )
        file = root / f'sequence-{index}.npz'
        save_sequence(file, sequence)
        report = root / f'preparation-{index}.json'
        write_json(report, {'schema': 'station-satellite-sequence-1',
                            'status': 'sequence_prepared', 'data_kind': 'real',
                            'meteorologically_validated': False, 'sensor': sensor.id, 'frames': 1, 'shape': [1, 1, 8, 8],
                            'sha256': sha256(file), 'plan': reference(root, root / 'plan.json'),
                            'frame_index': [{'id': frame_id, 'scan_start': t-3600, 'scan_end': t-1800, 'available_at': seen}]})
        mapping[issue] = [{'sensor': sensor.id, 'file': reference(root, file), 'preparation': reference(root, report)}]
    manifest = root / 'input.json'
    write_json(manifest, {'schema': 'station-satellite-input-1', 'station_dataset_sha256': sha256(dataset / 'dataset.json'),
                          'sensors': reference(root, registry), 'scenes': mapping})
    return dataset, manifest


@pytest.fixture
def station_case(tmp_path, monkeypatch):
    # The pixel/split tests exercise the station bridge independently of source
    # evidence. The native GeoTIFF test below runs the actual proof checker.
    monkeypatch.setattr(bridge, '_capsule_provenance', lambda *args, **kwargs: None)
    return lambda **options: _fixture(tmp_path, **options)


def test_fit_reads_only_train_and_keeps_mask(tmp_path, monkeypatch, station_case):
    dataset, manifest = station_case()
    original = bridge.load_sequence
    reads = []

    def train_only(path, sensor):
        reads.append(path.name)
        assert path.name == 'sequence-0.npz'
        return original(path, sensor)

    monkeypatch.setattr(bridge, 'load_sequence', train_only)
    output = tmp_path / 'fixed.json'
    result = bridge.fit_station_sensors(dataset, manifest, output)
    data = bridge.StationSatelliteData(output, dataset)
    assert reads == ['sequence-0.npz']
    assert result['heldout_pixels_read'] is False
    assert data.indices('train') == [0] and data.indices('test') == [2]
    assert data.covered('validation', [0, 1]) == 1
    values = torch.arange(241, 304, dtype=torch.float64)
    channel = data.sensors[0].channels[0]
    assert channel.mean == pytest.approx(values.mean().item())
    assert channel.std == pytest.approx(values.std(unbiased=False).item())
    sequence, = data.sequences(0)
    normalized, mask = sequence.normalized(data.sensors[0])
    assert not mask[..., 0, 0].any() and not normalized[..., 0, 0].any()
    assert data.sequences(3) == ()
    with pytest.raises(FileExistsError):
        bridge.fit_station_sensors(dataset, manifest, output)


@pytest.mark.parametrize(('fault', 'message'), [
    ('duplicate', 'Повторное усвоение'), ('future', 'причинность'), ('empty', 'пустой'),
])
def test_fit_rejects_duplicate_future_and_empty_frames(tmp_path, station_case, fault, message):
    dataset, manifest = station_case(**{fault: True})
    output = tmp_path / 'fixed.json'
    with pytest.raises(ValueError, match=message):
        bridge.fit_station_sensors(dataset, manifest, output)
    assert not output.exists()


def test_holdout_hash_checked_when_opened(tmp_path, station_case):
    dataset, manifest = station_case()
    output = tmp_path / 'fixed.json'
    bridge.fit_station_sensors(dataset, manifest, output)
    sequence_path = tmp_path / 'sequence-2.npz'
    sequence_path.write_bytes(b'corrupt held-out pixels')
    data = bridge.StationSatelliteData(output, dataset)
    with pytest.raises(ValueError, match='Изменённый'):
        data.sequences(2)


def test_norm_provenance_cannot_use_a_holdout_issue(tmp_path, station_case):
    dataset, manifest = station_case()
    output = tmp_path / 'fixed.json'
    bridge.fit_station_sensors(dataset, manifest, output)
    stats, registry = tmp_path / 'fixed.fixture-ir.norms.json', tmp_path / 'fixed.sensors.json'
    stat_data = read_json(stats)
    stat_data['sample_ids'].append(utc(ISSUES[2]).isoformat())
    stats.write_text(json.dumps(stat_data), encoding='utf-8')
    reg_data = read_json(registry)
    reg_data['sensors'][0]['normalization_sha256'] = sha256(stats)
    reg_data['statistics']['fixture-ir'] = reference(tmp_path, stats)
    registry.write_text(json.dumps(reg_data), encoding='utf-8')
    campaign_data = read_json(output)
    campaign_data['sensors'] = reference(tmp_path, registry)
    output.write_text(json.dumps(campaign_data), encoding='utf-8')
    with pytest.raises(ValueError, match='точным списком'):
        bridge.StationSatelliteData(output, dataset)


def test_manifest_rejects_native_input_with_only_a_generic_report(tmp_path):
    dataset, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match='Каналы паспорта'):
        bridge.fit_station_sensors(dataset, manifest, tmp_path / 'fixed.json')


def test_electro_proof_checks_heldout_metadata_and_opens_train_raw_only(tmp_path, monkeypatch):
    dataset, manifest = _fixture(tmp_path, electro=True)
    calls = []

    def verify(path, report, sensor, *, verify_raw, sequence):
        assert sensor.platform == 'EL3' and sensor.channels[0].id == '10'
        assert report['schema'] == 'station-satellite-sequence-1'
        calls.append((path.name, verify_raw))

    monkeypatch.setattr(bridge, '_capsule_provenance', verify)
    bridge.fit_station_sensors(dataset, manifest, tmp_path / 'fixed.json')
    assert calls == [('preparation-0.json', False), ('preparation-1.json', False),
                     ('preparation-2.json', False), ('preparation-0.json', True)]


def test_prepare_wraps_native_capsule_contract(tmp_path):
    from test_multimodal_capsules import native_ir_fixture, write_native_plan

    sensor, plan, admission = native_ir_fixture(tmp_path)
    write_native_plan(tmp_path, plan, admission)
    output, report = tmp_path / 'sequence.npz', tmp_path / 'passport.json'
    passport = bridge.prepare_station_sequence(tmp_path / 'plan.json', output, report)
    assert read_json(report) == passport and passport['sensor'] == sensor.id
    assert passport['sha256'] == sha256(output) and passport['meteorologically_validated'] is False
    assert utc('2026-11-01T10:00:00Z').timestamp() == passport['frame_index'][0]['scan_start']
    bridge._capsule_provenance(report, passport, sensor, verify_raw=True)
    sequence = bridge.load_sequence(output, sensor)
    bridge._capsule_provenance(report, passport, sensor, verify_raw=True, sequence=sequence)
    with pytest.raises(ValueError, match='values'):
        bridge._capsule_provenance(report, passport, sensor, verify_raw=True,
                                   sequence=bridge.replace(sequence, values=sequence.values + 1))
    passport['frame_index'][0]['scan_start'] += 1
    with pytest.raises(ValueError, match='нативной сцене'):
        bridge._capsule_provenance(report, passport, sensor, verify_raw=False)
    with pytest.raises(ValueError, match='новый паспорт'):
        bridge.prepare_station_sequence(tmp_path / 'plan.json', output, report)
