from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from global_weather.mesh8_final_check import verify
from global_weather.mesh8_preflight import CANONICAL_MESH8_FINGERPRINT, _cuda_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value)+'\n')


def test_cuda_preflight_can_be_inspected_without_claiming_qualification(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: False)
    config = {'physical_policy': {'memory_budget_mib': 22528}}
    report = _cuda_identity(config, require_cuda=False)
    assert report['required'] is False and report['available'] is False
    with pytest.raises(ValueError, match='requires CUDA'):
        _cuda_identity(config, require_cuda=True)


def test_cuda_preflight_rejects_device_below_declared_budget(monkeypatch):
    monkeypatch.setattr('torch.cuda.is_available', lambda: True)
    monkeypatch.setattr('torch.cuda.init', lambda: None)
    monkeypatch.setattr('torch.cuda.get_device_properties',
                        lambda _: SimpleNamespace(total_memory=8*1024**3, name='fixture', major=8, minor=9))
    with pytest.raises(ValueError, match='below declared'):
        _cuda_identity({'physical_policy': {'memory_budget_mib': 22528}}, require_cuda=True)


def test_final_gate_accepts_only_complete_hashed_q4_q5_evidence(tmp_path):
    revision = 'a'*40
    write(tmp_path/'preflight.json', {
        'status': 'ready_for_q4_q5', 'source_commit': revision,
        'mesh': {'fingerprint': CANONICAL_MESH8_FINGERPRINT}})
    training = tmp_path/'training'; epoch = training/'epoch-0001'; epoch.mkdir(parents=True)
    (epoch/'state.pt').write_bytes(b'checkpoint')
    checkpoint_hash = sha(epoch/'state.pt')
    write(training/'best.json', {'directory': 'epoch-0001', 'epoch': 1, 'sha256': checkpoint_hash})
    write(training/'complete.json', {
        'status': 'joint_measured_observation_research_trained', 'mesh_level': 8,
        'grid_cells': 655362, 'rollout_hours': 3, 'scientific_acceptance': False,
        'best_epoch': 1, 'identity': {'grid_fingerprint': CANONICAL_MESH8_FINGERPRINT,
                                     'experiment_id': 'joint-mesh8-qualification-r10'}})
    write(epoch/'metrics.json', {
        'cuda_peak_allocated_bytes': 1024, 'cuda_peak_reserved_bytes': 2048,
        'mesh_level': 8, 'rollout_hours': 3,
        'nonzero_gradients': {'profile': True, 'surface': True}})

    forecast = tmp_path/'forecast'; issue = forecast/'20220101T000000Z'; issue.mkdir(parents=True)
    (issue/'forecast.npz').write_bytes(b'npz')
    (issue/'forecast.grib2').write_bytes(b'grib')
    (issue/'forecast.grib2.json').write_bytes(b'inventory')
    write(issue/'completion.json', {
        'issue_time': '2022-01-01T00:00:00+00:00', 'source_revision': revision,
        'grid_fingerprint': CANONICAL_MESH8_FINGERPRINT, 'lead_hours': [0, 3],
        'scientific_acceptance': False,
        'forecast_npz_sha256': sha(issue/'forecast.npz'),
        'grib_sha256': sha(issue/'forecast.grib2'),
        'grib_inventory_sha256': sha(issue/'forecast.grib2.json'),
        'diagnostic_physics': {
            'specific_humidity': {'physical_domain_violations': 0},
            'hydrostatic_residual': {'count': 10, 'max_abs_m2_s2': .2},
            'continuity_residual': {'count': 10, 'max_abs_s-1': 1e-10}},
        'telemetry': {'device': 'cuda:0'}})
    write(forecast/'batch.json', {
        'status': 'complete', 'source_revision': revision, 'mesh_level': 8,
        'grid_fingerprint': CANONICAL_MESH8_FINGERPRINT, 'horizon_hours': 3,
        'scientific_acceptance': False, 'products': [{'directory': issue.name}]})

    report = verify(tmp_path, revision)
    assert report['status'] == 'r10_q4_q5_technically_qualified'
    assert report['scientific_acceptance'] is False
    assert report['next_stage_authorized'] is False

    manifest = json.loads((issue/'completion.json').read_text())
    manifest['diagnostic_physics']['specific_humidity']['physical_domain_violations'] = 1
    write(issue/'completion.json', manifest)
    with pytest.raises(ValueError, match='humidity'):
        verify(tmp_path, revision, tmp_path/'another.json')
