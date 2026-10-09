"""Pinned real statistics and synthetic model tests; no weather skill claim."""
import json
from datetime import timedelta

import numpy as np
import pytest
import torch

from global_weather import profile_graphcast_normalization as graphcast
from global_weather.profile_normalization import load_normalization
from global_weather.profile_model_v2 import PressureProfileModel
from global_weather.profile_training import VARIABLES, START
from global_weather.grid import build_pyramid
from global_weather.import_climatology import PINNED_HASHES
from global_weather.vertical import PROFILE_UNITS


def payload():
    return {**graphcast._reference(), 'source_identity': {
        name: '0' * 64 for name in ('database_sha256', 'dataset_manifest_sha256',
                                   'source_sha256', 'admission_sha256')}}


def test_actual_pinned_statistics_affine_roundtrip_all_levels_and_intermediate_pressure():
    norms = load_normalization(payload())
    assert norms.payload['provenance']['artifact_sha256'] == PINNED_HASHES
    assert norms.payload['provenance']['fit_period']['start'].startswith('1979-01-02')
    assert norms.payload['provenance']['fit_period']['end'].startswith('2015-12-31')
    assert norms.support.all() and norms.mean.shape == (37, 5)
    for index, value in enumerate((270., .001, 4., -3., 30000.)):
        for pressure in (*norms.pressure_pa, 68000.):
            normalized = norms.normalize(index, value, pressure)
            assert float(norms.inverse(index, normalized, pressure)) == pytest.approx(value, abs=1e-10)
    assert float(norms.inverse(1, norms.normalize(1, 0., 70000.), 70000.)) == pytest.approx(0., abs=1e-18)
    assert not norms.at(0, 100001.)[2]
    assert not norms.at(0, 99.)[2]


@pytest.mark.parametrize('field', ['mean', 'std', 'units', 'provenance', 'source_identity'])
def test_artifact_tampering_rejected(field):
    data = payload()
    if field in ('mean', 'std'): data[field][0][0] += 1
    elif field == 'units': data[field][0] = 'C'
    elif field == 'provenance': data[field]['artifact_sha256']['mean_by_level.nc'] = '0' * 64
    else: data[field].pop('admission_sha256')
    with pytest.raises(ValueError): graphcast.GraphCastNormalization(data)


def test_wrong_actual_source_hash_rejected(monkeypatch):
    monkeypatch.setattr(graphcast, 'PINNED_HASHES', {name: '0' * 64 for name in PINNED_HASHES})
    with pytest.raises(ValueError, match='pinned'): graphcast._reference()


def test_graphcast_q_no_softplus_floor_and_affine_objective_gradients():
    from global_weather.profile_training_v2 import objective
    norms = graphcast.GraphCastNormalization(payload())
    model = PressureProfileModel(build_pyramid(0)[0], norms, 8)
    issue = START + timedelta(days=3)
    records = [dict(variable=name, value=[270., .001, 4., -3., 30000.][i],
                    units=PROFILE_UNITS[i], latitude=10., longitude=20., pressure_pa=70000.,
                    observed_at=issue.isoformat(), available_at=issue.isoformat(), profile_id='synthetic')
               for i, name in enumerate(VARIABLES)]
    frames = model(records, issue)
    targets = [dict(row, observed_at=(issue + timedelta(hours=12)).isoformat()) for row in records]
    loss, counts = objective(model, frames, targets)
    assert counts == [1] * 5
    loss.backward()
    assert torch.isfinite(model.head.weight.grad).all()
    assert (model.head.weight.grad.abs().sum(dim=1) > 0).all()
    with torch.no_grad(): model.head.weight.zero_(); model.head.bias[1] = -1000.
    frame = model._decode(torch.zeros(12, 37, 8), issue, 0)
    assert torch.equal(frame.profiles[..., 1], torch.zeros(12, 37))
    raw = torch.tensor([0., 1e-6], requires_grad=True)
    raw.clamp_min(0.).sum().backward()
    assert torch.equal(raw.grad, torch.ones(2))
    assert torch.isnan(frame.profiles[..., 5]).all()


def test_immutable_import_dataset_drift_rejected_without_fitting(tmp_path, monkeypatch):
    from global_weather import profile_training
    current = {'database_sha256': '0' * 64, 'dataset_manifest_sha256': '1' * 64,
               'source_sha256': '2' * 64, 'admission_sha256': '3' * 64}
    monkeypatch.setattr(profile_training, 'ProfileDataset', lambda path: object())
    monkeypatch.setattr(graphcast, '_identity', lambda dataset: dict(current))
    artifact = tmp_path / 'norms.json'
    first = graphcast.create(tmp_path / 'dataset', artifact)
    assert graphcast.create(tmp_path / 'dataset', artifact) == first
    original_bytes = artifact.read_bytes()
    current['admission_sha256'] = '4' * 64
    with pytest.raises(ValueError, match='identity changed'):
        graphcast.create(tmp_path / 'dataset', artifact)
    assert artifact.read_bytes() == original_bytes


def test_frozen_forecast_reads_only_causal_records_and_never_sample(tmp_path, monkeypatch):
    from global_weather import profile_training_v2 as training
    from global_weather.observation_training import digest
    norms = graphcast.GraphCastNormalization(payload())
    model = PressureProfileModel(build_pyramid(0)[0], norms, 8).eval()
    issue = START + timedelta(days=3)
    class Dataset:
        def records(self, start, end, *, issue):
            assert start == end - timedelta(hours=12) and end == issue
            return []
        def sample(self, *args):
            raise AssertionError('Forecast may not read targets')
        def verify(self):
            pass
    training_root = tmp_path / 'training'; training_root.mkdir()
    norm_path = tmp_path / 'norms.json'; norm_path.write_text(json.dumps(norms.payload))
    identity = {'config': {'max_records_per_window': 50}, 'norm_path': str(norm_path),
                'norm_sha256': digest(norm_path)}
    (training_root / 'complete.json').write_text(json.dumps({'identity': identity}))
    checkpoint_dir=training_root/'epoch-0001';checkpoint_dir.mkdir()
    (checkpoint_dir/'state.pt').write_bytes(b'synthetic checkpoint: frozen loader mocked')
    (training_root / 'best.json').write_text(json.dumps({'directory':'epoch-0001','epoch':1,
        'sha256':digest(checkpoint_dir/'state.pt')}))
    monkeypatch.setattr(training, 'load_frozen', lambda *args: (model, Dataset()))
    output = tmp_path / 'forecast'
    training.forecast(tmp_path / 'dataset', training_root, issue.isoformat(), output)
    metadata = json.loads((output / 'forecast.json').read_text())
    assert metadata['future_targets_read'] is False and metadata['input_count'] == 0
    with np.load(output / 'forecast.npz') as arrays:
        assert arrays['profiles'].shape == (25, 12, 37, 6)
        assert not arrays['profile_variable_mask'][..., 5].any()
    training.forecast(tmp_path / 'dataset', training_root, issue.isoformat(), output)
    with (output / 'forecast.npz').open('ab') as file:file.write(b'tampered')
    with pytest.raises(ValueError,match='artifact changed'):
        training.forecast(tmp_path / 'dataset', training_root, issue.isoformat(), output)
    original_forward=model.forward
    def mutating_forward(*args):
        result=original_forward(*args)
        completion_path=training_root/'complete.json'
        completion_path.write_text(completion_path.read_text()+'\n')
        return result
    monkeypatch.setattr(model,'forward',mutating_forward)
    with pytest.raises(ValueError,match='Frozen training artifacts changed during inference'):
        training.forecast(tmp_path/'dataset',training_root,issue.isoformat(),tmp_path/'mutated-forecast')
    assert not (tmp_path/'mutated-forecast').exists()


def test_checkpoint_cannot_replace_fixed_graphcast_buffers():
    from global_weather.profile_training_v2 import _load_fixed_state
    model=PressureProfileModel(build_pyramid(0)[0],graphcast.GraphCastNormalization(payload()),8)
    state=model.state_dict();state['mean']=state['mean'].clone()+1
    with pytest.raises(ValueError,match='fixed normalization'):_load_fixed_state(model,state)
