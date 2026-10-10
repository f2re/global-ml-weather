"""Both independently published R9 paths survive integration without mixing."""
from datetime import timedelta
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from global_weather import profile_training_v2 as training
from global_weather.profile_physics import PhysicalPolicy
from global_weather.profile_training import START
from global_weather.vertical import PRESSURE_HPA


@pytest.mark.parametrize('filename', [
    'profile_r9_training.json', 'profile_r9_loss.json',
    'profile_r9_hydrostatic.json', 'profile_r9_mesh3.json',
])
def test_published_configurations_are_distinct_and_idempotent(filename):
    root = Path(__file__).resolve().parents[1]
    config = training._configuration(json.loads((root / 'configs' / filename).read_text()))
    assert training._configuration(config) == config
    if filename == 'profile_r9_training.json':
        assert 'physical_policy' not in config
        assert config['min_humidity_pressure_pa'] == 30000.
        assert config['use_vertical_weights'] is True
        assert config['hydrostatic_weight'] == .01
    else:
        assert not training.LEGACY_LOSS_KEYS.intersection(config)
        assert config['physical_policy']['humidity_min_pressure_pa'] == 0.
        assert training._objective_options(config) == {}


@pytest.mark.parametrize('key,value', [
    ('min_humidity_pressure_pa', 30000.),
    ('use_vertical_weights', True), ('hydrostatic_weight', .01),
])
def test_explicit_policies_cannot_be_silently_combined(key, value):
    config = {'physical_policy': {}, key: value}
    with pytest.raises(ValueError, match='mix'):
        training._configuration(config)
    with pytest.raises(ValueError, match='mix'):
        training._objective_options(config)


@pytest.mark.parametrize('key,value', [
    ('min_humidity_pressure_pa', -1.),
    ('min_humidity_pressure_pa', float('nan')),
    ('min_humidity_pressure_pa', 100001.),
    ('hydrostatic_weight', -1.),
    ('hydrostatic_weight', float('inf')),
    ('use_vertical_weights', 'false'),
    ('use_vertical_weights', 1),
])
def test_legacy_options_do_not_coerce_invalid_settings(key, value):
    with pytest.raises(ValueError):
        training._configuration({key: value})


def test_only_legacy_default_objective_has_the_300hpa_cutoff(monkeypatch):
    from global_weather.analysis import observation_operator
    predicted = torch.tensor(0., requires_grad=True)
    prediction = lambda *args: (predicted, None)
    monkeypatch.setattr(training, '_prediction', prediction)
    monkeypatch.setattr(observation_operator, '_prediction', prediction)
    class Norm:
        humidity_transform = 'identity'
        def at(self, *args):
            return 0., 1., True
    model = SimpleNamespace(normalization=Norm(), grid=None,
                            pressure_pa=torch.tensor(PRESSURE_HPA) * 100.)
    base = dict(profile_id='coordinate-only-launch', pressure_pa=10000.,
                observed_at=(START + timedelta(hours=3)).isoformat(),
                latitude=61., longitude=31.)
    targets = [dict(base, variable='temperature', value=230.),
               dict(base, variable='specific_humidity', value=0.000001)]
    frames = [SimpleNamespace(valid_time=START)]
    _, legacy_counts = training.objective(model, frames, targets)
    assert legacy_counts[:2] == [1, 0]
    _, unmasked_counts = training.objective(model, frames, targets, min_humidity_pressure_pa=0.)
    assert unmasked_counts[:2] == [1, 1]
    model.physical_policy = PhysicalPolicy()
    _, physical_counts = training.objective(model, frames, targets)
    assert physical_counts[:2] == [1, 1]
    with pytest.raises(ValueError, match='mix'):
        training.objective(model, frames, targets, min_humidity_pressure_pa=30000.)


@pytest.mark.parametrize('physical', [False, True])
def test_score_routes_the_same_policy_to_forecast_and_reconstruction(monkeypatch, physical):
    calls = []
    class Model:
        def eval(self):
            pass
        def __call__(self, inputs, issue):
            return ['forecast']
    class Dataset:
        def issues(self, split, maximum):
            return [START]
        def sample(self, issue, maximum):
            return [], ['target']
    def objective(model, frames, targets, **options):
        calls.append(('forecast', options))
        return torch.tensor(1.), [1] * 5
    def reconstruction(model, dataset, targets, limit, split, **options):
        calls.append(('reconstruction', options))
        return torch.tensor(2.)
    monkeypatch.setattr(training, 'objective', objective)
    monkeypatch.setattr(training, 'reconstruction', reconstruction)
    options = {} if physical else dict(min_humidity_pressure_pa=40000., use_vertical_weights=False)
    config = dict(max_validation_issues=1, max_records_per_window=20, **options)
    if physical:
        config['physical_policy'] = {}
    result = training.score(Model(), Dataset(), config, 'validation')
    assert calls == [('forecast', options), ('reconstruction', options)]
    assert result == {'forecast': 1., 'reconstruction': 2.}


def test_internal_type_error_does_not_retry_with_another_objective(monkeypatch):
    calls = []
    class Model:
        def eval(self):
            pass
        def __call__(self, inputs, issue):
            calls.append('model')
            return ['forecast']
    class Dataset:
        def issues(self, split, maximum):
            return [START]
        def sample(self, issue, maximum):
            return [], ['target']
    def broken(model, frames, targets, **options):
        calls.append('objective')
        raise TypeError('internal numerical error')
    monkeypatch.setattr(training, 'objective', broken)
    with pytest.raises(TypeError, match='internal numerical error'):
        training.score(Model(), Dataset(), dict(max_validation_issues=1,
                       max_records_per_window=20, min_humidity_pressure_pa=30000.), 'validation')
    assert calls == ['model', 'objective']
