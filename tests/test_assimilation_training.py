"""R11 trainer: configuration bounds, synthetic refusal, self-test and strict resume."""
import json

import pytest
import torch

from global_weather.assimilation.selftest import SELFTEST_CONFIG, run
from global_weather.assimilation.training import DEFAULT_CONFIG, Experiment, learning_rate, load_config, train


def test_shipped_configurations_validate_and_keep_problem_definition():
    for path in ('configs/physical_assimilation_r11.json', 'configs/physical_assimilation_smoke.json'):
        config = load_config(json.loads(open(path).read()))
        assert config['input_hours'] == 12 and config['horizon_hours'] == 72
        assert config['allow_synthetic'] is False
        assert config['data']['satellite_supervision'] is False
        hours = [stage['rollout_hours'] for stage in config['stages']]
        assert hours == sorted(hours) and hours[-1] == 72
    r11 = load_config(json.loads(open('configs/physical_assimilation_r11.json').read()))
    assert r11['mesh_level'] == 5 and r11['multimesh'] and r11['issue_stride_hours'] == 6


@pytest.mark.parametrize('change', [{'horizon_hours': 240}, {'input_hours': 6}, {'unknown': 1},
                                    {'stages': [{**DEFAULT_CONFIG['stages'][0], 'rollout_hours': 5}]},
                                    {'noise_dim': 0, 'stages': [{**DEFAULT_CONFIG['stages'][0], 'ensemble_members': 2}]}])
def test_invalid_configuration_is_refused(change):
    with pytest.raises(ValueError):
        load_config(change)


def test_learning_rate_warmup_and_cosine():
    stage = {**DEFAULT_CONFIG['stages'][0], 'warmup_steps': 10, 'peak_lr': 1e-3, 'min_lr': 1e-5}
    assert learning_rate(stage, 0, 100) == pytest.approx(1e-4)
    assert learning_rate(stage, 10, 100) == pytest.approx(1e-3)
    assert learning_rate(stage, 100, 100) == pytest.approx(1e-5)


def test_real_configuration_refuses_synthetic_measurements():
    with pytest.raises(ValueError, match='Synthetic'):
        Experiment(load_config({}), synthetic=True)


def test_selftest_passes_physics_and_dependence_checks(tmp_path):
    report = run(tmp_path / 'selftest')
    assert report['passed'] and report['data_kind'] == 'synthetic_analytic'
    assert report['one_step_influence_radius_km'] >= 1620


def test_resume_requires_identical_identity(tmp_path):
    output = tmp_path / 'run'
    train(SELFTEST_CONFIG, output, synthetic=True, device='cpu', max_steps=1)
    first = json.loads((output / 'latest.json').read_text())
    changed = json.loads(json.dumps(SELFTEST_CONFIG))
    changed['hidden'] = 24
    with pytest.raises(ValueError, match='changed'):
        train(changed, output, synthetic=True, device='cpu', max_steps=2)
    assert json.loads((output / 'latest.json').read_text()) == first


def test_resume_with_identical_identity_completes_and_frozen_outputs(tmp_path):
    from global_weather.assimilation.forecast import forecast, load_frozen, score
    output = tmp_path / 'run'
    train(SELFTEST_CONFIG, output, synthetic=True, device='cpu', max_steps=1)
    result = train(SELFTEST_CONFIG, output, synthetic=True, device='cpu', max_steps=2)
    assert result['completion']['status'] == 'physical_assimilation_research_trained'
    model, experiment, provenance = load_frozen(output, synthetic=True)
    assert all(not p.requires_grad for p in model.parameters())
    scores = score(model, experiment, 'validation', 6, limit=1)
    assert scores['issues'] == 1 and scores['scientific_acceptance'] is False
    manifest = forecast(model, experiment, '2021-03-01T12:00:00Z', tmp_path / 'forecast', hours=6, members=2,
                        provenance=provenance)
    import numpy as np
    with np.load(tmp_path / 'forecast' / 'forecast.npz') as data:
        assert data['profiles'].shape == (2, 3, experiment.mesh.n_nodes, 37, 6)
        assert np.isnan(data['profiles'][~data['profile_mask']]).all()
    assert manifest['members'] == 2 and manifest['data_kind'] == 'synthetic'


def test_cycled_analysis_never_reassimilates_an_earlier_report():
    import numpy as np
    from datetime import datetime, timezone
    from global_weather.assimilation.training import background_state
    experiment = Experiment(load_config(SELFTEST_CONFIG), synthetic=True)
    model = experiment.build_model()
    issue = datetime(2021, 3, 1, 12, tzinfo=timezone.utc)
    background, previous = background_state(model, experiment, issue, 'train')
    assert previous == datetime(2021, 3, 1, 6, tzinfo=timezone.utc) and background.shape[0] == experiment.mesh.n_nodes
    cycled = experiment.sample(issue, 'train', 6, rng=np.random.default_rng(0), training=False, background_issue=previous)
    fresh = experiment.sample(issue, 'train', 6, rng=np.random.default_rng(0), training=False)
    assert 0 < cycled.inputs < fresh.inputs
    rollout = model(cycled.tokens, issue, 6, background=background)
    assert len(rollout.states) == 3
