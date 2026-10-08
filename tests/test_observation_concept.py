"""Concept wiring and regression cases; no live data or forecast-skill claims."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from global_weather.analysis.observation_operator import observation_space_loss, predict_observations
from global_weather.grid import build_grid, unit_xyz
from global_weather.observation_identity import observation_identity, observation_group_identity
from global_weather.observation_training import default_training_stages, stage_plan_payload
from global_weather.providers.upper_air import normalize_profile
from test_observation_driven import anonymous_profile, frames_for_operator, ISSUE, PRESSURE

ROOT = Path(__file__).resolve().parents[1]
ROLES = ('coordinator', 'data-steward', 'radiometry', 'normalization', 'physics',
         'model-engineer', 'executor', 'verification', 'release-auditor')


@pytest.mark.parametrize('role', ROLES)
def test_every_role_reaches_observation_contract(role):
    for folder in ('agents', '.claude/agents'):
        assert 'agents/OPERATING_CONTRACT.md' in (ROOT / folder / f'{role}.md').read_text()
    for path in ('agents/OPERATING_CONTRACT.md', 'agents/PIPELINE_CONTRACT.md',
                 'agents/CONTINUOUS_TRAINING_CONTRACT.md', 'AGENTS.md', 'agents/AGENTS.md'):
        text = (ROOT / path).read_text()
        assert 'GLOBAL-WEATHER-OBSERVATIONS-1' in text and 'OBSERVATION_CONTRACT.md' in text
    contract = (ROOT / 'agents/OBSERVATION_CONTRACT.md').read_text()
    assert 'Индекс станции необязателен' in contract
    assert 'ERA5 не является обязательным входом' in contract
    assert 'observed_at <= issue_time' in contract and 'available_at <= issue_time' in contract


def test_config_matches_runtime_stage_policy():
    policy = json.loads((ROOT / 'configs/observation_training_stages.json').read_text())
    stages = default_training_stages()
    assert [row['id'] for row in policy['stages']] == [s.id for s in stages]
    assert policy['observations']['station_identifier_required'] is False
    assert policy['era5']['allowed_operational_input'] is False
    for configured, stage in zip(policy['stages'], stages):
        assert configured['deployment_equivalent'] == stage.deployment_equivalent
        if stage.deployment_equivalent:
            assert 'era5' not in stage.inputs and stage.era5_role != 'teacher'
    assert policy['stages'][-1]['gradient'] is False
    assert stage_plan_payload()['execution_implemented'] is False


def record(variable='t2m', units='K', **kw):
    return dict({'source': 'station', 'variable': variable, 'value': 280., 'units': units,
                 'latitude': 0., 'longitude': 0., 'observed_at': ISSUE.isoformat()}, **kw)


def test_masked_nan_does_not_poison_horizontal_interpolation_or_gradient():
    grid = build_grid(0); frames = frames_for_operator(grid)
    cell = int(grid.tree.query(unit_xyz(0., 0.), k=3)[1][0])
    with torch.no_grad():
        frames[0].surface[cell, 0] = float('nan')
        frames[0].surface_mask[cell, 0] = False
    predicted, rejected = predict_observations(frames, grid, PRESSURE, [record()])
    assert not rejected and predicted[0].predicted == pytest.approx(280.)
    loss, _ = observation_space_loss(frames, grid, PRESSURE, [record(value=281.)])
    loss.backward()
    assert torch.isfinite(frames[0].surface.grad).all()
    assert frames[0].surface.grad[cell, 0] == 0


def test_missing_bracketing_level_is_not_vertical_extrapolation():
    grid = build_grid(0); frames = frames_for_operator(grid)
    frames[0].profile_mask[:, 0] = False
    p = float(np.sqrt(PRESSURE[0] * PRESSURE[1]))
    predicted, rejected = predict_observations(frames, grid, PRESSURE,
        [record('temperature', pressure_pa=p)])
    assert not predicted and rejected == {'outside_supported_forecast': 1}


@pytest.mark.parametrize('axis', [PRESSURE[::-1], np.zeros_like(PRESSURE), np.r_[PRESSURE[0], PRESSURE[:-1]]])
def test_invalid_pressure_axis_is_rejected(axis):
    grid = build_grid(0)
    predicted, rejected = predict_observations(frames_for_operator(grid), grid, axis,
        [record('temperature', pressure_pa=85000.)])
    assert not predicted and rejected == {'invalid_contract': 1}


def test_accumulations_need_separate_interval_operator():
    grid = build_grid(0)
    predicted, rejected = predict_observations(frames_for_operator(grid), grid, PRESSURE,
        [record('precipitation_step', 'kg m-2')])
    assert not predicted and rejected == {'accumulation_operator_required': 1}


def test_loss_does_not_mix_kelvin_and_pascal_without_normalization():
    grid = build_grid(0)
    with pytest.raises(ValueError, match='Разные единицы'):
        observation_space_loss(frames_for_operator(grid), grid, PRESSURE,
                               [record(), record('surface_pressure', 'Pa')])


def test_provider_namespaces_and_later_station_assignment_preserve_identity():
    a = record(source='radiosonde', provider='provider_a', provider_message_id='flight-1',
               profile_id='profile-1', pressure_pa=85000.)
    b = dict(a, provider='provider_b')
    assert observation_identity(a) != observation_identity(b)
    assert observation_group_identity(a) != observation_group_identity(b)
    assert observation_identity(a) == observation_identity(dict(a, station_id='newly-registered'))


def test_partial_wind_does_not_discard_other_profile_measurements():
    profile = anonymous_profile()
    profile['levels'] = [dict(pressure_pa=85000., temperature_k=270., u_ms=12.)]
    records = normalize_profile(profile, acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    assert {r['variable'] for r in records} == {'temperature', 'u'}
    profile['levels'] = [dict(pressure_pa=85000., temperature_k=270., wind_speed_ms=12.)]
    records = normalize_profile(profile, acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    assert {r['variable'] for r in records} == {'temperature'}
    assert records[0]['omitted_derivations'] == ['wind_components_require_speed_and_direction']


def test_nominal_time_fallback_is_not_claimed_to_be_launch_time():
    profile = anonymous_profile(); profile['nominal_time'] = profile.pop('launch_time')
    profile['levels'] = [dict(pressure_pa=85000., temperature_k=270.)]
    rows = normalize_profile(profile, acquired_at='2026-01-01T00:00:00Z',
                             availability_mode='assumed_latency', latency_minutes=15)
    assert rows[0]['launch_time'] is None and rows[0]['time_basis'] == 'nominal_time_fallback'


@pytest.mark.parametrize('change', ['negative_elapsed', 'unknown_mode', 'future_measurement', 'before_launch'])
def test_contradictory_upper_air_time_is_rejected(change):
    profile = anonymous_profile(); args = dict(acquired_at='2026-01-01T00:00:00Z',
        availability_mode='assumed_latency', latency_minutes=15)
    if change == 'negative_elapsed': profile['levels'][0]['elapsed_seconds'] = -1
    if change == 'unknown_mode': args['availability_mode'] = 'unknown'
    if change == 'future_measurement': args['acquired_at'] = '2019-01-01T00:00:00Z'
    if change == 'before_launch': profile['levels'][0]['observed_at'] = '2020-01-01T03:00:00Z'
    with pytest.raises(ValueError): normalize_profile(profile, **args)
