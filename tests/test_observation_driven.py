from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from global_weather.analysis.observation_operator import observation_space_loss, predict_observations
from global_weather.grid import build_grid
from global_weather.model import ForecastFrame
from global_weather.observation_identity import observation_identity, profile_identity
from global_weather.observation_training import (default_training_stages, partition_observation_groups,
                                                  stage_plan_payload, validate_stage_transition)
from global_weather.observations import pack_observations
from global_weather.providers.upper_air import normalize_profile, specific_humidity, wind_components
from global_weather.vertical import PRESSURE_HPA

UTC = timezone.utc
ISSUE = datetime(2020, 1, 1, 6, tzinfo=UTC)
PRESSURE = np.asarray(PRESSURE_HPA, dtype=float) * 100


def anonymous_profile(**changes):
    value = {
        'provider': 'field_campaign',
        'provider_message_id': 'flight-20200101-0337',
        'launch_time': '2020-01-01T03:37:00Z',
        'launch_latitude': 61.125,
        'launch_longitude': 31.25,
        'revision': 0,
        'levels': [
            {'pressure_pa': 100000., 'elapsed_seconds': 120,
             'latitude': 61.126, 'longitude': 31.251,
             'temperature_k': 271.15, 'dewpoint_k': 269.15,
             'wind_speed_ms': 10., 'wind_direction_deg': 270.,
             'geopotential_height_m': 120.},
            {'pressure_pa': 85000., 'observed_at': '2020-01-01T04:12:30Z',
             'latitude': 61.45, 'longitude': 31.90,
             'temperature_k': 265.15, 'u_ms': 12., 'v_ms': -3.},
            {'pressure_pa': 70000., 'elapsed_seconds': 3600,
             'temperature_k': 253.15},
        ],
    }
    value.update(changes)
    return value


def test_stationless_profile_keeps_actual_level_time_and_position():
    records = normalize_profile(anonymous_profile(), acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    assert records
    assert all(r['station_id'] is None and r['station_registered'] is False for r in records)
    first = next(r for r in records if r['pressure_pa'] == 100000 and r['variable'] == 'temperature')
    assert first['observed_at'] == '2020-01-01T03:39:00+00:00'
    assert first['latitude'] == pytest.approx(61.126)
    assert first['position_basis'] == 'reported_level_position'
    fallback = next(r for r in records if r['pressure_pa'] == 70000 and r['variable'] == 'temperature')
    assert fallback['latitude'] == pytest.approx(61.125)
    assert fallback['position_basis'] == 'launch_position_fallback'
    assert fallback['observed_at'] == '2020-01-01T04:37:00+00:00'


def test_existing_provider_identity_with_timezone_offset_remains_compatible():
    record = {
        'observation_id': 'GHCNh/USW00014933/2020-01-01T00:00:00+00:00/t2m',
        'source': 'station', 'variable': 't2m', 'observed_at': '2020-01-01T00:00:00Z',
    }
    assert observation_identity(record) == record['observation_id']


def test_profile_and_observation_ids_do_not_require_station_index():
    profile = anonymous_profile()
    pid = profile_identity(profile)
    records = normalize_profile(profile, acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    assert pid.startswith('profile:')
    assert all(r['profile_id'] == pid and r['observation_id'].startswith('coord:') for r in records)
    changed = anonymous_profile(revision=1)
    changed['levels'][0]['temperature_k'] = 272.15
    revised = normalize_profile(changed, acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    old = next(r for r in records if r['pressure_pa'] == 100000 and r['variable'] == 'temperature')
    new = next(r for r in revised if r['pressure_pa'] == 100000 and r['variable'] == 'temperature')
    assert old['observation_id'] == new['observation_id']
    assert old['value'] != new['value'] and new['revision'] == 1


def test_missing_variables_and_standard_levels_are_not_filled():
    records = normalize_profile(anonymous_profile(), acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    variables_700 = {r['variable'] for r in records if r['pressure_pa'] == 70000}
    assert variables_700 == {'temperature'}
    assert not any(r['pressure_pa'] in (92500, 50000) for r in records)


def test_upper_air_physical_conversions():
    assert specific_humidity(100000, temperature_k=293.15, relative_humidity=.5) > 0
    u, v = wind_components(10, 270)
    assert u == pytest.approx(10, abs=1e-12)
    assert v == pytest.approx(0, abs=1e-12)


def test_reported_availability_is_required_unless_policy_explicit():
    with pytest.raises(ValueError, match='времени готовности'):
        normalize_profile(anonymous_profile(), acquired_at='2026-01-01T00:00:00Z')
    records = normalize_profile(anonymous_profile(available_at='2020-01-01T05:00:00Z'),
                                acquired_at='2026-01-01T00:00:00Z')
    assert {r['availability_basis'] for r in records} == {'reported'}


def test_pack_observations_generates_coordinate_identity_and_preserves_fractional_age():
    grid = build_grid(0)
    record = {
        'source': 'radiosonde', 'variable': 'temperature', 'value': 270., 'units': 'K',
        'latitude': 60., 'longitude': 30., 'pressure_pa': 85000.,
        'observed_at': '2020-01-01T03:37:30Z', 'available_at': '2020-01-01T04:00:00Z',
        'revision': 0, 'valid': True,
    }
    packed = pack_observations([record], grid, PRESSURE, ISSUE)
    assert packed.accepted_records == 1
    assert len(packed.evidence_ids) == 1
    age = (ISSUE - datetime(2020, 1, 1, 3, 37, 30, tzinfo=UTC)).total_seconds() / 3600
    assert packed.features[:, 1].tolist() == pytest.approx([age / 12])
    assert observation_identity(record).startswith('coord:')


def frames_for_operator(grid):
    n, l = grid.n_cells, len(PRESSURE)
    p0 = torch.full((n, l, 6), 260., requires_grad=True)
    p1 = torch.full((n, l, 6), 266., requires_grad=True)
    s0 = torch.full((n, 8), 280., requires_grad=True)
    s1 = torch.full((n, 8), 286., requires_grad=True)
    pm = torch.ones((n, l), dtype=torch.bool)
    sm = torch.ones((n, 8), dtype=torch.bool)
    return [ForecastFrame(0, ISSUE, p0, s0, pm, sm),
            ForecastFrame(3, ISSUE + timedelta(hours=3), p1, s1, pm, sm)]


def test_observation_operator_uses_arbitrary_coordinate_pressure_and_time():
    grid = build_grid(0)
    lat, lon = np.rad2deg(np.arcsin(grid.xyz[0, 2])), np.rad2deg(np.arctan2(grid.xyz[0, 1], grid.xyz[0, 0]))
    records = [
        {'data_kind': 'synthetic', 'source': 'station', 'variable': 't2m', 'value': 284., 'units': 'K',
         'latitude': float(lat), 'longitude': float(lon),
         'observed_at': (ISSUE + timedelta(hours=1, minutes=30)).isoformat(),
         'available_at': (ISSUE + timedelta(hours=2)).isoformat(), 'valid': True},
        {'source': 'radiosonde', 'variable': 'temperature', 'value': 264., 'units': 'K',
         'latitude': float(lat), 'longitude': float(lon), 'pressure_pa': 92500.,
         'observed_at': (ISSUE + timedelta(hours=1, minutes=30)).isoformat(),
         'available_at': (ISSUE + timedelta(hours=2)).isoformat(), 'valid': True},
    ]
    frames = frames_for_operator(grid)
    predicted, rejected = predict_observations(frames, grid, PRESSURE, records)
    assert not rejected and len(predicted) == 2
    assert predicted[0].predicted == pytest.approx(283.)
    assert predicted[0].time_fraction == pytest.approx(.5)
    assert predicted[1].predicted == pytest.approx(263.)
    loss, report = observation_space_loss(frames, grid, PRESSURE, records)
    assert report['accepted'] == 2
    loss.backward()
    assert frames[0].surface.grad is not None and frames[1].profiles.grad is not None


def test_late_arrival_is_excluded_then_admitted_at_later_issue():
    grid = build_grid(0)
    record = {
        'source': 'radiosonde', 'variable': 'temperature', 'value': 270., 'units': 'K',
        'latitude': 60., 'longitude': 30., 'pressure_pa': 85000.,
        'observed_at': '2020-01-01T03:37:30Z', 'available_at': '2020-01-01T06:15:00Z',
        'revision': 0, 'valid': True,
    }
    early = pack_observations([record], grid, PRESSURE, ISSUE)
    assert early.accepted_records == 0
    assert early.rejected == {'not_available_in_12h_window': 1}
    late = pack_observations([record], grid, PRESSURE, ISSUE + timedelta(hours=1))
    assert late.accepted_records == 1


def test_observation_operator_does_not_extrapolate_in_time():
    grid = build_grid(0)
    records = [{'source': 'station', 'variable': 't2m', 'value': 280., 'units': 'K',
                'latitude': 0., 'longitude': 0.,
                'observed_at': (ISSUE - timedelta(seconds=1)).isoformat(),
                'available_at': ISSUE.isoformat(), 'valid': True}]
    predicted, rejected = predict_observations(frames_for_operator(grid), grid, PRESSURE, records)
    assert predicted == [] and rejected == {'outside_supported_forecast': 1}


def test_training_stages_keep_era5_out_of_operational_inputs():
    stages = default_training_stages()
    assert [s.id for s in stages] == ['S0-theory', 'S1-observation-encoder',
                                      'S2-observation-space', 'S3-cycling',
                                      'S4-independent-validation']
    assert all('era5' not in s.inputs for s in stages if s.deployment_equivalent)
    payload = stage_plan_payload()
    assert payload['era5_operational_input'] is False and len(payload['fingerprint']) == 64
    assert validate_stage_transition(['S0-theory', 'S1-observation-encoder'], 'S2-observation-space')
    with pytest.raises(ValueError):
        validate_stage_transition(['S0-theory'], 'S2-observation-space')


def test_profile_group_partition_never_splits_levels():
    records = normalize_profile(anonymous_profile(), acquired_at='2026-01-01T00:00:00Z',
                                availability_mode='assumed_latency', latency_minutes=15)
    parts = partition_observation_groups(records, seed=4, validation_fraction=.2, test_fraction=.2)
    occupied = [name for name, rows in parts.items() if rows]
    assert len(occupied) == 1
    assert len(parts[occupied[0]]) == len(records)
