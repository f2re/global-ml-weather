"""Monthly causality, complete inventories and paired scalar metrics."""
from datetime import datetime, timedelta, timezone
import json
import math
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import global_weather.joint_monthly_verification as module


def test_full_month_and_actual_pressure_time_reporting():
    dates = module.monthly_issues()
    assert len(dates) == 30
    assert dates[0] == datetime(2022, 9, 1, 6, tzinfo=timezone.utc)
    assert dates[-1]+timedelta(hours=72) == datetime(2022, 10, 3, 6, tzinfo=timezone.utc)
    record = {'observed_at': (dates[0]+timedelta(hours=1.5)).isoformat(), 'pressure_pa': 29900.}
    lead, level = module.reporting_bins(dates[0], record)
    assert lead == 1
    assert module.PRESSURE_HPA[level] == 300


def test_paired_controls_bias_rmse_and_aggregation_are_not_mean_rmse():
    cell = np.zeros((5, 4))
    module.accumulate(cell, 12., 10., persistence=11., climatology=None)
    module.accumulate(cell, 8., 10., persistence=None, climatology=10.)
    assert module.metrics(cell[0]) == {'mae': 2., 'rmse': 2., 'bias': 0., 'count': 2}
    assert module.metrics(cell[1])['count'] == module.metrics(cell[2])['count'] == 1
    assert module.metrics(cell[3])['count'] == module.metrics(cell[4])['count'] == 1
    assert module.metrics(np.zeros(4)) == {'mae': None, 'rmse': None, 'bias': None, 'count': 0}
    other = np.zeros((5, 4)); module.accumulate(other, 14., 10.)
    pooled = module.metrics((cell+other)[0])
    assert pooled['rmse'] == pytest.approx(math.sqrt(24/3))
    assert pooled['bias'] == pytest.approx(4/3)
    assert pooled['rmse'] != (2.+4.)/2


def test_complete_empty_inventory_and_undefined_skill():
    profiles = module.empty_accumulator()
    native = module.empty_accumulator(False)
    rows = list(module.inventory_rows(profiles))
    assert len(rows) == 37*25*6
    assert len(list(module.inventory_rows(native, profile=False))) == 25*6
    assert all(row['metrics']['model']['count'] == 0 for row in rows)
    assert all(row['status'] == 'no_admitted_omega_targets' for row in rows if row['variable'] == 'omega')
    module.accumulate(profiles[1, 0, 0], 12., 10., persistence=11., climatology=10.)
    row = next(row for row in module.inventory_rows(profiles) if row['lead_bin_upper_hours'] == 3
               and row['pressure_reporting_bin_hpa'] == 1000 and row['variable'] == 'temperature')
    assert row['skill_rmse']['persistence'] == -1.
    assert row['skill_rmse']['climatology'] is None


def test_sql_targets_are_uncapped_and_native_primary_key_bounds(tmp_path):
    issue = module.monthly_issues()[0]
    profiles = tmp_path/'profiles'; profiles.mkdir()
    with sqlite3.connect(profiles/'observations.sqlite') as database:
        database.execute('CREATE TABLE records(id TEXT PRIMARY KEY, observed TEXT, record TEXT)')
        when = (issue+timedelta(hours=12)).isoformat()
        database.executemany('INSERT INTO records VALUES(?,?,?)',
                             [(str(i), when, json.dumps({'observed_at': when, 'profile_id': str(i//10)})) for i in range(2003)])
    assert sum(len(group) for group in module.iter_target_groups(profiles, issue, chunk_size=7)) == 2003
    native = tmp_path/'native'; native.mkdir()
    with sqlite3.connect(native/'observations.sqlite') as database:
        database.execute('CREATE TABLE observations(id TEXT PRIMARY KEY, record TEXT)')
        for station in ('known', 'unknown'):
            for hour in (0, 1, 72, 73):
                when = (issue+timedelta(hours=hour)).isoformat()
                key = f'GHCNh/{station}/{when}/t2m'
                database.execute('INSERT INTO observations VALUES(?,?)',
                                 (key, json.dumps({'observation_id': key, 'observed_at': when})))
    targets = [row for group in module.iter_native_targets(native, issue, ['known'], chunk_size=1) for row in group]
    assert len(targets) == 2
    assert {row['observed_at'] for row in targets} == {(issue+timedelta(hours=h)).isoformat() for h in (1, 72)}


def test_persistence_is_last_frozen_causal_input_and_native_actual_time():
    issue = module.monthly_issues()[0]
    rows = [{'variable': 'temperature', 'latitude': 1., 'longitude': 2., 'pressure_pa': 29900.,
             'observed_at': (issue-timedelta(hours=hour)).isoformat(), 'value': value} for hour, value in [(4, 270.), (1, 275.)]]
    controls = module.persistence_profiles(rows)
    assert controls[('temperature', 1., 2., 29900.)] == 275.
    assert controls.get(('temperature', 1., 2., 30000.)) is None
    raw = np.arange(12*6).reshape(12, 1, 6); mask = np.ones(raw.shape, bool); mask[-2:, 0, 0] = False
    assert module.persistence_surface(raw, mask)[0, 0] == raw[-3, 0, 0]
    forecast = np.broadcast_to(np.arange(25)[:, None, None], (25, 1, 6))
    assert module._native_prediction(forecast, 1.5, 0, 0) == .5
    assert module._native_prediction(forecast, 73., 0, 0) is None


def test_incomplete_month_blocks_before_target_reads_or_model_load(tmp_path, monkeypatch):
    forecasts = tmp_path/'forecasts'; forecasts.mkdir()
    (forecasts/'month.json').write_text(json.dumps({'schema': module.SCHEMA, 'stage': 'forecast', 'status': 'running'}))
    monkeypatch.setattr(module, 'iter_target_groups', lambda *args, **kwargs: pytest.fail('Targets opened before freeze.'))
    with pytest.raises(ValueError, match='frozen before verification'):
        module.verify_month('unused', 'unused', 'unused', forecasts, tmp_path/'scores')
    assert not (tmp_path/'scores').exists()


def test_forecast_stage_never_samples_targets_and_preserves_preterrain_state(tmp_path, monkeypatch):
    import global_weather.joint_observation_training as training
    import global_weather.joint_static as static
    from global_weather.grid import build_grid
    from global_weather.profile_physics import parse_policy
    issue = module.monthly_issues()[0]; grid = build_grid(0)
    inputs = [{'variable': 'temperature', 'observed_at': issue.isoformat(), 'available_at': issue.isoformat()}]
    class PastOnly:
        surface = SimpleNamespace(latlon=np.array([[1., 2.]]))
        fingerprint = {'data': 'fixed'}
        def forecast_history(self, when, limit):
            assert when == issue and limit == 2000
            return np.ones((12, 1, 6)), np.ones((12, 1, 6), bool), inputs
        def sample(self, *args): pytest.fail('Forecast called target-bearing sample.')
        def verify(self): pass
    state = {'epoch': 12, 'identity': {'static_fingerprint': 'static'},
             'config': {'max_records_per_window': 2000}}
    model = SimpleNamespace(xyz=torch.as_tensor(grid.xyz), grid=grid,
                            frozen_metadata=state, frozen_checkpoint_sha256='c'*64,
                            physical_policy=parse_policy({'humidity_qc': False}))
    profiles = torch.ones((grid.n_cells, 37, 6)); profiles[..., 5] = float('nan')
    frames = [SimpleNamespace(profiles=profiles, profile_variable_mask=torch.isfinite(profiles),
                              native_physical=torch.ones((1, 6)), lead_hours=int(lead)) for lead in module.LEADS]
    monkeypatch.setattr(training, 'load_frozen', lambda *args, **kwargs: (model, PastOnly()))
    monkeypatch.setattr(static, 'load_static', lambda *args: {'static_sha256': 'static'})
    monkeypatch.setattr(module, 'monthly_issues', lambda *args: [issue])
    monkeypatch.setattr(module, 'timed_forecast', lambda *args, **kwargs: (frames, {}))
    monkeypatch.setattr(module, 'save_issue', lambda *args, **kwargs: {'grib_sha256': 'g'*64})
    report = module.forecast_month('unused', 'unused', 'unused', tmp_path/'forecasts')
    assert report['status'] == 'complete' and report['future_targets_read'] is False
    folder = tmp_path/'forecasts'/issue.strftime('%Y%m%dT%H%M%SZ')
    with np.load(folder/'verification-state.npz', allow_pickle=False) as archive:
        assert np.isfinite(archive['profiles'][..., :5]).all()
        assert archive['profile_norm_mask'][..., :5].all()


def test_resource_budget_and_checkpoint_pin(tmp_path, monkeypatch):
    budget = module.Budget(seconds=10, maximum_bytes=2)
    (tmp_path/'large').write_bytes(b'123')
    with pytest.raises(RuntimeError, match='disk budget'): budget.check(tmp_path, size=True)
    budget.deadline = 0
    with pytest.raises(TimeoutError): budget.check(tmp_path)
    state = {'epoch': 12, 'config': {'max_records_per_window': 2000}}
    with pytest.raises(ValueError, match='checkpoint'):
        module._bind_config(state, 'a'*64, {'checkpoint_epoch': 12, 'checkpoint_sha256': 'b'*64, 'input_profile_limit': 2000})


def test_two_stages_raw_error_pairs_and_complete_aggregate(tmp_path, monkeypatch):
    import gzip
    import global_weather.joint_observation_training as training
    import global_weather.joint_static as static
    from global_weather.grid import build_grid
    from global_weather.profile_physics import parse_policy
    issue = module.monthly_issues()[0]; when = issue+timedelta(hours=3)
    grid = build_grid(0)
    profile_root, native_root = tmp_path/'profiles', tmp_path/'native'
    profile_root.mkdir(); native_root.mkdir()
    old = {'observation_id': 'past', 'profile_id': 'past', 'provider': 'IGRA', 'variable': 'temperature',
           'observed_at': issue.isoformat(), 'available_at': issue.isoformat(), 'units': 'K',
           'latitude': 0., 'longitude': 0., 'pressure_pa': 95000., 'value': 279.}
    target = {**old, 'observation_id': 'target', 'profile_id': 'target',
              'observed_at': when.isoformat(), 'value': 285.}
    with sqlite3.connect(profile_root/'observations.sqlite') as database:
        database.execute('CREATE TABLE records(id TEXT PRIMARY KEY, observed TEXT, record TEXT)')
        database.execute('INSERT INTO records VALUES(?,?,?)', ('target', when.isoformat(), json.dumps(target)))
    key = 'GHCNh/known/'+when.isoformat()+'/t2m'
    native = {'observation_id': key, 'variable': 't2m', 'observed_at': when.isoformat(),
              'available_at': (when+timedelta(hours=1)).isoformat(), 'latitude': 0., 'longitude': 0.,
              'value': 285., 'units': 'K', 'valid': True, 'provider': 'NOAA_GHCNh',
              'provider_qc': {'measured': {'Quality_Code': '', 'Source_Code': '', 'Measurement_Code': ''}}}
    with sqlite3.connect(native_root/'observations.sqlite') as database:
        database.execute('CREATE TABLE observations(id TEXT PRIMARY KEY, record TEXT)')
        database.execute('INSERT INTO observations VALUES(?,?)', (key, json.dumps(native)))
    policy = parse_policy({'architecture': 'hydrostatic_flow', 'humidity_min_pressure_pa': 30000.})
    config = {'max_records_per_window': 2000, 'physical_policy': policy.payload()}
    state = {'epoch': 12, 'identity': {'static_fingerprint': 'static', 'config': config},
             'config': config, 'trained_variable_mask': np.ones((37, 5), bool).tolist()}
    class Joint:
        surface = SimpleNamespace(root=native_root, latlon=np.array([[0., 0.]], np.float32),
                    manifest={'stations': [{'id': 'known'}]}, mask=np.ones((8760, 1, 6), bool))
        profile = SimpleNamespace(root=profile_root)
        fingerprint = {'source': 'fixed'}
        def forecast_history(self, when, limit):
            return np.full((12, 1, 6), 279.), np.ones((12, 1, 6), bool), [old]
        def sample(self, *args): pytest.fail('Forecast opened target-bearing sample.')
        def verify(self): pass
    model = SimpleNamespace(grid=grid, xyz=torch.as_tensor(grid.xyz), frozen_metadata=state,
              frozen_checkpoint_sha256='c'*64, physical_policy=policy,
              pressure_pa=torch.tensor(module.PRESSURE_HPA)*100,
              surface_monthly=torch.full((12, 1, 6), 275.),
              climatology=SimpleNamespace(sample=lambda when: (np.full((grid.n_cells, 37, 5), 270.),
                                                                np.ones((grid.n_cells, 37, 5), bool))))
    profiles = torch.full((grid.n_cells, 37, 6), 280.)
    # An issuance mask could hide these low-geopotential predictions; raw
    # verification must still score admitted observations independently.
    profiles[..., 4] = -10000.; profiles[..., 5] = float('nan')
    frames = [SimpleNamespace(profiles=profiles, profile_variable_mask=torch.isfinite(profiles),
                              native_physical=torch.full((1, 6), 280.), lead_hours=int(lead)) for lead in module.LEADS]
    monkeypatch.setattr(training, 'load_frozen', lambda *args, **kwargs: (model, Joint()))
    monkeypatch.setattr(static, 'load_static', lambda *args: {'static_sha256': 'static'})
    monkeypatch.setattr(module, 'monthly_issues', lambda *args: [issue])
    monkeypatch.setattr(module, 'timed_forecast', lambda *args, **kwargs: (frames, {}))
    monkeypatch.setattr(module, 'save_issue', lambda *args, **kwargs: {'grib_sha256': 'g'*64})
    forecasts = tmp_path/'forecasts'; output = tmp_path/'scores'
    module.forecast_month('unused', 'unused', 'unused', forecasts)
    report = module.verify_month('unused', 'unused', 'unused', forecasts, output)
    assert report['status'] == 'complete'
    assert report['evaluation_qc_policy']['humidity_min_pressure_pa'] == 0.
    day = report['days'][0]
    assert day['counts']['profile_accepted'] == day['counts']['native_accepted'] == 1
    with gzip.open(output/day['pairs'], 'rt') as stream:
        pairs = [json.loads(line) for line in stream]
    assert len(pairs) == 2
    assert all(row['predicted'] == pytest.approx(280., abs=1e-3) and row['observed'] == 285. for row in pairs)
    rows = [json.loads(line) for line in (output/'aggregate.jsonl').read_text().splitlines()]
    assert len(rows) == 37*25*6+25*6
    scored = [row for row in rows if row['metrics']['model']['count']]
    assert len(scored) == 2
    assert all(row['metrics']['model']['bias'] == pytest.approx(-5., abs=1e-3) for row in scored)
    assert all(row['metrics']['persistence']['count'] == row['metrics']['persistence_model_paired']['count'] == 1 for row in scored)
