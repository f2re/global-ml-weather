"""R11 adapters for the admitted GHCNh and IGRA archives (structure and causality only)."""
from datetime import timedelta
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from global_weather.assimilation.sources import GhcnhSource, IgraSource, seconds
from global_weather.assimilation.tokens import VARIABLES
from global_weather.observation_data import START


def ghcnh(tmp_path, elevation=(120.0, np.nan)):
    source = GhcnhSource.__new__(GhcnhSource)
    hours = 48
    rng = np.random.default_rng(3)
    values = rng.normal(0., 1., (hours, 2, 6)).astype(np.float32)
    values[..., 0] += 280.
    values[..., 4] = 99000. + 50. * values[..., 4]
    mask = np.zeros_like(values, bool)
    mask[10:20, 0, :] = True
    mask[10:20, 1, 0] = True
    mask[10:20, 1, 4] = True
    available = np.tile(np.arange(hours)[:, None, None] * 3600 + 1800, (1, 2, 6)).astype(np.int64)
    source.dataset = SimpleNamespace(values=values, mask=mask, available=available,
                                     latlon=np.array([[60., 30.], [-10., 100.]], np.float32))
    source.root = tmp_path
    source.start_seconds = seconds(START)
    source.train_hours = hours
    source.station_ids = ['A', 'B']
    source.groups = np.array([11, 22], np.int64)
    source.elevation = np.array(elevation)
    return source


def test_ghcnh_window_uses_trailing_hour_and_availability(tmp_path):
    source = ghcnh(tmp_path)
    issue = source.start_seconds + 15 * 3600
    batch = source.window(issue - 12 * 3600, issue, split='train', available_before=issue).validate()
    # Hour 15 ends at the issue but becomes available 30 minutes later: excluded.
    assert batch.observed.max() == issue - 3600
    assert np.all(batch.available <= issue)
    future = source.window(issue, issue + 3 * 3600, split='train')
    assert future.observed.min() == issue + 3600
    station_b = batch.select(batch.group == 22)
    assert np.isnan(station_b.height_m).all()
    assert set(VARIABLES[v] for v in station_b.variable) == {'station_temperature', 'station_pressure'}


def test_ghcnh_pressure_statistics_use_height_reduction_and_known_elevations_only(tmp_path):
    stats = ghcnh(tmp_path).statistics()
    pressure = stats['surface_station|station_pressure|']
    assert pressure['transform'] == 'height_reduced_log'
    assert pressure['count'] == 10  # station B has no elevation and is excluded
    source = ghcnh(tmp_path)
    expected = np.log(source.dataset.values[10:20, 0, 4].astype(float)) + 120. / 8000.
    assert pressure['mean'] == pytest.approx(expected.mean(), rel=1e-9)


def test_ghcnh_elevation_read_from_records_and_inconsistency_refused(tmp_path):
    source = GhcnhSource.__new__(GhcnhSource)
    source.root = tmp_path
    source.station_ids = ['A', 'B', 'C']
    with sqlite3.connect(tmp_path / 'observations.sqlite') as database:
        database.execute('CREATE TABLE observations (id TEXT PRIMARY KEY, revision INTEGER, record TEXT, hash TEXT)')
        rows = [('GHCNh/A/1/t2m', {'elevation_m': 120.}), ('GHCNh/A/2/t2m', {'elevation_m': 120.}),
                ('GHCNh/B/1/t2m', {}), ('GHCNh/C/1/t2m', {'elevation_m': 5.}), ('GHCNh/C/2/t2m', {'elevation_m': 7.})]
        database.executemany('INSERT INTO observations VALUES (?,0,?,?)', [(k, json.dumps(r), 'h') for k, r in rows])
    elevation = source._elevations()
    assert elevation[0] == 120. and np.isnan(elevation[1]) and np.isnan(elevation[2])


def igra(tmp_path, launches=3, levels=4):
    source = IgraSource.__new__(IgraSource)
    source.root = tmp_path
    with sqlite3.connect(tmp_path / 'observations.sqlite') as database:
        database.execute('CREATE TABLE records(id TEXT PRIMARY KEY, observed TEXT, available TEXT, variable INTEGER, '
                         'value REAL, revision INTEGER, split TEXT, record TEXT)')
        for launch in range(launches):
            launch_time = START + timedelta(hours=60 + launch)
            for level in range(levels):
                observed = launch_time + timedelta(minutes=10 * level)
                record = {'pressure_pa': 90000. - 10000 * level, 'latitude': 50., 'longitude': 10. + launch,
                          'profile_id': f'launch-{launch}', 'observation_error': None}
                database.execute('INSERT INTO records VALUES (?,?,?,?,?,0,?,?)',
                                 (f'id-{launch}-{level}', observed.isoformat(),
                                  (observed + timedelta(minutes=30)).isoformat(), 0, 270. - level, 'train',
                                  json.dumps(record)))
    return source


def test_igra_window_keeps_actual_pressure_time_and_samples_whole_launches(tmp_path):
    source = igra(tmp_path)
    begin, end = seconds(START + timedelta(hours=59)), seconds(START + timedelta(hours=64))
    batch = source.window(begin, end, split='train').validate()
    assert len(batch) == 12 and set(np.unique(batch.pressure_pa)) == {90000., 80000., 70000., 60000.}
    sampled = source.window(begin, end, split='train', limit=5, rng=np.random.default_rng(0))
    sizes = np.unique(sampled.group, return_counts=True)[1]
    assert np.all(sizes == 4)  # never a partial sounding
    cut = seconds(START + timedelta(hours=61, minutes=40))
    causal = source.window(begin, cut, split='train', available_before=cut)
    assert np.all(causal.available <= cut) and len(causal) < 12
    assert source.window(begin, end, split='validation').value.size == 0
