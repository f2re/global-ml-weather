"""Radiosonde drift reconstruction from the sounding's own winds and times (R11)."""
from datetime import timedelta
import json
import sqlite3

import numpy as np
import pytest

from global_weather.assimilation import drift
from global_weather.assimilation.sources import IgraSource, seconds
from global_weather.observation_data import START

LEVELS = np.array([1000., 925., 850., 700., 500., 400., 300., 250., 200., 150., 100.]) * 100.
G0 = 9.80665


def sounding(*, u=10.0, v=0.0, elapsed=True, launch=1_000_000.0, heights=None, temperature=True, winds=True):
    """Rows for one launch; heights default to a 250 K isothermal atmosphere."""
    z = 287.05 * 250 / G0 * np.log(LEVELS[0] / LEVELS) if heights is None else heights
    t_level = z / drift.ASCENT_RATE_M_S
    rows = {k: [] for k in ('group', 'pressure', 'variable', 'value', 'observed', 'available', 'latitude',
                            'longitude', 'time_basis', 'position_basis', 'launch')}
    names = (['temperature'] if temperature else []) + ['geopotential'] + (['u', 'v'] if winds else [])
    for k, p in enumerate(LEVELS):
        for name in names:
            value = {'temperature': 250.0, 'geopotential': G0 * z[k], 'u': u, 'v': v}[name]
            when = launch + (t_level[k] if elapsed else 0.0)
            for key, item in (('group', 'L1'), ('pressure', p), ('variable', name), ('value', value),
                              ('observed', when), ('available', when + 7200.), ('latitude', 0.0),
                              ('longitude', 0.0),
                              ('time_basis', 'launch_plus_elapsed_seconds' if elapsed else 'launch_time_fallback'),
                              ('position_basis', 'launch_position_fallback'), ('launch', launch)):
                rows[key].append(item)
    out = {k: np.array(v, dtype=object if k in ('group', 'variable') else None) for k, v in rows.items()}
    out['time_basis'] = list(rows['time_basis'])
    out['position_basis'] = list(rows['position_basis'])
    for key in ('pressure', 'value', 'observed', 'available', 'latitude', 'longitude', 'launch'):
        out[key] = out[key].astype(float)
    return out, z, t_level


def test_constant_wind_drift_matches_analytic_track_with_reported_times():
    rows, z, t_level = sounding(u=10.0, v=-4.0)
    lat, lon, observed, available, basis = drift.reconstruct(**rows)
    level = np.searchsorted(-LEVELS, -rows['pressure'])
    east, north = 10.0 * t_level[level], -4.0 * t_level[level]
    expected_lat = np.degrees(north / drift.EARTH_RADIUS_M)
    expected_lon = np.degrees(east / (drift.EARTH_RADIUS_M * np.cos(np.radians(0.5 * expected_lat))))
    assert np.allclose(lat, expected_lat, atol=1e-9) and np.allclose(lon, expected_lon, atol=1e-9)
    assert np.array_equal(observed, rows['observed']) and np.array_equal(available, rows['available'])
    assert set(basis) == {drift.DRIFT_REPORTED_TIME}
    top = drift.great_circle_km(0, 0, lat[level == len(LEVELS) - 1][0], lon[level == len(LEVELS) - 1][0])
    assert top == pytest.approx(np.hypot(10, 4) * t_level[-1] / 1000, rel=1e-3)


def test_launch_time_fallback_estimates_level_time_and_moves_availability_with_it():
    rows, z, t_level = sounding(elapsed=False)
    lat, lon, observed, available, basis = drift.reconstruct(**rows)
    level = np.searchsorted(-LEVELS, -rows['pressure'])
    assert np.allclose(observed - rows['observed'], t_level[level], atol=1e-6)
    assert np.allclose(available - observed, 7200.0, atol=1e-6)
    assert np.all(available >= observed)
    assert set(basis) == {drift.DRIFT_ESTIMATED_TIME}


def test_nominal_time_keeps_time_but_still_drifts():
    rows, z, t_level = sounding(elapsed=False)
    rows['launch'] = np.full(len(rows['launch']), np.nan)
    rows['time_basis'] = ['nominal_time_fallback'] * len(rows['launch'])
    lat, lon, observed, available, _ = drift.reconstruct(**rows)
    assert np.array_equal(observed, rows['observed'])
    assert lon.max() == pytest.approx(np.degrees(10.0 * t_level[-1] / drift.EARTH_RADIUS_M), rel=1e-9)


def test_heights_follow_temperature_when_geopotential_is_missing():
    rows, z, t_level = sounding(elapsed=False)
    keep = rows['variable'] != 'geopotential'
    reduced = {k: (np.asarray(v)[keep] if not isinstance(v, list) else [x for x, m in zip(v, keep) if m])
               for k, v in rows.items()}
    _, _, observed, _, basis = drift.reconstruct(**reduced)
    level = np.searchsorted(-LEVELS, -reduced['pressure'])
    # Hypsometric heights from the measured 250 K reproduce the isothermal ascent times.
    assert np.allclose(observed - reduced['observed'], t_level[level], rtol=1e-6, atol=1e-6)
    assert set(basis) == {drift.DRIFT_ESTIMATED_TIME}


def test_no_wind_keeps_launch_position_and_reported_level_positions_are_respected():
    rows, _, _ = sounding(winds=False)
    lat, lon, *_, basis = drift.reconstruct(**rows)
    assert np.all(lat == 0) and np.all(lon == 0) and set(basis) == {drift.LAUNCH}
    rows, _, _ = sounding()
    rows['position_basis'] = ['reported_level_position'] * len(rows['launch'])
    rows['latitude'] = np.linspace(0, 1, len(rows['launch']))
    lat, _, *_, basis = drift.reconstruct(**rows)
    assert np.array_equal(lat, rows['latitude']) and set(basis) == {drift.REPORTED}


def test_lower_levels_do_not_depend_on_unavailable_upper_levels():
    rows, _, _ = sounding(u=25.0)
    full = drift.reconstruct(**rows)
    low = rows['pressure'] >= 50000.
    cut = {k: (np.asarray(v)[low] if not isinstance(v, list) else [x for x, m in zip(v, low) if m])
           for k, v in rows.items()}
    partial = drift.reconstruct(**cut)
    assert np.allclose(partial[1], full[1][low], atol=1e-12)


def test_igra_window_applies_drift_and_records_the_position_model(tmp_path):
    source = IgraSource.__new__(IgraSource)
    source.root, source.drift = tmp_path, True
    launch = START + timedelta(hours=60)
    with sqlite3.connect(tmp_path / 'observations.sqlite') as database:
        database.execute('CREATE TABLE records(id TEXT PRIMARY KEY, observed TEXT, available TEXT, variable INTEGER, '
                         'value REAL, revision INTEGER, split TEXT, record TEXT)')
        for k, p in enumerate(LEVELS):
            observed = launch  # launch-time fallback: every level stamped at launch
            z = 287.05 * 250 / G0 * np.log(LEVELS[0] / p)
            for variable, value in ((0, 250.), (2, 20.), (3, 0.), (4, G0 * z)):
                record = {'pressure_pa': p, 'latitude': 45., 'longitude': 179.9, 'profile_id': 'L',
                          'observation_error': None, 'time_basis': 'launch_time_fallback',
                          'position_basis': 'launch_position_fallback', 'launch_time': launch.isoformat()}
                database.execute('INSERT INTO records VALUES (?,?,?,?,?,0,?,?)',
                                 (f'{k}-{variable}', observed.isoformat(),
                                  (observed + timedelta(minutes=120)).isoformat(), variable, value, 'train',
                                  json.dumps(record)))
    end = seconds(launch) + 6 * 3600
    batch = source.window(seconds(launch) - 3600, end, split='train').validate()
    assert len(batch) == 4 * len(LEVELS)
    assert batch.observed.max() > seconds(launch) + 3000  # upper levels later than launch
    assert np.all(batch.available >= batch.observed)
    assert np.all(np.abs(batch.longitude) <= 180) and (batch.longitude < 0).any()  # crossed the date line
    # A window that ends 30 minutes after launch keeps only the levels measured by then.
    early = source.window(seconds(launch) - 3600, seconds(launch) + 1800, split='train')
    assert 0 < len(early) < len(batch)
    assert early.observed.max() <= seconds(launch) + 1800
