"""Desroziers observation-error diagnostics and table rewrite (R11)."""
import json

import numpy as np
import pytest

from global_weather.assimilation import calibration
from global_weather.assimilation.tokens import PLATFORMS, VARIABLES


def optimal_cycle(n, sigma_o, sigma_b, *, platform='radiosonde', variable='temperature', pressure=50000., seed=0):
    """Scalar BLUE analyses with known errors: Desroziers relations hold in expectation."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(250., 5., n)
    background = truth + rng.normal(0., sigma_b, n)
    observation = truth + rng.normal(0., sigma_o, n)
    gain = sigma_b ** 2 / (sigma_b ** 2 + sigma_o ** 2)
    analysis = background + gain * (observation - background)
    return {'platform': np.full(n, PLATFORMS.index(platform), np.int16),
            'variable': np.full(n, VARIABLES.index(variable), np.int16), 'channel': np.full(n, -1, np.int32),
            'pressure': np.full(n, pressure), 'value': observation, 'sigma': np.full(n, 1.0),
            'hxa': analysis, 'hxb': background, 'issue': np.zeros(n)}


def merge(*parts):
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def test_desroziers_recovers_known_observation_and_background_errors():
    rows = calibration.diagnose(optimal_cycle(200_000, 0.8, 1.2))
    (row,) = rows
    assert row['sigma_o'] == pytest.approx(0.8, rel=0.02)
    assert row['sigma_b'] == pytest.approx(1.2, rel=0.02)
    assert row['consistency'] == pytest.approx(1.0, abs=0.02)
    assert row['pressure_hpa'] == 500.0 and row['enough']


def test_copying_inputs_is_detected_by_the_fit_ratio():
    data = optimal_cycle(10_000, 0.8, 1.2)
    data['hxa'] = data['value'].copy()  # an analysis that reproduces its inputs exactly
    (row,) = calibration.diagnose(data)
    assert row['sigma_o'] is None and row['fit_ratio'] == 0.0


def test_table_rewrite_keeps_unusable_bands_and_bounds_changes():
    payload = {'schema': 'observation-errors-1', 'basis': 'test', 'platforms': {
        'radiosonde': {'temperature': {'pressure_hpa': [1000, 500, 100], 'sigma': [1.0, 1.0, 1.0]}},
        'surface_station': {'station_temperature': {'sigma': 2.0}},
        'satellite_radiance': {'brightness_temperature': {'from_record': True}}}}
    data = merge(optimal_cycle(50_000, 0.8, 1.2, pressure=50000.),
                 optimal_cycle(50_000, 0.1, 1.2, pressure=10000., seed=1),   # below the floor ratio: kept
                 optimal_cycle(50, 0.6, 1.0, pressure=100000., seed=2),      # too few values: kept
                 optimal_cycle(50_000, 1.5, 1.0, platform='surface_station', variable='station_temperature',
                               pressure=np.nan, seed=3))
    data['sigma'][data['platform'] == PLATFORMS.index('surface_station')] = 2.0
    rows = calibration.diagnose(data, minimum_count=200)
    table, applied = calibration.calibrated_table(payload, rows)
    sonde = table['platforms']['radiosonde']['temperature']
    levels = dict(zip(sonde['pressure_hpa'], sonde['sigma']))
    assert levels[500.0] == pytest.approx(0.8, rel=0.03)
    assert levels[100] == 1.0 and levels[1000] == 1.0
    assert list(sonde['pressure_hpa']) == sorted(sonde['pressure_hpa'], reverse=True)
    assert table['platforms']['surface_station']['station_temperature']['sigma'] == pytest.approx(1.5, rel=0.03)
    assert table['platforms']['satellite_radiance']['brightness_temperature'] == {'from_record': True}
    assert {a['platform'] for a in applied} == {'radiosonde', 'surface_station'}
    from global_weather.assimilation.tokens import ErrorModel
    ErrorModel(json.loads(json.dumps(table)))  # the rewritten table is admissible


def test_cycled_departures_on_the_synthetic_program_check():
    from global_weather.assimilation.selftest import SELFTEST_CONFIG
    from global_weather.assimilation.training import Experiment, load_config
    experiment = Experiment(load_config(SELFTEST_CONFIG), synthetic=True)
    model = experiment.build_model()
    data = calibration.departures(model, experiment, issues=2)
    assert len(data['value']) > 0 and np.isfinite(data['hxa']).all() and np.isfinite(data['hxb']).all()
    rows = calibration.diagnose(data, minimum_count=1)
    assert rows and all(r['omb_rms'] >= 0 for r in rows)
