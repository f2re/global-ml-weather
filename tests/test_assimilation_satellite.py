"""Physical admission of satellite channels and the weighting-function operator (R11)."""
import copy
import json

import numpy as np
import pytest
import torch

from global_weather.assimilation import satellite
from global_weather.assimilation.operator import build_plan, observe
from global_weather.assimilation.sources import prepare_store, TokenStore
from global_weather.assimilation.tokens import PLATFORMS, VARIABLES, TokenBatch, vertical_slots
from global_weather.vertical import PRESSURE_HPA

from test_assimilation_model import ISSUE_S, analytic_rollout, model

PASSPORT = 'ab' * 32


def weighting(peak_hpa=500., width=0.35, surface=0.0):
    log_p = np.log(np.asarray(PRESSURE_HPA, float))
    w = np.exp(-0.5 * ((log_p - np.log(peak_hpa)) / width) ** 2)
    w = (1 - surface) * w / w.sum()
    return w


def entry(name='AMSU-A/N19:5', **change):
    base = {'channel': name, 'quantity': 'brightness_temperature', 'units': 'K', 'central_frequency_ghz': 53.596,
            'passport_sha256': PASSPORT, 'noise_equivalent_dt_k': 0.25, 'observation_error_k': 0.35,
            'roles': ['input', 'target'], 'clear_sky_screened': True,
            'weighting_function': {'pressure_hpa': [float(p) for p in PRESSURE_HPA], 'weights': weighting().tolist(),
                                   'surface_weight': 0.0, 'source': 'RTTOV 13.2 nadir, US standard atmosphere'}}
    for key, value in change.items():
        if key.startswith('wf_'):
            base['weighting_function'][key[3:]] = value
        else:
            base[key] = value
    return base


def write(tmp_path, *entries, minimum=10):
    path = tmp_path / 'admission.json'
    path.write_text(json.dumps({'schema': satellite.SCHEMA, 'minimum_train_count': minimum, 'channels': list(entries)}))
    return path


def test_valid_admission_and_encoder_slots_follow_the_weighting_function(tmp_path):
    admission = satellite.load(write(tmp_path, entry(), entry('AMSU-A/N19:9', **{'wf_weights': weighting(100.).tolist()})))
    tables = satellite.channel_tables(admission, ['AMSU-A/N19:5', 'AMSU-A/N19:9', 'other:1'])
    assert tables['physical'].tolist() == [False, True, True, False]
    assert tables['target'].tolist() == [False, True, True, False]
    centres = np.exp(np.linspace(np.log(100000.), np.log(100.), 8)) / 100.
    assert centres[np.argmax(tables['slots'][1, :8])] == pytest.approx(500., rel=0.35)
    assert np.argmax(tables['slots'][2, :8]) > np.argmax(tables['slots'][1, :8])  # 100 hPa channel sits higher
    batch = TokenBatch(platform=np.full(2, PLATFORMS.index('satellite_radiance'), np.int16),
                       variable=np.full(2, VARIABLES.index('brightness_temperature'), np.int16),
                       channel=np.array([0, 2], np.int32), group=np.arange(2), identity=np.arange(2),
                       observed=np.zeros(2), available=np.zeros(2), latitude=np.zeros(2), longitude=np.zeros(2),
                       pressure_pa=np.full(2, np.nan), height_m=np.full(2, np.nan), value=np.full(2, 250.),
                       error=np.full(2, .3), quality=np.ones(2, np.float32))
    slots = vertical_slots(batch, channel_slots=tables['slots'])
    assert slots[0, :8].sum() == pytest.approx(1.0) and slots[0, 10] == 0
    assert slots[1, 10] == 1.0  # not admitted: generic column slot


@pytest.mark.parametrize('change, message', [
    ({'units': 'counts'}, 'brightness temperature in K'),
    ({'central_frequency_ghz': None}, 'frequency or wavenumber'),
    ({'observation_error_k': 0.1}, 'not below NEdT'),
    ({'wf_pressure_hpa': [1000., 500.]}, '37 model levels'),
    ({'wf_weights': (weighting() * 0.5).tolist()}, 'sum to 1'),
    ({'wf_weights': weighting(surface=0.3).tolist(), 'wf_surface_weight': 0.3}, 'surface-sensitive'),
    ({'clear_sky_screened': False}, 'clear-sky'),
    ({'passport_sha256': 'X' * 64}, 'SHA256'),
    ({'wf_source': ''}, 'radiative-transfer'),
])
def test_admission_refuses_physically_incomplete_channels(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        satellite.load(write(tmp_path, entry(**change)))


def test_surface_sensitive_channel_may_still_enter_as_input_only(tmp_path):
    admission = satellite.load(write(tmp_path, entry(roles=['input'], clear_sky_screened=False,
                                                     wf_weights=weighting(surface=0.4).tolist(), wf_surface_weight=0.4)))
    tables = satellite.channel_tables(admission, ['AMSU-A/N19:5'])
    assert tables['physical'][1] and not tables['target'][1]
    assert tables['slots'][1, 8] == pytest.approx(0.4)


def test_store_passport_and_train_count_are_checked(tmp_path):
    rows = []
    for i in range(30):
        observed = f'2021-02-{1 + i % 27:02d}T00:00:00+00:00'
        rows.append({'observation_id': f'o{i}', 'granule_id': 'g', 'variable': 'brightness_temperature', 'units': 'K',
                     'observed_at': observed, 'available_at': observed, 'latitude': 10., 'longitude': 20.,
                     'value': 240. + i % 5, 'instrument': 'AMSU-A/N19', 'channel': 5, 'observation_error': 0.35,
                     'zenith_angle_deg': 30., 'radiometric_admission': {'calibrated': True, 'passport_sha256': PASSPORT}})
    source = tmp_path / 'tb.jsonl'
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    bounds = {'train': ('2021-01-01T00:00:00+00:00', '2022-01-01T00:00:00+00:00')}
    manifest = prepare_store(source, 'satellite_radiance', tmp_path / 'store', train_end='2022-01-01T00:00:00+00:00',
                             split_bounds=bounds)
    assert manifest['channel_passports'] == {'AMSU-A/N19:5': [PASSPORT]}
    store = TokenStore(tmp_path / 'store')
    admission = satellite.load(write(tmp_path, entry()))
    report = satellite.check_against_stores(admission, [store], store.statistics())
    assert report['AMSU-A/N19:5']['train_count'] == 30
    wrong = copy.deepcopy(admission)
    wrong['channels']['AMSU-A/N19:5']['passport_sha256'] = 'cd' * 32
    with pytest.raises(ValueError, match='passports differ'):
        satellite.check_against_stores(wrong, [store], store.statistics())
    strict = dict(admission, minimum_train_count=1000)
    with pytest.raises(ValueError, match='too few train values'):
        satellite.check_against_stores(strict, [store], store.statistics())
    missing = satellite.load(write(tmp_path, entry('AMSU-A/N19:7')))
    with pytest.raises(ValueError, match='not present'):
        satellite.check_against_stores(missing, [store], store.statistics())


def test_weighting_function_operator_reproduces_layer_temperature_and_learns_bias():
    weights = np.zeros(37)
    k = list(PRESSURE_HPA).index(500)
    weights[k] = 1.0
    tables = {'weights': np.stack([np.zeros(37), weights]), 'surface': np.zeros(2), 'physical': np.array([False, True])}
    net = model(channels=1, channel_tables=tables)
    rollout = analytic_rollout(net, temperature=lambda p, f: 200. + 10. * torch.log(p / 100.))
    batch = TokenBatch(platform=np.array([PLATFORMS.index('satellite_radiance')], np.int16),
                       variable=np.array([VARIABLES.index('brightness_temperature')], np.int16),
                       channel=np.array([0], np.int32), group=np.array([1]), identity=np.array([1]),
                       observed=np.array([ISSUE_S + 3 * 3600.]), available=np.array([ISSUE_S + 3 * 3600.]),
                       latitude=np.array([30.]), longitude=np.array([40.]), pressure_pa=np.array([np.nan]),
                       height_m=np.array([np.nan]), value=np.array([250.]), error=np.array([0.35]),
                       quality=np.ones(1, np.float32), zenith_cos=np.array([1.0], np.float32))
    stats = {'satellite_radiance|brightness_temperature|c': {'mean': 250., 'std': 10.}}
    plan = build_plan(batch, np.array([0.35]), ISSUE_S, 3, net.mesh, analysis=np.array([False]),
                      column_statistics=stats, channels=['c'])
    value = observe(net, rollout, plan)
    assert float(value[0]) == pytest.approx(200. + 10. * np.log(500.), abs=1e-3)
    value.sum().backward()
    assert net.channel_bias.weight.grad[1].abs().sum() > 0
    assert net.channel_predictors.weight.grad[1].abs().sum() > 0  # thickness predictors are non-zero
    assert net.column_operator[0].weight.grad is None  # the free network is not used for admitted channels


def test_satellite_targets_require_an_admission_file():
    from global_weather.assimilation.training import load_config, CONFIG_SCHEMA
    with pytest.raises(ValueError, match='admission'):
        load_config({'schema': CONFIG_SCHEMA, 'data': {'satellite_supervision': True}})


def test_shipped_admission_template_is_valid_and_contains_no_channels():
    from pathlib import Path
    payload = json.loads((Path(__file__).resolve().parents[1] / 'configs/satellite_channels_r11.json').read_text())
    assert payload['schema'] == satellite.SCHEMA and payload['channels'] == []
    assert 'example' in payload and payload['example']['channel'].count(':') == 1
    path = Path(__file__).resolve().parents[1] / 'configs/satellite_channels_r11.json'
    assert satellite.load(path)['channels'] == {}
