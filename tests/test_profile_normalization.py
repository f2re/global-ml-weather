"""Synthetic software fixtures; no actual observation or weather skill evidence."""
from datetime import timedelta
import hashlib
import json

import numpy as np
import pytest

from global_weather.profile_normalization import PressureNormalization, fit
from global_weather.profile_training import START, TRAIN_END, VARIABLES, prepare
from global_weather.vertical import PROFILE_UNITS


def dataset(tmp_path, validation_shift=0., single=False, outside=False):
    rows = []
    for split, beginning, shift in (('train', START, 0.), ('validation', TRAIN_END, validation_shift)):
        pressures = (70000., 75000., 100001., 99.) if outside else (70000., 75000.)
        for level, pressure in enumerate(pressures):
            for sample in range(1 if single and level == 1 else 2):
                when = beginning + timedelta(days=3 + sample, hours=level)
                values = [250. + level * 20 + sample * 2 + shift,
                          .001 + level * .001 + sample * .0002,
                          level * 3 + sample * 2., -level * 3 + sample * 2.,
                          30000. - level * 1000 + sample * 100.]
                for i, variable in enumerate(VARIABLES):
                    rows.append(dict(source='radiosonde', provider='NOAA_IGRA2', valid=True,
                                     observation_id=f'fixture/{when.isoformat()}/{variable}',
                                     profile_id=f'fixture/{when.isoformat()}', observed_at=when.isoformat(),
                                     available_at=(when + timedelta(hours=2)).isoformat(),
                                     latitude=10., longitude=20., pressure_pa=pressure,
                                     variable=variable, value=values[i], units=PROFILE_UNITS[i],
                                     group_split=split, revision=0,
                                     provider_qc={'software_fixture': True},
                                     archive_sha256='0' * 64, format_sha256='1' * 64,
                                     time_basis='reported_level_time',
                                     position_basis='reported_level_position'))
    source = tmp_path / 'records.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    admission = dict(schema='igra-observation-archive-1', provider='NOAA_IGRA2', data_kind='real',
                     observations_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                     software_fixture=True)
    (tmp_path / 'manifest.json').write_text(json.dumps(admission))
    root = tmp_path / 'dataset'
    prepare(source, root)
    return root


def test_train_only_level_statistics_and_immutable_reuse(tmp_path):
    first = tmp_path / 'first'; first.mkdir()
    second = tmp_path / 'second'; second.mkdir()
    a = fit(dataset(first, 0.), first / 'norms.json')
    b = fit(dataset(second, 100.), second / 'norms.json')
    for key in ('mean', 'std', 'support', 'count', 'q_scale'):
        assert a[key] == b[key]
    assert a['count'][11] == [2] * 5
    assert a['mean'][11][0] == 251.
    assert a['mean'][10][0] == 271.
    assert a['std'][11][0] == 1.
    assert a['mean'][0] == [None] * 5
    assert a['support'][0] == [False] * 5
    assert fit(first / 'dataset', first / 'norms.json') == a
    assert 'NaN' not in (first / 'norms.json').read_text()
    assert all(len(value) == 64 for value in a['source_identity'].values())


def test_log_pressure_interpolation_requires_both_neighbours(tmp_path):
    normal = PressureNormalization(fit(dataset(tmp_path), tmp_path / 'norms.json'))
    p = np.sqrt(70000. * 75000.)
    mean, std, supported = normal.at('temperature', p)
    assert supported
    assert mean == pytest.approx(261.)
    assert std == pytest.approx(1.)
    means, stds, support = normal.at(0, [70000., p, 76000., 100001., 99.])
    assert support.tolist() == [True, True, False, False, False]
    assert np.isnan(means[2:]).all() and np.isnan(stds[2:]).all()
    with pytest.raises(ValueError): normal.at(0, 0.)
    with pytest.raises(ValueError): normal.at('omega', 70000.)


def test_humidity_exact_roundtrip_zero_and_other_variables(tmp_path):
    normal = PressureNormalization(fit(dataset(tmp_path), tmp_path / 'norms.json'))
    pressure = [70000., np.sqrt(70000. * 75000.), 75000.]
    for variable, values in enumerate(([250., 261., 272.], [0., .0012, .003], [2., 3., 4.], [-2., 0., 2.], [30000., 29500., 29000.])):
        encoded = normal.normalize(variable, values, pressure)
        np.testing.assert_allclose(normal.inverse(variable, encoded, pressure), values, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match='negative'):
        normal.normalize('specific_humidity', -.001, 70000.)
    assert np.isnan(normal.normalize(0, 250., 65000.))
    assert np.isnan(normal.inverse(0, 0., 65000.))
    mean, std, support = normal.arrays()
    assert mean.shape == std.shape == support.shape == (37, 5)
    assert np.isnan(mean[~support]).all()
    assert np.isnan(std[~support]).all()


def test_one_sample_bin_stays_unsupported_and_no_gap_fill(tmp_path):
    normal = PressureNormalization(fit(dataset(tmp_path, single=True), tmp_path / 'norms.json'))
    assert normal.count[10, 0] == 1
    assert not normal.support[10, 0]
    assert not normal.at(0, 75000.)[2]
    assert not normal.at(0, np.sqrt(70000. * 75000.))[2]
    assert normal.at(0, 70000.)[2]


def test_artifact_identity_and_mask_fail_closed(tmp_path):
    root = dataset(tmp_path)
    output = tmp_path / 'norms.json'
    payload = fit(root, output)
    broken = json.loads(json.dumps(payload))
    broken['mean'][0][0] = 1.
    with pytest.raises(ValueError, match='missing'):
        PressureNormalization(broken)
    broken = json.loads(json.dumps(payload)); broken['support'][0][0] = 1
    with pytest.raises(ValueError, match='mask'):
        PressureNormalization(broken)
    broken = json.loads(json.dumps(payload)); broken['source_identity']['source_sha256'] = '0' * 64
    output.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='identity'):
        fit(root, output)
    (tmp_path / 'records.jsonl').write_text((tmp_path / 'records.jsonl').read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        fit(root, tmp_path / 'new.json')


def test_output_symlink_is_rejected(tmp_path):
    root = dataset(tmp_path)
    output = tmp_path / 'norms.json'
    output.symlink_to(tmp_path / 'target.json')
    with pytest.raises(ValueError, match='symlink'):
        fit(root, output)


def test_measurements_outside_output_pressure_do_not_contaminate_edge_bins(tmp_path):
    payload = fit(dataset(tmp_path, outside=True), tmp_path / 'norms.json')
    normal = PressureNormalization(payload)
    assert payload['excluded_pressure_count'] == [4] * 5
    assert normal.count[0].tolist() == [0] * 5
    assert normal.count[-1].tolist() == [0] * 5
    assert not normal.support[0].any() and not normal.support[-1].any()
    assert normal.count[10].tolist() == normal.count[11].tolist() == [2] * 5
