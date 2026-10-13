"""R11 acceptance gates: pinned thresholds, baselines, spectra and the end-to-end report."""
import json

import numpy as np
import pytest
import torch

from global_weather.assimilation import gates
from global_weather.assimilation.mesh import LatentMesh
from global_weather.assimilation.tokens import PLATFORMS, VARIABLES, TokenBatch

from test_assimilation_model import model as small_model

THRESHOLDS = 'configs/physical_assimilation_gates_r11.json'


def test_pinned_thresholds_validate_and_weaker_gates_are_refused(tmp_path):
    payload = gates.load_thresholds(THRESHOLDS)
    assert payload['gate_1']['min_reach_km'] == 1620 and payload['gate_4']['leads_hours'] == [24, 48, 72]
    weaker = json.loads(open(THRESHOLDS).read())
    weaker['gate_1']['min_reach_km'] = 500
    (tmp_path / 'weak.json').write_text(json.dumps(weaker))
    with pytest.raises(ValueError, match='Gate 1'):
        gates.load_thresholds(tmp_path / 'weak.json')


def test_harmonic_basis_is_orthonormal_under_area_quadrature():
    mesh = LatentMesh(4, multimesh=False)
    basis = gates.HarmonicBasis(mesh.xyz, mesh.areas_m2, 8)
    gram = (basis.matrix * basis.weights) @ basis.matrix.T
    assert np.abs(gram - np.eye(len(gram))).max() < 2e-2


def test_rotational_and_divergent_flows_are_separated():
    net = small_model(level=4)
    xyz = net.xyz.double()
    rotation = 20.0 * torch.cross(torch.tensor([0., 0., 1.], dtype=torch.float64).expand_as(xyz), xyz, dim=-1)
    basis = gates.HarmonicBasis(net.mesh.xyz, net.mesh.areas_m2, 8)
    solid = gates.energy_spectrum(net, rotation, basis)
    assert solid['divergent_fraction'] < 1e-3 and np.argmax(solid['total']) == 1
    z = xyz[:, 2]
    gradient = torch.stack((-z * xyz[:, 0], -z * xyz[:, 1], 1 - z * z), -1) * 20.0  # grad of z on the sphere
    potential = gates.energy_spectrum(net, gradient, basis)
    assert potential['divergent_fraction'] > 0.999


def batch(group, variable, observed, value, pressure=np.nan):
    n = len(value)
    return TokenBatch(platform=np.full(n, PLATFORMS.index('surface_station'), np.int16),
                      variable=np.full(n, VARIABLES.index(variable), np.int16), channel=np.full(n, -1, np.int32),
                      group=np.asarray(group, np.int64), identity=np.arange(n, dtype=np.int64),
                      observed=np.asarray(observed, float), available=np.asarray(observed, float),
                      latitude=np.zeros(n), longitude=np.zeros(n), pressure_pa=np.full(n, pressure),
                      height_m=np.zeros(n), value=np.asarray(value, float), error=np.full(n, np.nan),
                      quality=np.ones(n, np.float32))


def test_persistence_uses_only_reports_older_than_the_analysis_window_for_withheld_rows():
    issue = 100_000.0
    history = batch([1, 1, 1], 'station_temperature', [issue - 6 * 3600, issue - 3 * 3600, issue - 600], [280., 281., 282.])
    targets = batch([1, 1], 'station_temperature', [issue - 300, issue + 6 * 3600], [282., 285.])
    out = gates.persistence(history, targets, issue, np.array([True, False]), 1.5)
    assert out.tolist() == [281., 282.]


def test_gate_report_on_the_synthetic_program_check(tmp_path):
    from global_weather.assimilation.selftest import SELFTEST_CONFIG
    from global_weather.assimilation.training import train
    train(SELFTEST_CONFIG, tmp_path / 'run', synthetic=True, device='cpu', max_steps=1)
    selftest = tmp_path / 'selftest.json'
    selftest.write_text(json.dumps({'passed': True, 'checks': {'a': True}}))
    thresholds = json.loads(open(THRESHOLDS).read())
    thresholds['gate_5']['l_max'] = 10
    thresholds['gate_5']['l_high'] = 10
    thresholds['gate_5']['l_low'] = 3
    (tmp_path / 'gates.json').write_text(json.dumps(thresholds))
    report = gates.run(tmp_path / 'run', tmp_path / 'gates.json', tmp_path / 'report.json', split='validation',
                       issues=2, selftest_path=selftest, climate_stride_days=73, synthetic=True)
    assert set(report['gates']) == {'0', '1', '2', '3', '4', '5'}
    assert report['gates']['0']['passed'] and report['gates']['1']['passed']
    assert report['gates']['3']['max_relative_drift'] < 1e-6 and report['gates']['3']['passed']
    assert report['era5_used'] is False and report['data_kind'] == 'synthetic'
    assert report['gates']['5']['ratios']['24h']['samples'] > 0
    # Two issues cannot satisfy the pinned 1000-target minimum; the gate must say so.
    assert report['passed'] is False
    with pytest.raises(FileExistsError):
        gates.run(tmp_path / 'run', tmp_path / 'gates.json', tmp_path / 'report.json', split='validation',
                  issues=2, synthetic=True)


def test_train_climatology_is_monthly_with_a_month_free_fallback():
    from types import SimpleNamespace
    from global_weather.assimilation.sources import seconds as to_seconds

    class Source:
        fingerprint = {'kind': 'fixture'}

        def window(self, begin, end, *, split):
            assert split == 'train'
            month = utc_month(begin)
            return batch([7], 'station_temperature', [begin + 3600], [1.0 if month == 1 else 3.0])

    def utc_month(value):
        from datetime import datetime, timezone
        return datetime.fromtimestamp(value, tz=timezone.utc).month

    experiment = SimpleNamespace(sources=[Source()], channel_index={})
    climate = gates.TrainClimatology(experiment, stride_days=200)   # 1 January and 20 July 2021
    january = to_seconds('2021-01-10T00:00:00Z'); july = to_seconds('2021-07-10T00:00:00Z')
    march = to_seconds('2021-03-10T00:00:00Z')
    values = climate.lookup(batch([7, 7, 7, 8], 'station_temperature', [january, july, march, march], [0, 0, 0, 0]))
    assert values[:3].tolist() == [1.0, 3.0, 2.0] and np.isnan(values[3])
