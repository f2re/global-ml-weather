"""R11 model, observation operators and objective: analytic and structural checks.

These are program checks on synthetic fields; they do not certify weather skill.
"""
from datetime import datetime, timedelta, timezone
import math

import numpy as np
import pytest
import torch

from global_weather.assimilation import thermo
from global_weather.assimilation.mesh import LatentMesh
from global_weather.assimilation.model import CHANNELS, PhysicalAssimilationModel, Rollout, StateNormalization
from global_weather.assimilation.objective import fair_crps, observation_objective
from global_weather.assimilation.operator import build_plan, observe
from global_weather.assimilation.selftest import reach_after_one_step
from global_weather.assimilation.tokens import PLATFORMS, VARIABLES, TokenBatch
from global_weather.grid import unit_xyz
from global_weather.vertical import PRESSURE_HPA

ISSUE = datetime(2021, 3, 1, 12, tzinfo=timezone.utc)
ISSUE_S = (ISSUE - datetime(1970, 1, 1, tzinfo=timezone.utc)).total_seconds()


def normalization():
    mean = np.tile([250., 1e-3, 5., 0., 5e4], (37, 1))
    std = np.tile([10., 1e-3, 10., 10., 1e4], (37, 1))
    return StateNormalization.from_arrays(mean, std, mslp_mean=101000., mslp_std=1300., t2m_mean=280.,
                                          t2m_std=20., u10_std=5.)


def model(level=2, elevation=0., **kwargs):
    mesh = LatentMesh(level)
    static = {'elevation_m': np.full(mesh.n_nodes, elevation), 'land_fraction': np.full(mesh.n_nodes, 0.5)}
    options = dict(hidden=16, message_dim=8, processor_layers=3, analysis_layers=3)
    options.update(kwargs)
    torch.manual_seed(0)
    return PhysicalAssimilationModel(mesh, normalization(), static, **options)


def token(variable, platform, *, pressure=np.nan, height=np.nan, hours=3., lat=30., lon=40., value=0.):
    return TokenBatch(platform=np.array([PLATFORMS.index(platform)], np.int16),
                      variable=np.array([VARIABLES.index(variable)], np.int16), channel=np.array([-1], np.int32),
                      group=np.array([1], np.int64), identity=np.array([hash((variable, pressure, height, hours))], np.int64),
                      observed=np.array([ISSUE_S + hours * 3600.]), available=np.array([ISSUE_S + hours * 3600.]),
                      latitude=np.array([lat]), longitude=np.array([lon]), pressure_pa=np.array([pressure]),
                      height_m=np.array([height]), value=np.array([value]), error=np.array([np.nan]),
                      quality=np.ones(1, np.float32))


def analytic_rollout(net, frames=2, temperature=None, surface_pressure=95000., wind_east=0.):
    """Physically consistent fields built through the model's own diagnostics."""
    n = net.mesh.n_nodes
    p = net.pressure
    physics = []
    for f in range(frames):
        t = temperature(p, f) if temperature else torch.full((37,), 250.)
        lat_east = torch.stack((-torch.sin(torch.atan2(net.xyz[:, 1], net.xyz[:, 0])),
                                torch.cos(torch.atan2(net.xyz[:, 1], net.xyz[:, 0])), torch.zeros(n)), -1)
        fields = {'surface_pressure': torch.full((n,), surface_pressure), 'temperature': t.expand(n, 37).clone(),
                  'humidity': torch.zeros(n, 37), 'wind': (wind_east * lat_east)[:, None, :].expand(n, 37, 3).clone(),
                  'surface_temperature': torch.full((n,), 280.), 'surface_humidity': torch.full((n,), 5e-3),
                  'surface_wind': wind_east * lat_east}
        physics.append(net._diagnose(fields))
    states = [torch.zeros(n, CHANNELS) for _ in range(frames)]
    return Rollout(ISSUE, [3 * f for f in range(frames)], states, physics)


def plan_for(net, batch, horizon=3, analysis=False):
    sigma = np.ones(len(batch))
    return build_plan(batch, sigma, ISSUE_S, horizon, net.mesh, analysis=np.full(len(batch), analysis))


def test_pressure_operator_is_linear_in_log_pressure_and_time():
    net = model()
    rollout = analytic_rollout(net, temperature=lambda p, f: 200. + 10. * torch.log(p / 100.) + 10. * f)
    plan = plan_for(net, token('temperature', 'radiosonde', pressure=55000., hours=1.5))
    expected = 200. + 10. * math.log(550.) + 5.
    assert float(observe(net, rollout, plan)[0]) == pytest.approx(expected, abs=1e-3)


def test_wind_component_is_projected_at_the_measurement_point():
    net = model()
    rollout = analytic_rollout(net, wind_east=20.)
    east = observe(net, rollout, plan_for(net, token('u', 'aircraft', pressure=25000.)))
    north = observe(net, rollout, plan_for(net, token('v', 'aircraft', pressure=25000.)))
    assert float(east[0]) == pytest.approx(20., rel=0.03)
    assert abs(float(north[0])) < 0.6


def test_station_operators_follow_hypsometry_and_refuse_unknown_elevation():
    net = model(elevation=500.)
    rollout = analytic_rollout(net, surface_pressure=95000.)
    same = observe(net, rollout, plan_for(net, token('station_pressure', 'surface_station', height=500.)))
    assert float(same[0]) == pytest.approx(95000., rel=1e-6)
    tv = float(thermo.virtual_temperature(torch.tensor(280.), torch.tensor(5e-3)))
    higher = observe(net, rollout, plan_for(net, token('station_pressure', 'surface_station', height=600.)))
    lapse = float(rollout.physics[0]['lapse_rate'][0])
    expected = 95000. * math.exp(-9.80665 * 100. / (287.05 * (tv - 0.5 * lapse * 100.)))
    assert float(higher[0]) == pytest.approx(expected, rel=1e-5)
    refused = plan_for(net, token('station_temperature', 'surface_station'))
    assert refused.count == 0 and refused.rejected == {'unknown_station_elevation': 1}
    assert plan_for(net, token('station_mean_sea_level_pressure', 'surface_station')).count == 1


def test_pressures_outside_model_levels_are_refused_not_extrapolated():
    net = model()
    plan = plan_for(net, token('temperature', 'radiosonde', pressure=101000.))
    assert plan.count == 0 and plan.rejected == {'pressure_outside_model_levels': 1}


def test_refractivity_operator_locates_height_in_hydrostatic_column():
    net = model()
    rollout = analytic_rollout(net, surface_pressure=100000.)
    height = 10000.
    phi = 9.80665 * height * 6_371_008.8 / (6_371_008.8 + height)
    pressure_hpa = 1000. * math.exp(-phi / (287.05 * 250.))
    predicted = observe(net, rollout, plan_for(net, token('refractivity', 'gnss_ro', height=height)))
    assert float(predicted[0]) == pytest.approx(77.6 * pressure_hpa / 250., rel=2e-3)


def test_gradients_reach_every_branch_and_state_constraints_hold():
    net = model(noise_dim=2)
    batch = TokenBatch.concatenate([token('temperature', 'radiosonde', pressure=50000., hours=0., value=250.),
                                    token('station_pressure', 'surface_station', height=100., hours=0., value=99000.)])
    from global_weather.assimilation.tokens import Normalizer, encode
    normalizer = Normalizer(np.asarray(net.normalization.profile_mean), np.asarray(net.normalization.profile_std),
                            {Normalizer.key('surface_station', 'station_pressure'):
                             {'mean': 11.5, 'std': 0.05, 'transform': 'height_reduced_log'}})
    encoded, _ = encode(batch, ISSUE_S, normalizer, np.ones(2), [])
    rollout = net(encoded, ISSUE, 6)
    targets = TokenBatch.concatenate([token('temperature', 'radiosonde', pressure=50000., hours=4., value=251.),
                                      token('u', 'radiosonde', pressure=30000., hours=6., value=20.),
                                      token('station_pressure', 'surface_station', height=100., hours=5., value=99100.)])
    plan = build_plan(targets, np.array([1., 2., 100.]), ISSUE_S, 6, net.mesh, analysis=np.zeros(3, bool))
    loss, report = observation_objective(observe(net, rollout, plan)[None], plan)
    loss.backward()
    exempt = ('column_operator', 'channel_bias', 'channel_predictors')
    for name, parameter in net.named_parameters():
        if not name.startswith(exempt):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    masses = [float(net.dry_mass(f)) for f in rollout.physics]
    assert max(abs(m / masses[0] - 1) for m in masses) < 1e-6
    for fields in rollout.physics:
        assert float((fields['wind'] * net.xyz[:, None]).sum(-1).abs().max()) < 1e-4
        assert bool((fields['humidity'] <= 1.05 * thermo.saturation_specific_humidity(fields['temperature'], net.pressure) + 1e-9).all())
    assert report['targets'] == 3


def test_one_step_domain_of_dependence_exceeds_jet_transport():
    net = model(level=3, processor_layers=4)
    assert reach_after_one_step(net, ISSUE) >= 1620.


def test_frames_mask_below_ground_and_carry_physical_omega():
    net = model(elevation=3000.)
    rollout = net(None, ISSUE, 6)
    frames = net.frames(rollout)
    assert [f.lead_hours for f in frames] == [0, 3, 6]
    frame = frames[-1]
    ps = frame.surface[:, 4]
    assert torch.equal(frame.profile_mask, net.pressure[None] <= ps[:, None])
    assert torch.isnan(frame.profiles[~frame.profile_mask]).all()
    assert torch.isfinite(frame.profiles[frame.profile_mask]).all()


def test_fair_crps_and_huber_objective():
    identical = torch.tensor([[1.0, -2.0], [1.0, -2.0]])
    torch.testing.assert_close(fair_crps(identical), torch.tensor([1.0, 2.0]))
    spread = torch.tensor([[1.0], [-1.0]])
    torch.testing.assert_close(fair_crps(spread), torch.tensor([0.0]))
    with pytest.raises(FloatingPointError):
        net = model()
        plan = plan_for(net, token('temperature', 'radiosonde', pressure=50000.))
        observation_objective(torch.tensor([[float('nan')]]), plan)
