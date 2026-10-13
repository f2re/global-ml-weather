"""Synthetic program checks; these do not certify weather forecast skill."""
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from global_weather.grid import build_pyramid
from global_weather.joint_observation_model import JointObservationModel


class Norm:
    humidity_transform = 'identity'

    def arrays(self):
        mean = np.tile([270., .003, 0., 0., 30000.], (37, 1))
        std = np.tile([10., .003, 10., 10., 10000.], (37, 1))
        return mean, std, np.ones((37, 5), bool)

    def at(self, variable, pressure):
        return 0., 1., True

    def normalize(self, variable, value, pressure):
        return np.asarray(float(value)/100.)


def model():
    return JointObservationModel(build_pyramid(0)[0], Norm(), [[0., 0.], [45., 20.]],
                                 np.ones(6), np.ones((12, 2, 6)), hidden=16)


def test_joint_surface_and_profile_decoder_gradients_and_nan_mask():
    net = model()
    x = torch.randn(12, 2, 6)
    mask = torch.ones_like(x, dtype=torch.bool)
    x[0, 0, 2] = float('nan'); mask[0, 0, 2] = False
    frames = net([], datetime(2021, 2, 5, tzinfo=timezone.utc),
                 surface_inputs=x, surface_mask=mask, horizon_hours=3)
    assert frames[-1].profiles.shape == (12, 37, 6)
    assert not frames[-1].profile_variable_mask[..., 5].any()
    assert torch.isnan(frames[-1].profiles[..., 5]).all()
    assert torch.isfinite(frames[-1].profiles[..., :5]).all()
    surface = frames[-1].native_physical
    assert (surface[..., 0] > 0).all()
    assert (surface[..., 1] > 0).all()
    assert (surface[..., 1] <= surface[..., 0]).all()
    assert (surface[..., 4:] > 0).all()
    reconstructed = net.surface_monthly[frames[-1].valid_time.month-1] + frames[-1].native_normalized*net.surface_scales
    assert torch.allclose(reconstructed, surface)
    loss = frames[-1].native_normalized.square().mean() + (frames[-1].profiles[..., :5]/net.std).square().mean()
    loss.backward()
    for branch in (net.surface_encoder, net.surface_ingest, net.surface_head, net.head, net.dynamic, net.tendency):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert sum(float(p.grad.abs().sum()) for p in branch.parameters()) > 0
    assert net._surface_history is None


def test_surface_projection_corrects_unphysical_dewpoint_and_pressures_smoothly():
    raw = torch.tensor([[250., 270., 3., -4., -100., -10.],
                        [250., -1000., 3., -4., -100., -10.]], requires_grad=True)
    physical = model()._physical_surface(raw)
    assert (physical[:, 0] > 0).all()
    assert (physical[:, 1] > 0).all()
    assert (physical[:, 1] <= physical[:, 0]).all()
    assert (physical[:, 4] > 0).all() and (physical[:, 5] > 0).all()
    physical.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_joint_history_rejects_admitted_nan():
    net = model()
    x = torch.zeros(12, 2, 6); x[0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='Finite admitted'):
        net([], '2021-02-05T00:00:00+00:00', surface_inputs=x,
            surface_mask=torch.ones_like(x, dtype=torch.bool), horizon_hours=3)


def test_joint_fixed_station_climate_buffers_cannot_be_replaced():
    from global_weather.joint_observation_training import _state
    net = model(); state = {key: value.clone() for key, value in net.state_dict().items()}
    state['surface_monthly'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='surface_monthly'):
        _state(net, state)


def test_joint_sparse_fixed_buffer_roundtrip_and_change_rejection():
    from global_weather.joint_observation_training import _state
    net = model()
    state = {key: value.clone() for key, value in net.state_dict().items()}
    _state(net, state)
    graph = state['graph.adjacency'].coalesce()
    values = graph.values().clone()
    values[0] += 1
    state['graph.adjacency'] = torch.sparse_coo_tensor(graph.indices(), values, graph.shape)
    with pytest.raises(ValueError, match='graph.adjacency'):
        _state(net, state)


def test_joint_profile_and_terrain_input_branches_are_active():
    grid = build_pyramid(0)[0]
    static = {'terrain_elevation_m': np.full(grid.n_cells, -50.),
              'terrain_land_fraction': np.ones(grid.n_cells),
              'terrain_slope': np.zeros(grid.n_cells)}
    net = JointObservationModel(grid, Norm(), [[0., 0.]], np.ones(6),
                                 np.ones((12, 1, 6)), hidden=16, static=static)
    record = {'variable': 'temperature', 'value': 270., 'pressure_pa': 85000.,
              'latitude': 0., 'longitude': 0., 'observed_at': '2021-02-04T23:00:00Z',
              'available_at': '2021-02-04T23:00:00Z'}
    frames = net([record], '2021-02-05T00:00:00Z', surface_inputs=torch.ones(12, 1, 6),
                 surface_mask=torch.ones(12, 1, 6, dtype=torch.bool), horizon_hours=3)
    (frames[-1].native_normalized.square().mean()+frames[-1].profiles[..., 0].mean()).backward()
    for branch in (net.token, net.variable, net.ingest, net.terrain_encoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
        assert sum(float(p.grad.abs().sum()) for p in branch.parameters()) > 0
    assert (net.terrain_elevation_m == -50.).all()
