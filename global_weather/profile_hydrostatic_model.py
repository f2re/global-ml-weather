"""Hydrostatic decoder with primitive-equation-conditioned learned dynamics.

The decoder satisfies the discrete hydrostatic relation by construction and
specific humidity is parameterised inside its physical domain. Horizontal
messages are conditioned by wind, geopotential-gradient acceleration and
Coriolis acceleration; vertical coupling is weakened across statically stable
potential-temperature jumps. The processor remains a learned surrogate, not a
finite-volume primitive-equation solver, so no exact mass/energy budget or CFL
claim is made here.
"""
from __future__ import annotations

from datetime import timedelta
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .grid import EARTH_RADIUS_M
from .observations import utc
from .profile_model_v2 import PressureProfileModel
from .profile_physics import hydrostatic_projection, RD
from .profile_seasonal_model import SeasonalProfileModel

ARCHITECTURE = 'hydrostatic-primitive-conditioned-v2'
STATUS = 'measured_hydrostatic_primitive_conditioned_profile_research_trained'
EARTH_ANGULAR_VELOCITY = 7.292115e-5
CP_DRY = 1004.0
KAPPA = RD / CP_DRY
# The graph update is still a learned 3 h tendency. This shorter interval is
# used only to nondimensionalise physical edge conditioning and is not an
# explicit integration step or a CFL guarantee.
FLOW_CONDITION_SECONDS = 900.0
STATIC_STABILITY_SCALE_K = 8.0
MIN_VERTICAL_GATE = 0.05


class HydrostaticFlowProfileModel(PressureProfileModel):
    """Opt-in physical architecture; it is checkpoint-incompatible with R7-R9."""
    def __init__(self, grid, normalization, hidden=64, climatology=None):
        if normalization.humidity_transform != 'identity':
            raise ValueError('Physical hydrostatic model requires physical affine normalization.')
        super().__init__(grid, normalization, hidden)
        if not bool(self.norm_support.all()):
            raise ValueError('Hydrostatic projection requires fixed norms for all five variables and 37 levels.')
        if not bool((self.mean[:, 1] > 0).all()):
            raise ValueError('A positive reference humidity is required by the smooth decoder.')
        if climatology is not None:
            self.climatology = climatology
            self.climate_encoder = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # Replace relaxation toward the GRU attractor with an explicit learned
        # residual tendency. It is bounded per 3 h output step, not CFL based.
        del self.step
        self.tendency = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.Tanh())
        nn.init.normal_(self.tendency[1].weight, std=.001)
        nn.init.zeros_(self.tendency[1].bias)
        src, dst = grid.edges
        a, b = grid.xyz[src], grid.xyz[dst]
        cosine = np.clip((a*b).sum(-1), -1., 1.)
        tangent = b - cosine[:, None]*a
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        distance = EARTH_RADIUS_M*np.arccos(cosine)
        self.register_buffer('edge_src', torch.tensor(src, dtype=torch.long))
        self.register_buffer('edge_dst', torch.tensor(dst, dtype=torch.long))
        self.register_buffer('edge_direction', torch.tensor(tangent, dtype=torch.float32))
        self.register_buffer('edge_length_m', torch.tensor(distance, dtype=torch.float32))
        lon = np.arctan2(grid.xyz[:, 1], grid.xyz[:, 0])
        lat = np.arctan2(grid.xyz[:, 2], np.hypot(grid.xyz[:, 0], grid.xyz[:, 1]))
        basis = np.stack((np.column_stack((-np.sin(lon), np.cos(lon), np.zeros(len(lon)))),
                         np.column_stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)))), axis=1)
        self.register_buffer('enu_basis', torch.tensor(basis, dtype=torch.float32))
        self.register_buffer('coriolis_f', torch.tensor(2*EARTH_ANGULAR_VELOCITY*grid.xyz[:, 2], dtype=torch.float32))
        # None keeps the historical one-shot edge path on small meshes. Mesh8
        # experiments set an explicit positive chunk size through their config.
        self.edge_chunk_size = None
        # q=std*softplus(z+inv_softplus(mean/std)) starts at the fixed mean and
        # has no hard dead zone. Fixed physical normalization is unchanged.
        ratio = self.mean[:, 1]/self.std[:, 1]
        self.register_buffer('humidity_bias', ratio + torch.log(-torch.expm1(-ratio)))

    _climate_features = SeasonalProfileModel._climate_features

    def _initial_state(self, issue, pressure):
        state = super()._initial_state(issue, pressure)
        if hasattr(self, 'climatology'):
            state = state + self.climate_encoder(self._climate_features(issue))
        return state

    def _forcing(self, issue, lead, pressure):
        state = super()._forcing(issue, lead, pressure)
        if hasattr(self, 'climatology'):
            state = state + self.climate_encoder(self._climate_features(utc(issue)+timedelta(hours=lead)))
        return state

    def _decode(self, state, issue, lead):
        # Preserve legacy output masks and explicitly unavailable omega.
        frame = super()._decode(state, issue, lead)
        z = self.head(state)
        t = F.softplus(z[..., 0]*self.std[:, 0] + self.mean[:, 0])
        q = self.std[:, 1]*F.softplus(z[..., 1] + self.humidity_bias)
        if not bool((q < 1).all()):
            raise FloatingPointError('Humidity mass fraction left its physical domain.')
        phi = hydrostatic_projection(t, q, frame.profiles[..., 4], self.pressure_pa,
                                     self.std[:, 4].reciprocal().square())
        frame.profiles = torch.stack((t, q, frame.profiles[..., 2], frame.profiles[..., 3],
                                      phi, frame.profiles[..., 5]), -1)
        return frame

    def _flow_neighbours(self, state, frame):
        """Directed messages conditioned by the horizontal momentum terms.

        The source wind provides advective direction. The geopotential-gradient
        and Coriolis accelerations perturb that direction over a short physical
        conditioning interval before it is converted to a bounded edge weight.
        This is deliberately not described as an explicit momentum solver.
        """
        src, dst = self.edge_src, self.edge_dst
        edge_count = int(src.numel())
        chunk = edge_count if self.edge_chunk_size is None else int(self.edge_chunk_size)
        if chunk < 1:
            raise ValueError('edge_chunk_size must be a positive integer or None.')
        total = state.new_zeros(state.shape)
        denominator = state.new_zeros(state.shape[:2])
        tau = FLOW_CONDITION_SECONDS
        for start in range(0, edge_count, chunk):
            stop = min(start + chunk, edge_count)
            source = src[start:stop]
            target = dst[start:stop]
            direction = self.edge_direction[start:stop]
            length = self.edge_length_m[start:stop]

            # Build only this edge chunk in Cartesian coordinates; materialising
            # all mesh8 winds as [edge,level,xyz] costs hundreds of MiB.
            source_vector = torch.einsum('elv,evd->eld',
                                         frame.profiles[source, :, 2:4], self.enu_basis[source])
            wind_along = (source_vector*direction[:, None, :]).sum(-1)
            phi = frame.profiles[..., 4]
            pressure_gradient_accel = -(phi[target]-phi[source])/length[:, None]
            radial = self.xyz[source][:, None, :].expand_as(source_vector)
            coriolis_vector = -self.coriolis_f[source][:, None, None] * torch.cross(
                radial, source_vector, dim=-1)
            coriolis_accel = (coriolis_vector*direction[:, None, :]).sum(-1)
            effective_along = wind_along + .5*tau*(pressure_gradient_accel+coriolis_accel)
            courant_like = effective_along*tau/length[:, None]
            weights = torch.sigmoid(courant_like.clamp(-8, 8))
            total = total.index_add(0, target, state[source]*weights[..., None])
            denominator = denominator.index_add(0, target, weights)
        return total/denominator.clamp_min(torch.finfo(state.dtype).tiny)[..., None]

    def _stability_neighbours(self, state, frame):
        """Local vertical coupling gated by dry potential-temperature stability."""
        temperature = frame.profiles[..., 0]
        factor = (100000.0/self.pressure_pa).pow(KAPPA)
        theta = temperature*factor[None, :]
        # Index increases upward because pressure decreases. Positive jumps are
        # statically stable and reduce cross-interface latent mixing; unstable
        # or neutral layers keep the full local coupling.
        stable_jump = F.relu(theta[:, 1:]-theta[:, :-1])
        gate = torch.exp(-stable_jump/STATIC_STABILITY_SCALE_K).clamp_min(MIN_VERTICAL_GATE)
        zero = gate.new_zeros((gate.shape[0], 1))
        lower_weight = torch.cat((zero, gate), 1)
        upper_weight = torch.cat((gate, zero), 1)
        lower = torch.cat((state[:, :1], state[:, :-1]), 1)
        upper = torch.cat((state[:, 1:], state[:, -1:]), 1)
        denominator = 1 + lower_weight + upper_weight
        return (state + lower*lower_weight[..., None] + upper*upper_weight[..., None]) / denominator[..., None]

    def _advance_state(self, state, pressure, issue, lead):
        frame = self._decode(state, issue, lead-3)
        horizontal = self._flow_neighbours(state, frame)
        vertical = self._stability_neighbours(state, frame)
        context = torch.cat((state, horizontal, vertical, self._forcing(issue, lead, pressure)), -1)
        return state + .1*self.tendency(torch.tanh(self.dynamic(context)))
