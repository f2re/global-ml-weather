"""Experimental R9 hydrostatic decoder and wind-conditioned graph dynamics.

The decoder satisfies the discrete hydrostatic relation by construction. The
processor is still LEARNED dynamics: wind-conditioned attention is not a
finite-volume advection solver, and no complete mass/energy budget is claimed.
Omega, surface and terrain are unsupported exactly as in the profile pilot.
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
from .profile_physics import hydrostatic_projection
from .profile_seasonal_model import SeasonalProfileModel

ARCHITECTURE = 'hydrostatic-flow-profile-v1'
STATUS = 'measured_hydrostatic_flow_profile_research_trained'


class HydrostaticFlowProfileModel(PressureProfileModel):
    """Opt-in new architecture. It cannot resume R7/R8 weights as the same model."""
    def __init__(self, grid, normalization, hidden=64, climatology=None):
        if normalization.humidity_transform != 'identity':
            raise ValueError('R9 hydrostatic model requires physical affine normalization.')
        super().__init__(grid, normalization, hidden)
        if not bool(self.norm_support.all()):
            raise ValueError('R9 projection requires fixed norms for all five variables and 37 levels.')
        if not bool((self.mean[:, 1] > 0).all()):
            raise ValueError('A positive reference humidity is required by the smooth decoder.')
        if climatology is not None:
            self.climatology = climatology
            self.climate_encoder = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # Replace relaxation toward the GRU attractor with an explicit learned
        # residual tendency. It is bounded per 3h step, NOT a CFL guarantee.
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
        # q=std*softplus(z+inv_softplus(mean/std)) starts at the fixed mean and
        # has no hard dead zone. The physical normalization remains unchanged;
        # this POSITIVITY PARAMETERIZATION is explicitly a new architecture.
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
        # Preserve the legacy output masks and explicitly unavailable channels.
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
        # Positive projection from source to destination increases upstream
        # information. Never average ENU vectors as though their bases coincide.
        vectors = torch.einsum('nlv,nvd->nld', frame.profiles[..., 2:4], self.enu_basis)
        src, dst = self.edge_src, self.edge_dst
        courant = (vectors[src]*self.edge_direction[:, None]).sum(-1)*10800/self.edge_length_m[:, None]
        weights = torch.sigmoid(courant.clamp(-8, 8))
        total = state.new_zeros(state.shape).index_add(0, dst, state[src]*weights[..., None])
        denominator = state.new_zeros(state.shape[:2]).index_add(0, dst, weights)
        return total/denominator.clamp_min(torch.finfo(state.dtype).tiny)[..., None]

    def _advance_state(self, state, pressure, issue, lead):
        frame = self._decode(state, issue, lead-3)
        horizontal = self._flow_neighbours(state, frame)
        # Local vertical coupling, not a whole-column mean erasing inversions.
        lower = torch.cat((state[:, :1], state[:, :-1]), 1)
        upper = torch.cat((state[:, 1:], state[:, -1:]), 1)
        vertical = .5*state + .25*(lower+upper)
        context = torch.cat((state, horizontal, vertical, self._forcing(issue, lead, pressure)), -1)
        return state + .1*self.tendency(torch.tanh(self.dynamic(context)))
