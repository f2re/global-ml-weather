"""Joint measured model with physically constrained profile and surface heads.

The spherical profile processor enforces hydrostatic consistency and uses
primitive-equation-conditioned learned exchange. The station-native readout is
projected to basic physical domains: temperature, dew point and pressures are
positive and dew point cannot exceed air temperature. Satellite parameters are
deliberately absent from this experiment. Diagnostic omega and below-terrain
issuance are computed only by the forecast-physics stage.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .observation_model import NearestSphericalOperator
from .profile_hydrostatic_model import HydrostaticFlowProfileModel

ARCHITECTURE = 'joint-measured-physical-v2'


class JointObservationModel(HydrostaticFlowProfileModel):
    def __init__(self, grid, normalization, station_coordinates, surface_scales,
                 surface_monthly, hidden=64, climatology=None, static=None):
        super().__init__(grid, normalization, hidden, climatology)
        self.station_operator = NearestSphericalOperator(grid, station_coordinates)
        scales = torch.as_tensor(surface_scales, dtype=torch.float32)
        monthly = torch.as_tensor(surface_monthly, dtype=torch.float32)
        if (scales.shape != (6,) or monthly.shape != (12, len(station_coordinates), 6)
                or not torch.isfinite(scales).all() or not torch.isfinite(monthly).all()
                or bool((scales <= 0).any())):
            raise ValueError('Immutable finite station climate and positive train scales required.')
        self.register_buffer('surface_scales', scales.clone())
        self.register_buffer('surface_monthly', monthly.clone())
        self.surface_encoder = nn.Sequential(nn.Linear(12, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.surface_ingest = nn.GRUCell(hidden, hidden)
        self.surface_head = nn.Sequential(nn.Linear(2 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 6))
        # Learned surface forcing tapers with log-pressure altitude; this is an
        # input embedding, not an asserted boundary-layer physical equation.
        self.register_buffer('surface_vertical_weight',
                             torch.exp(-torch.abs(torch.log(self.pressure_pa / 100000.)) * 4))
        if static is not None:
            fields = np.column_stack([static[key] for key in
                                      ('terrain_elevation_m', 'terrain_land_fraction', 'terrain_slope')])
            if fields.shape != (grid.n_cells, 3) or not np.isfinite(fields).all():
                raise ValueError('Verified terrain must cover every global cell.')
            if np.any((fields[:, 1] < 0) | (fields[:, 1] > 1)) or np.any(fields[:, 2] < 0):
                raise ValueError('Invalid independent land fraction or terrain slope.')
            self.register_buffer('terrain_elevation_m', torch.tensor(fields[:, 0], dtype=torch.float32))
            self.register_buffer('terrain_land_fraction', torch.tensor(fields[:, 1], dtype=torch.float32))
            self.register_buffer('terrain_slope', torch.tensor(fields[:, 2], dtype=torch.float32))
            self.terrain_encoder = nn.Sequential(nn.Linear(3, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self._surface_history = None

    def _initial_state(self, issue, pressure):
        state = super()._initial_state(issue, pressure)
        if hasattr(self, 'terrain_encoder'):
            terrain = torch.stack((self.terrain_elevation_m / 10000., self.terrain_land_fraction, self.terrain_slope), -1)
            state = state + self.terrain_encoder(terrain)[:, None, :]
        if self._surface_history is None:
            raise ValueError('Explicit masked station history is required.')
        history, masks = self._surface_history
        n, d = self.grid.n_cells, self.hidden
        for values, mask in zip(history, masks):
            clean = torch.where(mask, values, torch.zeros_like(values))
            tokens = self.surface_encoder(torch.cat((clean, mask.to(values.dtype)), -1))
            valid = mask.any(-1)
            pooled = state.new_zeros(n, d).index_add(0, self.station_operator.cells, tokens * valid[:, None])
            count = state.new_zeros(n).index_add(0, self.station_operator.cells, valid.to(state.dtype))
            forcing = pooled / count.clamp_min(1)[:, None]
            forcing = forcing[:, None, :] * self.surface_vertical_weight[None, :, None]
            updated = self.surface_ingest(forcing.expand_as(state).reshape(-1, d), state.reshape(-1, d)).reshape_as(state)
            state = torch.where((count > 0)[:, None, None], updated, state)
        return state

    @staticmethod
    def _physical_surface(raw):
        """Project six native station variables to their hard physical domains."""
        t = F.softplus(raw[..., 0])
        # First map the unconstrained dew-point candidate to Kelvin > 0, then
        # smoothly cap it below T. For ordinary atmospheric Kelvin values both
        # softplus operations are asymptotically identity, so realistic values
        # are not compressed merely to satisfy the constraint.
        td_candidate = F.softplus(raw[..., 1])
        td = (t - F.softplus(t-td_candidate)).clamp_min(torch.finfo(raw.dtype).tiny)
        ps = F.softplus(raw[..., 4])
        mslp = F.softplus(raw[..., 5])
        return torch.stack((t, td, raw[..., 2], raw[..., 3], ps, mslp), -1)

    def _decode(self, state, issue, lead):
        frame = super()._decode(state, issue, lead)
        cells = self.station_operator.cells
        context = torch.cat((state.mean(1)[cells], state[:, 0][cells], self.station_operator.station_xyz), -1)
        raw_normalized = self.surface_head(context)
        baseline = self.surface_monthly[frame.valid_time.month - 1]
        raw_physical = baseline + raw_normalized * self.surface_scales
        frame.native_physical = self._physical_surface(raw_physical)
        # Training remains in the same immutable station-anomaly coordinates,
        # but the loss now sees the physically projected state.
        frame.native_normalized = (frame.native_physical-baseline)/self.surface_scales
        frame.native_variable_mask = torch.isfinite(frame.native_physical)
        return frame

    def forward(self, inputs, issue, *, surface_inputs, surface_mask, horizon_hours=72):
        expected = (12, self.station_operator.cells.numel(), 6)
        if (surface_inputs.shape != expected or surface_mask.shape != expected
                or surface_mask.dtype != torch.bool or not torch.isfinite(surface_inputs[surface_mask]).all()):
            raise ValueError('Finite admitted 12-hour native station anomalies required.')
        self._surface_history = (surface_inputs, surface_mask)
        try:
            return super().forward(inputs, issue, horizon_hours=horizon_hours)
        finally:
            self._surface_history = None
