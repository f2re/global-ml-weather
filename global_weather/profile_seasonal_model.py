"""Measured-profile dynamics conditioned on frozen seasonal climate features.

GraphCast remains the affine normalization and physical output inverse. Monthly
NOAA means are masked context, never observations or additional training targets.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import torch
from torch import nn

from .observations import utc
from .profile_model_v2 import PressureProfileModel

ARCHITECTURE = 'pressure-profile-seasonal-v1'
STATUS = 'measured_seasonal_profile_research_trained'


class SeasonalProfileModel(PressureProfileModel):
    def __init__(self, grid, normalization, climatology, hidden: int = 32):
        if normalization.humidity_transform != 'identity':
            raise ValueError('Seasonal profiles require pinned physical GraphCast normalization.')
        super().__init__(grid, normalization, hidden)
        self.climatology = climatology
        self.climate_encoder = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def _climate_features(self, valid_time: datetime) -> torch.Tensor:
        valid_time = utc(valid_time)
        mean, support = self.climatology.sample(valid_time)
        expected = (self.grid.n_cells, 37, 5)
        if mean.shape != expected or support.shape != expected or support.dtype != np.bool_:
            raise ValueError('Seasonal context shape or mask differs.')
        if not np.isfinite(mean[support]).all():
            raise ValueError('Nonfinite supported climate context.')
        # Unsupported climate context is a neutral network feature with its own
        # false mask; it never changes the observed inputs or target masks.
        anomalies = np.where(support, (np.where(support, mean, self.normalization.mean)
                                      - self.normalization.mean) / self.normalization.std, 0.)
        begin = datetime(valid_time.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(valid_time.year + 1, 1, 1, tzinfo=timezone.utc)
        phase = 2 * np.pi * (valid_time - begin).total_seconds() / (end - begin).total_seconds()
        longitude = np.rad2deg(np.arctan2(self.grid.xyz[:, 1], self.grid.xyz[:, 0]))
        utc_hours = valid_time.hour + valid_time.minute / 60 + valid_time.second / 3600
        solar_phase = 2 * np.pi * (utc_hours + longitude / 15) / 24
        calendar = np.broadcast_to(np.column_stack((np.full(len(longitude), np.sin(phase)),
                         np.full(len(longitude), np.cos(phase)), np.sin(solar_phase), np.cos(solar_phase)))[:, None, :],
                         (self.grid.n_cells, 37, 4))
        features = np.concatenate((anomalies, support.astype(float), calendar), axis=-1)
        if not np.isfinite(features).all():
            raise FloatingPointError('Nonfinite seasonal network features.')
        return torch.tensor(features, dtype=self.mean.dtype, device=self.mean.device)

    def _initial_state(self, issue, pressure):
        return super()._initial_state(issue, pressure) + self.climate_encoder(self._climate_features(issue))

    def _forcing(self, issue, lead, pressure):
        valid_time = utc(issue) + timedelta(hours=lead)
        return super()._forcing(issue, lead, pressure) + self.climate_encoder(self._climate_features(valid_time))
