"""Station climatology anomaly model.

The neural network ingests observed anomalies relative to station climatological
normals onto a global spherical grid, propagates the atmospheric dynamics, and
predicts exclusively the future anomaly (departure from the climatological normal).
Physical forecasts are obtained by adding the predicted anomaly to the frozen normal.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .grid import SphereGrid, unit_xyz
from .model import GraphOps
from .observation_model import LEAD_HOURS, NearestSphericalOperator
from .observations import utc
from .vertical import PRESSURE_HPA

NATIVE_VARIABLES = (
    "station_temperature",
    "station_dew_point",
    "station_eastward_wind",
    "station_northward_wind",
    "station_pressure",
    "station_mean_sea_level_pressure",
)
NATIVE_UNITS = ("K", "K", "m s-1", "m s-1", "Pa", "Pa")


@dataclass
class AnomalyForecast:
    predicted_anomalies: Tensor  # [24 leads, n_stations, 6] dimensionless
    lead_hours: tuple[int, ...]
    profile_diagnostics: Tensor
    profile_target_mask: Tensor
    scientific_acceptance: bool = False
    prediction_target: str = "dimensionless anomaly departure from climatology"


class StationAnomalyModel(nn.Module):
    """Global spherical atmospheric model predicting surface departures from climate normals."""

    def __init__(self, grids, station_coordinates, scales, *, hidden: int = 32, step_hours: int = 3):
        super().__init__()
        if not grids or hidden < 8 or step_hours != 3:
            raise ValueError("Need a global grid, hidden >= 8 and three-hour steps.")
        scale_tensor = torch.as_tensor(scales, dtype=torch.float32)
        if scale_tensor.shape != (6,) or not torch.isfinite(scale_tensor).all() or bool((scale_tensor <= 0).any()):
            raise ValueError("Six finite positive variability scales are required.")
        self.register_buffer("scales", scale_tensor.clone())
        self.register_buffer("pressure_hpa", torch.tensor(PRESSURE_HPA, dtype=torch.float32))
        self.operator = NearestSphericalOperator(grids[0], station_coordinates)
        self.graph = GraphOps(grids[0])
        self.register_buffer("xyz", torch.tensor(grids[0].xyz, dtype=torch.float32))
        self.hidden = hidden
        self.step_hours = step_hours

        # Embedding & Spatial Layers
        self.geometry = nn.Linear(3, hidden)
        self.level_embedding = nn.Parameter(torch.randn(37, hidden) * 0.01)
        # Input encoder takes normalized anomaly and mask: [6 + 6] = 12
        self.input_encoder = nn.Sequential(nn.Linear(12, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # Recurrent ingestion onto the 37-level spherical column
        self.ingest = nn.GRUCell(hidden, hidden)

        # Dynamic transition step
        self.dynamic = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.transition = nn.GRUCell(hidden, hidden)

        # Head predicting exclusively the normalized departure from climatology
        self.anomaly_head = nn.Sequential(nn.Linear(2 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 6))
        self.profile_head = nn.Linear(hidden, 6)
        self.grid_fingerprint = grids[0].fingerprint

    def reconstruct_physical(self, predicted_anomalies: Tensor, climatology_normals: Tensor) -> Tensor:
        """Add the predicted departure back to the climatological normal: y = y_clim + sigma * delta_y."""
        if predicted_anomalies.shape != climatology_normals.shape:
            raise ValueError("Predicted anomalies and climatology normals must have identical shapes.")
        return climatology_normals + predicted_anomalies * self.scales

    def forward(self, history_anomalies: Tensor, history_mask: Tensor) -> AnomalyForecast:
        """Ingest 12 hours of observed departures, evolve the global state, and predict future anomalies."""
        expected = (12, self.operator.cells.numel(), 6)
        if history_anomalies.shape != expected or history_mask.shape != expected or history_mask.dtype != torch.bool:
            raise ValueError("History departures and boolean mask must have shape [12, stations, 6].")
        if not bool(torch.isfinite(history_anomalies[history_mask]).all()):
            raise ValueError("An admitted anomaly measurement is nonfinite.")

        n, d = self.xyz.shape[0], self.hidden
        # Mask missing anomalies so NaNs cannot enter computation
        clean_anomalies = torch.where(history_mask, history_anomalies, torch.zeros_like(history_anomalies))
        state = self.geometry(self.xyz)[:, None, :] + self.level_embedding[None, :, :]

        for values, mask in zip(clean_anomalies, history_mask):
            tokens = self.input_encoder(torch.cat([values, mask.to(values.dtype)], dim=-1))
            valid = mask.any(-1)
            pooled = values.new_zeros(n, d).index_add(0, self.operator.cells, tokens * valid[:, None])
            counts = values.new_zeros(n).index_add(0, self.operator.cells, valid.to(values.dtype))
            pooled = pooled / counts.clamp_min(1)[:, None]
            updated = self.ingest(
                pooled[:, None, :].expand(-1, 37, -1).reshape(-1, d),
                state.reshape(-1, d),
            ).reshape(n, 37, d)
            state = torch.where((counts > 0)[:, None, None], updated, state)

        predictions = []
        for _ in LEAD_HOURS:
            vertical_mean = state.mean(1, keepdim=True).expand_as(state)
            forcing = self.dynamic(torch.cat([state, self.graph.neighbours(state), vertical_mean], -1))
            state = self.transition(forcing.reshape(-1, d), state.reshape(-1, d)).reshape(n, 37, d)
            station_context = torch.cat(
                [self.operator(state.mean(1)), self.operator(state[:, 0]), self.operator.station_xyz],
                dim=-1,
            )
            predictions.append(self.anomaly_head(station_context))

        profiles = self.profile_head(state)
        return AnomalyForecast(
            torch.stack(predictions),
            LEAD_HOURS,
            profiles,
            torch.zeros_like(profiles, dtype=torch.bool),
        )


def anomaly_loss(predicted_anomalies: Tensor, target_anomalies: Tensor, mask: Tensor) -> Tensor:
    """Equal-variable Huber loss evaluated strictly between predicted and true departures from climate."""
    if (predicted_anomalies.shape != target_anomalies.shape
            or predicted_anomalies.shape != mask.shape
            or predicted_anomalies.shape[-1] != 6
            or mask.dtype != torch.bool):
        raise ValueError("Shapes of predicted departures, target departures, and boolean mask must match.")
    if not bool(mask.any()):
        raise ValueError("Need at least one admitted target departure.")
    if not bool(torch.isfinite(target_anomalies[mask]).all()):
        raise ValueError("Nonfinite target anomaly.")
    if not bool(torch.isfinite(predicted_anomalies[mask]).all()):
        raise ValueError("Nonfinite predicted anomaly.")

    components = [
        F.huber_loss(predicted_anomalies[..., v][mask[..., v]], target_anomalies[..., v][mask[..., v]])
        for v in range(6)
        if bool(mask[..., v].any())
    ]
    return torch.stack(components).mean()
