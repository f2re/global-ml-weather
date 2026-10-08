"""Station-native spherical pilot; no reanalysis or fabricated atmospheric profile.

Instrument heights are unknown in the admitted archive. The six readouts retain
the station measurement convention and must not be marketed as 2 m/10 m fields.
The 37-level latent diagnostic is uncalibrated and has no supervised profile
targets. This geometry-only ablation deliberately has no terrain context.
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
from .observations import utc
from .vertical import PRESSURE_HPA

NATIVE_VARIABLES = ("station_temperature", "station_dew_point", "station_eastward_wind",
                    "station_northward_wind", "station_pressure", "station_mean_sea_level_pressure")
NATIVE_UNITS = ("K", "K", "m s-1", "m s-1", "Pa", "Pa")
LEAD_HOURS = tuple(range(3, 73, 3))


@dataclass
class ObservationForecast:
    native_normalized: Tensor
    lead_hours: tuple[int, ...]
    profile_diagnostics: Tensor
    profile_target_mask: Tensor
    scientific_acceptance: bool = False
    profile_units: str = "uncalibrated latent diagnostics; no physical units"


def exact_lead_index(issue_time: datetime, observed_at: datetime) -> int:
    """Match an admitted aggregate's endpoint; never round a future timestamp."""
    seconds = (utc(observed_at) - utc(issue_time)).total_seconds()
    for index, lead in enumerate(LEAD_HOURS):
        if seconds == timedelta(hours=lead).total_seconds():
            return index
    raise ValueError("Target endpoint does not match a forecast lead exactly.")


class NearestSphericalOperator(nn.Module):
    """Point readout at the nearest Voronoi centre, not an area average."""

    def __init__(self, grid: SphereGrid, station_coordinates):
        super().__init__()
        coordinates = np.asarray(station_coordinates, dtype=np.float64)
        if (coordinates.ndim != 2 or coordinates.shape[1] != 2 or not len(coordinates)
                or not np.isfinite(coordinates).all()
                or np.any(np.abs(coordinates[:, 0]) > 90)):
            raise ValueError("Finite station latitude/longitude pairs are required.")
        cells = grid.locate(coordinates[:, 0], coordinates[:, 1])
        self.register_buffer("cells", torch.as_tensor(cells, dtype=torch.long))
        self.register_buffer("station_xyz", torch.as_tensor(
            unit_xyz(coordinates[:, 0], coordinates[:, 1]), dtype=torch.float32))
        self.n_cells = grid.n_cells
        self.grid_fingerprint = grid.fingerprint

    def forward(self, field: Tensor) -> Tensor:
        if field.shape[0] != self.n_cells:
            raise ValueError("Field does not cover the global grid.")
        return field[self.cells]


class StationObservationModel(nn.Module):
    """12-hour causal ingestion followed by 24 autonomous three-hour updates.

    Missing values have no observation contribution. The learned initial latent
    is a model parameter, not an observation, ISA profile or climate normal.
    Upper-air diagnostics are deliberately uncalibrated and never loss targets.
    train_mean/train_std describe only the six admitted station-native channels.
    """

    def __init__(self, grids, station_coordinates, train_mean, train_std, *, hidden=32,
                 step_hours=3):
        super().__init__()
        if not grids or hidden < 8 or step_hours != 3:
            raise ValueError("Need a global grid, hidden >= 8 and three-hour steps.")
        mean = torch.as_tensor(train_mean, dtype=torch.float32)
        std = torch.as_tensor(train_std, dtype=torch.float32)
        if (mean.shape != (6,) or std.shape != (6,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or bool((std <= 0).any())):
            raise ValueError("Six finite, positive train-only station statistics are required.")
        self.register_buffer("train_mean", mean.clone())
        self.register_buffer("train_std", std.clone())
        self.register_buffer("pressure_hpa", torch.tensor(PRESSURE_HPA, dtype=torch.float32))
        self.operator = NearestSphericalOperator(grids[0], station_coordinates)
        self.graph = GraphOps(grids[0])
        self.register_buffer("xyz", torch.tensor(grids[0].xyz, dtype=torch.float32))
        self.hidden = hidden
        self.step_hours = step_hours
        self.geometry = nn.Linear(3, hidden)
        self.level_embedding = nn.Parameter(torch.randn(37, hidden) * .01)
        self.input_encoder = nn.Sequential(nn.Linear(12, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.ingest = nn.GRUCell(hidden, hidden)
        self.dynamic = nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.transition = nn.GRUCell(hidden, hidden)
        self.native_head = nn.Sequential(nn.Linear(2 * hidden + 3, hidden), nn.GELU(), nn.Linear(hidden, 6))
        self.profile_head = nn.Linear(hidden, 6)
        self.grid_fingerprint = grids[0].fingerprint

    def physical_native(self, normalized: Tensor) -> Tensor:
        return normalized * self.train_std + self.train_mean

    def forward(self, history: Tensor, history_mask: Tensor) -> ObservationForecast:
        expected = (12, self.operator.cells.numel(), 6)
        if history.shape != expected or history_mask.shape != expected or history_mask.dtype != torch.bool:
            raise ValueError("History and boolean mask must have shape [12, stations, 6].")
        if not bool(torch.isfinite(history[history_mask]).all()):
            raise ValueError("An admitted measurement is nonfinite.")
        # Mask before normalization so missing NaNs cannot enter arithmetic.
        normalized = (torch.where(history_mask, history, self.train_mean) - self.train_mean) / self.train_std
        n, d = self.xyz.shape[0], self.hidden
        state = self.geometry(self.xyz)[:, None, :] + self.level_embedding[None, :, :]
        for values, mask in zip(normalized, history_mask):
            tokens = self.input_encoder(torch.cat([values, mask.to(values.dtype)], dim=-1))
            valid = mask.any(-1)
            pooled = values.new_zeros(n, d).index_add(0, self.operator.cells,
                                                     tokens * valid[:, None])
            counts = values.new_zeros(n).index_add(0, self.operator.cells, valid.to(values.dtype))
            pooled = pooled / counts.clamp_min(1)[:, None]
            updated = self.ingest(pooled[:, None, :].expand(-1, 37, -1).reshape(-1, d),
                                 state.reshape(-1, d)).reshape(n, 37, d)
            state = torch.where((counts > 0)[:, None, None], updated, state)
        predictions = []
        for _ in LEAD_HOURS:
            vertical_mean = state.mean(1, keepdim=True).expand_as(state)
            forcing = self.dynamic(torch.cat([state, self.graph.neighbours(state), vertical_mean], -1))
            state = self.transition(forcing.reshape(-1, d), state.reshape(-1, d)).reshape(n, 37, d)
            native_context = torch.cat([self.operator(state.mean(1)), self.operator(state[:, 0]),
                                        self.operator.station_xyz], -1)
            predictions.append(self.native_head(native_context))
        profiles = self.profile_head(state)
        return ObservationForecast(torch.stack(predictions), LEAD_HOURS, profiles,
                                   torch.zeros_like(profiles, dtype=torch.bool))


def station_loss(prediction: Tensor, physical_target: Tensor, mask: Tensor,
                 train_mean: Tensor, train_std: Tensor) -> Tensor:
    """Equal-variable Huber mean over admitted station aggregates; no cell areas.

    A station scalar has equal weight inside its variable. Missing channels are
    excluded. Variables without any admitted targets have no loss term.
    """
    if (prediction.shape != physical_target.shape or prediction.shape != mask.shape
            or prediction.shape[-1] != 6 or mask.dtype != torch.bool):
        raise ValueError("Prediction, target and boolean mask shapes differ.")
    if (train_mean.shape != (6,) or train_std.shape != (6,)
            or not bool(torch.isfinite(train_mean).all())
            or not bool(torch.isfinite(train_std).all()) or bool((train_std <= 0).any())):
        raise ValueError("Invalid station-only training statistics.")
    if not bool(mask.any()) or not bool(torch.isfinite(physical_target[mask]).all()):
        raise ValueError("Need finite admitted station targets.")
    if not bool(torch.isfinite(prediction[mask]).all()):
        raise ValueError("Nonfinite station prediction.")
    safe_target = torch.where(mask, physical_target, train_mean)
    target = (safe_target - train_mean) / train_std
    components = [F.huber_loss(prediction[..., v][mask[..., v]], target[..., v][mask[..., v]])
                  for v in range(6) if bool(mask[..., v].any())]
    return torch.stack(components).mean()
