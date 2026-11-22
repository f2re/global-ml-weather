"""Station climatology anomaly model.

The neural network ingests observed anomalies relative to station climatological
normals onto a global spherical grid, propagates the atmospheric dynamics, and
predicts exclusively the future anomaly (departure from the climatological normal).
Physical forecasts are obtained by adding the predicted anomaly to the frozen normal.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .model import GraphOps
from .multimodal.contracts import Sensor, Sequence
from .multimodal.fusion import SatelliteBank, SourceFusion
from .observation_model import LEAD_HOURS, NearestSphericalOperator
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
    satellite_support: Tensor | None = None  # bool [cells], at issue time


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

    def analysis_state(self, history_anomalies: Tensor, history_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Ingest 12 hours of observed departures and return the analysis state and diagnostic profiles at t0."""
        state = self._station_state(history_anomalies, history_mask)
        return state, self.profile_head(state)

    def _station_state(self, history_anomalies: Tensor, history_mask: Tensor) -> Tensor:
        expected = (12, self.operator.cells.numel(), 6)
        if history_anomalies.shape != expected or history_mask.shape != expected or history_mask.dtype != torch.bool:
            raise ValueError("History departures and boolean mask must have shape [12, stations, 6].")
        if not bool(torch.isfinite(history_anomalies[history_mask]).all()):
            raise ValueError("An admitted anomaly measurement is nonfinite.")

        n, d = self.xyz.shape[0], self.hidden
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

        return state

    def forward(self, history_anomalies: Tensor, history_mask: Tensor) -> AnomalyForecast:
        """Ingest 12 hours of observed departures, evolve the global state, and predict future anomalies."""
        return self._forecast_from_state(self._station_state(history_anomalies, history_mask))

    def _forecast_from_state(self, state: Tensor, *, satellite_support: Tensor | None = None) -> AnomalyForecast:
        n, _, d = state.shape
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
            satellite_support=satellite_support,
        )


def _supported_station_sensor(sensor: Sensor) -> bool:
    if sensor.kind != "imager":
        return False
    if sensor.source == "arktika_m":
        return True
    if sensor.source != "goes_abi" or sensor.platform != "GOES-18" or sensor.data_kind != "real":
        return False
    return tuple((channel.id, channel.quantity, channel.units) for channel in sensor.channels) == (
        ("13", "brightness_temperature", "K"),
    )


class SatelliteStationAnomalyModel(StationAnomalyModel):
    """Add a masked infrared feature update at issue time before the station rollout."""

    satellite_prefixes = ("satellite_bank.", "source_fusion.")

    def __init__(
        self, grids, station_coordinates, scales, *, sensors: Iterable[Sensor], hidden: int = 32,
        step_hours: int = 3, base_channels: int = 16,
    ):
        sensors = tuple(sensors)
        if (not sensors or len({sensor.id for sensor in sensors}) != len(sensors)
                or not all(_supported_station_sensor(sensor) for sensor in sensors)):
            raise ValueError("Station fusion requires unique Arctic imagers or the real GOES-18 ABI C13 channel.")
        if hidden % 4:
            raise ValueError("Station satellite fusion requires hidden divisible by four.")
        for sensor in sensors:
            sensor.require_normalization()
        super().__init__(grids, station_coordinates, scales, hidden=hidden, step_hours=step_hours)
        self.sensors = sensors
        self.output_grid = grids[0]
        self.satellite_bank = SatelliteBank(self.sensors, hidden, base_channels)
        self.source_fusion = SourceFusion(hidden)

    def get_extra_state(self):
        return {
            "architecture": "station-anomaly-satellite-fusion-1",
            "grid_fingerprint": self.grid_fingerprint,
            "sensor_contracts": [sensor.signature for sensor in self.sensors],
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Station satellite checkpoint has different sensor contracts or geometry.")

    def initialize_station_core(self, checkpoint: Mapping[str, Tensor]) -> None:
        """Warm start the unchanged station core and verify its static arrays before loading."""
        initialized = self.state_dict()
        legacy = {
            name: tensor for name, tensor in initialized.items()
            if name != "_extra_state" and not name.startswith(self.satellite_prefixes)
        }
        if legacy.keys() != checkpoint.keys():
            raise ValueError("Unexpected station core checkpoint keys.")
        static = dict(self.named_buffers())
        for name, expected in legacy.items():
            saved = checkpoint[name]
            if (not isinstance(saved, Tensor) or saved.dtype != expected.dtype or saved.layout != expected.layout
                    or saved.shape != expected.shape):
                raise ValueError(f"Incompatible station core tensor: {name}.")
            if saved.layout == torch.sparse_coo:
                actual = saved.cpu().coalesce()
                wanted = expected.cpu().coalesce()
                if not (torch.equal(actual.indices(), wanted.indices()) and torch.equal(actual.values(), wanted.values())):
                    raise ValueError(f"Station core geometry or normalization differs: {name}.")
            elif name in static:
                if not torch.equal(saved.cpu(), expected.cpu()):
                    raise ValueError(f"Station core geometry or normalization differs: {name}.")
            elif not torch.isfinite(saved).all():
                raise ValueError(f"Nonfinite station core tensor: {name}.")
        initialized.update(checkpoint)
        self.load_state_dict(initialized, strict=True)

    def _satellite_analysis(
        self, history_anomalies: Tensor, history_mask: Tensor, issue_time: datetime | str | None,
        satellite_sequences: Iterable[Sequence],
    ) -> tuple[Tensor, Tensor]:
        state = self._station_state(history_anomalies, history_mask)
        by_id = {sensor.id: sensor for sensor in self.sensors}
        admitted = []
        seen = set()
        for sequence in satellite_sequences:
            if issue_time is None:
                raise ValueError("Satellite frames require an issue time.")
            if sequence.sensor_id not in by_id:
                raise ValueError("Unregistered station satellite sensor.")
            causal = sequence.causal(issue_time, by_id[sequence.sensor_id])
            if causal is None:
                continue
            if seen.intersection(causal.frame_ids):
                raise ValueError("Duplicate satellite frame in station analysis.")
            seen.update(causal.frame_ids)
            admitted.append(causal.to(state.device))
        fields, masks = self.satellite_bank(admitted, issue_time, self.output_grid)
        if not masks:
            return state, state.new_zeros((state.shape[0],), dtype=torch.bool)
        return self.source_fusion(state, fields, masks), torch.stack(masks).any(0)

    def analysis_state(
        self, history_anomalies: Tensor, history_mask: Tensor, *, issue_time: datetime | str | None = None,
        satellite_sequences: Iterable[Sequence] = (),
    ) -> tuple[Tensor, Tensor]:
        state, _ = self._satellite_analysis(history_anomalies, history_mask, issue_time, satellite_sequences)
        return state, self.profile_head(state)

    def forward(
        self, history_anomalies: Tensor, history_mask: Tensor, *, issue_time: datetime | str | None = None,
        satellite_sequences: Iterable[Sequence] = (),
    ) -> AnomalyForecast:
        state, support = self._satellite_analysis(history_anomalies, history_mask, issue_time, satellite_sequences)
        return self._forecast_from_state(state, satellite_support=support)


def anomaly_loss(
    predicted_anomalies: Tensor,
    target_anomalies: Tensor,
    mask: Tensor,
    bias_weight: float = 0.0,
) -> Tensor:
    """Equal-variable Huber loss evaluated strictly between predicted and true departures from climate.

    When bias_weight > 0, an explicit quadratic penalty on mean domain bias is added to prevent
    unphysical thermal and baric drift.
    """
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

    components = []
    for v in range(6):
        mv = mask[..., v]
        if bool(mv.any()):
            pred_v = predicted_anomalies[..., v][mv]
            targ_v = target_anomalies[..., v][mv]
            h_loss = F.huber_loss(pred_v, targ_v)
            if bias_weight > 0.0:
                mean_bias = (pred_v - targ_v).mean()
                h_loss = h_loss + bias_weight * mean_bias.pow(2)
            components.append(h_loss)
    return torch.stack(components).mean()
