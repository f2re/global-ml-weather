"""Joint observation model assimilating surface stations and upper-air soundings.

Conditioned on official 30-year multi-decadal NetCDF climatology (NOAA 1991-2020 LTM).
Predicts strictly dimensionless departures from climatology across 37 pressure levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .grid import build_grid, SphereGrid
from .model import GraphOps
from .observations import utc
from .station_anomaly_model import NearestSphericalOperator
from .vertical import PRESSURE_HPA

LEAD_HOURS = list(range(3, 73, 3))


@dataclass(frozen=True)
class UnifiedForecast:
    surface_anomalies: Tensor          # [24, n_stations, 6]
    profile_anomalies: Tensor          # [24, n_cells, 37, 5]
    lead_hours: list[int]
    prediction_target: str = "joint dimensionless anomaly departure from official 30-year climatology"
    scientific_acceptance: bool = False


class UnifiedObservationModel(nn.Module):
    """Assimilates surface and upper-air observations onto a 37-level spherical column."""

    def __init__(
        self,
        grid: SphereGrid,
        station_latlon: np.ndarray,
        surface_scales: Tensor | np.ndarray,
        profile_scales: Tensor | np.ndarray,
        hidden: int = 64,
        climatology_grid_mean: np.ndarray | None = None,
    ):
        super().__init__()
        if hidden < 16:
            raise ValueError("Hidden dimension must be >= 16.")

        self.grid = grid
        self.hidden = hidden
        self.n_cells = grid.xyz.shape[0]
        self.n_levels = len(PRESSURE_HPA)

        # Station geometry & Voronoi operator
        self.station_operator = NearestSphericalOperator(grid, station_latlon)
        self.graph = GraphOps(grid)

        # Buffers
        self.register_buffer("xyz", torch.tensor(grid.xyz, dtype=torch.float32))
        self.register_buffer("pressure_hpa", torch.tensor(PRESSURE_HPA, dtype=torch.float32))
        self.register_buffer("surface_scales", torch.as_tensor(surface_scales, dtype=torch.float32))
        self.register_buffer("profile_scales", torch.as_tensor(profile_scales, dtype=torch.float32))

        # Climatology background buffer: [12, n_cells, 37, 5]
        if climatology_grid_mean is not None:
            c_mean = np.asarray(climatology_grid_mean, dtype=np.float32)
            if c_mean.shape != (12, self.n_cells, self.n_levels, 5):
                raise ValueError("Climatology mean shape must be [12, n_cells, 37, 5].")
            self.register_buffer("climatology_mean", torch.tensor(c_mean))
        else:
            self.register_buffer("climatology_mean", torch.zeros(12, self.n_cells, self.n_levels, 5))

        # Spatial geometry embedding
        self.geometry = nn.Linear(3, hidden)
        self.level_embedding = nn.Parameter(torch.randn(self.n_levels, hidden) * 0.01)

        # Climate background encoder: takes 5 physical variables + 1 valid mask = 6
        self.climate_encoder = nn.Sequential(
            nn.Linear(5, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # 1. Surface observation encoder: takes [6 values + 6 mask] = 12
        self.surface_encoder = nn.Sequential(
            nn.Linear(12, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.surface_ingest = nn.GRUCell(hidden, hidden)

        # 2. Upper-air sounding encoder: takes [5 profile values + 5 mask] = 10
        self.profile_encoder = nn.Sequential(
            nn.Linear(10, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.profile_ingest = nn.GRUCell(hidden, hidden)

        # Dynamics: spherical horizontal neighbours + vertical mean + local state
        self.dynamic = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.transition = nn.GRUCell(hidden, hidden)

        # Prediction heads
        # Surface head maps station context to 6 surface anomalies
        self.surface_head = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6),
        )
        # Profile head maps cell state directly to 5 profile anomalies across 37 levels
        self.profile_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 5),
        )

    def forward(
        self,
        surface_anomalies: Tensor,      # [12, n_stations, 6]
        surface_mask: Tensor,           # [12, n_stations, 6]
        month_idx: int = 0,
        profile_anomalies: Tensor | None = None, # [12, n_cells, 37, 5] optional
        profile_mask: Tensor | None = None,      # [12, n_cells, 37, 5] optional
    ) -> UnifiedForecast:
        n, levels, d = self.n_cells, self.n_levels, self.hidden

        # Base atmospheric column state: geometry + level embedding + climate context
        base_geom = self.geometry(self.xyz)[:, None, :] + self.level_embedding[None, :, :]
        clim_features = self.climate_encoder(self.climatology_mean[month_idx])
        state = base_geom + clim_features

        # 1. Ingest surface station observations
        clean_surf = torch.where(surface_mask, surface_anomalies, torch.zeros_like(surface_anomalies))
        for vals, mask in zip(clean_surf, surface_mask):
            tokens = self.surface_encoder(torch.cat([vals, mask.to(vals.dtype)], dim=-1))
            valid = mask.any(-1)
            pooled = vals.new_zeros(n, d).index_add(0, self.station_operator.cells, tokens * valid[:, None])
            counts = vals.new_zeros(n).index_add(0, self.station_operator.cells, valid.to(vals.dtype))
            pooled = pooled / counts.clamp_min(1)[:, None]

            # Ingest onto surface level (index 0)
            updated_surf = self.surface_ingest(pooled, state[:, 0])
            new_surf = torch.where((counts > 0)[:, None], updated_surf, state[:, 0])
            state = torch.cat([new_surf[:, None, :], state[:, 1:]], dim=1)

        # 2. Ingest aerological sounding observations (if provided)
        if profile_anomalies is not None and profile_mask is not None:
            clean_prof = torch.where(profile_mask, profile_anomalies, torch.zeros_like(profile_anomalies))
            for p_vals, p_mask in zip(clean_prof, profile_mask):
                p_tokens = self.profile_encoder(torch.cat([p_vals, p_mask.to(p_vals.dtype)], dim=-1))
                p_valid = p_mask.any(-1) # [n, 37]
                updated_prof = self.profile_ingest(
                    p_tokens.reshape(-1, d),
                    state.reshape(-1, d),
                ).reshape(n, levels, d)
                state = torch.where(p_valid[..., None], updated_prof, state)

        # 3. Dynamic evolution over lead times
        surface_preds = []
        profile_preds = []
        for _ in LEAD_HOURS:
            vertical_mean = state.mean(1, keepdim=True).expand_as(state)
            forcing = self.dynamic(torch.cat([state, self.graph.neighbours(state), vertical_mean], dim=-1))
            state = self.transition(forcing.reshape(-1, d), state.reshape(-1, d)).reshape(n, levels, d)

            # Surface prediction at station locations
            station_context = torch.cat(
                [self.station_operator(state.mean(1)), self.station_operator(state[:, 0]), self.station_operator.station_xyz],
                dim=-1,
            )
            surface_preds.append(self.surface_head(station_context))

            # Full 3D profile prediction across grid cells and levels
            profile_preds.append(self.profile_head(state))

        return UnifiedForecast(
            surface_anomalies=torch.stack(surface_preds),
            profile_anomalies=torch.stack(profile_preds),
            lead_hours=LEAD_HOURS,
        )


def unified_loss(
    forecast: UnifiedForecast,
    target_surface: Tensor,
    mask_surface: Tensor,
    target_profiles: Tensor | None = None,
    mask_profiles: Tensor | None = None,
    surface_weight: float = 0.5,
    profile_weight: float = 0.5,
) -> Tensor:
    """Joint Huber loss for surface departures and vertical sounding departures."""
    loss_parts = []

    # Surface component
    if mask_surface.any():
        surf_components = [
            F.huber_loss(forecast.surface_anomalies[..., v][mask_surface[..., v]], target_surface[..., v][mask_surface[..., v]])
            for v in range(6)
            if bool(mask_surface[..., v].any())
        ]
        if surf_components:
            loss_parts.append(surface_weight * torch.stack(surf_components).mean())

    # Profile component
    if target_profiles is not None and mask_profiles is not None and mask_profiles.any():
        prof_components = [
            F.huber_loss(forecast.profile_anomalies[..., v][mask_profiles[..., v]], target_profiles[..., v][mask_profiles[..., v]])
            for v in range(5)
            if bool(mask_profiles[..., v].any())
        ]
        if prof_components:
            loss_parts.append(profile_weight * torch.stack(prof_components).mean())

    if not loss_parts:
        raise ValueError("No valid targets found in joint observation loss computation.")

    return sum(loss_parts)
