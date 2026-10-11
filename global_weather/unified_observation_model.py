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
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden < 16:
            raise ValueError("Hidden dimension must be >= 16.")

        self.grid = grid
        self.hidden = hidden
        self.dropout = dropout
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
            c_mean = np.nan_to_num(np.asarray(climatology_grid_mean, dtype=np.float32), nan=0.0)
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
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        self.surface_ingest = nn.GRUCell(hidden, hidden)

        # Boundary layer vertical projection from surface observations
        self.surface_to_column = nn.Linear(hidden, hidden)
        decay = torch.exp(-torch.arange(self.n_levels).float() / 4.0)
        self.register_buffer("bl_decay", decay / decay.sum())

        # 2. Upper-air sounding encoder: takes [5 profile values + 5 mask] = 10
        self.profile_encoder = nn.Sequential(
            nn.Linear(10, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        self.profile_ingest = nn.GRUCell(hidden, hidden)

        # 1D Vertical stratification coupling along 37 pressure levels
        self.vertical_coupling = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
        )

        # Solar diurnal cycle encoder
        self.solar_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        # Dynamics: spherical horizontal neighbours + vertical coupling + local state + solar forcing
        self.dynamic = nn.Sequential(
            nn.Linear(4 * hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        self.transition = nn.GRUCell(hidden, hidden)
        self.layer_norm = nn.LayerNorm(hidden)

        # Prediction heads
        # Surface head maps station context to 6 surface anomalies
        self.surface_head = nn.Sequential(
            nn.Linear(2 * hidden + 3, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 6),
        )
        # Profile head maps cell state directly to 5 profile anomalies across 37 levels
        self.profile_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 5),
        )

    def forward(
        self,
        surface_anomalies: Tensor,      # [12, n_stations, 6]
        surface_mask: Tensor,           # [12, n_stations, 6]
        month_idx: int = 0,
        profile_anomalies: Tensor | None = None, # [12, n_cells, 37, 5] optional
        profile_mask: Tensor | None = None,      # [12, n_cells, 37, 5] optional
        issue_hour: int = 0,
    ) -> UnifiedForecast:
        n, levels, d = self.n_cells, self.n_levels, self.hidden

        # Base atmospheric column state: geometry + level embedding + climate context
        base_geom = self.geometry(self.xyz)[:, None, :] + self.level_embedding[None, :, :]
        clim_features = self.climate_encoder(self.climatology_mean[month_idx])
        state = base_geom + clim_features

        # Station dropout during training for robust spatial generalization
        if self.training and self.dropout > 0:
            drop_prob = min(0.15, self.dropout * 2.0)
            st_keep = torch.rand(surface_anomalies.shape[1], device=surface_anomalies.device) > drop_prob
            clean_mask = surface_mask & st_keep[None, :, None]
        else:
            clean_mask = surface_mask

        # 1. Ingest surface station observations
        clean_surf = torch.where(clean_mask, surface_anomalies, torch.zeros_like(surface_anomalies))
        for vals, mask in zip(clean_surf, clean_mask):
            tokens = self.surface_encoder(torch.cat([vals, mask.to(vals.dtype)], dim=-1))
            valid = mask.any(-1)
            pooled = vals.new_zeros(n, d).index_add(0, self.station_operator.cells, tokens * valid[:, None])
            counts = vals.new_zeros(n).index_add(0, self.station_operator.cells, valid.to(vals.dtype))
            pooled = pooled / counts.clamp_min(1)[:, None]

            # Ingest onto surface level (index 0)
            updated_surf = self.surface_ingest(pooled, state[:, 0])
            new_surf = torch.where((counts > 0)[:, None], updated_surf, state[:, 0])
            state = torch.cat([new_surf[:, None, :], state[:, 1:]], dim=1)

            # Inject boundary layer influence across lower tropospheric levels (1000 - 850 hPa)
            bl_inject = self.surface_to_column(pooled)[:, None, :] * self.bl_decay[None, :, None]
            state = state + torch.where((counts > 0)[:, None, None], bl_inject, torch.zeros_like(state))

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
        for lead in LEAD_HOURS:
            # 1D Vertical stratification along 37 levels
            vert_forcing = self.vertical_coupling(state.permute(0, 2, 1)).permute(0, 2, 1)

            # Solar diurnal cycle
            target_hour = (issue_hour + lead) % 24
            hour_rad = 2.0 * np.pi * target_hour / 24.0
            solar_feats = state.new_tensor([np.sin(hour_rad), np.cos(hour_rad)])
            solar_forcing = self.solar_encoder(solar_feats)[None, None, :].expand(n, levels, d)

            # Horizontal neighbours from spherical graph
            horiz_neighbours = self.graph.neighbours(state)

            # Joint forcing and Pre-LN residual transition
            forcing = self.dynamic(torch.cat([state, horiz_neighbours, vert_forcing, solar_forcing], dim=-1))
            delta = self.transition(forcing.reshape(-1, d), state.reshape(-1, d)).reshape(n, levels, d)
            state = self.layer_norm(state + delta)

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
