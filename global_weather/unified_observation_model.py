"""Joint observation model assimilating surface stations, upper-air soundings, and satellite radiances.

Conditioned on official 30-year multi-decadal NetCDF climatology (NOAA 1991-2020 LTM)
and verified GMT Earth Relief 30s DEM (896 MB) topography.
Predicts strictly dimensionless departures from climatology across 37 pressure levels up to 10 days (240 h).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .grid import build_grid, SphereGrid
from .model import GraphOps
from .station_anomaly_model import NearestSphericalOperator
from .vertical import PRESSURE_HPA
from .terrain_features import H_SCALE, barometric_pressure_ratio

LEAD_HOURS = list(range(3, 243, 3))  # 80 lead steps: 3, 6, 9, ..., 240 hours (10 days)


@dataclass(frozen=True)
class UnifiedForecast:
    surface_anomalies: Tensor          # [n_leads, n_stations, 6] or [B, n_leads, n_stations, 6]
    profile_anomalies: Tensor          # [n_leads, n_cells, 37, 5] or [B, n_leads, n_cells, 37, 5]
    lead_hours: list[int]
    prediction_target: str = "joint dimensionless anomaly departure from official 30-year climatology"
    scientific_acceptance: bool = False


class UnifiedObservationModel(nn.Module):
    """Assimilates surface, sounding, and satellite observations onto a 37-level spherical column."""

    def __init__(
        self,
        grid: SphereGrid,
        station_latlon: np.ndarray,
        surface_scales: Tensor | np.ndarray,
        profile_scales: Tensor | np.ndarray,
        hidden: int = 64,
        climatology_grid_mean: np.ndarray | None = None,
        dropout: float = 0.0,
        terrain_elevation_m: Tensor | np.ndarray | None = None,
        terrain_land_fraction: Tensor | np.ndarray | None = None,
        terrain_slope: Tensor | np.ndarray | None = None,
        station_elevation_m: Tensor | np.ndarray | None = None,
        lead_hours: list[int] | None = None,
    ):
        super().__init__()
        if hidden < 16:
            raise ValueError("Hidden dimension must be >= 16.")

        self.grid = grid
        self.hidden = hidden
        self.dropout = dropout
        self.n_cells = grid.xyz.shape[0]
        self.n_levels = len(PRESSURE_HPA)
        self.lead_hours = list(lead_hours) if lead_hours is not None else list(LEAD_HOURS)

        # Station geometry & Voronoi operator
        self.station_operator = NearestSphericalOperator(grid, station_latlon)
        self.graph = GraphOps(grid)

        # Basic buffers
        self.register_buffer("xyz", torch.tensor(grid.xyz, dtype=torch.float32))
        self.register_buffer("pressure_hpa", torch.tensor(PRESSURE_HPA, dtype=torch.float32))
        self.register_buffer("surface_scales", torch.as_tensor(surface_scales, dtype=torch.float32))
        self.register_buffer("profile_scales", torch.as_tensor(profile_scales, dtype=torch.float32))

        # Terrain buffers (896 MB DEM features)
        n_st = len(station_latlon)
        t_elev = np.zeros(self.n_cells, dtype=np.float32) if terrain_elevation_m is None else np.asarray(terrain_elevation_m, dtype=np.float32)
        t_land = np.zeros(self.n_cells, dtype=np.float32) if terrain_land_fraction is None else np.asarray(terrain_land_fraction, dtype=np.float32)
        t_slope = np.zeros(self.n_cells, dtype=np.float32) if terrain_slope is None else np.asarray(terrain_slope, dtype=np.float32)
        st_elev = np.zeros(n_st, dtype=np.float32) if station_elevation_m is None else np.asarray(station_elevation_m, dtype=np.float32)

        self.register_buffer("terrain_elevation", torch.tensor(t_elev))
        self.register_buffer("terrain_land_fraction", torch.tensor(t_land))
        self.register_buffer("terrain_slope", torch.tensor(t_slope))
        self.register_buffer("station_elevation", torch.tensor(st_elev))

        # Climatology background buffer: [12, n_cells, 37, 5]
        if climatology_grid_mean is not None:
            c_mean = np.nan_to_num(np.asarray(climatology_grid_mean, dtype=np.float32), nan=0.0)
            if c_mean.shape != (12, self.n_cells, self.n_levels, 5):
                raise ValueError("Climatology mean shape must be [12, n_cells, 37, 5].")
            self.register_buffer("climatology_mean", torch.tensor(c_mean))
        else:
            self.register_buffer("climatology_mean", torch.zeros(12, self.n_cells, self.n_levels, 5))

        # Spatial geometry + DEM embedding: (x, y, z, elevation/H_SCALE, land_fraction, slope) = 6 features
        self.geometry = nn.Linear(6, hidden)
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

        # 3. Satellite infrared radiometer encoder (Arktika-M / MSU-GS): takes [1 radiance/Tb + 1 mask] = 2
        self.satellite_encoder = nn.Sequential(
            nn.Linear(2, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, hidden),
        )
        # Upper-troposphere / cloud-top vertical influence centered around ~300-400 hPa (indices 16-20)
        sat_vert = torch.exp(-((torch.arange(self.n_levels).float() - 18.0) / 4.0) ** 2)
        self.register_buffer("sat_decay", sat_vert / sat_vert.sum())

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
        # Surface head maps station context (mean_column, surface_level, station_xyz, station_elevation) to 6 anomalies
        self.surface_head = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden),
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

    def transition_step(self, state: Tensor, solar_feats: Tensor) -> Tensor:
        """Autoregressive step with horizontal, vertical, and diurnal dynamic coupling."""
        B, n, levels, d = state.shape

        # 1D Vertical stratification along 37 levels
        vert_forcing = self.vertical_coupling(
            state.reshape(B * n, levels, d).permute(0, 2, 1)
        ).permute(0, 2, 1).reshape(B, n, levels, d)

        # Solar diurnal cycle forcing
        solar_forcing = self.solar_encoder(solar_feats)[:, None, None, :].expand(B, n, levels, d)

        # Horizontal neighbours from spherical graph
        horiz_neighbours = self.graph.neighbours(state)

        # Joint forcing and Pre-LN residual transition
        forcing = self.dynamic(torch.cat([state, horiz_neighbours, vert_forcing, solar_forcing], dim=-1))
        delta = self.transition(forcing.reshape(B * n * levels, d), state.reshape(B * n * levels, d)).reshape(B, n, levels, d)
        return self.layer_norm(state + delta)

    def forward(
        self,
        surface_anomalies: Tensor,      # [12, n_stations, 6] or [B, 12, n_stations, 6]
        surface_mask: Tensor,           # [12, n_stations, 6] or [B, 12, n_stations, 6]
        month_idx: int | list[int] | Tensor = 0,
        profile_anomalies: Tensor | None = None,   # [12, n_cells, 37, 5] or [B, 12, n_cells, 37, 5]
        profile_mask: Tensor | None = None,        # [12, n_cells, 37, 5] or [B, 12, n_cells, 37, 5]
        satellite_anomalies: Tensor | None = None, # [12, n_cells, 1] or [B, 12, n_cells, 1]
        satellite_mask: Tensor | None = None,      # [12, n_cells, 1] or [B, 12, n_cells, 1]
        issue_hour: int | float | list[int | float] | Tensor = 0,
    ) -> UnifiedForecast:
        is_unbatched = surface_anomalies.dim() == 3
        if is_unbatched:
            surface_anomalies = surface_anomalies.unsqueeze(0)
            surface_mask = surface_mask.unsqueeze(0)
            if profile_anomalies is not None:
                profile_anomalies = profile_anomalies.unsqueeze(0)
            if profile_mask is not None:
                profile_mask = profile_mask.unsqueeze(0)
            if satellite_anomalies is not None:
                satellite_anomalies = satellite_anomalies.unsqueeze(0)
            if satellite_mask is not None:
                satellite_mask = satellite_mask.unsqueeze(0)
            if isinstance(month_idx, int):
                month_idx = [month_idx]
            if isinstance(issue_hour, (int, float)):
                issue_hour = [issue_hour]

        B = surface_anomalies.shape[0]
        n, levels, d = self.n_cells, self.n_levels, self.hidden
        device = surface_anomalies.device

        if not isinstance(month_idx, torch.Tensor):
            month_idx = torch.as_tensor(month_idx, dtype=torch.long, device=device)
        if not isinstance(issue_hour, torch.Tensor):
            issue_hour = torch.as_tensor(issue_hour, dtype=torch.float32, device=device)

        # Base atmospheric column state: geometry (3D coords + 3D DEM) + level embedding + climate context
        terrain_feats = torch.stack([
            self.terrain_elevation / H_SCALE,
            self.terrain_land_fraction,
            self.terrain_slope,
        ], dim=-1) # [n, 3]
        cell_geom = torch.cat([self.xyz, terrain_feats], dim=-1) # [n, 6]
        base_geom = (self.geometry(cell_geom)[:, None, :] + self.level_embedding[None, :, :])[None, ...].expand(B, n, levels, d)
        clim_features = self.climate_encoder(self.climatology_mean[month_idx])
        state = base_geom + clim_features

        # Station dropout during training for robust spatial generalization
        if self.training and self.dropout > 0:
            drop_prob = min(0.15, self.dropout * 2.0)
            st_keep = torch.rand(B, surface_anomalies.shape[2], device=device) > drop_prob
            clean_mask = surface_mask & st_keep[:, None, :, None]
        else:
            clean_mask = surface_mask

        # 1. Ingest surface station observations
        clean_surf = torch.where(clean_mask, surface_anomalies, torch.zeros_like(surface_anomalies))
        for step in range(clean_surf.shape[1]):
            vals = clean_surf[:, step]   # [B, n_stations, 6]
            mask = clean_mask[:, step]   # [B, n_stations, 6]
            tokens = self.surface_encoder(torch.cat([vals, mask.to(vals.dtype)], dim=-1))
            valid = mask.any(-1)         # [B, n_stations]

            pooled_list = []
            counts_list = []
            for b in range(B):
                p = vals.new_zeros(n, d).index_add(0, self.station_operator.cells, tokens[b] * valid[b, :, None])
                c = vals.new_zeros(n).index_add(0, self.station_operator.cells, valid[b].to(vals.dtype))
                pooled_list.append(p / c.clamp_min(1)[:, None])
                counts_list.append(c)
            pooled = torch.stack(pooled_list) # [B, n, d]
            counts = torch.stack(counts_list) # [B, n]

            # Ingest onto surface level (index 0)
            updated_surf = self.surface_ingest(pooled.reshape(B * n, d), state[:, :, 0].reshape(B * n, d)).reshape(B, n, d)
            new_surf = torch.where((counts > 0)[..., None], updated_surf, state[:, :, 0])
            state = torch.cat([new_surf[:, :, None, :], state[:, :, 1:]], dim=2)

            # Inject boundary layer influence across lower tropospheric levels (1000 - 850 hPa)
            bl_inject = self.surface_to_column(pooled)[:, :, None, :] * self.bl_decay[None, None, :, None]
            state = state + torch.where((counts > 0)[..., None, None], bl_inject, torch.zeros_like(state))

        # 2. Ingest aerological sounding observations (if provided)
        if profile_anomalies is not None and profile_mask is not None:
            clean_prof = torch.where(profile_mask, profile_anomalies, torch.zeros_like(profile_anomalies))
            for step in range(clean_prof.shape[1]):
                p_vals = clean_prof[:, step]
                p_mask = profile_mask[:, step]
                p_tokens = self.profile_encoder(torch.cat([p_vals, p_mask.to(p_vals.dtype)], dim=-1))
                p_valid = p_mask.any(-1)
                updated_prof = self.profile_ingest(
                    p_tokens.reshape(B * n * levels, d),
                    state.reshape(B * n * levels, d),
                ).reshape(B, n, levels, d)
                state = torch.where(p_valid[..., None], updated_prof, state)

        # 3. Ingest real satellite radiances by sensor visibility mask (e.g. Arktika-M high-elliptic disc)
        if satellite_anomalies is not None and satellite_mask is not None:
            clean_sat = torch.where(satellite_mask, satellite_anomalies, torch.zeros_like(satellite_anomalies))
            for step in range(clean_sat.shape[1]):
                sat_vals = clean_sat[:, step]
                sat_m = satellite_mask[:, step]
                sat_tok = self.satellite_encoder(torch.cat([sat_vals, sat_m.to(sat_vals.dtype)], dim=-1))
                sat_valid = sat_m.any(-1)
                sat_inject = sat_tok[:, :, None, :] * self.sat_decay[None, None, :, None]
                state = state + torch.where(sat_valid[..., None, None], sat_inject, torch.zeros_like(state))

        # 4. Dynamic evolution over lead times (up to 10 days / 240 hours) with activation checkpointing
        surface_preds = []
        profile_preds = []
        for lead in self.lead_hours:
            target_hours = (issue_hour + lead) % 24.0
            hour_rads = 2.0 * np.pi * target_hours / 24.0
            solar_feats = torch.stack([torch.sin(hour_rads), torch.cos(hour_rads)], dim=-1) # [B, 2]

            # Autoregressive rollout with gradient checkpointing to prevent VRAM overflow
            if self.training and state.requires_grad:
                state = checkpoint.checkpoint(self.transition_step, state, solar_feats, use_reentrant=False)
            else:
                state = self.transition_step(state, solar_feats)

            # Surface prediction at station locations conditioned on station elevation
            st_mean = state.mean(2)[:, self.station_operator.cells]
            st_surf = state[:, :, 0][:, self.station_operator.cells]
            st_xyz = self.station_operator.station_xyz[None, ...].expand(B, -1, -1)
            st_elev = (self.station_elevation[None, :, None] / H_SCALE).expand(B, -1, -1)
            station_context = torch.cat([st_mean, st_surf, st_xyz, st_elev], dim=-1)
            surface_preds.append(self.surface_head(station_context))

            # Full 3D profile prediction across grid cells and levels
            profile_preds.append(self.profile_head(state))

        pred_surf = torch.stack(surface_preds, dim=1) # [B, n_leads, n_stations, 6]
        pred_prof = torch.stack(profile_preds, dim=1) # [B, n_leads, n, levels, 5]

        if is_unbatched:
            pred_surf = pred_surf.squeeze(0)
            pred_prof = pred_prof.squeeze(0)

        return UnifiedForecast(
            surface_anomalies=pred_surf,
            profile_anomalies=pred_prof,
            lead_hours=self.lead_hours,
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

    # Surface component (slice matching lead horizon)
    if mask_surface.any():
        n_s = min(forecast.surface_anomalies.shape[-3] if forecast.surface_anomalies.dim() == 4 else forecast.surface_anomalies.shape[0],
                  target_surface.shape[-3] if target_surface.dim() == 4 else target_surface.shape[0])
        f_surf = forecast.surface_anomalies[:, :n_s] if forecast.surface_anomalies.dim() == 4 else forecast.surface_anomalies[:n_s]
        t_surf = target_surface[:, :n_s] if target_surface.dim() == 4 else target_surface[:n_s]
        m_surf = mask_surface[:, :n_s] if mask_surface.dim() == 4 else mask_surface[:n_s]

        surf_components = [
            F.huber_loss(f_surf[..., v][m_surf[..., v]], t_surf[..., v][m_surf[..., v]])
            for v in range(6)
            if bool(m_surf[..., v].any())
        ]
        if surf_components:
            loss_parts.append(surface_weight * torch.stack(surf_components).mean())

    # Profile component (slice matching lead horizon)
    if target_profiles is not None and mask_profiles is not None and mask_profiles.any():
        n_p = min(forecast.profile_anomalies.shape[-4] if forecast.profile_anomalies.dim() == 5 else forecast.profile_anomalies.shape[0],
                  target_profiles.shape[-4] if target_profiles.dim() == 5 else target_profiles.shape[0])
        f_prof = forecast.profile_anomalies[:, :n_p] if forecast.profile_anomalies.dim() == 5 else forecast.profile_anomalies[:n_p]
        t_prof = target_profiles[:, :n_p] if target_profiles.dim() == 5 else target_profiles[:n_p]
        m_prof = mask_profiles[:, :n_p] if mask_profiles.dim() == 5 else mask_profiles[:n_p]

        prof_components = [
            F.huber_loss(f_prof[..., v][m_prof[..., v]], t_prof[..., v][m_prof[..., v]])
            for v in range(5)
            if bool(m_prof[..., v].any())
        ]
        if prof_components:
            loss_parts.append(profile_weight * torch.stack(prof_components).mean())

    if not loss_parts:
        raise ValueError("No valid targets found in joint observation loss computation.")

    return sum(loss_parts)
