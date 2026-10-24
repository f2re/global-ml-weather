"""Couple multimodal analysis to the hydrostatic forecast core.

The analysis branch is used only to assimilate observations.  Its compact latent
state is translated to the 37-level hidden state expected by the hydrostatic
forecast branch.  The forecast branch then advances autonomously; observations
are not ingested a second time.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch
from torch import nn

from .observations import utc

ARCHITECTURE = "coupled-multimodal-hydrostatic-v1"


@dataclass(frozen=True)
class CoupledAnalysisState:
    """Hydrostatic initial state produced from a causal multimodal analysis."""

    hidden: torch.Tensor
    valid_time: datetime
    evidence: tuple
    analysis_schema: dict


class LatentToPressureBridge(nn.Module):
    """Translate K latent slots into a residual correction on 37 pressure levels.

    Query vectors come from the hydrostatic background state.  Keys/values come
    from the assimilated latent slots.  A bounded gate keeps a fresh bridge near
    the established forecast background until it has been trained on paired
    analysis/forecast examples.
    """

    def __init__(self, source_hidden: int, target_hidden: int, *, heads: int = 4,
                 max_residual: float = 0.25):
        super().__init__()
        if min(source_hidden, target_hidden, heads) < 1 or target_hidden % heads:
            raise ValueError("Bridge dimensions are incompatible with attention heads.")
        if not 0 < max_residual <= 1:
            raise ValueError("max_residual must be within (0, 1].")
        self.source_hidden = int(source_hidden)
        self.target_hidden = int(target_hidden)
        self.heads = int(heads)
        self.max_residual = float(max_residual)
        self.source_projection = nn.Linear(source_hidden, target_hidden)
        self.query_norm = nn.LayerNorm(target_hidden)
        self.source_norm = nn.LayerNorm(target_hidden)
        self.attention = nn.MultiheadAttention(target_hidden, heads, batch_first=True)
        self.delta = nn.Sequential(
            nn.LayerNorm(target_hidden),
            nn.Linear(target_hidden, 2 * target_hidden),
            nn.GELU(),
            nn.Linear(2 * target_hidden, target_hidden),
            nn.Tanh(),
        )
        self.gate = nn.Linear(2 * target_hidden, target_hidden)
        nn.init.constant_(self.gate.bias, -2.0)
        # Start as an exact no-op.  The last layer learns first; deeper bridge
        # parameters then receive gradient without perturbing an existing model.
        nn.init.zeros_(self.delta[-2].weight)
        nn.init.zeros_(self.delta[-2].bias)

    def extra_repr(self):
        return (f"source_hidden={self.source_hidden}, target_hidden={self.target_hidden}, "
                f"heads={self.heads}, max_residual={self.max_residual}")

    def forward(self, latent: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or background.ndim != 3:
            raise ValueError("Need latent [N,K,H] and background [N,L,H].")
        if latent.shape[0] != background.shape[0] or latent.shape[-1] != self.source_hidden:
            raise ValueError("Analysis latent geometry differs from bridge contract.")
        if background.shape[-1] != self.target_hidden:
            raise ValueError("Forecast hidden width differs from bridge contract.")
        if not torch.isfinite(latent).all() or not torch.isfinite(background).all():
            raise ValueError("Coupled analysis state must be finite.")
        source = self.source_norm(self.source_projection(latent))
        query = self.query_norm(background)
        update, _ = self.attention(query, source, source, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat((query, update), -1)))
        correction = self.delta(update)
        result = background + self.max_residual * gate * correction
        if not torch.isfinite(result).all():
            raise FloatingPointError("Nonfinite coupled analysis state.")
        return result


class CoupledAnalysisForecastModel(nn.Module):
    """Use Multimodal/Adaptive for analysis and Joint/Hydrostatic for forecast.

    The forecast model's observation-ingest path is deliberately bypassed.  This
    prevents the same station/profile observation from being assimilated once in
    Multimodal and then again in Joint.  Only deterministic pressure/climate/
    terrain background features are retained before the learned bridge is added.
    """

    def __init__(self, analysis_model: nn.Module, forecast_model: nn.Module, *,
                 bridge_heads: int = 4, bridge_max_residual: float = 0.25):
        super().__init__()
        self.analysis_model = analysis_model
        self.forecast_model = forecast_model
        source_hidden = getattr(analysis_model, "hidden", None)
        target_hidden = getattr(forecast_model, "hidden", None)
        if type(source_hidden) is not int or type(target_hidden) is not int:
            raise ValueError("Both branches must expose integer hidden widths.")
        analysis_cells = int(getattr(analysis_model, "xyz").shape[0])
        forecast_cells = int(forecast_model.grid.n_cells)
        if analysis_cells != forecast_cells:
            raise ValueError("Analysis and forecast branches must use the same global grid.")
        analysis_grid = getattr(analysis_model, "grid_fingerprint", None)
        forecast_grid = getattr(forecast_model.grid, "fingerprint", None)
        if analysis_grid is not None and forecast_grid is not None and analysis_grid != forecast_grid:
            raise ValueError("Analysis and forecast grid fingerprints differ.")
        self.grid_fingerprint = forecast_grid
        self.hidden = target_hidden
        self.bridge = LatentToPressureBridge(source_hidden, target_hidden,
                                             heads=bridge_heads,
                                             max_residual=bridge_max_residual)

    def get_extra_state(self):
        analysis_identity = (self.analysis_model.get_extra_state()
                             if hasattr(self.analysis_model, "get_extra_state") else None)
        return {
            "architecture": ARCHITECTURE,
            "grid": self.grid_fingerprint,
            "analysis": analysis_identity,
            "forecast": getattr(self.forecast_model, "ARCHITECTURE", None),
            "bridge": {
                "source_hidden": self.bridge.source_hidden,
                "target_hidden": self.bridge.target_hidden,
                "heads": self.bridge.heads,
                "max_residual": self.bridge.max_residual,
            },
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("Coupled checkpoint architecture differs from the live model.")

    def _pressure_features(self):
        model = self.forecast_model
        return model.pressure_encoder(model.log_pressure) + model.levels

    def _forecast_background(self, issue_time):
        """Build deterministic forecast background without ingesting observations."""
        model = self.forecast_model
        issue = utc(issue_time)
        pressure = self._pressure_features()
        state = model.geometry(model.xyz)[:, None] + pressure[None]
        if hasattr(model, "climatology"):
            state = state + model.climate_encoder(model._climate_features(issue))
        if hasattr(model, "terrain_encoder"):
            terrain = torch.stack((model.terrain_elevation_m / 10000.0,
                                   model.terrain_land_fraction,
                                   model.terrain_slope), -1)
            state = state + model.terrain_encoder(terrain)[:, None, :]
        return state, pressure, issue

    def analysis_state(self, observations, elevation_m, land_fraction, *, background=None):
        analysed = self.analysis_model.analysis_state(
            observations, elevation_m, land_fraction, background
        )
        base, _, issue = self._forecast_background(analysed.valid_time)
        hidden = self.bridge(analysed.latent, base)
        return CoupledAnalysisState(hidden, issue, analysed.evidence, analysed.schema)

    def forecast_from_analysis(self, analysed: CoupledAnalysisState, *, horizon_hours: int = 72):
        if type(horizon_hours) is not int or not 0 <= horizon_hours <= 72 or horizon_hours % 3:
            raise ValueError("Coupled forecast horizon must be 0..72 hours in 3-hour steps.")
        model = self.forecast_model
        state = analysed.hidden
        expected = (model.grid.n_cells, 37, model.hidden)
        if state.shape != expected or not torch.isfinite(state).all():
            raise ValueError("Coupled hydrostatic state has invalid shape or values.")
        pressure = self._pressure_features()
        issue = utc(analysed.valid_time)
        frames = [model._decode(state, issue, 0)]
        for lead in range(3, horizon_hours + 1, 3):
            state = model._advance_state(state, pressure, issue, lead)
            frames.append(model._decode(state, issue, lead))
        return frames

    def forward(self, observations, elevation_m, land_fraction, *, horizon_hours: int = 72,
                background=None):
        analysed = self.analysis_state(observations, elevation_m, land_fraction,
                                       background=background)
        return self.forecast_from_analysis(analysed, horizon_hours=horizon_hours)

    @staticmethod
    def _requires(module, enabled):
        for parameter in module.parameters():
            parameter.requires_grad_(enabled)

    def set_trainable(self, scope: str = "fast"):
        """Select the part updated by an offline or online learning step.

        fast: bridge + hydrostatic tendency/dynamic/output heads.
        slow: fast + multimodal encoders/fusion/analysis processor.
        bridge/forecast/analysis/all are explicit research scopes.
        """
        if scope not in {"fast", "slow", "bridge", "forecast", "analysis", "all"}:
            raise ValueError("Unknown coupled training scope.")
        self._requires(self, False)
        if scope == "all":
            self._requires(self, True)
        elif scope == "bridge":
            self._requires(self.bridge, True)
        elif scope == "forecast":
            self._requires(self.forecast_model, True)
        elif scope == "analysis":
            self._requires(self.analysis_model, True)
        else:
            self._requires(self.bridge, True)
            for name in ("dynamic", "tendency", "head", "surface_head"):
                module = getattr(self.forecast_model, name, None)
                if isinstance(module, nn.Module):
                    self._requires(module, True)
            if scope == "slow":
                for name in ("encoder", "processor", "satellite_bank", "source_fusion"):
                    module = getattr(self.analysis_model, name, None)
                    if isinstance(module, nn.Module):
                        self._requires(module, True)
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
