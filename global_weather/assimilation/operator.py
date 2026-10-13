"""Differentiable observation operators H(state) for every admitted token type.

Horizontal: barycentric weights in the latent spherical triangle that contains
the measurement. Time: linear between 3-hour frames (analysis targets use the
analysis frame inside +-analysis_window_hours). Vertical:
* pressure tokens: linear in ln p between the two bracketing standard levels;
  measured pressures outside 1..1000 hPa are refused (no extrapolation);
* surface tokens: per node, then interpolated (avoids orography interpolation):
  station pressure by the hypsometric relation at the station elevation,
  MSLP by the conventional reduction, temperature by the model's own local
  lapse rate between model surface and station elevation, dew point from the
  near-surface humidity at station pressure, wind from the near-surface wind;
* GNSS-RO refractivity at a tangent height: N = 77.6 p/T + 3.73e5 e/T^2
  (hPa, K; Smith-Weintraub) after locating the height in the hydrostatic Phi;
* column tokens (radiances, products): a learned operator on the full
  interpolated state with a per-channel bias (no radiative transfer claimed).
Below-ground handling never uses the predicted surface pressure to drop a
target: the vertical interpolation continues below the model surface, so a
target cannot be hidden by moving p_s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from ..grid import EARTH_RADIUS_M
from ..vertical import PRESSURE_HPA
from . import thermo
from .mesh import LatentMesh
from .model import CHANNELS, PhysicalAssimilationModel, Rollout
from .tokens import OPERATOR, PLATFORMS, VARIABLES, FAMILY, SECONDS_PER_HOUR, TokenBatch, wind_directions

KIND_CODES = {'pressure': 0, 'surface': 1, 'height': 2, 'column': 3}
NEEDS_STATION_HEIGHT = ('station_temperature', 'station_dew_point', 'station_pressure')


@dataclass
class Plan:
    """Static (state-independent) operator data for one window of target tokens."""
    count: int
    kind: Tensor
    variable: Tensor
    frame0: Tensor
    frame1: Tensor
    alpha: Tensor
    nodes: Tensor
    weights: Tensor
    level: Tensor
    beta: Tensor
    pressure: Tensor
    height: Tensor
    direction: Tensor
    channel: Tensor
    zenith: Tensor
    column_mean: Tensor
    column_std: Tensor
    value: Tensor
    sigma: Tensor
    weight: Tensor
    lead_hours: Tensor
    is_analysis: Tensor
    rows: np.ndarray = None  # indices of the planned targets in the input batch
    rejected: dict = field(default_factory=dict)
    variable_names: list = field(default_factory=list)


def _reject(counts: dict, reason: str, number: int):
    if number:
        counts[reason] = counts.get(reason, 0) + int(number)


def density_weights(nodes: np.ndarray, weights: np.ndarray, variable: np.ndarray, platform: np.ndarray,
                    pressure: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """1 / count of targets sharing (nearest node, variable, family, ln p band, frame).

    Dense networks therefore contribute one unit per occupied latent cell,
    variable, layer and lead instead of one unit per report (superobbing).
    """
    nearest = nodes[np.arange(len(nodes)), weights.argmax(1)]
    family = np.array([FAMILY[PLATFORMS[p]] for p in platform]) if len(platform) else np.zeros(0, str)
    band = np.where(np.isfinite(pressure), np.floor(np.log(np.where(np.isfinite(pressure), pressure, 1.)) / 0.35), -1)
    keys = np.rec.fromarrays([nearest, variable, np.unique(family, return_inverse=True)[1].reshape(-1),
                              band.astype(np.int64), frame])
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse.reshape(-1)]


def build_plan(batch: TokenBatch, sigma: np.ndarray, issue_seconds: float, horizon_hours: int,
               mesh: LatentMesh, *, analysis: np.ndarray, analysis_window_hours: float = 1.5,
               column_statistics: dict | None = None, channels: list[str] | None = None,
               analysis_weight: float = 0.5, family_weights: dict | None = None,
               device=None, dtype=torch.float32) -> Plan:
    """Prepare operator indices; refuse unsupported targets with explicit reasons."""
    rejected: dict = {}
    hours = (batch.observed - issue_seconds) / SECONDS_PER_HOUR
    analysis = np.asarray(analysis, bool)
    keep = np.ones(len(batch), bool)
    in_window = np.where(analysis, np.abs(hours) <= analysis_window_hours, (hours > 0) & (hours <= horizon_hours))
    _reject(rejected, 'outside_supported_forecast_time', np.sum(keep & ~in_window))
    keep &= in_window
    names = np.array(VARIABLES, dtype=object)[batch.variable] if len(batch) else np.zeros(0, object)
    kinds = np.array([OPERATOR[n] for n in names]) if len(batch) else np.zeros(0, str)
    pressure_rows = kinds == 'pressure'
    p = batch.pressure_pa
    outside = pressure_rows & ((p > PRESSURE_HPA[0] * 100.) | (p < PRESSURE_HPA[-1] * 100.))
    _reject(rejected, 'pressure_outside_model_levels', np.sum(keep & outside))
    keep &= ~outside
    unknown_height = np.isin(names, NEEDS_STATION_HEIGHT) & ~np.isfinite(batch.height_m)
    _reject(rejected, 'unknown_station_elevation', np.sum(keep & unknown_height))
    keep &= ~unknown_height
    column_rows = kinds == 'column'
    column_mean = np.zeros(len(batch))
    column_std = np.ones(len(batch))
    if column_rows.any():
        stats = column_statistics or {}
        for row in np.flatnonzero(column_rows & keep):
            key = f'{PLATFORMS[batch.platform[row]]}|{names[row]}|{channels[batch.channel[row]] if batch.channel[row] >= 0 else ""}'
            stat = stats.get(key)
            if stat is None:
                keep[row] = False
                _reject(rejected, 'column_channel_without_train_statistics', 1)
            else:
                column_mean[row], column_std[row] = stat['mean'], stat['std']
    kept_rows = np.flatnonzero(keep)
    batch, sigma, hours, kinds, names = batch.select(keep), sigma[keep], hours[keep], kinds[keep], names[keep]
    column_mean, column_std, analysis = column_mean[keep], column_std[keep], analysis[keep]
    interpolation = mesh.interpolation(batch.latitude, batch.longitude) if len(batch) else None
    nodes = interpolation.nodes if interpolation else np.zeros((0, 3), np.int64)
    weights = interpolation.weights if interpolation else np.zeros((0, 3))
    frames = horizon_hours // 3 + 1
    frame0 = np.where(analysis, 0, np.floor(hours / 3).astype(np.int64))
    frame0 = np.clip(frame0, 0, frames - 1)
    frame1 = np.where(analysis, 0, np.clip(frame0 + 1, 0, frames - 1))
    alpha = np.where(analysis, 0., np.clip(hours / 3 - frame0, 0, 1))
    log_levels = np.log(np.asarray(PRESSURE_HPA, float) * 100.)
    level = np.zeros(len(batch), np.int64)
    beta = np.zeros(len(batch))
    rows = kinds == 'pressure'
    if rows.any():
        x = np.log(batch.pressure_pa[rows])
        k = np.clip(np.searchsorted(-log_levels, -x, side='right') - 1, 0, len(log_levels) - 2)
        level[rows] = k
        beta[rows] = np.clip((log_levels[k] - x) / (log_levels[k] - log_levels[k + 1]), 0, 1)
    family_weights = family_weights or {}
    weight = density_weights(nodes, weights, batch.variable, batch.platform, batch.pressure_pa,
                             frame0 + np.round(alpha).astype(np.int64))
    weight *= np.where(analysis, analysis_weight, 1.0)
    weight *= np.array([family_weights.get(FAMILY[PLATFORMS[p]], 1.0) for p in batch.platform]) if len(batch) else 1.0
    weight *= batch.quality
    t = lambda array, kind=dtype: torch.tensor(np.asarray(array), dtype=kind, device=device)
    return Plan(len(batch), t([KIND_CODES[k] for k in kinds], torch.long), t(batch.variable, torch.long),
                t(frame0, torch.long), t(frame1, torch.long), t(alpha), t(nodes, torch.long), t(weights),
                t(level, torch.long), t(beta), t(np.nan_to_num(batch.pressure_pa, nan=1.)),
                t(np.nan_to_num(batch.height_m)), t(wind_directions(batch)), t(batch.channel + 1, torch.long),
                t(np.nan_to_num(batch.zenith_cos)), t(column_mean), t(column_std), t(batch.value), t(sigma),
                t(weight), t(np.where(analysis, 0., hours)), t(analysis, torch.bool), kept_rows, rejected, list(names))


def _horizontal(field: Tensor, plan: Plan, rows: Tensor) -> Tensor:
    """Interpolate a per-frame node quantity [F, N, ...] to rows in space and time."""
    nodes, weights = plan.nodes[rows], plan.weights[rows]
    shape = weights.shape + (1,) * (field.dim() - 2)
    a = (field[plan.frame0[rows][:, None], nodes] * weights.view(shape)).sum(1)
    b = (field[plan.frame1[rows][:, None], nodes] * weights.view(shape)).sum(1)
    alpha = plan.alpha[rows].view((-1,) + (1,) * (a.dim() - 1))
    return (1 - alpha) * a + alpha * b


def _per_node(values: Tensor, plan: Plan, rows: Tensor) -> Tensor:
    """Combine per-(row, node) values [R, 2, 3] (two frames) into [R]."""
    weights = plan.weights[rows]
    a, b = (values[:, 0] * weights).sum(1), (values[:, 1] * weights).sum(1)
    return (1 - plan.alpha[rows]) * a + plan.alpha[rows] * b


def _gather_nodes(field: Tensor, plan: Plan, rows: Tensor) -> Tensor:
    """[F, N] field -> [R, 2, 3] at both frames and the three triangle nodes."""
    nodes = plan.nodes[rows]
    return torch.stack((field[plan.frame0[rows][:, None], nodes], field[plan.frame1[rows][:, None], nodes]), 1)


def observe(model: PhysicalAssimilationModel, rollout: Rollout, plan: Plan, *, states: Tensor | None = None) -> Tensor:
    """Return H(x) for every planned target in physical units, shape [P]."""
    dtype = rollout.states[0].dtype
    out = torch.zeros(plan.count, dtype=dtype, device=rollout.states[0].device)
    if plan.count == 0:
        return out
    stack = rollout.stack
    rows_by_kind = {kind: torch.nonzero(plan.kind == code).flatten() for kind, code in KIND_CODES.items()}
    names = VARIABLES
    rows = rows_by_kind['pressure']
    if len(rows):
        level, beta = plan.level[rows], plan.beta[rows]
        for name, key in (('temperature', 'temperature'), ('specific_humidity', 'humidity'),
                          ('geopotential', 'geopotential')):
            selected = rows[plan.variable[rows] == names.index(name)]
            if len(selected):
                field = stack(key)
                profile = _horizontal(field, plan, selected)
                k = plan.level[selected]
                b = plan.beta[selected]
                low = profile.gather(1, k[:, None]).squeeze(1)
                high = profile.gather(1, (k + 1)[:, None]).squeeze(1)
                out[selected] = (1 - b) * low + b * high
        winds = rows[(plan.variable[rows] == names.index('u')) | (plan.variable[rows] == names.index('v'))]
        if len(winds):
            profile = _horizontal(stack('wind'), plan, winds)  # [R, 37, 3]
            k, b = plan.level[winds], plan.beta[winds]
            low = profile[torch.arange(len(winds)), k]
            high = profile[torch.arange(len(winds)), k + 1]
            vector = (1 - b)[:, None] * low + b[:, None] * high
            out[winds] = (vector * plan.direction[winds]).sum(-1)
    rows = rows_by_kind['surface']
    if len(rows):
        elevation = model.elevation.to(dtype)
        variable = plan.variable[rows]
        height = plan.height[rows]
        ps = _gather_nodes(stack('surface_pressure'), plan, rows)
        tv = _gather_nodes(stack('surface_virtual_temperature'), plan, rows)
        lapse = _gather_nodes(stack('lapse_rate'), plan, rows)
        z = elevation[plan.nodes[rows]][:, None, :].expand_as(ps)
        station_height = height[:, None, None].expand_as(ps)
        station_pressure = thermo.pressure_at_height(ps, tv, z, station_height, lapse)
        values = torch.zeros_like(ps)
        code = {name: names.index(name) for name in ('station_pressure', 'station_mean_sea_level_pressure',
                                                       'station_temperature', 'station_dew_point')}
        choose = lambda name: (variable == code[name])[:, None, None]
        values = torch.where(choose('station_pressure'), station_pressure, values)
        values = torch.where(choose('station_mean_sea_level_pressure'), _gather_nodes(stack('mslp'), plan, rows), values)
        temperature = _gather_nodes(stack('surface_temperature'), plan, rows) - lapse * (station_height - z)
        values = torch.where(choose('station_temperature'), temperature, values)
        humidity = _gather_nodes(stack('surface_humidity'), plan, rows)
        values = torch.where(choose('station_dew_point'),
                             thermo.dew_point_from_humidity(humidity, station_pressure), values)
        scalar = _per_node(values, plan, rows)
        is_wind = (variable == names.index('station_eastward_wind')) | (variable == names.index('station_northward_wind'))
        if bool(is_wind.any()):
            vector = _horizontal(stack('surface_wind'), plan, rows[is_wind])
            scalar = scalar.clone()
            scalar[is_wind] = (vector * plan.direction[rows[is_wind]]).sum(-1)
        out[rows] = scalar
    rows = rows_by_kind['height']
    if len(rows):
        phi = _horizontal(stack('geopotential'), plan, rows)
        temperature = _horizontal(stack('temperature'), plan, rows)
        humidity = _horizontal(stack('humidity'), plan, rows)
        h = plan.height[rows]
        target = thermo.G0 * h * EARTH_RADIUS_M / (EARTH_RADIUS_M + h)
        k = (torch.searchsorted(phi.detach().contiguous(), target.detach()[:, None]).squeeze(1) - 1).clamp(0, phi.shape[1] - 2)
        index = torch.arange(len(rows), device=phi.device)
        f0, f1 = phi[index, k], phi[index, k + 1]
        b = ((target - f0) / (f1 - f0)).clamp(0, 1)
        log_p = torch.log(model.pressure.to(dtype))
        pressure = torch.exp((1 - b) * log_p[k] + b * log_p[k + 1]) / 100.
        t = (1 - b) * temperature[index, k] + b * temperature[index, k + 1]
        q = (1 - b) * humidity[index, k] + b * humidity[index, k + 1]
        vapour = q * pressure / (thermo.EPSILON + (1 - thermo.EPSILON) * q)
        out[rows] = 77.6 * pressure / t + 3.73e5 * vapour / t ** 2
    rows = rows_by_kind['column']
    if len(rows):
        state = torch.stack(rollout.states) if states is None else states
        interpolated = _horizontal(state, plan, rows)
        features = torch.cat((interpolated, model.channel_embedding(plan.channel[rows]),
                              plan.zenith[rows, None], (plan.zenith[rows] != 0).to(dtype)[:, None]), -1)
        normalised = model.column_operator(features).squeeze(-1) + model.channel_bias(plan.channel[rows]).squeeze(-1)
        out[rows] = plan.column_mean[rows] + plan.column_std[rows] * normalised
    return out
