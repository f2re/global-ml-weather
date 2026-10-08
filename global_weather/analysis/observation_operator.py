"""Differentiable forecast-to-observation operator in space, pressure and time.

The operator evaluates model frames at real observation coordinates and their
actual timestamps.  It supports stationless and moving platforms.  It performs
no temporal extrapolation and does not turn missing model support into zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable, Sequence

import numpy as np
import torch

from ..grid import SphereGrid, unit_xyz
from ..observation_identity import observation_identity
from ..vertical import PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS

UTC = timezone.utc


def _utc(value):
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, str):
        try:
            result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError('Неверное время наблюдения.') from exc
    else:
        raise ValueError('Нужно время наблюдения.')
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Время наблюдения должно иметь часовой пояс.')
    return result.astimezone(UTC)


def _finite(value, label, lower=-math.inf, upper=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not lower <= value <= upper:
        raise ValueError(f'Неверное значение: {label}.')
    return float(value)


@dataclass(frozen=True)
class ObservationEquivalent:
    observation_id: str
    variable: str
    observed_at: str
    latitude: float
    longitude: float
    pressure_pa: float | None
    observed: float
    predicted: float
    units: str
    residual: float
    horizontal_cells: tuple[int, ...]
    time_fraction: float


def _horizontal(grid: SphereGrid, latitude, longitude, neighbours=3):
    latitude = _finite(latitude, 'широта', -90, 90)
    longitude = _finite(longitude, 'долгота', -180, 180)
    if type(neighbours) is not int or neighbours < 1:
        raise ValueError('Число соседей должно быть положительным целым.')
    k = min(neighbours, grid.n_cells)
    distances, indices = grid.tree.query(unit_xyz(latitude, longitude), k=k)
    indices = np.atleast_1d(indices).astype(np.int64)
    distances = np.atleast_1d(distances).astype(np.float64)
    if distances[0] < 1e-12:
        return indices[:1], np.array([1.], dtype=np.float64)
    weights = 1. / np.maximum(distances, 1e-9) ** 2
    weights /= weights.sum()
    return indices, weights


def _vertical(pressure_axis, pressure):
    if isinstance(pressure_axis, torch.Tensor):
        pressure_axis = pressure_axis.detach().cpu().numpy()
    p = np.asarray(pressure_axis, dtype=float)
    if p.ndim != 1 or len(p) < 2 or not np.isfinite(p).all() or (p <= 0).any() or not (np.diff(p) < 0).all():
        raise ValueError('Ось давления должна быть конечной, положительной и убывающей.')
    pressure = _finite(pressure, 'давление', 1, 120000)
    if pressure > p[0] or pressure < p[-1]:
        return None
    x = -np.log(p)
    target = -math.log(pressure)
    if np.isclose(target, x).any():
        j = int(np.argmin(np.abs(x - target)))
        return np.array([j]), np.array([1.])
    j = int(np.searchsorted(x, target))
    if j <= 0 or j >= len(p):
        return None
    alpha = (target - x[j - 1]) / (x[j] - x[j - 1])
    return np.array([j - 1, j]), np.array([1 - alpha, alpha], dtype=np.float64)


def _frame_pairs(frames, when):
    if not frames:
        raise ValueError('Нет кадров прогноза.')
    rows = sorted(frames, key=lambda frame: _utc(frame.valid_time))
    times = [_utc(frame.valid_time) for frame in rows]
    if len(set(times)) != len(times):
        raise ValueError('Повторное время кадра прогноза.')
    when = _utc(when)
    for i, time in enumerate(times):
        if when == time:
            return rows[i], rows[i], 0.0
        if when < time:
            if i == 0:
                return None
            left, right = rows[i - 1], rows[i]
            total = (times[i] - times[i - 1]).total_seconds()
            if total <= 0:
                raise ValueError('Нарушен порядок кадров.')
            return left, right, (when - times[i - 1]).total_seconds() / total
    return None


def _sample_frame(frame, grid, record, neighbours):
    variable = record.get('variable')
    if variable == 'precipitation_step':
        raise ValueError('accumulation_operator_required')
    indices, horizontal = _horizontal(grid, record.get('latitude'), record.get('longitude'), neighbours)
    device = frame.profiles.device
    cells = torch.as_tensor(indices, dtype=torch.long, device=device)
    hw = torch.as_tensor(horizontal, dtype=frame.profiles.dtype, device=device)
    if variable in SURFACE_VARIABLES:
        expected = SURFACE_UNITS[SURFACE_VARIABLES.index(variable)]
        if record.get('units') != expected:
            raise ValueError('Единицы наблюдения не совпадают с выходом модели.')
        k = SURFACE_VARIABLES.index(variable)
        values = frame.surface[cells, k]
        valid = frame.surface_mask[cells, k]
        weights = hw * valid.to(hw.dtype)
    elif variable in PROFILE_VARIABLES:
        expected = PROFILE_UNITS[PROFILE_VARIABLES.index(variable)]
        if record.get('units') != expected:
            raise ValueError('Единицы наблюдения не совпадают с выходом модели.')
        # ForecastFrame deliberately carries only fields. The caller injects
        # the model pressure axis through the private record key below.
        p = record.get('_pressure_axis_pa')
        if p is None:
            raise ValueError('Для профильного оператора нужна ось давления модели.')
        links = _vertical(p, record.get('pressure_pa'))
        if links is None:
            return None
        level_indices, vertical = links
        level_tensor = torch.as_tensor(level_indices, dtype=torch.long, device=device)
        vw = torch.as_tensor(vertical, dtype=frame.profiles.dtype, device=device)
        k = PROFILE_VARIABLES.index(variable)
        values = frame.profiles[cells[:, None], level_tensor[None, :], k]
        valid = frame.profile_mask[cells[:, None], level_tensor[None, :]]
        # Both bracketing levels are required. Renormalising just one level
        # would silently turn interpolation into vertical extrapolation.
        valid = valid.all(dim=1, keepdim=True).expand_as(values)
        weights = hw[:, None] * vw[None, :] * valid.to(hw.dtype)
    else:
        raise ValueError('Величина отсутствует в выходе модели.')
    total = weights.sum()
    if not bool(total > 0):
        return None
    if not bool(torch.isfinite(values[valid]).all()):
        raise ValueError('nonfinite_prediction')
    values = torch.where(valid, values, torch.zeros_like(values))
    return (values * weights).sum() / total, tuple(int(x) for x in indices)


def _sample_frame_with_pressure(frame, grid, record, pressure_axis, neighbours):
    local = dict(record)
    local['_pressure_axis_pa'] = pressure_axis
    return _sample_frame(frame, grid, local, neighbours)


def _prediction(frames, grid, record, pressure_axis, neighbours):
    pair = _frame_pairs(frames, record.get('observed_at'))
    if pair is None:
        return None
    left, right, alpha = pair
    a = _sample_frame_with_pressure(left, grid, record, pressure_axis, neighbours)
    if a is None:
        return None
    if left is right:
        return a[0], a[1], 0.0
    b = _sample_frame_with_pressure(right, grid, record, pressure_axis, neighbours)
    if b is None:
        return None
    return (1 - alpha) * a[0] + alpha * b[0], tuple(sorted(set(a[1] + b[1]))), float(alpha)


def predict_observations(frames: Sequence, grid: SphereGrid, pressure_axis_pa,
                         records: Iterable[dict], *, neighbours=3):
    result, rejected = [], {}
    for record in records:
        try:
            if not isinstance(record, dict) or record.get('valid', True) is not True:
                raise ValueError('invalid_record')
            prediction = _prediction(frames, grid, record, pressure_axis_pa, neighbours)
            if prediction is None:
                raise ValueError('outside_supported_forecast')
            value = _finite(record.get('value'), 'наблюдаемое значение')
            predicted, cells, alpha = prediction
            if not bool(torch.isfinite(predicted)):
                raise ValueError('nonfinite_prediction')
            observed_at = _utc(record.get('observed_at')).isoformat()
            result.append(ObservationEquivalent(
                observation_identity(record), record['variable'], observed_at,
                float(record['latitude']), float(record['longitude']),
                float(record['pressure_pa']) if record.get('pressure_pa') not in (None, 0, 0.0) else None,
                value, float(predicted.detach().cpu()), record['units'],
                value - float(predicted.detach().cpu()), cells, alpha))
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc) if str(exc) in ('outside_supported_forecast', 'invalid_record', 'accumulation_operator_required', 'nonfinite_prediction') else 'invalid_contract'
            rejected[reason] = rejected.get(reason, 0) + 1
    return result, rejected


def observation_space_loss(frames: Sequence, grid: SphereGrid, pressure_axis_pa,
                           records: Iterable[dict], *, neighbours=3):
    """Weighted MSE at actual observation locations and timestamps."""
    errors, weights, accepted, rejected = [], [], 0, {}
    units = set()
    for record in records:
        try:
            if not isinstance(record, dict) or record.get('valid', True) is not True:
                raise ValueError('invalid_record')
            prediction = _prediction(frames, grid, record, pressure_axis_pa, neighbours)
            if prediction is None:
                raise ValueError('outside_supported_forecast')
            predicted = prediction[0]
            if not bool(torch.isfinite(predicted)):
                raise ValueError('nonfinite_prediction')
            units.add(record['units'])
            observed = predicted.new_tensor(_finite(record.get('value'), 'наблюдаемое значение'))
            quality = _finite(record.get('quality', 1.0), 'качество', 1e-12, 1)
            sigma = record.get('observation_error')
            weight = quality if sigma is None else quality / _finite(sigma, 'ошибка наблюдения', 1e-12) ** 2
            errors.append((predicted - observed) ** 2)
            weights.append(predicted.new_tensor(weight))
            accepted += 1
        except (KeyError, TypeError, ValueError) as exc:
            reason = str(exc) if str(exc) in ('outside_supported_forecast', 'invalid_record', 'accumulation_operator_required', 'nonfinite_prediction') else 'invalid_contract'
            rejected[reason] = rejected.get(reason, 0) + 1
    if not errors:
        raise ValueError('Нет поддержанных наблюдений для функции ошибки.')
    if len(units) > 1:
        raise ValueError('Разные единицы требуют раздельных потерь и фиксированной нормировки.')
    e, w = torch.stack(errors), torch.stack(weights)
    loss = (e * w).sum() / w.sum()
    return loss, {'accepted': accepted, 'rejected': rejected,
                  'weighted_rmse': float(torch.sqrt(loss.detach()).cpu())}
