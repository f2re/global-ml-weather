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
import torch.nn as nn

from ..grid import SphereGrid, unit_xyz
from ..observation_identity import observation_identity
from ..vertical import PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS

UTC = timezone.utc
SURFACE_ELEVATION_TOLERANCE_M = 1e-3

G_STD = 9.80665          # м/с^2
R_DRY = 287.058          # Дж/(кг·К)
LAPSE_RATE_T = 0.0065    # К/м (стандартный вертикальный градиент температуры)
LAPSE_RATE_TD = 0.0020   # К/м (градиент для точки росы)
MAX_DZ_METERS = 2500.0   # защита от выбросов в метаданных высоты станции


def apply_orographic_station_reduction(
    interp_surface: dict[str, torch.Tensor],
    mesh_elevation_m: torch.Tensor,
    station_elevation_m: torch.Tensor,
    station_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Приводит интерполированные с сетки Mesh-8 приземные поля к фактической высоте станции.

    interp_surface: словарь тензоров на точках станций:
        - 'station_temperature' (K)
        - 'station_dew_point' (K)
        - 'station_pressure' (Pa)
    mesh_elevation_m: интерполированная с тех же k=3 ячеек высота рельефа сетки (м)
    station_elevation_m: фактическая высота станции из метаданных WMO/GHCNh (м)
    """
    valid_elev = torch.isfinite(station_elevation_m) & torch.isfinite(mesh_elevation_m)
    if station_mask is not None:
        valid_elev = valid_elev & station_mask

    dz = torch.where(
        valid_elev,
        torch.clamp(station_elevation_m - mesh_elevation_m, -MAX_DZ_METERS, MAX_DZ_METERS),
        torch.zeros_like(mesh_elevation_m),
    )

    t_mesh = interp_surface["station_temperature"]
    td_mesh = interp_surface["station_dew_point"]
    p_mesh = interp_surface["station_pressure"]

    # 1. Адиабатическая поправка приземной температуры и точки росы
    t_station = t_mesh - LAPSE_RATE_T * dz
    td_station = torch.minimum(td_mesh - LAPSE_RATE_TD * dz, t_station)

    # 2. Барометрическая формула (гипсометрическое уравнение) со средней температурой слоя
    t_layer_mean = torch.clamp(0.5 * (t_mesh + t_station), min=200.0, max=330.0)
    exponent = -(G_STD * dz) / (R_DRY * t_layer_mean)
    p_station = p_mesh * torch.exp(exponent)

    out = dict(interp_surface)
    out["station_temperature"] = t_station
    out["station_dew_point"] = td_station
    out["station_pressure"] = p_station
    return out


def _surface_reference(frame, record, cells):
    """Require matching measured and model references; perform no height correction.

    Synthetic structural fixtures opt in explicitly. A real GHCNh provider
    cannot bypass unknown instrument heights by relabelling its record.
    The strict elevation equality is a numerical check, not a physical model
    of representativeness error over unequal terrain.
    """
    if record.get('data_kind') == 'synthetic' and record.get('provider') != 'NOAA_GHCNh':
        return
    variable = record['variable']
    height = {'t2m': 2., 'td2m': 2., 'u10': 10., 'v10': 10.}.get(variable)
    if record.get('instrument_height_verified') is not True:
        raise ValueError('surface_reference_not_admitted')
    if height is not None:
        if (record.get('height_reference') != 'height_above_ground' or
                _finite(record.get('instrument_height_m'), 'высота прибора', 0, 1000) != height):
            raise ValueError('surface_reference_not_admitted')
        observed_elevation = _finite(record.get('elevation_m'), 'высота станции')
    elif variable == 'surface_pressure':
        if record.get('height_reference') != 'model_lower_boundary':
            raise ValueError('surface_reference_not_admitted')
        observed_elevation = _finite(record.get('pressure_reference_elevation_m'), 'высота отсчёта давления')
    else:
        # A dedicated admitted sea-level reduction or column operator is needed.
        raise ValueError('surface_reference_not_admitted')
    elevation = getattr(frame, 'surface_elevation_m', None)
    if elevation is None:
        raise ValueError('surface_reference_not_admitted')
    elevation = torch.as_tensor(elevation, device=cells.device)
    if elevation.shape != (frame.surface.shape[0],) or not bool(torch.isfinite(elevation[cells]).all()):
        raise ValueError('surface_reference_not_admitted')
    if not bool((torch.abs(elevation[cells] - observed_elevation) <= SURFACE_ELEVATION_TOLERANCE_M).all()):
        raise ValueError('surface_reference_not_admitted')


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
    if (p == pressure).any():
        j = int(np.flatnonzero(p == pressure)[0])
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
        _surface_reference(frame, record, cells)
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
        variable_mask = getattr(frame, 'profile_variable_mask', None)
        if variable_mask is not None:
            if (not isinstance(variable_mask, torch.Tensor) or variable_mask.dtype != torch.bool
                    or variable_mask.shape != frame.profiles.shape):
                raise ValueError('invalid_profile_variable_mask')
            valid = valid & variable_mask[cells[:, None], level_tensor[None, :], k]
        if k in (2, 3) and getattr(frame, 'wind_basis', None) == 'local_enu_vector':
            # Components live in different local ENU bases. Interpolate the
            # Cartesian tangent vector, then project at the measurement point.
            pair = frame.profiles[cells[:, None], level_tensor[None, :], 2:4]
            if variable_mask is not None:
                valid = valid & variable_mask[cells[:, None], level_tensor[None, :], 2:4].all(-1)
            if not bool(torch.isfinite(pair[valid]).all()):
                raise ValueError('nonfinite_vector_prediction')
            xyz = grid.xyz[indices]
            lon = np.arctan2(xyz[:, 1], xyz[:, 0]); lat = np.arctan2(xyz[:, 2], np.hypot(xyz[:, 0], xyz[:, 1]))
            east = np.column_stack((-np.sin(lon), np.cos(lon), np.zeros(len(lon))))
            north = np.column_stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)))
            basis = torch.as_tensor(np.stack((east, north), axis=1), dtype=values.dtype, device=device)
            vectors = torch.einsum('clv,cvd->cld', torch.where(valid[..., None], pair, torch.zeros_like(pair)), basis)
            qlat, qlon = np.deg2rad([record['latitude'], record['longitude']])
            direction = ([-np.sin(qlon), np.cos(qlon), 0.] if k == 2 else
                         [-np.sin(qlat)*np.cos(qlon), -np.sin(qlat)*np.sin(qlon), np.cos(qlat)])
            values = vectors @ torch.as_tensor(direction, dtype=values.dtype, device=device)
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
            reason = str(exc) if str(exc) in ('outside_supported_forecast', 'invalid_record', 'accumulation_operator_required', 'nonfinite_prediction', 'surface_reference_not_admitted', 'invalid_profile_variable_mask') else 'invalid_contract'
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
            reason = str(exc) if str(exc) in ('outside_supported_forecast', 'invalid_record', 'accumulation_operator_required', 'nonfinite_prediction', 'surface_reference_not_admitted', 'invalid_profile_variable_mask') else 'invalid_contract'
            rejected[reason] = rejected.get(reason, 0) + 1
    if not errors:
        raise ValueError('Нет поддержанных наблюдений для функции ошибки.')
    if len(units) > 1:
        raise ValueError('Разные единицы требуют раздельных потерь и фиксированной нормировки.')
    e, w = torch.stack(errors), torch.stack(weights)
    loss = (e * w).sum() / w.sum()
    return loss, {'accepted': accepted, 'rejected': rejected,
                  'weighted_rmse': float(torch.sqrt(loss.detach()).cpu())}


class VariationalBiasCorrection(nn.Module):
    """
    Обучаемый модуль VarBC с 7 физико-геометрическими предикторами.
    Веса beta регуляризуются относительно фонового состояния в `continuous/store.py`.
    """

    def __init__(self, num_channels: int, num_predictors: int = 7) -> None:
        super().__init__()
        self.beta_0 = nn.Parameter(torch.zeros(num_channels))
        self.beta_w = nn.Parameter(torch.zeros(num_channels, num_predictors))

    def forward(self, predictors: torch.Tensor) -> torch.Tensor:
        """
        predictors: [N_obs, 7] -> (m_air-1, (m_air-1)^2, cos_scan, sin_scan, dZ_trop, dZ_strat, cos_orb)
        Возвращает поправку смещения: [N_obs, num_channels]
        """
        return self.beta_0.unsqueeze(0) + torch.matmul(predictors, self.beta_w.t())


class IKFS2CloudClearedOperator(nn.Module):
    """
    Оператор наблюдения для гиперспектрального ИК-зондировщика ИКФС-2 в связке с МСУ-МР.
    Выполняет проекцию на главные компоненты (PCA) и маскирование каналов ниже верхней границы облаков.
    """

    def __init__(
        self,
        num_selected_channels: int = 180,
        num_pcs: int = 32,
        num_model_levels: int = 37,
    ) -> None:
        super().__init__()
        self.num_pcs = num_pcs
        # Матрица собственных векторов U_K и шум NEDR загружаются из assets/normalization/
        self.register_buffer("pca_basis", torch.eye(num_selected_channels, num_pcs))
        self.register_buffer("nedr_inv", torch.ones(num_selected_channels))
        self.register_buffer("clim_mean", torch.zeros(num_selected_channels))
        # Давление пика весовой функции каждого из 180 каналов ИКФС-2 (гПа)
        self.register_buffer(
            "channel_peak_pressure_hpa",
            torch.linspace(30.0, 980.0, num_selected_channels),
        )

        # Быстрый нейросетевой суррогат прямого оператора переноса излучения H_ikfs2(T(p), q(p), T_skin, mu_sat)
        in_features = num_model_levels * 2 + 2
        self.forward_rt_surrogate = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, num_selected_channels),
        )
        self.varbc = VariationalBiasCorrection(num_channels=num_selected_channels)

    @torch.no_grad()
    def compute_above_cloud_mask(
        self,
        msumr_cloud_fraction: torch.Tensor,  # [N_obs] доля облаков в 35-км пятне ИКФС-2 по МСУ-МР
        msumr_cth_pressure_hpa: torch.Tensor,  # [N_obs] давление ВГО (гПа)
        margin_hpa: float = 50.0,
        tau_hpa: float = 15.0,
    ) -> torch.Tensor:
        """
        Возвращает весовую маску допустимости каналов [N_obs, 180].
        Каналы, чья функция вклада пикует ниже (p_cld - 50 гПа), подавляются.
        """
        is_clear = (msumr_cloud_fraction < 0.05).float().unsqueeze(-1)  # [N_obs, 1]
        p_cutoff = msumr_cth_pressure_hpa.unsqueeze(-1) - margin_hpa    # [N_obs, 1]
        p_peak = self.channel_peak_pressure_hpa.unsqueeze(0)            # [1, 180]
        above_cloud_gate = torch.sigmoid((p_cutoff - p_peak) / tau_hpa)
        return torch.maximum(is_clear, above_cloud_gate)

    def forward(
        self,
        obs_spectra: torch.Tensor,           # [N_obs, 180]
        model_t_profile: torch.Tensor,       # [N_obs, L] интерполированный профиль T
        model_q_profile: torch.Tensor,       # [N_obs, L] интерполированный профиль log(q)
        model_t_skin: torch.Tensor,          # [N_obs, 1]
        mu_sat: torch.Tensor,                # [N_obs, 1]
        varbc_predictors: torch.Tensor,      # [N_obs, 7]
        msumr_cloud_fraction: torch.Tensor,  # [N_obs]
        msumr_cth_pressure_hpa: torch.Tensor,  # [N_obs]
    ) -> dict[str, torch.Tensor]:
        x_col = torch.cat([model_t_profile, model_q_profile, model_t_skin, mu_sat], dim=-1)
        sim_spectra = self.forward_rt_surrogate(x_col)
        bias = self.varbc(varbc_predictors)

        channel_mask = self.compute_above_cloud_mask(msumr_cloud_fraction, msumr_cth_pressure_hpa)
        # Нормированная невязка в спектральном пространстве с маскированием облачных слоев
        masked_innovation = (obs_spectra - sim_spectra - bias) * self.nedr_inv.unsqueeze(0) * channel_mask
        # Проекция очищенной невязки в подпространство главных компонент (PC scores)
        pc_innovation = torch.matmul(masked_innovation, self.pca_basis)

        return {
            "pc_innovation": pc_innovation,
            "masked_channel_innovation": masked_innovation,
            "channel_mask": channel_mask,
            "varbc_bias": bias,
        }


class MeteorMTVZAAllSkyOperator(nn.Module):
    """
    Всепогодный (All-Sky) оператор наблюдения для 29 каналов СВЧ-радиометра МТВЗА-ГЯ (КА «Метеор-М»).
    Динамически масштабирует матрицу ошибок R(C_sym) и фильтрует поверхностно-чувствительные каналы.
    """

    def __init__(self, num_channels: int = 29, num_model_levels: int = 37) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.register_buffer("sigma_clr", torch.full((num_channels,), 0.6))
        self.register_buffer("sigma_cld", torch.full((num_channels,), 4.5))
        self.register_buffer("c_clr", torch.full((num_channels,), 0.05))
        self.register_buffer("c_cld", torch.full((num_channels,), 0.45))

        # Маска каналов, чувствительных к поверхности (окна 10.6-36.5, 89 ГГц и нижние каналы 52-53 ГГц)
        surface_sensitive = torch.zeros(num_channels, dtype=torch.bool)
        surface_sensitive[:12] = True
        self.register_buffer("surface_sensitive_mask", surface_sensitive)

        in_features = num_model_levels * 3 + 4  # T(p), q(p), clwc(p) + T_skin, wind_10m, lsm, sea_ice
        self.allsky_rt_surrogate = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, num_channels),
        )
        self.varbc = VariationalBiasCorrection(num_channels=num_channels)

    def forward(
        self,
        obs_tb: torch.Tensor,             # [N_obs, 29]
        model_profiles: torch.Tensor,     # [N_obs, 3*L] (T, q, cloud_liquid_water)
        surface_state: torch.Tensor,      # [N_obs, 4] (T_skin, wind_10m, land_mask, sea_ice_fraction)
        varbc_predictors: torch.Tensor,    # [N_obs, 7]
        idx_36v: int = 6,
        idx_36h: int = 7,
        delta_tb_clear_ocean: float = 65.0,
    ) -> dict[str, torch.Tensor]:
        x_in = torch.cat([model_profiles, surface_state], dim=-1)
        sim_tb = self.allsky_rt_surrogate(x_in)
        bias = self.varbc(varbc_predictors)

        # Расчет симметричного индекса облачности C_sym по поляризационной разности 36.5 ГГц
        c_obs = (1.0 - (obs_tb[:, idx_36v] - obs_tb[:, idx_36h]) / delta_tb_clear_ocean).clamp(0.0, 1.5)
        c_fg = (1.0 - (sim_tb[:, idx_36v] - sim_tb[:, idx_36h]) / delta_tb_clear_ocean).clamp(0.0, 1.5)
        c_sym = 0.5 * (c_obs + c_fg).unsqueeze(-1)  # [N_obs, 1]

        # Кусочно-линейное масштабирование ошибки наблюдения sigma_obs(C_sym)
        ramp = ((c_sym - self.c_clr) / (self.c_cld - self.c_clr)).clamp(0.0, 1.0)
        sigma_obs = self.sigma_clr + (self.sigma_cld - self.sigma_clr) * ramp  # [N_obs, 29]

        # Отключение низковысотных каналов над сушей и морским льдом для атмосферного анализа
        land_or_ice = ((surface_state[:, 2] > 0.1) | (surface_state[:, 3] > 0.05)).unsqueeze(-1)
        admissible_mask = ~(land_or_ice & self.surface_sensitive_mask.unsqueeze(0))

        innovation = obs_tb - sim_tb - bias
        normalized_innovation = (innovation / sigma_obs) * admissible_mask.float()

        return {
            "normalized_innovation": normalized_innovation,
            "sigma_obs": sigma_obs,
            "admissible_mask": admissible_mask.float(),
            "c_sym": c_sym.squeeze(-1),
            "varbc_bias": bias,
        }
