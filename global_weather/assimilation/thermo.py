"""Differentiable thermodynamics, hydrostatics and mass for the R11 state.

Every function states its formula, units and assumptions. All inputs are SI:
temperature K, pressure Pa, specific humidity kg/kg, geopotential m2/s2,
height m. Pressure axes are strictly descending (surface -> top).

These operators close specific relations exactly (hydrostatic balance above
the surface, the surface geopotential boundary condition, global dry-air mass,
omega boundary values). They do not make the learned tendency a solution of
the primitive equations, and they do not close the moisture or energy budget.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import torch
from torch import Tensor

RD = 287.05          # J kg-1 K-1, dry air
EPSILON = 0.622      # Rd / Rv, as used across this project
G0 = 9.80665         # m s-2, standard gravity (geopotential convention)
STANDARD_LAPSE = 0.0065  # K m-1, conventional fictitious column for MSLP reduction only
SOLAR_CONSTANT = 1361.0  # W m-2, total solar irradiance at 1 AU
MAX_RELATIVE_HUMIDITY = 1.05  # w.r.t. liquid water; permits ordinary ice supersaturation


def saturation_vapour_pressure_water(temperature: Tensor) -> Tensor:
    """Murphy & Koop (2005) Eq. 10 over liquid water, Pa.

    DOI 10.1256/qj.04.94. The formula domain is 123 < T < 332 K; inputs are
    clamped to that domain so gradients stay finite at extreme model states.
    """
    t = temperature.clamp(123.5, 331.5)
    return torch.exp(54.842763 - 6763.22 / t - 4.210 * torch.log(t) + 0.000367 * t
                     + torch.tanh(0.0415 * (t - 218.8))
                     * (53.878 - 1331.22 / t - 9.44523 * torch.log(t) + 0.014025 * t))


def _soft_minimum(value: Tensor, limit: Tensor, sharpness: float = 4.0) -> Tensor:
    """Smooth min(value, limit) for positive arguments; < 1e-4 relative change if value < limit/10."""
    return value / (1 + (value / limit) ** sharpness) ** (1 / sharpness)


def saturation_specific_humidity(temperature: Tensor, pressure: Tensor) -> Tensor:
    """q_s = eps e_s / (p - (1 - eps) e_s), kg/kg.

    In the upper stratosphere e_s(T) can exceed the ambient pressure, where the
    textbook expression is undefined. e_s is therefore softly capped at p/2;
    for e_s < p/10 (the entire troposphere) the cap changes q_s by < 1e-4.
    The resulting q_s is below 0.39 everywhere, so q = RH q_s stays < 1.
    """
    vapour = _soft_minimum(saturation_vapour_pressure_water(temperature), 0.5 * pressure)
    return EPSILON * vapour / (pressure - (1 - EPSILON) * vapour)


def specific_humidity_from_logit(logit: Tensor, temperature: Tensor, pressure: Tensor) -> Tensor:
    """q = RH_max * sigmoid(z) * q_s(T, p); 0 < q <= 1.05 q_s by construction."""
    return MAX_RELATIVE_HUMIDITY * torch.sigmoid(logit) * saturation_specific_humidity(temperature, pressure)


def virtual_temperature(temperature: Tensor, humidity: Tensor) -> Tensor:
    """Tv = T (1 + (1/eps - 1) q), K. Condensate loading is not represented."""
    return temperature * (1 + (1 / EPSILON - 1) * humidity)


def dew_point_from_humidity(humidity: Tensor, pressure: Tensor) -> Tensor:
    """Dew point over water from vapour pressure e = q p / (eps + (1 - eps) q), K.

    Inverse Magnus form with Alduchov & Eskridge (1996) coefficients
    (a = 17.625, b = 243.04 C, c = 610.94 Pa). Its difference from the
    Murphy-Koop saturation curve is below 0.1 K between -40 and +50 C.
    """
    vapour = humidity * pressure / (EPSILON + (1 - EPSILON) * humidity)
    ratio = torch.log(vapour.clamp_min(1e-6) / 610.94)
    return 243.04 * ratio / (17.625 - ratio) + 273.15


def _first_level_above(pressure: Tensor, surface_pressure: Tensor) -> Tensor:
    """Index j of the first descending level with p_j <= p_s (0 if p_s > p_0)."""
    # Levels are descending, so the below-ground levels form a prefix.
    below = (pressure > surface_pressure[..., None]).sum(-1)
    if bool((below >= pressure.shape[-1]).any()):
        raise ValueError('Surface pressure is above the model top.')
    return below


def _bracket(values: Tensor, index: Tensor) -> Tensor:
    return torch.gather(values, -1, index[..., None]).squeeze(-1)


def interpolate_to_surface(values: Tensor, pressure: Tensor, surface_pressure: Tensor) -> Tensor:
    """Linear-in-ln(p) value at p_s from the two levels bracketing it.

    Below the lowest level (p_s > p_0) the lowest-level value is held constant.
    The lower bracket can be below ground; its weight vanishes as p_s -> p_j.
    """
    j = _first_level_above(pressure, surface_pressure)
    lower = (j - 1).clamp_min(0)
    log_p = torch.log(pressure).expand_as(values)
    upper_value, lower_value = _bracket(values, j), _bracket(values, lower)
    upper_log, lower_log = _bracket(log_p, j), _bracket(log_p, lower)
    span = (lower_log - upper_log)
    alpha = torch.where(j > 0, (torch.log(surface_pressure) - upper_log) / torch.where(j > 0, span, torch.ones_like(span)),
                        torch.zeros_like(span))
    return upper_value + alpha.clamp(0, 1) * (lower_value - upper_value)


def geopotential_from_surface(temperature: Tensor, humidity: Tensor, pressure: Tensor,
                              surface_pressure: Tensor, surface_geopotential: Tensor) -> Tensor:
    """Hypsometric geopotential anchored to the lower boundary, m2/s2.

    Phi(p_s) = Phi_s exactly. With j the first level above ground,
        Phi_j = Phi_s + Rd * 0.5 (Tv(p_s) + Tv_j) * ln(p_s / p_j),
        Phi_{k+1} - Phi_k = Rd * 0.5 (Tv_k + Tv_{k+1}) * ln(p_k / p_{k+1}).
    Tv(p_s) is interpolated linearly in ln p, so Phi is continuous in p_s.
    Levels below ground continue the same formula downward and are diagnostic
    only; above-ground values depend on below-ground temperature solely through
    the partial-layer interpolation weight.
    """
    if temperature.shape != humidity.shape or temperature.shape[-1] != pressure.shape[-1]:
        raise ValueError('T, q and pressure axes differ.')
    tv = virtual_temperature(temperature, humidity)
    log_p = torch.log(pressure)
    thickness = RD * 0.5 * (tv[..., :-1] + tv[..., 1:]) * (log_p[:-1] - log_p[1:])
    cumulative = torch.cat((torch.zeros_like(tv[..., :1]), thickness.cumsum(-1)), -1)
    j = _first_level_above(pressure, surface_pressure)
    tv_surface = interpolate_to_surface(tv, pressure, surface_pressure)
    partial = RD * 0.5 * (tv_surface + _bracket(tv, j)) * (torch.log(surface_pressure) - log_p[j])
    phi_j = surface_geopotential + partial
    return phi_j[..., None] + cumulative - _bracket(cumulative, j)[..., None]


def mean_sea_level_pressure(surface_pressure: Tensor, surface_virtual_temperature: Tensor,
                            surface_height: Tensor) -> Tensor:
    """Conventional reduction p_msl = p_s exp(g z_s / (Rd (Tv_s + Gamma z_s / 2))), Pa.

    Gamma = 6.5 K/km defines the fictitious column below the surface; this is
    the reduction convention, not a measurement or an atmospheric profile.
    """
    mean_tv = surface_virtual_temperature + 0.5 * STANDARD_LAPSE * surface_height
    return surface_pressure * torch.exp(G0 * surface_height / (RD * mean_tv))


def pressure_at_height(surface_pressure: Tensor, surface_virtual_temperature: Tensor,
                       surface_height: Tensor, height: Tensor, lapse_rate: Tensor) -> Tensor:
    """Hypsometric pressure at a nearby height (station barometer), Pa.

    p = p_s exp(-g (z - z_s) / (Rd Tv_mean)), Tv_mean = Tv_s - lapse (z - z_s) / 2.
    The lapse rate is the model's own local rate, not a standard atmosphere.
    """
    offset = height - surface_height
    mean_tv = (surface_virtual_temperature - 0.5 * lapse_rate * offset).clamp_min(150.)
    return surface_pressure * torch.exp(-G0 * offset / (RD * mean_tv))


def local_lapse_rate(temperature: Tensor, geopotential: Tensor, pressure: Tensor,
                     surface_pressure: Tensor, minimum: float = -0.03, maximum: float = 0.0098) -> Tensor:
    """-dT/dz from the two lowest above-ground levels, K/m, bounded to [-0.03, g/cp]."""
    j = _first_level_above(pressure, surface_pressure)
    k = (j + 1).clamp_max(pressure.shape[-1] - 1)
    dz = (_bracket(geopotential, k) - _bracket(geopotential, j)) / G0
    dt = _bracket(temperature, j) - _bracket(temperature, k)
    return (dt / dz.clamp_min(1.0)).clamp(minimum, maximum)


def column_integral(values: Tensor, pressure: Tensor, surface_pressure: Tensor) -> Tensor:
    """int_{p_top}^{p_s} x dp over above-ground air, trapezoidal in p, units [x] Pa."""
    above = (pressure <= surface_pressure[..., None]).to(values.dtype)
    layer = 0.5 * (values[..., :-1] + values[..., 1:]) * (pressure[:-1] - pressure[1:])
    full = (layer * above[..., :-1]).sum(-1)
    j = _first_level_above(pressure, surface_pressure)
    surface_value = interpolate_to_surface(values, pressure, surface_pressure)
    partial = 0.5 * (surface_value + _bracket(values, j)) * (surface_pressure - pressure[j])
    return full + partial


def dry_air_mass_per_area(surface_pressure: Tensor, humidity: Tensor, pressure: Tensor) -> Tensor:
    """(p_s - p_top - int q dp) / g, kg/m2. Air above the 1 hPa top is excluded."""
    water = column_integral(humidity, pressure, surface_pressure)
    return (surface_pressure - pressure[-1] - water) / G0


def global_dry_air_mass(surface_pressure: Tensor, humidity: Tensor, pressure: Tensor, areas: Tensor) -> Tensor:
    """Sum over cells of dry mass per area times cell area, kg (leading batch dims kept)."""
    return (dry_air_mass_per_area(surface_pressure, humidity, pressure) * areas).sum(-1)


def correct_dry_air_mass(surface_pressure: Tensor, humidity: Tensor, pressure: Tensor, areas: Tensor,
                         target_mass: Tensor, iterations: int = 2) -> Tensor:
    """Uniform surface-pressure offset restoring the global dry-air mass.

    Newton step on M(p_s + delta) = M_target with dM/d delta = sum A (1 - q_s) / g.
    The uniform offset follows the ACE corrector; it changes no horizontal
    gradient of p_s. Moisture sources/sinks are not represented here.
    """
    corrected = surface_pressure
    for _ in range(iterations):
        mass = global_dry_air_mass(corrected, humidity, pressure, areas)
        q_surface = interpolate_to_surface(humidity, pressure, corrected)
        slope = ((1 - q_surface) * areas).sum(-1) / G0
        corrected = corrected + ((target_mass - mass) / slope)[..., None]
    return corrected


def omega_with_lower_boundary(divergence: Tensor, pressure: Tensor, surface_pressure: Tensor,
                              surface_omega: Tensor) -> tuple[Tensor, Tensor]:
    """Continuity omega with O'Brien (1970) correction, Pa/s; returns (omega, raw omega).

    Raw: d(omega)/dp = -div(V), omega(p_top) = 0, trapezoidal in p.
    Corrected: omega = raw - (p - p_top) / (p_s - p_top) * (raw(p_s) - omega_s),
    so omega(p_top) = 0 and omega(p_s) = omega_s = dp_s/dt (kinematic boundary).
    Values at levels below ground are continuations and must be masked.
    """
    top = pressure[-1]
    layers = 0.5 * (divergence[..., :-1] + divergence[..., 1:]) * (pressure[:-1] - pressure[1:])
    # Integrate from the top downward: omega_k = -sum_{m >= k} layers_m.
    from_top = torch.flip(torch.cumsum(torch.flip(layers, (-1,)), -1), (-1,))
    raw = -torch.cat((from_top, torch.zeros_like(divergence[..., :1])), -1)
    j = _first_level_above(pressure, surface_pressure)
    div_surface = interpolate_to_surface(divergence, pressure, surface_pressure)
    raw_surface = _bracket(raw, j) - 0.5 * (_bracket(divergence, j) + div_surface) * (surface_pressure - pressure[j])
    weight = (pressure - top) / (surface_pressure - top)[..., None]
    corrected = raw - weight * (raw_surface - surface_omega)[..., None]
    return corrected, raw


def _spencer(day_of_year: float) -> tuple[float, float, float]:
    """Spencer (1971): declination [rad], eccentricity factor, equation of time [h]."""
    g = 2 * math.pi * (day_of_year - 1) / 365.0
    declination = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
                   + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eccentricity = (1.000110 + 0.034221 * math.cos(g) + 0.001280 * math.sin(g)
                    + 0.000719 * math.cos(2 * g) + 0.000077 * math.sin(2 * g))
    equation_minutes = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g)
                                 - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    return declination, eccentricity, equation_minutes / 60.0


def toa_insolation(xyz: Tensor, when: datetime) -> Tensor:
    """Instantaneous top-of-atmosphere incident shortwave flux, W/m2.

    S0 * E0 * max(cos(zenith), 0) with the Spencer (1971) declination,
    eccentricity and equation of time. Deterministic astronomy, not a
    radiation model and not an observation.
    """
    when = when.astimezone(timezone.utc)
    day = when.timetuple().tm_yday + (when.hour + when.minute / 60 + when.second / 3600) / 24
    declination, eccentricity, equation_hours = _spencer(day)
    latitude = torch.asin(xyz[..., 2].clamp(-1, 1))
    longitude = torch.atan2(xyz[..., 1], xyz[..., 0])
    solar_hours = when.hour + when.minute / 60 + when.second / 3600 + equation_hours
    hour_angle = 2 * math.pi * solar_hours / 24 + longitude - math.pi
    cosine = (torch.sin(latitude) * math.sin(declination)
              + torch.cos(latitude) * math.cos(declination) * torch.cos(hour_angle))
    return SOLAR_CONSTANT * eccentricity * cosine.clamp_min(0)


def mean_insolation(xyz: Tensor, start: datetime, hours: float, samples: int = 7) -> Tensor:
    """Average of toa_insolation over [start, start + hours] with Simpson-like sampling."""
    from datetime import timedelta
    if samples < 2:
        return toa_insolation(xyz, start)
    weights = np.ones(samples)
    weights[0] = weights[-1] = 0.5
    weights /= weights.sum()
    total = torch.zeros(xyz.shape[:-1], dtype=xyz.dtype, device=xyz.device)
    for weight, fraction in zip(weights, np.linspace(0., 1., samples)):
        total = total + float(weight) * toa_insolation(xyz, start + timedelta(hours=hours * float(fraction)))
    return total
