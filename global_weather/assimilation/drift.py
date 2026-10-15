"""Radiosonde balloon drift and level time from the sounding's own measurements.

IGRA-normalized records carry the launch position for every level. A balloon
rising at about 5 m/s through a 30-60 m/s jet travels 50-150 km before
100 hPa, so the launch position misplaces upper levels by one to several
latent cells. Following Laroche and Sarrazin (2013, Weather and Forecasting
28, 772-782), the horizontal track is integrated from the sounding's own
winds:

    x(t_k) = x_0 + sum_{i<k} 0.5 (V_i + V_{i+1}) (t_{i+1} - t_i),

with level time from the reported elapsed time (IGRA ETIME) when present.
Without it, the time is interpolated in height between reported times or
estimated from a nominal ascent rate of 5 m/s. Heights come from the measured
geopotential; gaps use the hypsometric relation with the measured
temperature. Only when a sounding reports no temperature at all is an
isothermal 250 K layer used to space the levels, and the record says so.

Nothing here creates a measurement: values are untouched; only the position
and, for launch-time fallbacks with an actual launch, the level time move.
Availability moves by the same shift, so a reconstructed later time can never
make a measurement causal earlier. Positions use only rows the caller passed,
so applying the availability cut first keeps the reconstruction causal.
"""
from __future__ import annotations

import numpy as np

from .thermo import G0, RD

MODEL = 'sonde-drift-laroche-sarrazin-2013-v1'
ASCENT_RATE_M_S = 5.0
EARTH_RADIUS_M = 6_371_000.0
MAX_ASCENT_SECONDS = 3 * 3600.0
FALLBACK_TEMPERATURE_K = 250.0

# Position basis codes stored with each token for audit.
LAUNCH = 0              # no wind in the sounding: launch position kept
REPORTED = 1            # the record reported its own level position
DRIFT_REPORTED_TIME = 2  # drift integrated over reported level times
DRIFT_ESTIMATED_TIME = 3  # drift integrated over heights / nominal ascent rate

# A missing basis is never trusted as a measured level time.
REPORTED_TIME_BASES = ('reported_level_time', 'launch_plus_elapsed_seconds')


def _interp_log_p(pressure, known_pressure, known_value):
    """Linear in ln p, constant beyond the measured range."""
    order = np.argsort(-known_pressure)  # descending pressure = ascending ln(1/p)
    x = -np.log(known_pressure[order])
    return np.interp(-np.log(pressure), x, known_value[order])


def _heights(levels, temperature, geopotential):
    """Geopotential height of each level (descending pressure), from the sounding's own data."""
    z = geopotential / G0
    known_t = np.isfinite(temperature)
    if known_t.any():
        t = _interp_log_p(levels, levels[known_t], temperature[known_t])
        fallback = False
    else:
        t = np.full(len(levels), FALLBACK_TEMPERATURE_K)
        fallback = True
    # Layer thickness between consecutive levels (hypsometric, mean T).
    thickness = RD * 0.5 * (t[1:] + t[:-1]) / G0 * np.log(levels[:-1] / levels[1:])
    relative = np.concatenate([[0.0], np.cumsum(thickness)])
    known_z = np.isfinite(z)
    if known_z.any():
        # Anchor the hypsometric shape to every measured height; between two
        # measured heights distribute the residual linearly in the relative coordinate.
        offset = z[known_z] - relative[known_z]
        heights = relative + np.interp(relative, relative[known_z], offset)
    else:
        heights = relative
    return heights, fallback


def _elapsed(levels_time, heights):
    """Seconds after the first level: reported times where known, else height-interpolated."""
    known = np.isfinite(levels_time)
    if known.sum() >= 2:
        order = np.argsort(heights[known])
        hk, tk = heights[known][order], levels_time[known][order]
        out = np.interp(heights, hk, tk)
        below, above = heights < hk[0], heights > hk[-1]
        out[below] = tk[0] - (hk[0] - heights[below]) / ASCENT_RATE_M_S
        out[above] = tk[-1] + (heights[above] - hk[-1]) / ASCENT_RATE_M_S
        return out, False
    if known.sum() == 1:
        anchor = int(np.flatnonzero(known)[0])
        return levels_time[anchor] + (heights - heights[anchor]) / ASCENT_RATE_M_S, True
    return (heights - heights[0]) / ASCENT_RATE_M_S, True


def reconstruct(*, group, pressure, variable, value, observed, available, latitude, longitude,
                time_basis, position_basis, launch):
    """Return (latitude, longitude, observed, available, basis) for each row.

    ``variable`` holds names ('temperature', 'u', 'v', 'geopotential', ...);
    ``time_basis`` and ``position_basis`` are the normalized record bases (None if absent);
    ``launch`` holds the actual launch time in seconds or NaN when only nominal time is known.
    """
    n = len(group)
    lat, lon = latitude.astype(float).copy(), longitude.astype(float).copy()
    obs, avail = observed.astype(float).copy(), available.astype(float).copy()
    basis = np.full(n, LAUNCH, np.int8)
    reported_position = np.array([b == 'reported_level_position' for b in position_basis], bool)
    basis[reported_position] = REPORTED
    reported_time = np.array([b in REPORTED_TIME_BASES for b in time_basis], bool)
    names = np.asarray(variable)
    keys, inverse = np.unique(np.asarray(group), return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.argsort(inverse, kind='stable')
    bounds = np.searchsorted(inverse[order], np.arange(len(keys) + 1))
    for g in range(len(keys)):
        rows = order[bounds[g]:bounds[g + 1]]  # ascending row indices (stable sort)
        if reported_position[rows].all():
            continue
        p_rows = pressure[rows]
        levels, level_of = np.unique(-p_rows, return_inverse=True)
        levels = -levels  # descending pressure: bottom of the ascent first
        k = len(levels)

        def per_level(name):
            out = np.full(k, np.nan)
            mask = names[rows] == name
            out[level_of[mask]] = value[rows][mask]
            return out

        u, v = per_level('u'), per_level('v')
        heights, isothermal = _heights(levels, per_level('temperature'), per_level('geopotential'))
        level_time = np.full(k, np.nan)
        timed = reported_time[rows]
        level_time[level_of[timed]] = obs[rows][timed]
        launch_time = launch[rows][np.isfinite(launch[rows])]
        start = float(launch_time[0]) if len(launch_time) else float(obs[rows].min())
        elapsed, estimated = _elapsed(level_time - start, heights)
        elapsed = np.maximum.accumulate(np.clip(elapsed, 0.0, MAX_ASCENT_SECONDS))
        wind = np.isfinite(u) & np.isfinite(v)
        if not wind.any():
            new_time = start + elapsed
        else:
            uu = _interp_log_p(levels, levels[wind], u[wind])
            vv = _interp_log_p(levels, levels[wind], v[wind])
            dt = np.diff(elapsed)
            east = np.concatenate([[0.0], np.cumsum(0.5 * (uu[1:] + uu[:-1]) * dt)])
            north = np.concatenate([[0.0], np.cumsum(0.5 * (vv[1:] + vv[:-1]) * dt)])
            lat0, lon0 = float(latitude[rows][0]), float(longitude[rows][0])
            level_lat = np.clip(lat0 + np.degrees(north / EARTH_RADIUS_M), -89.999, 89.999)
            mid = np.radians(0.5 * (lat0 + level_lat))
            level_lon = lon0 + np.degrees(east / (EARTH_RADIUS_M * np.maximum(np.cos(mid), 1e-3)))
            level_lon = (level_lon + 180.0) % 360.0 - 180.0
            move = ~reported_position[rows]
            lat[rows[move]] = level_lat[level_of[move]]
            lon[rows[move]] = level_lon[level_of[move]]
            basis[rows[move]] = DRIFT_ESTIMATED_TIME if (estimated or isothermal) else DRIFT_REPORTED_TIME
            new_time = start + elapsed
        # Level time moves only for launch-time fallbacks with an actual launch.
        shift_rows = rows[(~timed) & np.isfinite(launch[rows])]
        if len(shift_rows):
            shifted = new_time[level_of[np.searchsorted(rows, shift_rows)]]
            delta = np.maximum(shifted - obs[shift_rows], 0.0)
            obs[shift_rows] += delta
            avail[shift_rows] += delta
    return lat, lon, obs, avail, basis


def great_circle_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    d = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M / 1000 * np.arcsin(np.sqrt(np.clip(d, 0, 1)))


__all__ = ['MODEL', 'reconstruct', 'great_circle_km', 'ASCENT_RATE_M_S', 'LAUNCH', 'REPORTED',
           'DRIFT_REPORTED_TIME', 'DRIFT_ESTIMATED_TIME']
