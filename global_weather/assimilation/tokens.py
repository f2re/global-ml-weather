"""Observation tokens for every platform: registries, columnar batches, norms, errors.

A token is one measured scalar at its own position, time and vertical
coordinate. Nothing is interpolated onto standard levels or rounded to
synoptic times. Vector winds keep their component direction so the encoder
can rotate them into each node's local frame.

Input normalisation (what the encoder sees) and observation error (what the
loss divides by) are different objects:
* pressure-coordinate T, q, u, v and geopotential use the pinned GraphCast
  mu/sigma interpolated in ln p at the measured pressure (decision 0010);
* every other platform/variable/channel uses train-only statistics computed
  by its source; station pressure is normalised after a fixed height
  reduction ln p + z / 8000 m, so the scale is not dominated by elevation;
* observation errors come from configs/observation_errors_r11.json, or from
  an admitted radiometric passport for satellite channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import hashlib
import math

import numpy as np
import torch

from ..vertical import PRESSURE_HPA

PLATFORMS = ('surface_station', 'ship', 'buoy', 'radiosonde', 'dropsonde', 'pilot_balloon',
             'wind_profiler', 'aircraft', 'satellite_amv', 'scatterometer', 'gnss_ro',
             'satellite_radiance', 'satellite_product')

PRESSURE_VARIABLES = ('temperature', 'specific_humidity', 'u', 'v', 'geopotential')
SURFACE_VARIABLES = ('station_temperature', 'station_dew_point', 'station_eastward_wind',
                     'station_northward_wind', 'station_pressure', 'station_mean_sea_level_pressure')
HEIGHT_VARIABLES = ('refractivity',)
COLUMN_VARIABLES = ('brightness_temperature', 'satellite_product')
VARIABLES = PRESSURE_VARIABLES + SURFACE_VARIABLES + HEIGHT_VARIABLES + COLUMN_VARIABLES
UNITS = {'temperature': 'K', 'specific_humidity': 'kg kg-1', 'u': 'm s-1', 'v': 'm s-1',
         'geopotential': 'm2 s-2', 'station_temperature': 'K', 'station_dew_point': 'K',
         'station_eastward_wind': 'm s-1', 'station_northward_wind': 'm s-1', 'station_pressure': 'Pa',
         'station_mean_sea_level_pressure': 'Pa', 'refractivity': '1', 'brightness_temperature': 'K',
         'satellite_product': None}
OPERATOR = {**{v: 'pressure' for v in PRESSURE_VARIABLES}, **{v: 'surface' for v in SURFACE_VARIABLES},
            **{v: 'height' for v in HEIGHT_VARIABLES}, **{v: 'column' for v in COLUMN_VARIABLES}}
PLATFORM_VARIABLES = {
    'surface_station': SURFACE_VARIABLES, 'ship': SURFACE_VARIABLES, 'buoy': SURFACE_VARIABLES,
    'scatterometer': ('station_eastward_wind', 'station_northward_wind'),
    'radiosonde': PRESSURE_VARIABLES, 'dropsonde': PRESSURE_VARIABLES,
    'pilot_balloon': ('u', 'v'), 'wind_profiler': ('u', 'v'),
    'aircraft': ('temperature', 'specific_humidity', 'u', 'v'), 'satellite_amv': ('u', 'v'),
    'gnss_ro': ('refractivity',), 'satellite_radiance': ('brightness_temperature',),
    'satellite_product': ('satellite_product',),
}
FAMILY = {'surface_station': 'surface', 'ship': 'surface', 'buoy': 'surface', 'scatterometer': 'surface',
          'radiosonde': 'upper_air', 'dropsonde': 'upper_air', 'pilot_balloon': 'upper_air',
          'wind_profiler': 'upper_air', 'aircraft': 'upper_air', 'satellite_amv': 'upper_air',
          'gnss_ro': 'limb', 'satellite_radiance': 'radiance', 'satellite_product': 'product'}
# Component direction of each wind scalar in the local tangent frame.
EAST_COMPONENTS = ('u', 'station_eastward_wind')
NORTH_COMPONENTS = ('v', 'station_northward_wind')
HEIGHT_REDUCTION_M = 8000.0  # fixed reference scale height for station-pressure normalisation only
SECONDS_PER_HOUR = 3600.0


def stable_hash(text: str) -> int:
    """Deterministic signed 63-bit identity used for groups and duplicate detection."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'big') >> 1


@dataclass
class TokenBatch:
    """Columnar observations; all arrays have identical length P."""
    platform: np.ndarray          # int16 index into PLATFORMS
    variable: np.ndarray          # int16 index into VARIABLES
    channel: np.ndarray           # int32 index into the experiment channel table, -1 if none
    group: np.ndarray             # int64 station / launch / flight / granule identity hash
    identity: np.ndarray          # int64 measurement identity hash (duplicates are refused)
    observed: np.ndarray          # float64 seconds since 1970-01-01 UTC
    available: np.ndarray         # float64 seconds since 1970-01-01 UTC
    latitude: np.ndarray          # float64 degrees
    longitude: np.ndarray         # float64 degrees
    pressure_pa: np.ndarray       # float64, NaN when not a pressure coordinate
    height_m: np.ndarray          # float64 station elevation / tangent height, NaN if unknown
    value: np.ndarray             # float64 physical value in UNITS
    error: np.ndarray             # float64 explicit observation error, NaN -> configured model
    quality: np.ndarray           # float32 in (0, 1]
    zenith_cos: np.ndarray = field(default=None)  # float32, NaN when not applicable

    def __post_init__(self):
        if self.zenith_cos is None:
            self.zenith_cos = np.full(len(self.value), np.nan, dtype=np.float32)
        lengths = {len(getattr(self, f.name)) for f in fields(self)}
        if len(lengths) != 1:
            raise ValueError('Token columns differ in length.')

    def __len__(self) -> int:
        return len(self.value)

    @classmethod
    def empty(cls) -> 'TokenBatch':
        kinds = {'platform': np.int16, 'variable': np.int16, 'channel': np.int32, 'group': np.int64,
                 'identity': np.int64, 'quality': np.float32, 'zenith_cos': np.float32}
        return cls(**{f.name: np.zeros(0, dtype=kinds.get(f.name, np.float64)) for f in fields(cls)})

    def select(self, mask) -> 'TokenBatch':
        return TokenBatch(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})

    @staticmethod
    def concatenate(batches) -> 'TokenBatch':
        batches = [b for b in batches if len(b)]
        if not batches:
            return TokenBatch.empty()
        return TokenBatch(**{f.name: np.concatenate([getattr(b, f.name) for b in batches])
                             for f in fields(TokenBatch)})

    def validate(self) -> 'TokenBatch':
        """Physical admission common to all sources; raises instead of repairing."""
        if len(self) == 0:
            return self
        if (self.platform.min() < 0 or self.platform.max() >= len(PLATFORMS)
                or self.variable.min() < 0 or self.variable.max() >= len(VARIABLES)):
            raise ValueError('Unknown platform or variable index.')
        for p in np.unique(self.platform):
            used = {VARIABLES[v] for v in np.unique(self.variable[self.platform == p])}
            if not used <= set(PLATFORM_VARIABLES[PLATFORMS[p]]):
                raise ValueError(f'Variables {sorted(used)} are not admitted for platform {PLATFORMS[p]}.')
        if not (np.isfinite(self.value).all() and np.isfinite(self.observed).all() and np.isfinite(self.available).all()):
            raise ValueError('Nonfinite value or time; a missing measurement is not a token.')
        if np.any(self.available < self.observed):
            raise ValueError('Observation available before it was measured.')
        if (not np.isfinite(self.latitude).all() or np.any(np.abs(self.latitude) > 90)
                or not np.isfinite(self.longitude).all() or np.any(np.abs(self.longitude) > 180)):
            raise ValueError('Invalid observation geometry.')
        if np.any((self.quality <= 0) | (self.quality > 1)) or not np.isfinite(self.quality).all():
            raise ValueError('Quality must be within (0, 1].')
        explicit = np.isfinite(self.error)
        if np.any(self.error[explicit] <= 0):
            raise ValueError('Observation error must be positive.')
        kinds = np.array([OPERATOR[VARIABLES[v]] for v in self.variable])
        pressure = kinds == 'pressure'
        if np.any(~np.isfinite(self.pressure_pa[pressure])) or np.any(self.pressure_pa[pressure] <= 0):
            raise ValueError('Pressure-coordinate observation needs its measured pressure.')
        height = kinds == 'height'
        if np.any(~np.isfinite(self.height_m[height])):
            raise ValueError('Height-coordinate observation needs its tangent height.')
        q = self.variable == VARIABLES.index('specific_humidity')
        if np.any((self.value[q] < 0) | (self.value[q] >= 1)):
            raise ValueError('Specific humidity outside [0, 1).')
        if len(np.unique(self.identity)) != len(self):
            raise ValueError('The same measurement version appears twice in one window.')
        return self


def wind_directions(batch: TokenBatch) -> np.ndarray:
    """Global Cartesian unit vector of each wind component (zeros for scalars)."""
    lat, lon = np.deg2rad(batch.latitude), np.deg2rad(batch.longitude)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), -1)
    north = np.stack((-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)), -1)
    names = np.array(VARIABLES, dtype=object)[batch.variable]
    is_east = np.isin(names, EAST_COMPONENTS)[:, None]
    is_north = np.isin(names, NORTH_COMPONENTS)[:, None]
    return np.where(is_east, east, 0.) + np.where(is_north, north, 0.)


class ErrorModel:
    """Observation error sigma per platform/variable; pressure tables are linear in ln p."""

    def __init__(self, payload: dict):
        if payload.get('schema') != 'observation-errors-1':
            raise ValueError('Unknown observation error schema.')
        self.payload = payload
        self.table = payload['platforms']
        for platform, variables in self.table.items():
            if platform not in PLATFORMS:
                raise ValueError('Unknown platform in error table: ' + platform)
            for variable, spec in variables.items():
                if variable not in PLATFORM_VARIABLES[platform]:
                    raise ValueError(f'{variable} is not admitted for {platform}.')
                self._check(spec)

    @staticmethod
    def _check(spec):
        keys = set(spec)
        if keys == {'from_record'}:
            if spec['from_record'] is not True:
                raise ValueError('from_record must be true.')
        elif keys == {'sigma'}:
            if not (isinstance(spec['sigma'], (int, float)) and math.isfinite(spec['sigma']) and spec['sigma'] > 0):
                raise ValueError('Positive constant sigma required.')
        elif keys == {'pressure_hpa', 'sigma'}:
            p, s = np.asarray(spec['pressure_hpa'], float), np.asarray(spec['sigma'], float)
            if p.shape != s.shape or len(p) < 2 or not (np.diff(p) < 0).all() or (s <= 0).any():
                raise ValueError('Pressure table needs descending pressures and positive sigmas.')
        elif keys == {'relative', 'floor'}:
            if not (0 < spec['relative'] < 1 and spec['floor'] > 0):
                raise ValueError('Relative error needs 0 < relative < 1 and a positive floor.')
        else:
            raise ValueError('Unknown observation error specification: ' + str(sorted(keys)))

    def sigma(self, batch: TokenBatch) -> np.ndarray:
        out = np.full(len(batch), np.nan)
        for p in np.unique(batch.platform):
            platform = PLATFORMS[p]
            for v in np.unique(batch.variable[batch.platform == p]):
                variable = VARIABLES[v]
                rows = (batch.platform == p) & (batch.variable == v)
                spec = self.table.get(platform, {}).get(variable)
                if spec is None:
                    raise ValueError(f'No observation error for {platform}/{variable}.')
                if 'from_record' in spec:
                    out[rows] = batch.error[rows]
                elif set(spec) == {'sigma'}:
                    out[rows] = spec['sigma']
                elif 'pressure_hpa' in spec:
                    x = -np.log(np.asarray(spec['pressure_hpa'], float) * 100.)
                    out[rows] = np.interp(-np.log(batch.pressure_pa[rows]), x, np.asarray(spec['sigma'], float))
                else:
                    out[rows] = np.maximum(spec['relative'] * np.abs(batch.value[rows]), spec['floor'])
        # An explicit per-record error (instrument passport) always wins when present.
        explicit = np.isfinite(batch.error)
        out[explicit] = batch.error[explicit]
        if not np.isfinite(out).all() or (out <= 0).any():
            raise ValueError('Missing observation error; refuse instead of inventing one.')
        return out


class Normalizer:
    """Input scaling for the encoder; never used as the loss metric."""

    def __init__(self, profile_mean: np.ndarray, profile_std: np.ndarray, statistics: dict):
        profile_mean, profile_std = np.asarray(profile_mean, float), np.asarray(profile_std, float)
        if profile_mean.shape != (37, 5) or profile_std.shape != (37, 5) or (profile_std <= 0).any():
            raise ValueError('Pinned profile norms must cover 37 levels and 5 variables.')
        self.profile_mean, self.profile_std = profile_mean, profile_std
        self.log_pressure = -np.log(np.asarray(PRESSURE_HPA, float) * 100.)  # ascending
        self.statistics = {}
        for key, stat in statistics.items():
            if (not isinstance(stat, dict) or set(stat) - {'mean', 'std', 'transform', 'count', 'period'}
                    or not math.isfinite(stat['mean']) or not stat['std'] > 0
                    or stat.get('transform', 'identity') not in ('identity', 'height_reduced_log')):
                raise ValueError('Invalid train-only statistic for ' + key)
            self.statistics[key] = stat

    @staticmethod
    def key(platform: str, variable: str, channel: str | None = None) -> str:
        return f'{platform}|{variable}|{channel or ""}'

    def payload(self) -> dict:
        return {'profile_mean': self.profile_mean.tolist(), 'profile_std': self.profile_std.tolist(),
                'statistics': self.statistics}

    def normalise(self, batch: TokenBatch, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (normalised value, usable mask); unknown statistics are reported, not invented."""
        out = np.full(len(batch), np.nan)
        for v in np.unique(batch.variable):
            name = VARIABLES[v]
            rows = batch.variable == v
            if name in PRESSURE_VARIABLES:
                k = PRESSURE_VARIABLES.index(name)
                x = -np.log(batch.pressure_pa[rows])
                mean = np.interp(x, self.log_pressure, self.profile_mean[::-1, k])
                std = np.interp(x, self.log_pressure, self.profile_std[::-1, k])
                out[rows] = (batch.value[rows] - mean) / std
                continue
            for p in np.unique(batch.platform[rows]):
                for c in np.unique(batch.channel[rows & (batch.platform == p)]):
                    subset = rows & (batch.platform == p) & (batch.channel == c)
                    stat = self.statistics.get(self.key(PLATFORMS[p], name, channels[c] if c >= 0 else None))
                    if stat is None:
                        continue
                    value = batch.value[subset]
                    if stat.get('transform') == 'height_reduced_log':
                        value = np.log(value) + batch.height_m[subset] / HEIGHT_REDUCTION_M
                    out[subset] = (value - stat['mean']) / stat['std']
        return out, np.isfinite(out)


@dataclass
class EncodedTokens:
    """Torch view of a token batch for the encoder."""
    xyz: torch.Tensor          # [P,3]
    features: torch.Tensor     # [P,F]
    direction: torch.Tensor    # [P,3] global Cartesian component direction * normalised value
    platform: torch.Tensor     # [P] long
    variable: torch.Tensor     # [P] long
    channel: torch.Tensor      # [P] long (0 reserved for "no channel")
    slot: torch.Tensor         # [P,S] soft vertical slot assignment
    latitude: np.ndarray
    longitude: np.ndarray

    def __len__(self) -> int:
        return len(self.platform)


FEATURE_NAMES = ('value', 'has_pressure', 'log_pressure', 'has_height', 'height', 'age_hours',
                 'log_relative_error', 'quality', 'zenith_cos', 'has_zenith')
PRESSURE_BANDS = 8


def vertical_slots(batch: TokenBatch, bands: int = PRESSURE_BANDS) -> np.ndarray:
    """Triangular weights over `bands` ln p bands + surface, limb and column slots."""
    centres = np.linspace(np.log(100000.), np.log(100.), bands)
    width = centres[0] - centres[1]
    slots = np.zeros((len(batch), bands + 3))
    kinds = np.array([OPERATOR[VARIABLES[v]] for v in batch.variable]) if len(batch) else np.zeros(0, str)
    pressure = kinds == 'pressure'
    if pressure.any():
        x = np.log(np.clip(batch.pressure_pa[pressure], 100., 100000.))
        slots[pressure, :bands] = np.clip(1 - np.abs(x[:, None] - centres[None]) / width, 0, 1)
    slots[kinds == 'surface', bands] = 1
    slots[kinds == 'height', bands + 1] = 1
    slots[kinds == 'column', bands + 2] = 1
    return slots / np.maximum(slots.sum(1, keepdims=True), 1e-12)


def encode(batch: TokenBatch, issue_seconds: float, normalizer: Normalizer, sigma: np.ndarray,
           channels: list[str], device=None, dtype=torch.float32) -> tuple[EncodedTokens, np.ndarray]:
    """Return encoder tensors and the boolean mask of tokens with known input statistics."""
    value, usable = normalizer.normalise(batch, channels)
    batch = batch.select(usable)
    value, sigma = value[usable], sigma[usable]
    has_pressure = np.isfinite(batch.pressure_pa)
    has_height = np.isfinite(batch.height_m)
    has_zenith = np.isfinite(batch.zenith_cos)
    log_pressure = np.where(has_pressure, np.log(np.where(has_pressure, batch.pressure_pa, 1e5) / 1e5) / 7., 0.)
    scale = np.abs(batch.value) + 1e-12
    features = np.stack((np.clip(value, -10, 10), has_pressure, log_pressure, has_height,
                         np.where(has_height, batch.height_m, 0.) / 5000.,
                         (batch.observed - issue_seconds) / (12 * SECONDS_PER_HOUR),
                         np.clip(np.log(sigma / scale), -10, 5) / 5., batch.quality,
                         np.where(has_zenith, batch.zenith_cos, 0.), has_zenith), 1)
    from ..grid import unit_xyz
    tensors = EncodedTokens(
        xyz=torch.tensor(unit_xyz(batch.latitude, batch.longitude), dtype=dtype, device=device),
        features=torch.tensor(features, dtype=dtype, device=device),
        direction=torch.tensor(wind_directions(batch) * np.clip(value, -10, 10)[:, None], dtype=dtype, device=device),
        platform=torch.tensor(batch.platform.astype(np.int64), device=device),
        variable=torch.tensor(batch.variable.astype(np.int64), device=device),
        channel=torch.tensor(batch.channel.astype(np.int64) + 1, device=device),
        slot=torch.tensor(vertical_slots(batch), dtype=dtype, device=device),
        latitude=batch.latitude, longitude=batch.longitude)
    return tensors, usable
