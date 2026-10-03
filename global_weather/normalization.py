"""Frozen, provenance-bearing reanalysis statistics; never fit during a forecast.

Pressure interpolation is explicit and bounded. A global level-wise mean is not
local/monthly climatology, and a normalisation constant is not an observation.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ZStat:
    units: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    pressure_pa: tuple[float, ...] = ()
    interval_hours: int | None = None

    def __post_init__(self):
        for name in ('mean', 'std', 'pressure_pa'):
            object.__setattr__(self, name, tuple(float(v) for v in getattr(self, name)))
        if not self.units or not self.mean or len(self.mean) != len(self.std):
            raise ValueError('Statistics require units and matching nonempty mean/std.')
        if not np.isfinite(self.mean + self.std).all() or min(self.std) <= 0:
            raise ValueError('Statistics must be finite and standard deviations positive.')
        p = self.pressure_pa
        if p and (len(p) != len(self.mean) or not np.isfinite(p).all()
                  or min(p) <= 0 or not np.all(np.diff(p) < 0)):
            raise ValueError('Pressure coordinates must match statistics and descend.')
        if not p and len(self.mean) != 1:
            raise ValueError('A surface statistic must be scalar.')
        if self.interval_hours is not None and (not isinstance(self.interval_hours, int)
                                              or self.interval_hours <= 0):
            raise ValueError('Accumulation interval must be a positive integer.')

    def at(self, pressure_pa=None, *, interval_hours=None):
        if self.interval_hours != interval_hours:
            raise ValueError('Accumulation interval mismatch; never rescale 6h stats to 3h.')
        if not self.pressure_pa:
            return self.mean[0], self.std[0]
        if pressure_pa is None:
            raise ValueError('Pressure required for level-dependent normalisation.')
        p = np.asarray(pressure_pa, dtype=float)
        if not np.isfinite(p).all() or (p < self.pressure_pa[-1]).any() or (p > self.pressure_pa[0]).any():
            raise ValueError('Normalisation pressure outside source coverage.')
        axis = -np.log(self.pressure_pa)
        return (np.interp(-np.log(p), axis, self.mean),
                np.interp(-np.log(p), axis, self.std))


class NormalizationBundle:
    """Numerical statistics plus immutable provenance; no implicit fallbacks."""
    def __init__(self, payload):
        self._payload = json.loads(json.dumps(payload, allow_nan=False))
        if payload.get('schema_version') != 1 or payload.get('kind') != 'global_level_zscore':
            raise ValueError('Unsupported normalisation schema or statistical meaning.')
        source = payload.get('provenance', {})
        for key in ('repository', 'revision', 'data_family', 'artifact_sha256', 'fit_period', 'license'):
            if key not in source:
                raise ValueError(f'Missing normalisation provenance: {key}')
        if not all(source[k] for k in ('repository', 'revision', 'data_family', 'license')):
            raise ValueError('Source identity and license must be explicit.')
        hashes = source['artifact_sha256']
        if not isinstance(hashes, dict) or not hashes or any(
                not isinstance(h,str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in hashes.values()):
            raise ValueError('Actual SHA256 of source artifacts required.')
        self.stats = {k: ZStat(**v) for k, v in payload['variables'].items()}
        if not self.stats:
            raise ValueError('Empty normalisation bundle.')
        self.fingerprint = digest(self._payload)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + '.tmp')
        tmp.write_text(json.dumps(self._payload, ensure_ascii=False, indent=2,
                                  allow_nan=False) + '\n', encoding='utf-8')
        tmp.replace(target)

    def get(self, name, units):
        if name not in self.stats:
            raise ValueError(f'No verified normalisation for {name}. Fit on training reanalysis only.')
        stat = self.stats[name]
        if stat.units != units:
            raise ValueError(f'Normalisation units mismatch for {name}.')
        return stat

    def normalise(self, name, value, units, pressure_pa=None, *, interval_hours=None):
        mean, std = self.get(name, units).at(pressure_pa, interval_hours=interval_hours)
        return (np.asarray(value) - mean) / std

    def unnormalise(self, name, value, units, pressure_pa=None, *, interval_hours=None):
        mean, std = self.get(name, units).at(pressure_pa, interval_hours=interval_hours)
        return np.asarray(value) * std + mean

    def assert_independent_test(self, test_start):
        """Unknown fit period must not be represented as a clean retrospective test."""
        period = self._payload['provenance']['fit_period']
        if not isinstance(period, dict) or not period.get('end'):
            raise ValueError('Source fit period unknown: independent retrospective test not certified.')
        end = datetime.fromisoformat(period['end'].replace('Z', '+00:00'))
        start = datetime.fromisoformat(test_start.replace('Z', '+00:00'))
        if end.tzinfo is None or start.tzinfo is None or end >= start:
            raise ValueError('Normalisation fitting overlaps test period or lacks timezone.')


def weighted_statistics(values, mask, weights):
    """Population mean/std over sample axis 0, with area weights and missingness.

    Caller selects TRAINING times only and stores their provenance. Use float64;
    these estimates are not online adaptation and must be frozen for evaluation.
    """
    x = np.asarray(values, dtype=np.float64)
    m = np.asarray(mask)
    if m.dtype != bool or x.shape != m.shape or x.ndim < 1:
        raise ValueError('Values and Boolean masks must have identical shapes.')
    w = np.broadcast_to(np.asarray(weights, dtype=float), x.shape)
    if not np.isfinite(w).all() or (w < 0).any() or (m & ~np.isfinite(x)).any():
        raise ValueError('Invalid weights or nonfinite observed value.')
    w = np.where(m, w, 0.)
    count = w.sum(axis=0)
    if (count <= 0).any():
        raise ValueError('No training support for at least one statistic.')
    x = np.where(m, x, 0.)
    mean = (w*x).sum(axis=0)/count
    std = np.sqrt((w*(x-mean)**2).sum(axis=0)/count)
    if not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError('Degenerate variance; do not invent a scale silently.')
    return mean, std
