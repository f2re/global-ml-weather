"""Immutable train-only IGRA normalization on standard pressure bins.

Bin admission uses the nearest log-pressure output level. Interpolation needs
both adjacent supported bins; unsupported levels stay missing, never filled.
Humidity is normalized in log1p(q/q_scale) and decoded with its exact inverse.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import stat

import numpy as np

from .observation_training import digest, save
from .vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS

VARIABLES = PROFILE_VARIABLES[:5]
PERIOD = ['2021-01-01T00:00:00+00:00', '2022-01-01T00:00:00+00:00']
SCHEMA = 'igra-pressure-normalization-1'
PRESSURE_PA = np.asarray(PRESSURE_HPA, dtype=np.float64) * 100.


def _index(variable):
    if isinstance(variable, str):
        if variable not in VARIABLES:
            raise ValueError('Unknown pressure normalization variable.')
        return VARIABLES.index(variable)
    if type(variable) is not int or not 0 <= variable < len(VARIABLES):
        raise ValueError('Unknown pressure normalization variable.')
    return variable


class PressureNormalization:
    """Physical normalization with explicit unavailable (NaN) statistics."""

    def __init__(self, payload: dict) -> None:
        if (payload.get('schema') != SCHEMA or payload.get('period') != PERIOD
                or payload.get('split') != 'train'
                or payload.get('variables') != list(VARIABLES)
                or payload.get('units') != list(PROFILE_UNITS[:5])
                or payload.get('pressure_pa') != PRESSURE_PA.tolist()
                or payload.get('transforms') != ['identity', 'log1p(q/q_scale)', 'identity', 'identity', 'identity']):
            raise ValueError('Pressure normalization schema differs.')
        self.q_scale = float(payload['q_scale'])
        if not np.isfinite(self.q_scale) or self.q_scale <= 0:
            raise ValueError('Humidity scale must be positive empirical train std.')
        self.mean = np.asarray(payload['mean'], dtype=np.float64)
        self.std = np.asarray(payload['std'], dtype=np.float64)
        raw_support = np.asarray(payload['support'])
        raw_count = np.asarray(payload['count'])
        if (self.mean.shape != (37, 5) or self.std.shape != (37, 5)
                or raw_support.shape != (37, 5) or raw_support.dtype != np.bool_
                or raw_count.shape != (37, 5) or raw_count.dtype.kind not in 'iu'
                or (raw_count < 0).any()):
            raise ValueError('Invalid normalization array shape or admission mask.')
        self.support = raw_support.copy()
        self.count = raw_count.copy()
        if (not np.isfinite(self.mean[self.support]).all()
                or not np.isfinite(self.std[self.support]).all()
                or (self.std[self.support] <= 0).any()
                or (self.count[self.support] < 2).any()
                or not np.isnan(self.mean[~self.support]).all()
                or not np.isnan(self.std[~self.support]).all()):
            raise ValueError('Unavailable norms must remain missing; supported norms need train variance.')
        self.pressure_pa = PRESSURE_PA.copy()
        self.payload = payload

    def arrays(self) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Return buffers ordered (37 pressures, 5 variables), preserving NaN."""
        return self.mean.copy(), self.std.copy(), self.support.copy()

    def at(self, variable: str|int, pressure_pa: float|np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        """Interpolate mean/std in log(p); never extrapolate or bridge a gap."""
        variable = _index(variable)
        pressure = np.asarray(pressure_pa, dtype=np.float64)
        if not np.isfinite(pressure).all() or (pressure <= 0).any():
            raise ValueError('Pressure must be finite and positive in Pa.')
        flat = pressure.reshape(-1)
        means = np.full(flat.shape, np.nan)
        stds = np.full(flat.shape, np.nan)
        support = np.zeros(flat.shape, dtype=bool)
        xp = np.log(self.pressure_pa[::-1])
        for i, value in enumerate(flat):
            exact = np.flatnonzero(self.pressure_pa == value)
            if len(exact):
                j = int(exact[0])
                if self.support[j, variable]:
                    means[i], stds[i], support[i] = self.mean[j, variable], self.std[j, variable], True
                continue
            if value < self.pressure_pa[-1] or value > self.pressure_pa[0]:
                continue
            upper = int(np.searchsorted(xp, np.log(value), side='right'))
            lower = upper - 1
            j, k = 36 - lower, 36 - upper
            if self.support[j, variable] and self.support[k, variable]:
                weight = (np.log(value) - xp[lower]) / (xp[upper] - xp[lower])
                means[i] = (1 - weight) * self.mean[j, variable] + weight * self.mean[k, variable]
                stds[i] = (1 - weight) * self.std[j, variable] + weight * self.std[k, variable]
                support[i] = True
        return means.reshape(pressure.shape), stds.reshape(pressure.shape), support.reshape(pressure.shape)

    def normalize(self, variable: str|int, value: float|np.ndarray, pressure_pa: float|np.ndarray) -> np.ndarray:
        variable = _index(variable)
        value = np.asarray(value, dtype=np.float64)
        if not np.isfinite(value).all() or (variable == 1 and (value < 0).any()):
            raise ValueError('Measurements must be finite; specific humidity cannot be negative.')
        transformed = np.log1p(value / self.q_scale) if variable == 1 else value
        mean, std, support = self.at(variable, pressure_pa)
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(support, (transformed - mean) / std, np.nan)

    def inverse(self, variable: str|int, value: float|np.ndarray, pressure_pa: float|np.ndarray) -> np.ndarray:
        """Exact inverse. Negative predicted log-humidity is not silently clipped."""
        variable = _index(variable)
        mean, std, support = self.at(variable, pressure_pa)
        transformed = np.asarray(value, dtype=np.float64) * std + mean
        with np.errstate(over='ignore', invalid='ignore'):
            physical = self.q_scale * np.expm1(transformed) if variable == 1 else transformed
        return np.where(support, physical, np.nan)


def _identity(dataset):
    dataset.verify()
    return {'database_sha256': digest(dataset.root / 'observations.sqlite'),
            'dataset_manifest_sha256': digest(dataset.root / 'dataset.json'),
            'admission_sha256': digest(dataset.manifest['admission_manifest']),
            'source_sha256': digest(dataset.manifest['source'])}


@contextmanager
def _lock(destination):
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ValueError('Normalization symlinks are forbidden.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(destination) + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError('Unsafe normalization lock.')
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def fit(dataset_path: str|Path, output_file: str|Path) -> dict:
    """Fit once from immutable admitted train scalars, streaming read-only SQL."""
    from .profile_training import ProfileDataset

    destination = Path(output_file)
    with _lock(destination):
        dataset = ProfileDataset(dataset_path)
        identity = _identity(dataset)
        q_scale = float(dataset.manifest['statistics'][1]['std'])
        if not np.isfinite(q_scale) or q_scale <= 0:
            raise ValueError('Invalid existing empirical train humidity scale.')
        if destination.exists():
            old = json.loads(destination.read_text())
            PressureNormalization(old)
            if old.get('source_identity') != identity or old['q_scale'] != q_scale:
                raise ValueError('Immutable normalization source identity changed.')
            if _identity(dataset) != identity:
                raise ValueError('Normalization sources changed during verification.')
            return old
        count = np.zeros((37, 5), dtype=np.int64)
        excluded_pressure_count = np.zeros(5, dtype=np.int64)
        mean = np.zeros((37, 5), dtype=np.float64)
        m2 = np.zeros((37, 5), dtype=np.float64)
        uri = (dataset.root / 'observations.sqlite').resolve().as_uri() + '?mode=ro'
        with sqlite3.connect(uri, uri=True) as database:
            rows = database.execute("SELECT variable,value,record FROM records WHERE split='train' AND observed>=? AND observed<? ORDER BY id", PERIOD)
            for variable, value, text in rows:
                if type(variable) is not int or not 0 <= variable < 5:
                    raise ValueError('Unexpected train scalar variable.')
                record = json.loads(text)
                pressure = float(record['pressure_pa'])
                if (record['variable'] != VARIABLES[variable] or record['value'] != value
                        or not np.isfinite(value) or not np.isfinite(pressure) or pressure <= 0
                        or (variable == 1 and value < 0)):
                    raise ValueError('Invalid admitted normalization measurement.')
                if pressure < PRESSURE_PA[-1] or pressure > PRESSURE_PA[0]:
                    excluded_pressure_count[variable] += 1
                    continue
                level = int(np.abs(np.log(PRESSURE_PA) - np.log(pressure)).argmin())
                transformed = np.log1p(value / q_scale) if variable == 1 else value
                count[level, variable] += 1
                delta = transformed - mean[level, variable]
                mean[level, variable] += delta / count[level, variable]
                m2[level, variable] += delta * (transformed - mean[level, variable])
        support = (count >= 2) & (m2 > 0)
        std = np.full((37, 5), np.nan)
        std[support] = np.sqrt(m2[support] / count[support])
        mean[~support] = np.nan
        nullable = lambda array: [[float(x) if np.isfinite(x) else None for x in row] for row in array]
        payload = {'schema': SCHEMA, 'split': 'train', 'period': PERIOD,
                   'variables': list(VARIABLES), 'units': list(PROFILE_UNITS[:5]),
                   'pressure_pa': PRESSURE_PA.tolist(), 'q_scale': q_scale,
                   'q_scale_origin': 'existing pooled train specific-humidity standard deviation',
                   'transforms': ['identity', 'log1p(q/q_scale)', 'identity', 'identity', 'identity'],
                   'binning': 'nearest standard pressure in log(Pa); each admitted scalar once; outside [100,100000] Pa excluded',
                   'excluded_pressure_count': excluded_pressure_count.tolist(),
                   'interpolation': 'linear log-pressure, two adjacent supported bins or exact supported level; no extrapolation',
                   'variance': 'population Welford m2/count', 'source_identity': identity,
                   'count': count.tolist(), 'mean': nullable(mean), 'std': nullable(std),
                   'support': support.tolist(), 'scientific_acceptance': False}
        PressureNormalization(payload)
        if _identity(dataset) != identity:
            raise ValueError('Normalization sources changed during fitting.')
        if destination.with_suffix('.tmp').exists() or destination.with_suffix('.tmp').is_symlink():
            raise ValueError('Ambiguous unfinished normalization; preserve for inspection.')
        save(destination, payload)
        return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = fit(args.dataset, args.output)
    print(json.dumps({'schema': result['schema'], 'supported_bins': int(np.asarray(result['support']).sum()),
                      'output': args.output, 'sha256': digest(args.output)}, allow_nan=False))


if __name__ == '__main__':
    main()
