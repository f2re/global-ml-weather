"""Observation sources for R11: GHCNh, IGRA and prepared stores for every other platform.

Each source returns TokenBatch windows (begin, end] in observation time, with
an optional availability cut for causal inputs, restricted to one frozen
chronological split. No source interpolates, fills or rounds measurements.

* GhcnhSource reads the admitted hourly station archive (trailing-hour means,
  the archive's own operator) and the station elevations recorded by GHCNh.
  A station without a recorded elevation keeps its winds and MSLP but its
  temperature, dew point and station pressure operators are refused.
* IgraSource reads admitted radiosonde scalars with actual pressure/time and
  caps windows by sampling WHOLE launches, never by thinning one profile.
* TokenStore holds any other platform (aircraft, ships, buoys, AMV,
  scatterometer, GNSS-RO, satellite radiances/products) after explicit
  admission. Satellite channels require a radiometric passport and receive
  train-only statistics; brightness temperatures never use air-temperature norms.
* SyntheticSource generates an analytic atmosphere for program checks only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
import tempfile

import numpy as np

from ..observation_training import digest, save
from ..observations import utc
from .tokens import (HEIGHT_REDUCTION_M, PLATFORMS, PLATFORM_VARIABLES, SURFACE_VARIABLES, UNITS, VARIABLES,
                     Normalizer, TokenBatch, stable_hash)

EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
GHCNH_VARIABLES = ('t2m', 'td2m', 'u10', 'v10', 'surface_pressure', 'mslp')
STORE_SCHEMA = 'assimilation-token-store-1'


def seconds(when) -> float:
    return (utc(when) - EPOCH).total_seconds()


class GhcnhSource:
    kind = 'ghcnh_surface'
    data_kind = 'real'

    def __init__(self, root):
        from ..observation_data import ObservationDataset, START, TRAIN_END
        self.dataset = ObservationDataset(root)
        self.root = self.dataset.root
        self.start_seconds = seconds(START)
        self.train_hours = int((TRAIN_END - START).total_seconds() / 3600)
        stations = self.dataset.manifest['stations']
        self.station_ids = [s['id'] for s in stations]
        self.groups = np.array([stable_hash('GHCNh/' + s) for s in self.station_ids], dtype=np.int64)
        self.elevation = self._elevations()
        self.fingerprint = {'kind': self.kind, 'dataset_sha256': digest(self.root / 'dataset.json'),
                            'records_sha256': self.dataset.manifest['records_sha256'],
                            'hourly_sha256': self.dataset.manifest['hourly_sha256'],
                            'stations_with_elevation': int(np.isfinite(self.elevation).sum())}

    def _elevations(self) -> np.ndarray:
        """First recorded GHCNh elevation per admitted station; NaN if never reported or inconsistent."""
        uri = (self.root / 'observations.sqlite').resolve().as_uri() + '?mode=ro'
        out = np.full(len(self.station_ids), np.nan)
        with sqlite3.connect(uri, uri=True) as database:
            for index, station in enumerate(self.station_ids):
                rows = database.execute(
                    "SELECT DISTINCT json_extract(record, '$.elevation_m') FROM observations "
                    'WHERE id >= ? AND id < ? LIMIT 3', (f'GHCNh/{station}/', f'GHCNh/{station}0')).fetchall()
                values = {float(r[0]) for r in rows if r[0] is not None}
                if len(values) == 1:
                    value = next(iter(values))
                    if math.isfinite(value) and -500 < value < 9000:
                        out[index] = value
        return out

    def verify(self):
        from ..pipeline.io import sha256
        for file, key in (('hourly.npz', 'hourly_sha256'), ('observations.sqlite', 'records_sha256')):
            if sha256(self.root / file) != self.dataset.manifest[key]:
                raise ValueError('GHCNh archive changed during the experiment.')

    def statistics(self) -> dict:
        """Train-only mean/std per native variable; station pressure after height reduction."""
        values = self.dataset.values[:self.train_hours]
        mask = self.dataset.mask[:self.train_hours]
        out = {}
        for index, variable in enumerate(SURFACE_VARIABLES):
            data = values[..., index][mask[..., index]].astype(np.float64)
            transform = 'identity'
            if variable == 'station_pressure':
                known = mask[..., index] & np.isfinite(self.elevation)[None, :]
                data = np.log(values[..., index][known].astype(np.float64)) + np.broadcast_to(self.elevation, values.shape[:2])[known] / HEIGHT_REDUCTION_M
                transform = 'height_reduced_log'
            if len(data) < 2 or not data.std() > 0:
                raise ValueError('Insufficient train statistics for ' + variable)
            out[Normalizer.key('surface_station', variable)] = {
                'mean': float(data.mean()), 'std': float(data.std()), 'transform': transform,
                'count': int(len(data)), 'period': 'train'}
        return out

    def window(self, begin: float, end: float, *, split: str, available_before: float | None = None,
               rng=None, limit: int | None = None) -> TokenBatch:
        first = max(1, int(math.floor((begin - self.start_seconds) / 3600)) + 1)
        last = min(len(self.dataset.values) - 1, int(math.floor((end - self.start_seconds) / 3600)))
        if last < first:
            return TokenBatch.empty()
        hours = np.arange(first, last + 1)
        mask = self.dataset.mask[hours].copy()
        if available_before is not None:
            mask &= self.dataset.available[hours] <= available_before - self.start_seconds
        h, s, v = np.nonzero(mask)
        if limit is not None and len(h) > limit:
            keep = (rng or np.random.default_rng(0)).choice(len(h), limit, replace=False)
            keep.sort()
            h, s, v = h[keep], s[keep], v[keep]
        hour = hours[h]
        observed = self.start_seconds + hour * 3600.
        available = self.start_seconds + self.dataset.available[hour, s, v].astype(np.float64)
        identity = np.array([stable_hash(f'GHCNh/{self.station_ids[a]}/{b}/{c}') for a, b, c in zip(s, hour, v)],
                            dtype=np.int64)
        return TokenBatch(
            platform=np.full(len(h), PLATFORMS.index('surface_station'), np.int16),
            variable=np.array([VARIABLES.index(SURFACE_VARIABLES[c]) for c in v], np.int16),
            channel=np.full(len(h), -1, np.int32), group=self.groups[s], identity=identity,
            observed=observed, available=np.maximum(available, observed),
            latitude=self.dataset.latlon[s, 0].astype(np.float64), longitude=self.dataset.latlon[s, 1].astype(np.float64),
            pressure_pa=np.full(len(h), np.nan), height_m=self.elevation[s],
            value=self.dataset.values[hour, s, v].astype(np.float64), error=np.full(len(h), np.nan),
            quality=np.ones(len(h), np.float32))


class IgraSource:
    kind = 'igra_profiles'
    data_kind = 'real'
    PROFILE = ('temperature', 'specific_humidity', 'u', 'v', 'geopotential')

    def __init__(self, root):
        from ..profile_training import ProfileDataset
        self.dataset = ProfileDataset(root)
        self.root = self.dataset.root
        self.fingerprint = {'kind': self.kind, 'dataset_sha256': self.dataset.manifest_sha256,
                            'database_sha256': self.dataset.manifest['database_sha256']}

    def verify(self):
        self.dataset.verify()

    def statistics(self) -> dict:
        return {}  # pressure-coordinate variables use the pinned GraphCast norms

    def window(self, begin: float, end: float, *, split: str, available_before: float | None = None,
               rng=None, limit: int | None = None) -> TokenBatch:
        uri = (self.root / 'observations.sqlite').resolve().as_uri() + '?mode=ro'
        iso = lambda value: (EPOCH + timedelta(seconds=value)).isoformat()
        query = ("SELECT id, observed, available, variable, value, json_extract(record, '$.pressure_pa'), "
                 "json_extract(record, '$.latitude'), json_extract(record, '$.longitude'), "
                 "json_extract(record, '$.profile_id'), json_extract(record, '$.observation_error') "
                 'FROM records WHERE observed > ? AND observed <= ? AND split = ?')
        parameters = [iso(begin), iso(end), split]
        if available_before is not None:
            query += ' AND available <= ?'
            parameters.append(iso(available_before))
        with sqlite3.connect(uri, uri=True) as database:
            rows = database.execute(query + ' ORDER BY observed, id', parameters).fetchall()
        if not rows:
            return TokenBatch.empty()
        groups = np.array([stable_hash('IGRA/' + str(r[8])) for r in rows], dtype=np.int64)
        if limit is not None and len(rows) > limit:
            # Sample whole launches so a sounding is never partially thinned.
            unique, inverse = np.unique(groups, return_inverse=True)
            order = (rng or np.random.default_rng(0)).permutation(len(unique))
            sizes = np.bincount(inverse, minlength=len(unique))[order]
            chosen = order[:max(1, int(np.searchsorted(np.cumsum(sizes), limit, side='right')))]
            keep = np.isin(inverse, chosen)
            rows = [r for r, k in zip(rows, keep) if k]
            groups = groups[keep]
        name_index = [VARIABLES.index(self.PROFILE[int(r[3])]) for r in rows]
        error = np.array([np.nan if r[9] is None else float(r[9]) for r in rows])
        error[error <= 0] = np.nan
        return TokenBatch(
            platform=np.full(len(rows), PLATFORMS.index('radiosonde'), np.int16),
            variable=np.array(name_index, np.int16), channel=np.full(len(rows), -1, np.int32), group=groups,
            identity=np.array([stable_hash(str(r[0])) for r in rows], dtype=np.int64),
            observed=np.array([seconds(r[1]) for r in rows]), available=np.array([seconds(r[2]) for r in rows]),
            latitude=np.array([float(r[6]) for r in rows]), longitude=np.array([float(r[7]) for r in rows]),
            pressure_pa=np.array([float(r[5]) for r in rows]), height_m=np.full(len(rows), np.nan),
            value=np.array([float(r[4]) for r in rows]), error=error, quality=np.ones(len(rows), np.float32))


# ----- generic prepared stores -----------------------------------------------------------------

def _admit_record(record: dict, platform: str) -> tuple:
    variable = record.get('variable')
    if variable not in PLATFORM_VARIABLES[platform]:
        raise ValueError(f'{variable} is not admitted for {platform}.')
    if record.get('valid', True) is not True or record.get('data_kind', 'real') != 'real':
        raise ValueError('Only valid real measurements enter a store.')
    units = UNITS[variable]
    if units is not None and record.get('units') != units:
        raise ValueError(f'{variable} requires units {units}.')
    observed, available = seconds(record['observed_at']), seconds(record['available_at'])
    lat, lon, value = float(record['latitude']), float(record['longitude']), float(record['value'])
    if not (math.isfinite(lat) and math.isfinite(lon) and math.isfinite(value) and abs(lat) <= 90 and abs(lon) <= 180):
        raise ValueError('Invalid geometry or value.')
    if available < observed:
        raise ValueError('Observation available before measurement.')
    identity = record.get('observation_id')
    group = record.get('group_id') or record.get('profile_id') or record.get('platform_id') or record.get('granule_id')
    if not identity or not group:
        raise ValueError('Observation identity and group (platform, flight, launch or granule) are required.')
    pressure = record.get('pressure_pa')
    height = record.get('height_m', record.get('elevation_m'))
    channel = None
    if platform in ('satellite_radiance', 'satellite_product'):
        passport = record.get('radiometric_admission')
        if (not isinstance(passport, dict) or passport.get('calibrated') is not True
                or not isinstance(passport.get('passport_sha256'), str) or len(passport['passport_sha256']) != 64):
            raise ValueError('Satellite channel requires an admitted radiometric passport.')
        if not record.get('instrument') or record.get('channel') is None:
            raise ValueError('Satellite token requires instrument and channel.')
        channel = f"{record['instrument']}:{record['channel']}"
        if not (isinstance(record.get('observation_error'), (int, float)) and record['observation_error'] > 0):
            raise ValueError('Satellite channel requires a passport observation error.')
    if platform in ('ship', 'buoy') and variable in ('station_temperature', 'station_dew_point', 'station_pressure'):
        if height is None:
            raise ValueError('Marine thermodynamic reports require an explicit platform height.')
    error = record.get('observation_error')
    zenith = record.get('zenith_angle_deg')
    return (str(identity), observed, available, platform, variable, channel, str(group), lat, lon,
            None if pressure is None else float(pressure), None if height is None else float(height), value,
            None if error is None else float(error), float(record.get('quality', 1.0)),
            None if zenith is None else math.cos(math.radians(float(zenith))))


def prepare_store(source_jsonl, platform: str, output, *, train_end: str, split_bounds: dict) -> dict:
    """Admit a JSONL file into an immutable store; train statistics use observed < train_end only."""
    if platform not in PLATFORMS or platform in ('surface_station', 'radiosonde'):
        raise ValueError('Use the dedicated GHCNh/IGRA sources; store platforms: ' + platform)
    source = Path(source_jsonl)
    if source.is_symlink() or not source.is_file():
        raise ValueError('Store source must be a regular local file.')
    source_hash = digest(source)
    bounds = {name: (seconds(a), seconds(b)) for name, (a, b) in split_bounds.items()}
    cut = seconds(train_end)
    output = Path(output)
    if output.exists():
        raise FileExistsError('Store output exists; prepare a new store ID.')
    output.parent.mkdir(parents=True, exist_ok=True)
    statistics, channels, counts = {}, set(), {'admitted': 0, 'outside_splits': 0}
    with tempfile.TemporaryDirectory(prefix='.token-store-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'store'
        stage.mkdir()
        with sqlite3.connect(stage / 'tokens.sqlite') as database:
            database.execute('CREATE TABLE tokens(id TEXT PRIMARY KEY, observed REAL, available REAL, split TEXT, '
                             'platform TEXT, variable TEXT, channel TEXT, grp TEXT, latitude REAL, longitude REAL, '
                             'pressure REAL, height REAL, value REAL, error REAL, quality REAL, zenith_cos REAL)')
            sums = {}
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = _admit_record(json.loads(line), platform)
                    split = next((name for name, (a, b) in bounds.items() if a <= row[1] < b), None)
                    if split is None:
                        counts['outside_splits'] += 1
                        continue
                    database.execute('INSERT INTO tokens VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                     (row[0], row[1], row[2], split) + row[3:])
                    counts['admitted'] += 1
                    if row[5] is not None:
                        channels.add(row[5])
                    if row[1] < cut:
                        key = Normalizer.key(platform, row[4], row[5])
                        value = row[11]
                        if row[4] == 'station_pressure':
                            if row[10] is None:
                                continue
                            value = math.log(value) + row[10] / HEIGHT_REDUCTION_M
                        total = sums.setdefault(key, [0, 0.0, 0.0, 'height_reduced_log' if row[4] == 'station_pressure' else 'identity'])
                        total[0] += 1; total[1] += value; total[2] += value * value
            database.execute('CREATE INDEX tokens_observed ON tokens(observed)')
        for key, (count, first, second, transform) in sums.items():
            if count < 2:
                continue
            mean = first / count
            std = math.sqrt(max(second / count - mean * mean, 0.0))
            if std > 0:
                statistics[key] = {'mean': mean, 'std': std, 'transform': transform, 'count': count, 'period': 'train'}
        if source_hash != digest(source):
            raise ValueError('Store source changed during preparation.')
        manifest = {'schema': STORE_SCHEMA, 'platform': platform, 'data_kind': 'real',
                    'source': str(source.resolve()), 'source_sha256': source_hash,
                    'database_sha256': digest(stage / 'tokens.sqlite'), 'statistics': statistics,
                    'channels': sorted(channels), 'counts': counts, 'train_end': train_end,
                    'split_bounds': split_bounds, 'scientific_acceptance': False}
        save(stage / 'manifest.json', manifest)
        stage.rename(output)
    return manifest


class TokenStore:
    data_kind = 'real'

    def __init__(self, root):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError('Store symlinks are forbidden.')
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        if self.manifest.get('schema') != STORE_SCHEMA or self.manifest.get('data_kind') != 'real':
            raise ValueError('Unknown or non-real token store.')
        self.kind = 'store:' + self.manifest['platform']
        self.channels = list(self.manifest['channels'])
        self.fingerprint = {'kind': self.kind, 'manifest_sha256': digest(self.root / 'manifest.json'),
                            'database_sha256': self.manifest['database_sha256']}
        self.verify()

    def verify(self):
        if digest(self.root / 'tokens.sqlite') != self.manifest['database_sha256']:
            raise ValueError('Token store changed.')

    def statistics(self) -> dict:
        return dict(self.manifest['statistics'])

    def window(self, begin: float, end: float, *, split: str, available_before: float | None = None,
               rng=None, limit: int | None = None, channel_index: dict | None = None) -> TokenBatch:
        uri = (self.root / 'tokens.sqlite').resolve().as_uri() + '?mode=ro'
        query = ('SELECT id, observed, available, platform, variable, channel, grp, latitude, longitude, pressure, '
                 'height, value, error, quality, zenith_cos FROM tokens WHERE observed > ? AND observed <= ? AND split = ?')
        parameters = [begin, end, split]
        if available_before is not None:
            query += ' AND available <= ?'
            parameters.append(available_before)
        with sqlite3.connect(uri, uri=True) as database:
            rows = database.execute(query + ' ORDER BY observed, id', parameters).fetchall()
        if limit is not None and len(rows) > limit:
            keep = np.sort((rng or np.random.default_rng(0)).choice(len(rows), limit, replace=False))
            rows = [rows[i] for i in keep]
        if not rows:
            return TokenBatch.empty()
        channel_index = channel_index or {}
        number = lambda value: np.nan if value is None else float(value)
        return TokenBatch(
            platform=np.array([PLATFORMS.index(r[3]) for r in rows], np.int16),
            variable=np.array([VARIABLES.index(r[4]) for r in rows], np.int16),
            channel=np.array([channel_index.get(r[5], -1) if r[5] is not None else -1 for r in rows], np.int32),
            group=np.array([stable_hash(f'{r[3]}/{r[6]}') for r in rows], np.int64),
            identity=np.array([stable_hash(f'{r[3]}/{r[0]}') for r in rows], np.int64),
            observed=np.array([r[1] for r in rows], float), available=np.array([r[2] for r in rows], float),
            latitude=np.array([r[7] for r in rows], float), longitude=np.array([r[8] for r in rows], float),
            pressure_pa=np.array([number(r[9]) for r in rows]), height_m=np.array([number(r[10]) for r in rows]),
            value=np.array([r[11] for r in rows], float), error=np.array([number(r[12]) for r in rows]),
            quality=np.array([r[13] for r in rows], np.float32),
            zenith_cos=np.array([number(r[14]) for r in rows], np.float32))


class SyntheticSource:
    """Analytic, explicitly synthetic atmosphere for program checks; never a real experiment."""
    kind = 'synthetic_analytic'
    data_kind = 'synthetic'

    def __init__(self, seed: int = 0, stations: int = 60, soundings: int = 12, hourly: bool = True):
        rng = np.random.default_rng(seed)
        self.station_lat = np.degrees(np.arcsin(rng.uniform(-1, 1, stations)))
        self.station_lon = rng.uniform(-180, 180, stations)
        self.station_height = np.clip(rng.gamma(1.5, 300., stations), 0, 4000)
        self.sounding_lat = np.degrees(np.arcsin(rng.uniform(-0.9, 0.9, soundings)))
        self.sounding_lon = rng.uniform(-180, 180, soundings)
        self.levels = np.array([1000, 925, 850, 700, 500, 300, 250, 200, 100, 50, 20, 10], float) * 100
        self.hourly = hourly
        self.fingerprint = {'kind': self.kind, 'seed': seed, 'stations': stations, 'soundings': soundings}

    def verify(self):
        return None

    @staticmethod
    def temperature(lat, pressure, t_seconds):
        """Smooth analytic temperature with a travelling wave, K."""
        phase = 2 * np.pi * t_seconds / (86400 * 5)
        tropo = 300 - 45 * np.sin(np.radians(lat)) ** 2 - 60 * np.log(1e5 / np.clip(pressure, 2e4, 1e5)) / np.log(5)
        return np.where(pressure >= 2e4, tropo, 215.) + 2 * np.sin(np.radians(lat) * 3 + phase)

    @staticmethod
    def wind(lat, lon, pressure, t_seconds):
        jet = 30 * np.exp(-((np.abs(lat) - 40) / 15) ** 2) * np.clip(np.log(1e5 / pressure) / np.log(4), 0, 1.5)
        return jet + 3 * np.cos(np.radians(lon) + t_seconds / 86400), 2 * np.sin(np.radians(lon * 2))

    def statistics(self) -> dict:
        stats = {'station_temperature': (285., 15.), 'station_dew_point': (278., 12.),
                 'station_eastward_wind': (0., 5.), 'station_northward_wind': (0., 5.),
                 'station_pressure': (math.log(101000.), 0.02), 'station_mean_sea_level_pressure': (101000., 1200.)}
        return {Normalizer.key('surface_station', name): {
            'mean': value[0], 'std': value[1], 'count': 2, 'period': 'synthetic',
            'transform': 'height_reduced_log' if name == 'station_pressure' else 'identity'}
            for name, value in stats.items()}

    def window(self, begin: float, end: float, *, split: str, available_before: float | None = None,
               rng=None, limit: int | None = None) -> TokenBatch:
        step = 3600. if self.hourly else 10800.
        times = np.arange(math.floor(begin / step) * step + step, end + 1e-6, step)
        times = times[(times > begin) & (times <= end)]
        if available_before is not None:
            times = times[times + 600 <= available_before]
        batches = []
        for t in times:
            n = len(self.station_lat)
            t_surface = self.temperature(self.station_lat, 1e5 * np.exp(-self.station_height / 8000.), t)
            u, v = self.wind(self.station_lat, self.station_lon, np.full(n, 1e5), t)
            p_station = 101000. * np.exp(-self.station_height / (287.05 * t_surface / 9.80665))
            values = [t_surface, t_surface - 6, u * 0.4, v * 0.4, p_station, np.full(n, 101000.)]
            for k, data in enumerate(values):
                batches.append(TokenBatch(
                    platform=np.full(n, PLATFORMS.index('surface_station'), np.int16),
                    variable=np.full(n, VARIABLES.index(SURFACE_VARIABLES[k]), np.int16),
                    channel=np.full(n, -1, np.int32), group=np.arange(n, dtype=np.int64),
                    identity=np.array([stable_hash(f'syn/s/{i}/{t}/{k}') for i in range(n)], np.int64),
                    observed=np.full(n, t), available=np.full(n, t + 600.),
                    latitude=self.station_lat, longitude=self.station_lon, pressure_pa=np.full(n, np.nan),
                    height_m=self.station_height, value=np.asarray(data, float), error=np.full(n, np.nan),
                    quality=np.ones(n, np.float32)))
            if int(t) % 43200 == 0:
                lat = np.repeat(self.sounding_lat, len(self.levels))
                lon = np.repeat(self.sounding_lon, len(self.levels))
                p = np.tile(self.levels, len(self.sounding_lat))
                temperature = self.temperature(lat, p, t)
                u, v = self.wind(lat, lon, p, t)
                z = 287.05 * 250 * np.log(1e5 / p)
                q = np.clip(0.012 * np.exp(-np.log(1e5 / p) * 3) * np.cos(np.radians(lat)) ** 2, 2e-6, 0.02)
                group = 10_000 + np.repeat(np.arange(len(self.sounding_lat)), len(self.levels)) + int(t // 43200) * 1000
                for k, data in enumerate((temperature, q, u, v, z)):
                    m = len(p)
                    batches.append(TokenBatch(
                        platform=np.full(m, PLATFORMS.index('radiosonde'), np.int16),
                        variable=np.full(m, VARIABLES.index(IgraSource.PROFILE[k]), np.int16),
                        channel=np.full(m, -1, np.int32), group=group.astype(np.int64),
                        identity=np.array([stable_hash(f'syn/r/{i}/{t}/{k}') for i in range(m)], np.int64),
                        observed=np.full(m, t), available=np.full(m, t + 1800.), latitude=lat, longitude=lon,
                        pressure_pa=p, height_m=np.full(m, np.nan), value=np.asarray(data, float),
                        error=np.full(m, np.nan), quality=np.ones(m, np.float32)))
        batch = TokenBatch.concatenate(batches)
        if limit is not None and len(batch) > limit:
            batch = batch.select(np.sort((rng or np.random.default_rng(0)).choice(len(batch), limit, replace=False)))
        return batch


def main(argv=None):
    parser = argparse.ArgumentParser(description='Prepare an admitted token store for one platform.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--platform', required=True, choices=[p for p in PLATFORMS if p not in ('surface_station', 'radiosonde')])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    from ..observation_data import END, START, TRAIN_END, VAL_END
    bounds = {'train': (START.isoformat(), TRAIN_END.isoformat()), 'validation': (TRAIN_END.isoformat(), VAL_END.isoformat()),
              'test': (VAL_END.isoformat(), END.isoformat())}
    manifest = prepare_store(args.input, args.platform, args.output, train_end=TRAIN_END.isoformat(), split_bounds=bounds)
    print(json.dumps({'schema': manifest['schema'], 'platform': manifest['platform'], 'counts': manifest['counts']}))


if __name__ == '__main__':
    main()
