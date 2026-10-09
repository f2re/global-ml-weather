"""Immutable NOAA NCEP1 monthly means as optional seasonal context.

Offline preparation only. These external climatological means are neither
measured inputs/targets nor a replacement for canonical GraphCast scales.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import zipfile

import numpy as np

from .grid import build_grid, latlon
from .observation_training import digest, save, sync_directory
from .observations import utc
from .profile_normalization import _lock
from .vertical import PRESSURE_HPA, PROFILE_UNITS, PROFILE_VARIABLES

SCHEMA = 'ncep1-seasonal-context-1'
PERIOD = '1991/01/01 - 2020/12/31'
BASE_URL = 'https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/Monthlies/pressure/'
NAMES = ('air', 'shum', 'uwnd', 'vwnd', 'hgt')
FILES = tuple(name + '.mon.ltm.1991-2020.nc' for name in NAMES)
UNITS = ('degC', 'grams/kg', 'm/s', 'm/s', 'm')
PRESSURE_PA = np.asarray(PRESSURE_HPA, dtype=np.float64) * 100.
NATIVE_LEVELS = (1000., 925., 850., 700., 600., 500., 400., 300., 250.,
                 200., 150., 100., 70., 50., 30., 20., 10.)
GRAVITY = 9.80665


def _safe(path: Path) -> None:
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise ValueError('Seasonal source/artifact symlinks are forbidden.')


def _enu(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lat, lon = np.deg2rad(latitude), np.deg2rad(longitude)
    east = np.stack((-np.sin(lon), np.cos(lon), np.zeros_like(lon)), axis=-1)
    north = np.stack((-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)), axis=-1)
    return east, north


def _horizontal(values: np.ndarray, support: np.ndarray, latitudes: np.ndarray,
                longitudes: np.ndarray, query_lat: np.ndarray,
                query_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Strict four-corner periodic bilinear interpolation, leading dimensions preserved."""
    if (values.shape != support.shape or values.shape[-2:] != (len(latitudes), len(longitudes))
            or not np.isfinite(query_lat).all() or not np.isfinite(query_lon).all()
            or (np.abs(query_lat) > 90).any()):
        raise ValueError('Invalid seasonal interpolation geometry.')
    right = np.searchsorted(latitudes, query_lat, side='right').clip(1, len(latitudes)-1)
    left = right - 1
    lat_weight = (query_lat-latitudes[left])/(latitudes[right]-latitudes[left])
    query_lon = (query_lon-longitudes[0]) % 360. + longitudes[0]
    lon_right = np.searchsorted(longitudes, query_lon, side='right')
    lon_left = (lon_right-1) % len(longitudes)
    lon_right %= len(longitudes)
    low = longitudes[lon_left]
    high = np.where(lon_right == 0, longitudes[0]+360., longitudes[lon_right])
    lon_weight = (query_lon-low)/(high-low)
    corners = ((left, lon_left), (left, lon_right), (right, lon_left), (right, lon_right))
    weights = ((1-lat_weight)*(1-lon_weight), (1-lat_weight)*lon_weight,
               lat_weight*(1-lon_weight), lat_weight*lon_weight)
    valid = np.logical_and.reduce([support[..., y, x] for y, x in corners])
    total = np.zeros((*values.shape[:-2], len(query_lat)), dtype=np.float64)
    for (y, x), weight in zip(corners, weights):
        total += np.where(support[..., y, x], values[..., y, x], 0.) * weight
    return np.where(valid, total, np.nan), valid


def _vertical(values: np.ndarray, support: np.ndarray,
              native_pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Native adjacent log-pressure brackets only; no extrapolation or skipped gaps."""
    # Input [month,native_level,cell], output [month,cell,37].
    output = np.full((12, values.shape[-1], 37), np.nan)
    valid = np.zeros(output.shape, dtype=bool)
    axis = np.log(native_pressure[::-1])
    for index, pressure in enumerate(PRESSURE_PA):
        exact = np.flatnonzero(native_pressure == pressure)
        if len(exact):
            j = int(exact[0]); output[..., index] = values[:, j]; valid[..., index] = support[:, j]
        elif native_pressure[-1] <= pressure <= native_pressure[0]:
            upper = int(np.searchsorted(axis, np.log(pressure), side='right'))
            lower = upper-1; j = len(axis)-1-lower; k = len(axis)-1-upper
            weight = (np.log(pressure)-axis[lower])/(axis[upper]-axis[lower])
            admitted = support[:, j] & support[:, k]
            valid[..., index] = admitted
            output[..., index] = np.where(admitted, (1-weight)*values[:, j]+weight*values[:, k], np.nan)
    return output, valid


def _read(path: Path, name: str, units: str, minimum_years: int) -> dict:
    import xarray as xr
    from netCDF4 import num2date

    _safe(path)
    if not path.is_file() or path.stat().st_size > 20_000_000:
        raise ValueError('Missing or oversized approved seasonal source.')
    with xr.open_dataset(path, decode_times=False) as dataset:
        if name not in dataset or 'valid_yr_count' not in dataset:
            raise ValueError('Seasonal source needs a mean and valid-year counts.')
        value = dataset[name]; count = dataset['valid_yr_count']
        if (value.dims != ('time', 'level', 'lat', 'lon') or count.dims != value.dims
                or count.shape != value.shape or value.shape[0] != 12
                or value.shape[1]!=(8 if name=='shum' else 17)
                or not 3<=value.shape[2]<=73 or not 4<=value.shape[3]<=144):
            raise ValueError('Expected twelve monthly level/latitude/longitude climatological means.')
        if (value.attrs.get('units') != units or value.attrs.get('statistic') != 'Long Term Mean'
                or value.attrs.get('parent_stat') != 'Mean' or dataset.time.attrs.get('climo_period') != PERIOD):
            raise ValueError('Seasonal units, mean statistic or 1991–2020 fitting period differs.')
        if (dataset.level.attrs.get('units') != 'millibar'
                or dataset.level.attrs.get('positive') != 'down'
                or dataset.lat.attrs.get('units') != 'degrees_north'
                or dataset.lon.attrs.get('units') != 'degrees_east'):
            raise ValueError('Seasonal coordinate units differ.')
        dates = num2date(dataset.time.values, dataset.time.attrs['units'],
                         calendar=dataset.time.attrs.get('calendar', 'standard'),
                         only_use_cftime_datetimes=True)
        if [date.month for date in dates] != list(range(1, 13)):
            raise ValueError('Seasonal months must be ordered January through December.')
        levels = np.asarray(dataset.level.values, dtype=np.float64)
        expected = NATIVE_LEVELS[:8] if name == 'shum' else NATIVE_LEVELS
        if not np.array_equal(levels, expected):
            raise ValueError('Seasonal native pressure levels differ from approved NCEP1 product.')
        latitude = np.asarray(dataset.lat.values, dtype=np.float64)
        longitude = np.asarray(dataset.lon.values, dtype=np.float64)
        if (len(latitude) < 3 or len(longitude) < 4 or not np.isfinite(latitude).all()
                or not np.isfinite(longitude).all() or not np.all(np.diff(latitude) < 0)
                or latitude[0] != 90. or latitude[-1] != -90.
                or longitude[0] != 0. or not np.all(np.diff(longitude) > 0)
                or not np.allclose(np.diff(longitude), 360./len(longitude), atol=1e-8, rtol=0)
                or not np.isclose(longitude[-1], 360.-360./len(longitude))):
            raise ValueError('Expected global north-to-south latitude and periodic regular longitude axes.')
        years = np.asarray(count.values, dtype=np.float64)
        finite = np.isfinite(years)
        if ((years[finite] < 0).any() or (years[finite] > 30).any()
                or (years[finite] != np.floor(years[finite])).any()):
            raise ValueError('Valid-year counts must be integers between zero and thirty.')
        physical = np.asarray(value.values, dtype=np.float64)
        support = finite & (years >= minimum_years) & np.isfinite(physical)
        if name == 'air': physical += 273.15
        elif name == 'shum':
            physical *= .001
            support &= physical >= 0.
        elif name == 'hgt': physical *= GRAVITY
        physical = np.where(support, physical, np.nan)
        return {'values': physical[..., ::-1, :], 'support': support[..., ::-1, :],
                'latitude': latitude[::-1], 'longitude': longitude, 'pressure': levels*100.,
                'metadata': {'variable': name, 'native_units': units, 'statistic': 'Long Term Mean',
                             'parent_stat': 'Mean', 'climo_period': PERIOD,
                             'months': list(range(1, 13)), 'native_pressure_pa': (levels*100.).tolist(),
                             'valid_years_minimum': minimum_years, 'valid_years_maximum': 30,
                             'conversion': {'air': '+273.15', 'shum': '*0.001', 'hgt': '*9.80665'}.get(name, 'identity')}}


def prepare(source_dir: str | Path, output_dir: str | Path, mesh_level: int = 1,
            minimum_years: int = 25) -> dict:
    """Prepare fixed monthly spatial context offline; never fit or read target data."""
    if type(minimum_years) is not int or not 25 <= minimum_years <= 30:
        raise ValueError('Minimum valid years must be an integer in [25,30].')
    if type(mesh_level) is not int or not 0 <= mesh_level <= 4:
        raise ValueError('Seasonal preparation mesh level must be in [0,4].')
    source_dir = Path(source_dir); output = Path(output_dir)
    _safe(source_dir); _safe(output)
    source_dir = source_dir.resolve()
    sources = {filename: source_dir/filename for filename in FILES}
    for path in sources.values():
        _safe(path)
        if not path.is_file() or path.stat().st_size>20_000_000:
            raise ValueError('Missing or oversized approved seasonal source.')
    hashes = {name: digest(path) for name, path in sources.items()}
    grid = build_grid(mesh_level)
    with _lock(output):
        if output.exists():
            existing = SeasonalClimatology(output, grid)
            if (existing.payload['source_sha256'] != hashes
                    or existing.payload['minimum_valid_years'] != minimum_years
                    or existing.payload['source_directory'] != str(source_dir)):
                raise ValueError('Immutable seasonal context source or configuration changed.')
            return existing.payload
        stage = output.with_name('.'+output.name+'.incomplete')
        if stage.exists() or stage.is_symlink():
            raise ValueError('Ambiguous unfinished seasonal context; preserve for inspection.')
        data = [_read(sources[filename], name, units, minimum_years)
                for filename, name, units in zip(FILES, NAMES, UNITS)]
        lat, lon = latlon(grid.xyz).T
        means = np.full((12, grid.n_cells, 37, 5), np.nan)
        masks = np.zeros(means.shape, dtype=bool)
        for index in (0, 1, 4):
            item = data[index]
            horizontal, supported = _horizontal(item['values'], item['support'], item['latitude'], item['longitude'], lat, lon)
            means[..., index], masks[..., index] = _vertical(horizontal, supported, item['pressure'])
        u, v = data[2], data[3]
        if any(not np.array_equal(u[key], v[key]) for key in ('latitude', 'longitude', 'pressure')):
            raise ValueError('Seasonal vector components must share coordinates.')
        native_lat, native_lon = np.meshgrid(u['latitude'], u['longitude'], indexing='ij')
        east, north = _enu(native_lat, native_lon)
        paired = u['support'] & v['support']
        cartesian = u['values'][..., None]*east+v['values'][..., None]*north
        components = [_horizontal(cartesian[..., component], paired, u['latitude'], u['longitude'], lat, lon)
                      for component in range(3)]
        vector = np.stack([component[0] for component in components], axis=-1)
        east_grid, north_grid = _enu(lat, lon)
        wind_support = components[0][1]
        for index, basis in ((2, east_grid), (3, north_grid)):
            projected = (vector*basis).sum(axis=-1)
            means[..., index], masks[..., index] = _vertical(projected, wind_support, u['pressure'])
        if {name: digest(path) for name, path in sources.items()} != hashes:
            raise ValueError('Seasonal source changed during preparation.')
        stage.mkdir()
        with (stage/'seasonal-climatology.npz').open('xb') as file:
            np.savez_compressed(file, mean=means, support=masks, pressure_pa=PRESSURE_PA,
                                xyz=grid.xyz, months=np.arange(1, 13))
            file.flush()
            import os
            os.fsync(file.fileno())
        manifest = {'schema': SCHEMA, 'role': 'fixed_external_seasonal_context_not_observation',
                    'period': {'start': '1991-01-01', 'end': '2020-12-31'},
                    'source_directory': str(source_dir), 'source_sha256': hashes,
                    'source_urls': {filename: BASE_URL+filename for filename in FILES},
                    'source_metadata': {name: item['metadata'] for name, item in zip(NAMES, data)},
                    'mesh_level': mesh_level, 'grid_fingerprint': grid.fingerprint,
                    'minimum_valid_years': minimum_years, 'variables': list(PROFILE_VARIABLES[:5]),
                    'units': list(PROFILE_UNITS[:5]), 'pressure_pa': PRESSURE_PA.tolist(),
                    'interpolation': 'strict four-corner periodic bilinear; paired wind Cartesian projection; adjacent native log-pressure; cyclic actual-calendar month centres',
                    'artifact_sha256': digest(stage/'seasonal-climatology.npz'),
                    'canonical_normalization': 'fixed GraphCast mu/sigma remain unchanged',
                    'scientific_acceptance': False}
        save(stage/'manifest.json', manifest); sync_directory(stage)
        if {name: digest(path) for name, path in sources.items()} != hashes:
            raise ValueError('Seasonal source changed before publication.')
        stage.rename(output); sync_directory(output.parent)
        return manifest


class SeasonalClimatology:
    """Frozen monthly physical means with explicit support and verified sources."""

    def __init__(self, root: str | Path, grid=None) -> None:
        self.root = Path(root); _safe(self.root)
        _safe(self.root/'manifest.json')
        if not (self.root/'manifest.json').is_file() or (self.root/'manifest.json').stat().st_size>256_000:
            raise ValueError('Missing or oversized seasonal manifest.')
        self.manifest_sha256 = digest(self.root/'manifest.json')
        self.fingerprint = self.manifest_sha256
        self._payload = json.loads((self.root/'manifest.json').read_text())
        payload = self._payload
        if (payload.get('schema') != SCHEMA or payload.get('period') != {'start': '1991-01-01', 'end': '2020-12-31'}
                or payload.get('variables') != list(PROFILE_VARIABLES[:5])
                or payload.get('units') != list(PROFILE_UNITS[:5]) or payload.get('pressure_pa') != PRESSURE_PA.tolist()
                or payload.get('canonical_normalization')!='fixed GraphCast mu/sigma remain unchanged'
                or payload.get('role') != 'fixed_external_seasonal_context_not_observation'
                or payload.get('source_urls') != {filename:BASE_URL+filename for filename in FILES}
                or set(payload.get('source_sha256', {})) != set(FILES)
                or type(payload.get('mesh_level')) is not int or not 0<=payload['mesh_level']<=4
                or type(payload.get('minimum_valid_years')) is not int
                or not 25 <= payload['minimum_valid_years'] <= 30):
            raise ValueError('Seasonal artifact schema or fixed provenance differs.')
        for value in payload['source_sha256'].values():
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Seasonal original source SHA256 is required.')
        for name,units in zip(NAMES,UNITS):
            metadata=payload.get('source_metadata',{}).get(name,{})
            if (metadata.get('variable')!=name or metadata.get('native_units')!=units
                    or metadata.get('statistic')!='Long Term Mean' or metadata.get('parent_stat')!='Mean'
                    or metadata.get('climo_period')!=PERIOD
                    or metadata.get('valid_years_minimum')!=payload['minimum_valid_years']
                    or metadata.get('valid_years_maximum')!=30
                    or metadata.get('months')!=list(range(1,13))
                    or metadata.get('native_pressure_pa')!=[level*100. for level in (NATIVE_LEVELS[:8] if name=='shum' else NATIVE_LEVELS)]
                    or metadata.get('conversion')!={'air':'+273.15','shum':'*0.001','hgt':'*9.80665'}.get(name,'identity')):
                raise ValueError('Seasonal source metadata differs.')
        grid = build_grid(payload['mesh_level']) if grid is None else grid
        if grid.fingerprint != payload.get('grid_fingerprint') or grid.level != payload['mesh_level']:
            raise ValueError('Seasonal grid differs from the model grid.')
        self.verify_sources()
        with zipfile.ZipFile(self.root/'seasonal-climatology.npz') as archive:
            entries=archive.infolist()
            expected_names={'mean.npy','support.npy','pressure_pa.npy','xyz.npy','months.npy'}
            if (len(entries)!=5 or {entry.filename for entry in entries}!=expected_names
                    or sum(entry.file_size for entry in entries)>80_000_000):
                raise ValueError('Seasonal archive has unexpected entries or oversized expanded arrays.')
        with np.load(self.root/'seasonal-climatology.npz', allow_pickle=False) as data:
            self.mean = data['mean'].copy(); self.support = data['support'].copy()
            if (self.mean.shape != (12, grid.n_cells, 37, 5) or self.support.shape != self.mean.shape
                    or self.support.dtype != np.bool_ or not np.isfinite(self.mean[self.support]).all()
                    or not np.isnan(self.mean[~self.support]).all()
                    or not np.array_equal(data['pressure_pa'], PRESSURE_PA)
                    or not np.array_equal(data['months'], np.arange(1, 13))
                    or not np.array_equal(data['xyz'], grid.xyz)):
                raise ValueError('Seasonal arrays, support, pressure or geometry differ.')
        if self.support[:,:,PRESSURE_PA<30000.,1].any() or self.support[:,:,PRESSURE_PA<1000.][:,:,:,[0,2,3,4]].any():
            raise ValueError('Seasonal support exceeds native pressure coverage.')
        self.mean.setflags(write=False);self.support.setflags(write=False)
        self.verify_sources()

    @property
    def payload(self) -> dict:
        return json.loads(json.dumps(self._payload))

    def verify_sources(self) -> None:
        if digest(self.root/'manifest.json') != self.manifest_sha256:
            raise ValueError('Seasonal manifest changed.')
        _safe(self.root/'manifest.json')
        path = self.root/'seasonal-climatology.npz'; _safe(path)
        if not path.is_file() or path.stat().st_size>100_000_000:
            raise ValueError('Missing or oversized seasonal artifact.')
        if digest(path) != self._payload['artifact_sha256']:
            raise ValueError('Seasonal prepared artifact changed.')
        source = Path(self._payload['source_directory']); _safe(source)
        for filename in FILES:
            path = source/filename; _safe(path)
            if digest(path) != self._payload['source_sha256'][filename]:
                raise ValueError('Seasonal original source changed: '+filename)

    @staticmethod
    def _centre(year: int, month: int) -> datetime:
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = datetime(year+1, 1, 1, tzinfo=timezone.utc) if month == 12 else datetime(year, month+1, 1, tzinfo=timezone.utc)
        return start+(end-start)/2

    def sample(self, valid_time: str | datetime) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate adjacent monthly means about calendar centres, including leap years."""
        when = utc(valid_time); middle = self._centre(when.year, when.month)
        if when == middle:
            return self.mean[when.month-1].copy(), self.support[when.month-1].copy()
        if when < middle:
            right = middle
            left = self._centre(when.year-1, 12) if when.month == 1 else self._centre(when.year, when.month-1)
        else:
            left = middle
            right = self._centre(when.year+1, 1) if when.month == 12 else self._centre(when.year, when.month+1)
        weight = (when-left).total_seconds()/(right-left).total_seconds()
        a, b = left.month-1, right.month-1
        supported = self.support[a] & self.support[b]
        mean = (1-weight)*self.mean[a]+weight*self.mean[b]
        return np.where(supported, mean, np.nan), supported.copy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--mesh-level', type=int, default=1); parser.add_argument('--minimum-years', type=int, default=25)
    args = parser.parse_args(argv)
    result = prepare(args.source_dir, args.output, args.mesh_level, args.minimum_years)
    print(json.dumps({'schema': result['schema'], 'output': args.output,
                      'manifest_sha256': digest(Path(args.output)/'manifest.json')}, allow_nan=False))


if __name__ == '__main__':
    main()
