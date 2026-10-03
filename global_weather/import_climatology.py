"""Import GraphCast's published ERA5 mean/std without installing its neural model.

Download is an explicit setup action, not an inference dependency. Numerical
artifacts are not vendored. The source-code revision does NOT pin cloud bytes;
SHA256 of each actual local artifact is recorded in the resulting bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone
import numpy as np
from .normalization import NormalizationBundle
from .vertical import PRESSURE_HPA

SOURCE_REPO = 'https://github.com/google-deepmind/weathernext'
SOURCE_REVISION = 'f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3'
BASE_URL = 'https://storage.googleapis.com/dm_graphcast/graphcast/stats/'
# native name: (local name, canonical units, multiplier, accumulation interval)
VARIABLES = {
    'temperature': ('temperature', 'K', 1., None),
    'specific_humidity': ('specific_humidity', 'kg kg-1', 1., None),
    'u_component_of_wind': ('u', 'm s-1', 1., None),
    'v_component_of_wind': ('v', 'm s-1', 1., None),
    'geopotential': ('geopotential', 'm2 s-2', 1., None),
    'vertical_velocity': ('omega', 'Pa s-1', 1., None),
    '2m_temperature': ('t2m', 'K', 1., None),
    '10m_u_component_of_wind': ('u10', 'm s-1', 1., None),
    '10m_v_component_of_wind': ('v10', 'm s-1', 1., None),
    'mean_sea_level_pressure': ('mslp', 'Pa', 1., None),
    'total_precipitation_6hr': ('precipitation_step', 'kg m-2', 1000., 6),
}
NATIVE_UNITS = {
    'K': {'K'}, 'kg kg-1': {'kg kg-1', 'kg kg**-1', 'kg/kg', '1'},
    'm s-1': {'m s-1', 'm s**-1', 'm/s'},
    'm2 s-2': {'m2 s-2', 'm**2 s**-2', 'm^2/s^2'},
    'Pa s-1': {'Pa s-1', 'Pa s**-1', 'Pa/s'}, 'Pa': {'Pa'},
    'kg m-2': {'m'},  # GraphCast precipitation is liquid-water-equivalent metres.
}


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def download_artifact(name, directory):
    if name not in ('mean_by_level.nc', 'stddev_by_level.nc'):
        raise ValueError('Only approved GraphCast level statistics may be downloaded.')
    target = Path(directory)/name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'{target} exists; use the offline import instead of replacing it.')
    tmp = target.with_suffix('.nc.part')
    try:
        with urlopen(BASE_URL+name, timeout=30) as response, tmp.open('wb') as out:
            total = 0
            while block := response.read(1024*1024):
                total += len(block)
                if total > 64*1024*1024: raise ValueError('Unexpectedly large statistics artifact.')
                out.write(block)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target


def import_graphcast(mean_path, std_path, *, expected_hashes=None):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError('Install the optional data dependencies: pip install -e ".[data]"') from exc
    hashes = {'mean_by_level.nc': file_sha256(mean_path), 'stddev_by_level.nc': file_sha256(std_path)}
    if expected_hashes is not None and expected_hashes != hashes:
        raise ValueError('Statistics bytes differ from the pinned experiment hashes.')
    entries = {}
    missing_units = []
    with xr.open_dataset(mean_path) as means, xr.open_dataset(std_path) as stds:
        for original, (name, units, multiplier, interval) in VARIABLES.items():
            if original not in means or original not in stds:
                raise ValueError(f'Incomplete GraphCast artifact: {original}')
            mean, std = means[original], stds[original]
            for value in (mean, std):
                declared = value.attrs.get('units')
                if declared is not None and declared not in NATIVE_UNITS[units]:
                    raise ValueError(f'Unexpected native units for {original}: {declared}')
                if declared is None: missing_units.append(original)
            pressure = []
            if 'level' in mean.dims:
                if mean.dims != ('level',) or std.dims != ('level',):
                    raise ValueError('Only scalar or level-wise statistics are supported.')
                if set(mean.level.values.tolist()) != set(PRESSURE_HPA) or set(std.level.values.tolist()) != set(PRESSURE_HPA):
                    raise ValueError('GraphCast source must contain all 37 documented hPa levels.')
                mean, std = mean.sel(level=list(PRESSURE_HPA)), std.sel(level=list(PRESSURE_HPA))
                pressure = [p*100. for p in PRESSURE_HPA]
            elif mean.ndim or std.ndim:
                raise ValueError('Surface statistics must be scalar, not a climatology map.')
            entries[name] = dict(units=units,
                                 mean=(np.asarray(mean.values).reshape(-1)*multiplier).tolist(),
                                 std=(np.asarray(std.values).reshape(-1)*multiplier).tolist(),
                                 pressure_pa=pressure, interval_hours=interval)
    payload = dict(schema_version=1, kind='global_level_zscore', variables=entries,
                   provenance=dict(repository=SOURCE_REPO, revision=SOURCE_REVISION,
                                   data_family='GraphCast ERA5 normalization',
                                   artifact_sha256=hashes, fit_period=None,
                                   license='CC-BY-4.0; underlying ERA5 terms also apply',
                                   acquired_at=datetime.now(timezone.utc).isoformat(),
                                   verified_against_expected_hashes=expected_hashes is not None,
                                   acquisition='local_files; URLs identify intended upstream source',
                                   artifact_urls={name: BASE_URL+name for name in hashes},
                                   units_from_documented_schema=sorted(set(missing_units)),
                                   note='Fit period not established by these files; not local or monthly climatology.'))
    return NormalizationBundle(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mean', type=Path)
    parser.add_argument('--std', type=Path)
    parser.add_argument('--download', type=Path, metavar='CACHE_DIR')
    parser.add_argument('--expected-hashes', type=Path, help='JSON mapping both filenames to SHA256')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.download is not None:
        if args.mean or args.std: parser.error('Choose download or local mean/std, not both.')
        args.mean = download_artifact('mean_by_level.nc', args.download)
        args.std = download_artifact('stddev_by_level.nc', args.download)
    if not args.mean or not args.std: parser.error('Supply both --mean and --std, or --download.')
    expected = json.loads(args.expected_hashes.read_text()) if args.expected_hashes else None
    bundle = import_graphcast(args.mean, args.std, expected_hashes=expected)
    bundle.save(args.output)
    print(json.dumps(dict(fingerprint=bundle.fingerprint, variables=sorted(bundle.stats),
                          missing_for_3h_model=['td2m', 'surface_pressure', 'total_cloud_fraction',
                                               'precipitation_step: requires 3h training statistics'],
                          independent_test_period_verified=False), ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
