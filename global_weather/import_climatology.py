"""Import checksum-pinned ERA5 statistics published with GraphCast.

No neural weights are imported. The bundled NetCDF files contain global
level statistics, not local seasonal climatology. Download is explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone
import re
import sysconfig
import numpy as np
from .normalization import NormalizationBundle
from .vertical import PRESSURE_HPA

SOURCE_REPO='https://github.com/google-deepmind/weathernext'
SOURCE_REVISION='f2f2c5117d2f864e2d5e7c2f9f220db5e1049dd3'
BASE_URL='https://storage.googleapis.com/dm_graphcast/graphcast/stats/'
VARIABLES={
    'temperature':('temperature','K',1.,None),
    'specific_humidity':('specific_humidity','kg kg-1',1.,None),
    'u_component_of_wind':('u','m s-1',1.,None),
    'v_component_of_wind':('v','m s-1',1.,None),
    'geopotential':('geopotential','m2 s-2',1.,None),
    'vertical_velocity':('omega','Pa s-1',1.,None),
    '2m_temperature':('t2m','K',1.,None),
    '10m_u_component_of_wind':('u10','m s-1',1.,None),
    '10m_v_component_of_wind':('v10','m s-1',1.,None),
    'mean_sea_level_pressure':('mslp','Pa',1.,None),
    'total_precipitation_6hr':('precipitation_step','kg m-2',1000.,6),
}
PINNED_HASHES={
    'mean_by_level.nc':'e6e724b94cd27707903cdc5cf5784d420bab306f722e631f4455e25c96da9931',
    'stddev_by_level.nc':'29f941d8c34906b87a1745f93b2308d1e169ec9bbdc4c5491479c771fe92c3cb',
}
OPTIONAL_VARIABLES={
    'surface_pressure':('surface_pressure','Pa',1.,None),
    'total_cloud_cover':('total_cloud_fraction','1',1.,None),
    'sea_surface_temperature':('sea_surface_temperature','K',1.,None),
    'sea_ice_cover':('sea_ice_fraction','1',1.,None),
}
NATIVE_UNITS={
    '1':{'1','(0 - 1)','0-1'},'K':{'K'},'kg kg-1':{'kg kg-1','kg kg**-1','kg/kg','1'},
    'm s-1':{'m s-1','m s**-1','m/s'},'m2 s-2':{'m2 s-2','m**2 s**-2','m^2/s^2'},
    'Pa s-1':{'Pa s-1','Pa s**-1','Pa/s'},'Pa':{'Pa'},'kg m-2':{'m'},
}


def bundled_directory():
    candidates=[Path(__file__).resolve().parents[1]/'assets/normalization/graphcast',
                Path(sysconfig.get_path('data'))/'share/global-ml-weather/normalization/graphcast']
    for path in candidates:
        if all((path/name).is_file() for name in PINNED_HASHES):return path
    raise FileNotFoundError('Bundled statistics are missing; install data files or use explicit local paths.')


def source_fit_period(means,stds):
    """Year-only end includes the whole year; never invent an exact final sample."""
    attrs={k:str(means.attrs[k]) for k in ('date_start','date_end') if k in means.attrs}
    if len(attrs)!=2:return None,'missing_source_period'
    if any(str(stds.attrs.get(k))!=value for k,value in attrs.items()):
        raise ValueError('Mean and standard deviation have different fitting periods.')
    def bound(value,upper):
        if re.fullmatch(r'\d{4}',value):value+='-12-31' if upper else '-01-01'
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):raise ValueError('Unknown source fitting date format.')
        suffix='T23:59:59.999999+00:00' if upper else 'T00:00:00+00:00'
        return datetime.fromisoformat(value+suffix)
    start,end=bound(attrs['date_start'],False),bound(attrs['date_end'],True)
    if start>end:raise ValueError('Source fitting period is reversed.')
    return {'start':start.isoformat(),'end':end.isoformat()},'conservative_bounds_from_netcdf_attributes'


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download_artifact(name,directory):
    if name not in ('mean_by_level.nc','stddev_by_level.nc'):
        raise ValueError('Only approved GraphCast level statistics may be downloaded.')
    target=Path(directory)/name;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise FileExistsError(f'{target} exists; use the offline import instead of replacing it.')
    tmp=target.with_suffix('.nc.part');owned=False
    try:
        with urlopen(BASE_URL+name,timeout=30) as response,tmp.open('xb') as out:
            owned=True;total=0
            while block:=response.read(1024*1024):
                total+=len(block)
                if total>2_000_000:raise ValueError('Unexpectedly large statistics artifact.')
                out.write(block)
        tmp.replace(target)
    except Exception:
        if owned:tmp.unlink(missing_ok=True)
        raise
    return target


def import_graphcast(mean_path,std_path,*,expected_hashes=None):
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError('Install the optional data dependencies: pip install -e ".[data]"') from exc
    for p in (mean_path,std_path):
        if Path(p).stat().st_size>2_000_000:raise ValueError('Level statistics exceed the file-size limit.')
    hashes={'mean_by_level.nc':file_sha256(mean_path),'stddev_by_level.nc':file_sha256(std_path)}
    if expected_hashes is not None and expected_hashes!=hashes:
        raise ValueError('Statistics bytes differ from the pinned experiment hashes.')
    entries={};missing_units=[]
    with xr.open_dataset(mean_path) as means,xr.open_dataset(std_path) as stds:
        period,period_interpretation=source_fit_period(means,stds)
        attributes={k:str(v) for k,v in means.attrs.items()}
        mapping=dict(VARIABLES)
        for key,value in OPTIONAL_VARIABLES.items():
            if (key in means)!=(key in stds):raise ValueError('Incomplete optional statistic: '+key)
            if key in means:mapping[key]=value
        for original,(name,units,multiplier,interval) in mapping.items():
            if original not in means or original not in stds:raise ValueError(f'Incomplete GraphCast artifact: {original}')
            mean,std=means[original],stds[original]
            for value in (mean,std):
                declared=value.attrs.get('units')
                if declared is not None and declared not in NATIVE_UNITS[units]:raise ValueError(f'Unexpected native units for {original}: {declared}')
                if declared is None:missing_units.append(original)
            pressure=[]
            if 'level' in mean.dims:
                if mean.dims!=('level',) or std.dims!=('level',):raise ValueError('Only scalar or level-wise statistics are supported.')
                if set(mean.level.values.tolist())!=set(PRESSURE_HPA) or set(std.level.values.tolist())!=set(PRESSURE_HPA):
                    raise ValueError('GraphCast source must contain all 37 documented hPa levels.')
                mean,std=mean.sel(level=list(PRESSURE_HPA)),std.sel(level=list(PRESSURE_HPA))
                pressure=[p*100. for p in PRESSURE_HPA]
            elif mean.ndim or std.ndim:raise ValueError('Surface statistics must be scalar, not a climatology map.')
            entries[name]=dict(units=units,mean=(np.asarray(mean.values).reshape(-1)*multiplier).tolist(),
                std=(np.asarray(std.values).reshape(-1)*multiplier).tolist(),pressure_pa=pressure,interval_hours=interval)
    payload=dict(schema_version=1,kind='global_level_zscore',variables=entries,
        provenance=dict(repository=SOURCE_REPO,revision=SOURCE_REVISION,data_family='GraphCast ERA5 normalization',
            artifact_sha256=hashes,fit_period=period,source_attributes=attributes,period_interpretation=period_interpretation,
            license='CC-BY-4.0; underlying ERA5 terms also apply',
            verified_against_expected_hashes=expected_hashes is not None,
            acquisition='local_files; URLs identify intended upstream source',
            artifact_urls={name:BASE_URL+name for name in hashes},units_from_documented_schema=sorted(set(missing_units)),
            note='Global level statistics, not local/monthly climatology. Year-only date bounds are conservative.'))
    return NormalizationBundle(payload)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundled',action='store_true',help='Use checksum-pinned local reference files without network')
    parser.add_argument('--mean',type=Path);parser.add_argument('--std',type=Path)
    parser.add_argument('--download',type=Path,metavar='CACHE_DIR')
    parser.add_argument('--expected-hashes',type=Path,help='JSON mapping both filenames to SHA256')
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    if args.output.exists():raise FileExistsError('Normalization output is never overwritten.')
    if args.bundled:
        if args.mean or args.std or args.download or args.expected_hashes:parser.error('--bundled cannot be combined with a different source.')
        directory=bundled_directory();args.mean=directory/'mean_by_level.nc';args.std=directory/'stddev_by_level.nc'
    if args.download is not None:
        if args.mean or args.std:parser.error('Choose download or local mean/std, not both.')
        args.mean=download_artifact('mean_by_level.nc',args.download);args.std=download_artifact('stddev_by_level.nc',args.download)
    if not args.mean or not args.std:parser.error('Supply both --mean and --std, or --download.')
    expected=PINNED_HASHES if args.bundled else (json.loads(args.expected_hashes.read_text()) if args.expected_hashes else None)
    bundle=import_graphcast(args.mean,args.std,expected_hashes=expected);bundle.save(args.output)
    print(json.dumps(dict(fingerprint=bundle.fingerprint,variables=sorted(bundle.stats),
        missing_for_3h_model=[k for k in ('td2m','surface_pressure','total_cloud_fraction') if k not in bundle.stats]+['precipitation_step: requires 3h training statistics'],
        fit_period=bundle._payload['provenance']['fit_period'],independent_test_period_verified=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
