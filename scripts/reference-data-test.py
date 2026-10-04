"""Read the actually downloaded global DEM and pinned published normalization.

This test validates bytes, scale, geometry and numerical reader. It does not
claim a validated land mask or real weather training. Network is handled by CI.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import h5py
import numpy as np
from global_weather.terrain import dem_samples
from global_weather.grid import build_grid,latlon
from global_weather.import_climatology import bundled_directory,import_graphcast,PINNED_HASHES


def main():
    p=argparse.ArgumentParser();p.add_argument('--dem',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    source=json.loads(Path('assets/dem/earth_relief_30s_p.json').read_text())
    grid=build_grid(3);ll=latlon(grid.xyz)
    values=dem_samples(a.dem,source,ll[:,0],ll[:,1])
    if values.shape!=(642,) or not np.isfinite(values).all():raise RuntimeError('Global point sampling failed')
    if (values<source['min_m']).any() or (values>source['max_m']).any():raise RuntimeError('Sample outside recorded data range')
    with h5py.File(a.dem,'r') as ds:
        for i in (0,1,42,300,641):
            row=int(np.clip(np.floor((ll[i,0]+90)*120),0,21599))
            col=int(np.floor(((ll[i,1]+180)%360)*120))
            expected=float(ds['elevation'][row,col])*.5
            if values[i]!=expected:raise RuntimeError('Elevation packing was not applied exactly once')
    location=bundled_directory()
    bundle=import_graphcast(location/'mean_by_level.nc',location/'stddev_by_level.nc',expected_hashes=PINNED_HASHES)
    report={'status':'real_reference_reader_checked','dem_sha256':source['sha256'],'dem_bytes':source['bytes'],
            'max_dem_bytes':900000000,'sampled_global_cells':642,'sample_min_m':float(values.min()),
            'sample_max_m':float(values.max()),'grid_fingerprint':grid.fingerprint,
            'normalization_fingerprint':bundle.fingerprint,'fit_period':bundle._payload['provenance']['fit_period'],
            'normalization_variables':sorted(bundle.stats),'scale_applied_once':True,'weather_training_performed':False,
            'note':'Raw DEM includes bathymetry. Reader test only; atmospheric terrain requires independent land mask.'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
