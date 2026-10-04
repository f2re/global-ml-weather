"""Explicit public DEM acquisition. Does not write Git or use credentials.

Retains 30 arcsec sampling and original integer values. GMT's documented 0.5 m
packing is applied by CF readers, not guessed from JPEG2000 scale metadata.
"""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, re, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import rasterio
from netCDF4 import Dataset

BASE='https://www.star.nesdis.noaa.gov/data/socd3/lsa/gmtdata/'
DIRECTORY=BASE+'server/earth/earth_relief/earth_relief_30s_p/'
LIMIT=900_000_000


def get(url,path,limit):
    if not url.startswith(BASE): raise ValueError('Only this official GMT mirror is allowed')
    if path.exists(): raise FileExistsError(path)
    for attempt in range(3):
        temporary=path.with_suffix(path.suffix+'.part')
        try:
            with urlopen(url,timeout=90) as response,temporary.open('xb') as out:
                if int(response.headers.get('Content-Length','0'))>limit:raise ValueError('Source exceeds byte limit')
                total=0;h=hashlib.sha256()
                while chunk:=response.read(1024*1024):
                    total+=len(chunk)
                    if total>limit:raise ValueError('Source exceeds byte limit')
                    out.write(chunk);h.update(chunk)
            temporary.replace(path)
            return {'url':url,'bytes':total,'sha256':h.hexdigest()}
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt==2:raise
            time.sleep(2*(attempt+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--network',action='store_true');a=p.parse_args()
    if not a.network:p.error('Acquisition requires explicit --network')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    index=out/'index.html';index_info=get(DIRECTORY,index,2_000_000)
    names=sorted(set(re.findall(r'href="([NS]\d{2}[EW]\d{3}\.earth_relief_30s_p\.jp2)"',index.read_text())))
    if len(names)!=288:raise ValueError(f'Expected all 288 tiles, got {len(names)}')
    catalogue=out/'gmt_data_server.txt';catalog_info=get(BASE+'gmt_data_server.txt',catalogue,2_000_000)
    line=next((s for s in catalogue.read_text().splitlines() if 'earth_relief_30s_p/' in s and not s.lstrip().startswith('#')),None)
    if not line or not re.search(r'30s\s+p\s+0\.5\s+0\s+',line):raise ValueError('Official scale/registration metadata changed')
    dst_path=out/'earth_relief_30s_p.nc';temporary=out/'earth_relief_30s_p.nc.part'
    if dst_path.exists() or temporary.exists():raise FileExistsError('DEM is never overwritten')
    sources=[];seen=set();minimum=32767;maximum=-32768;total_bytes=0
    with Dataset(temporary,'w',format='NETCDF4') as dst:
        dst.createDimension('lat',21600);dst.createDimension('lon',43200)
        lat=dst.createVariable('lat','f8',('lat',));lon=dst.createVariable('lon','f8',('lon',))
        lat[:]=-90+(np.arange(21600)+.5)/120;lon[:]=-180+(np.arange(43200)+.5)/120
        lat.units='degrees_north';lon.units='degrees_east'
        z=dst.createVariable('elevation','i2',('lat','lon'),zlib=True,complevel=9,shuffle=True,chunksizes=(180,360),fill_value=-32768)
        z.units='m';z.scale_factor=.5;z.add_offset=0.;z.set_auto_maskandscale(False)
        z.long_name='Topographic and bathymetric relief, not atmospheric surface elevation'
        dst.Conventions='CF-1.8';dst.node_offset=1;dst.source=DIRECTORY
        dst.history='Lossless repacking of GMT 30s integer tiles. Original 0.5 m scale retained.'
        for start in range(0,len(names),8):
            group=names[start:start+8]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futures={n:pool.submit(get,DIRECTORY+n,out/n,20_000_000) for n in group}
                for name in group:
                    entry=futures[name].result();path=out/name
                    ns,lat0,ew,lon0=re.match(r'([NS])(\d{2})([EW])(\d{3})',name).groups()
                    south=int(lat0)*(1 if ns=='N' else -1);west=int(lon0)*(1 if ew=='E' else -1)
                    row=(south+90)*120;col=(west+180)*120
                    if (row,col) in seen or not 0<=row<=19800 or not 0<=col<=41400:raise ValueError('Duplicate/outside tile')
                    with rasterio.open(path) as src:
                        if src.count!=1 or src.shape!=(1800,1800) or src.dtypes[0]!='int16':raise ValueError('Unexpected tile type or dimensions')
                        expected=(1/120,0,west,0,-1/120,south+15)
                        if not np.allclose(tuple(src.transform)[:6],expected,atol=1e-7):raise ValueError('Incorrect 30 arcsec pixel coordinates')
                        raw=src.read(1)
                        if (raw==-32768).any() or src.read_masks(1).min()==0:raise ValueError('Global source has unaccounted missing elevations')
                        minimum=min(minimum,int(raw.min()));maximum=max(maximum,int(raw.max()))
                        z[row:row+1800,col:col+1800]=raw[::-1,:]
                    sources.append({'name':name,**entry});seen.add((row,col));total_bytes+=entry['bytes'];path.unlink()
            dst.sync();print(json.dumps({'tiles':len(seen),'source_bytes':total_bytes,'current_file_bytes':temporary.stat().st_size}),flush=True)
    if len(seen)!=288:raise ValueError('Incomplete global coverage')
    if temporary.stat().st_size>LIMIT:
        raise ValueError(f'Lossless 30s DEM is {temporary.stat().st_size} bytes, over the strict 900 MB limit. No lower-resolution substitution.')
    temporary.replace(dst_path)
    h=hashlib.sha256()
    with dst_path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    metadata={'schema':'global-dem-source-1','dataset':'GMT Earth Relief 30 arcsec','shape':[21600,43200],
        'registration':'pixel','scale_factor_m':.5,'add_offset_m':0,'dtype':'int16','spacing_arcsec':30,
        'filter_fullwidth_km':2.6,'bytes':dst_path.stat().st_size,'max_bytes':LIMIT,'sha256':h.hexdigest(),
        'min_m':minimum*.5,'max_m':maximum*.5,'source_bytes':total_bytes,'source_catalogue_line':line,
        'catalogue':catalog_info,'index':index_info,'tiles':sources,'acquired_at':datetime.now(timezone.utc).isoformat(),
        'reference':'https://www.generic-mapping-tools.org/remote-datasets/earth-relief.html',
        'license_reference':'https://topex.ucsd.edu/WWW_html/srtm15_plus.html',
        'bathymetry_present':True,'land_mask_required':True,
        'note':'Negative land exists. Never infer land/ocean from elevation sign. This is not atmospheric geopotential.'}
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'earth_relief_30s_p.nc.sha256').write_text(h.hexdigest()+'  earth_relief_30s_p.nc\n')
    print('DEM_RESULT='+json.dumps({k:v for k,v in metadata.items() if k!='tiles'}),flush=True)

if __name__=='__main__':main()
