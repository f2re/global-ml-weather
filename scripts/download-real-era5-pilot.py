"""Acquire real 37-level ERA5 pilot targets, not generated weather.

Four separated issues, analysis and +3h. Full hourly source chunks are sampled
at 12 global cells; the 6GB transfer bound is based on observed chunk sizes.
"""
from pathlib import Path
from datetime import datetime,timedelta,timezone
from urllib.request import urlopen,Request
import hashlib,json,struct,time,sys
import numpy as np
from numcodecs import get_codec
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from global_weather.grid import build_grid,latlon
from global_weather.vertical import PRESSURE_HPA
BASE='https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr'
OUT=Path('outputs/real-era5-pilot');OUT.mkdir(parents=True,exist_ok=True)
MAX=6_000_000_000
receipt={'schema':'provider-receipt-1','provider':'weatherbench2_era5_37','data_kind':'real','status':'acquiring','objects':[],
 'details':{'role':'real_reanalysis_targets_not_station_observations','license':'ECMWF/Copernicus ERA5; retain WeatherBench attribution',
 'operator':'nearest_native_ERA5_grid_cell; not conservative average','horizon_hours':3,'transfer_limit_bytes':MAX}}
used=0;cache={}
def get(key,limit=170_000_000):
 global used
 if key in cache:return cache[key]
 url=BASE+'/'+key
 for attempt in range(3):
  try:
   with urlopen(Request(url,headers={'User-Agent':'global-ml-weather real-data validation'}),timeout=120) as r:
    data=r.read(limit+1)
    if len(data)>limit or used+len(data)>MAX:raise ValueError('Explicit real-data download budget exceeded')
    row={'requested_url':url,'response_url':r.geturl(),'http_status':r.status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'acquired_at':datetime.now(timezone.utc).isoformat()}
   used+=len(data);receipt['objects'].append(row)
   if len(data)<2_000_000:cache[key]=data
   return data
  except (OSError,TimeoutError):
   if attempt==2:raise
   time.sleep(2+attempt)
metadata=get('.zmetadata',2_000_000);(OUT/'source-zmetadata.json').write_bytes(metadata);meta=json.loads(metadata)['metadata']
def chunk(name,ti=None):
 m=meta[name+'/.zarray'];dims=meta[name+'/.zattrs']['_ARRAY_DIMENSIONS'];shape=m['chunks'];size=int(np.prod(shape))*np.dtype(m['dtype']).itemsize
 if size>170_000_000 or m['order']!='C' or m['filters'] is not None or m['compressor']['id']!='blosc':raise ValueError('Unsupported source packing')
 indices=([ti//shape[0]] if 'time' in dims else [])+[0]*(len(shape)-(1 if 'time' in dims else 0))
 raw=get(name+'/'+'.'.join(map(str,indices)))
 if struct.unpack_from('<I',raw,4)[0]!=size:raise ValueError('Unsafe or inconsistent compressed header')
 decoded=get_codec(m['compressor']).decode(raw)
 if len(decoded)!=size:raise ValueError('Incomplete source chunk')
 a=np.frombuffer(decoded,dtype=np.dtype(m['dtype'])).reshape(shape)
 return a[ti%shape[0]] if 'time' in dims else a
lat=chunk('latitude');lon=chunk('longitude');levels=chunk('level')
if set(levels)!=set(PRESSURE_HPA):raise ValueError('Not all 37 levels')
order=[int(np.flatnonzero(levels==p)[0]) for p in PRESSURE_HPA]
grid=build_grid(0);ll=latlon(grid.xyz);iy=np.abs(lat[:,None]-ll[:,0]).argmin(0);ix=np.abs(lon[:,None]-(ll[:,1]%360)).argmin(0)
coords=dict(latitude=ll[:,0],longitude=ll[:,1],native_latitude=lat[iy],native_longitude=lon[ix],pressure_hpa=np.array(PRESSURE_HPA),grid_fingerprint=np.array(grid.fingerprint))
static={name:np.asarray(chunk(name)[iy,ix]).copy() for name in ('geopotential_at_surface','land_sea_mask')}
np.savez_compressed(OUT/'static-source.npz',**coords,**static)
PROFILE=['temperature','specific_humidity','u_component_of_wind','v_component_of_wind','geopotential','vertical_velocity']
SURFACE=['2m_temperature','2m_dewpoint_temperature','10m_u_component_of_wind','10m_v_component_of_wind','surface_pressure','mean_sea_level_pressure','total_cloud_cover','total_precipitation']
issues=[datetime(2020,1,day,12,tzinfo=timezone.utc) for day in (2,4,10,18)]
samples={}
for issue in issues:
 for lead in (0,3):
  when=issue+timedelta(hours=lead);ti=int((when-datetime(1959,1,1,tzinfo=timezone.utc)).total_seconds()/3600)
  if int(chunk('time',ti))!=ti:raise ValueError('Time coordinate mismatch')
  values={}
  for name in PROFILE:
   a=chunk(name,ti);values[name]=np.asarray(a[:,iy,ix].T[:,order]).copy();del a
   print('REAL_FIELD',when.isoformat(),name,'downloaded_bytes',used,flush=True)
  for name in SURFACE:values[name]=np.asarray(chunk(name,ti)[iy,ix]).copy()
  if lead:
   hourly=[values['total_precipitation']]
   for h in (1,2):hourly.append(np.asarray(chunk('total_precipitation',ti-h)[iy,ix]).copy())
   values['precipitation_3h_m']=np.sum(hourly,axis=0)
  if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Missing source target values')
  name=f'era5-{when:%Y%m%dT%H}.npz';np.savez_compressed(OUT/name,**coords,**values,time_utc=np.array(when.isoformat()))
  samples[when.isoformat()]={'path':name,'sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),'variables':list(values)}
receipt.update(status='verified',verified_at=datetime.now(timezone.utc).isoformat(),downloaded_bytes=used)
receipt['details'].update(sampled_grid_fingerprint=grid.fingerprint,issues=[v.isoformat() for v in issues],native_source_shape=[721,1440],pressure_levels=37)
(OUT/'provider-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'pilot-index.json').write_text(json.dumps({'schema':'real-era5-pilot-1','data_kind':'real','samples':samples,'grid_fingerprint':grid.fingerprint,
 'static':{'path':'static-source.npz','sha256':hashlib.sha256((OUT/'static-source.npz').read_bytes()).hexdigest()},
 'receipt':{'path':'provider-receipt.json','sha256':hashlib.sha256((OUT/'provider-receipt.json').read_bytes()).hexdigest()}},indent=2)+'\n')
print('REAL_ERA5_PILOT_COMPLETED',used,flush=True)
