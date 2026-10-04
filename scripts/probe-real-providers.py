"""Read actual public source files. No generated observations or credentials.

This is an explicit, bounded acquisition check, not a training data generator.
Every retained byte is accompanied by its URL, HTTP status and SHA256.
"""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib, json, re, time, zipfile
from pathlib import Path
from urllib.request import Request, urlopen

OUT = Path('outputs/provider-probe')
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
 'ghcnh': ['https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2020/psv/GHCNh_USW00014933_2020.psv', 'https://noaa-ghcnh-pds.s3.amazonaws.com/access/by-year/2020/psv/GHCNh_USW00014933_2020.psv'],
 'ghcnh_doc': ['https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf'],
 'ghcnh_stations': ['https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt'],
 'iem': ['https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station=DSM&data=tmpf&data=dwpf&data=drct&data=sknt&data=mslp&data=alti&sts=2020-01-01T00%3A00%3A00Z&ets=2020-01-21T00%3A00%3A00Z&tz=Etc%2FUTC&format=onlycomma&latlon=yes&elev=yes&missing=M&trace=T&report_type=3&report_type=4'],
 'ndbc': ['https://dods.ndbc.noaa.gov/thredds/fileServer/data/stdmet/46042/46042h2020.nc'],
 'ndbc_text': ['https://www.ndbc.noaa.gov/view_text_file.php?filename=46042h2020.txt.gz&dir=data/historical/stdmet/'],
 'igra': ['https://www.ncei.noaa.gov/pub/data/igra/data/data-por/USM00072558-data.txt.zip'],
 'igra_format': ['https://www.ncei.noaa.gov/pub/data/igra/data/igra2-data-format.txt'],
 'igra_stations': ['https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt'],
 'gfs_index': ['https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20200101/00/gfs.t00z.pgrb2.0p25.f000.idx', 'https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20200101/00/atmos/gfs.t00z.pgrb2.0p25.f000.idx'],
 'era5_37_metadata': ['https://storage.googleapis.com/weatherbench2/datasets/era5/1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr/.zmetadata'],
}
report = {'schema':'live-provider-probe-1','data_kind':'real','started_at':datetime.now(timezone.utc).isoformat(),'sources':{}}
for name, urls in SOURCES.items():
    attempts=[]
    for url in urls:
        try:
            with urlopen(Request(url,headers={'User-Agent':'global-ml-weather/0.7 provider validation'}),timeout=60) as r:
                data=r.read(40_000_001)
                if len(data)>40_000_000: raise ValueError('Bounded file exceeds 40 MB')
                if not data: raise ValueError('Empty response')
                info={'requested_url':url,'response_url':r.geturl(),'http_status':r.status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            suffix='.pdf' if name=='ghcnh_doc' else '.zip' if name=='igra' else '.nc' if name=='ndbc' else '.json' if name.endswith('metadata') else '.txt'
            path=OUT/(name+suffix);path.write_bytes(data)
            info.update(path=path.name,acquired_at=datetime.now(timezone.utc).isoformat(),status='downloaded',attempts=attempts)
            if suffix not in ('.pdf','.zip','.nc'): info['preview']=data[:1200].decode('utf-8',errors='replace')
            if name=='ndbc':
                import xarray as xr
                with xr.open_dataset(path) as ds: info['variables']={k:{'dims':list(v.dims),'units':v.attrs.get('units'),'shape':list(v.shape)} for k,v in ds.variables.items()}
            if name=='era5_37_metadata':
                m=json.loads(data)['metadata'];info['arrays']={k:v for k,v in m.items() if k.endswith('.zarray')}
            report['sources'][name]=info
            break
        except Exception as e: attempts.append({'url':url,'error':type(e).__name__+': '+str(e)[:200]})
    else: report['sources'][name]={'status':'failed','attempts':attempts}
    print(name,json.dumps(report['sources'][name],ensure_ascii=False),flush=True)
    (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    time.sleep(1.1)
report['completed_at']=datetime.now(timezone.utc).isoformat()
(OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
