"""Second-stage real source acquisition after inspecting actual provider metadata."""
import hashlib,json,time,xml.etree.ElementTree as ET,subprocess
from pathlib import Path
from urllib.parse import urlencode
import requests
OUT=Path('outputs/real-source-supplement');OUT.mkdir(parents=True,exist_ok=True)
report={}
def fetch(name,url,limit=80000000):
    path=OUT/name;entry={'url':url,'data_kind':'real','status':'failed'};report[name]=entry
    for attempt in range(3):
        try:
            with requests.get(url,stream=True,timeout=(20,120),headers={'User-Agent':'global-ml-weather/0.7 provider-validation'}) as r:
                r.raise_for_status();h=hashlib.sha256();size=0
                with path.open('wb') as f:
                    for b in r.iter_content(262144):
                        size+=len(b)
                        if size>limit:raise ValueError('Acquisition limit exceeded')
                        h.update(b);f.write(b)
                entry.update(status='downloaded',bytes=size,sha256=h.hexdigest(),acquired_unix=time.time())
            break
        except Exception as e:
            path.unlink(missing_ok=True);entry['error']=str(e)
            if isinstance(e,requests.HTTPError) and e.response.status_code in (403,404):break
            time.sleep(2)
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({name:entry}),flush=True)
fetch('ghcnh-2020.psv','https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/access/by-year/2020/psv/GHCNh_USW00094789_2020.psv')
fetch('igra-recent.zip','https://www.ncei.noaa.gov/pub/data/igra/data/data-y2d/USM00072249-data.txt.zip')
fetch('goes-list.xml','https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix=ABI-L1b-RadF/2020/001/00/OR_ABI-L1b-RadF-M6C13&max-keys=2')
if (OUT/'goes-list.xml').exists():
    keys=[e.text for e in ET.fromstring((OUT/'goes-list.xml').read_bytes()).iter() if e.tag.endswith('}Key')]
    if keys:fetch('goes-c13.nc','https://noaa-goes16.s3.amazonaws.com/'+keys[0])
query=dict(north=90,south=-90,west=0,east=360,horizStride=4,time_start='2020-01-01T00:00:00Z',time_end='2020-01-24T18:00:00Z',timeStride=1,accept='netcdf4')
for var in ('air','hgt','shum','uwnd'):
    fetch('ncep-'+var+'.nc','https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/pressure/'+var+'.2020.nc?'+urlencode({**query,'var':var}))
for name,var in [('air.2m.gauss','air'),('shum.2m.gauss','shum'),('uwnd.10m.gauss','uwnd'),('vwnd.10m.gauss','vwnd'),('tcdc.eatm.gauss','tcdc'),('prate.sfc.gauss','prate')]:
    fetch('ncep-'+name+'.nc','https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/surface_gauss/'+name+'.2020.nc?'+urlencode({**query,'var':var}))
for name in ('hgt.sfc','land'):
    fetch('ncep-'+name+'.nc','https://psl.noaa.gov/thredds/fileServer/Datasets/ncep.reanalysis/surface/'+name+'.nc')
subprocess.run(['python','-m','pip','download','--no-deps','--dest',str(OUT),'h5netcdf==1.6.4'],check=True)
