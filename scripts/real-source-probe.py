"""Explicit public-source acquisition for adapter acceptance, never synthetic data.

No keys, fabricated values or training are involved. Every downloaded byte is
retained with its URL and hash; failed sources remain failed in the report.
"""
import argparse,hashlib,json,time,xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlencode
import requests


def main():
    p=argparse.ArgumentParser();p.add_argument('--network',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not a.network:p.error('Use --network for this explicit download operation')
    a.output.mkdir(parents=True,exist_ok=True);report={}
    sources={
      'isd-2020.csv':'https://noaa-global-hourly-pds.s3.amazonaws.com/2020/74486094789.csv',
      'ndbc-44013-2020.txt.gz':'https://www.ndbc.noaa.gov/data/historical/stdmet/44013h2020.txt.gz',
      'ndbc-stations.xml':'https://www.ndbc.noaa.gov/activestations.xml',
      'igra-station.zip':'https://www.ncei.noaa.gov/pub/data/igra/data/data-por/USM00072249-data.txt.zip',
      'igra-format.txt':'https://www.ncei.noaa.gov/pub/data/igra/data/igra2-data-format.txt',
      'metar.json':'https://aviationweather.gov/api/data/metar?ids=KJFK&format=json&hours=24',
      'metar-openapi.yaml':'https://aviationweather.gov/data/schema/openapi.yaml',
      'ghcnh-index.html':'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/index.html',
      'ghcnh-format.pdf':'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh_DOCUMENTATION.pdf',
      'ghcnh-stations.txt':'https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt',
      'ghcnh-list.xml':'https://noaa-ghcnh-pds.s3.amazonaws.com/?list-type=2&prefix=access/&max-keys=10',
      'goes-list.xml':'https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix=ABI-L1b-RadF/2020/001/00/&max-keys=30',
    }
    query=dict(var='air',north=90,south=-90,west=0,east=360,horizStride=4,
        time_start='2020-01-01T00:00:00Z',time_end='2020-01-24T18:00:00Z',timeStride=1,accept='netcdf4')
    for var in ('air','hgt','shum','omega','uwnd','vwnd'):
        sources[f'ncep-{var}.nc']='https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/pressure/'+var+'.2020.nc?'+urlencode({**query,'var':var})
    for folder,name,var in [('surface','pres.sfc','pres'),('surface','slp','slp'),('surface_gauss','air.2m','air'),('surface_gauss','shum.2m','shum')]:
        sources[f'ncep-{name}.nc']='https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/'+folder+'/'+name+'.2020.nc?'+urlencode({**query,'var':var})
    session=requests.Session();session.headers['User-Agent']='global-ml-weather/0.7 real-source-validation'
    def fetch(name,url):
        entry={'url':url,'data_kind':'real','status':'failed'};report[name]=entry
        path=a.output/name
        try:
            with session.get(url,stream=True,timeout=(20,90)) as r:
                r.raise_for_status();entry['final_url']=r.url
                h=hashlib.sha256();size=0
                with path.open('xb') as f:
                    for b in r.iter_content(1048576):
                        size+=len(b)
                        if size>80000000:raise ValueError('Source exceeds 80 MB validation bound')
                        f.write(b);h.update(b)
                entry.update(status='downloaded',bytes=size,sha256=h.hexdigest(),content_type=r.headers.get('Content-Type'),acquired_unix=time.time())
        except Exception as e:
            path.unlink(missing_ok=True);entry['error']=str(e)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({name:entry}),flush=True)
    for name,url in sources.items():fetch(name,url);time.sleep(.3)
    path=a.output/'goes-list.xml'
    if path.exists():
        keys=[e.text for e in ET.fromstring(path.read_bytes()).iter() if e.tag.endswith('}Key')]
        key=next((k for k in keys if '-M6C13_' in k),None)
        if key:fetch('goes-c13.nc','https://noaa-goes16.s3.amazonaws.com/'+key)
    path=a.output/'ghcnh-list.xml'
    if path.exists():
        print('GHCNH_KEYS='+path.read_text()[:20000],flush=True)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
