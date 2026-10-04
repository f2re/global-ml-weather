"""Bounded public data verification. No invented observations or credentials."""
from pathlib import Path
from urllib.request import urlopen,Request
from datetime import datetime,timezone
import hashlib,json,zipfile
OUT=Path('outputs/additional-providers');OUT.mkdir(parents=True,exist_ok=True)
SOURCES={
 'igra':['https://www.ncei.noaa.gov/pub/data/igra/data/data-por/RSM00026075-data.txt.zip'],
 'gfs_index':['https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20220101/00/atmos/gfs.t00z.pgrb2.1p00.f000.idx','https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.20260901/00/atmos/gfs.t00z.pgrb2.1p00.f000.idx'],
 'goes_listing':['https://noaa-goes16.s3.amazonaws.com/?list-type=2&prefix=ABI-L2-MCMIPC/2020/001/00/&max-keys=2'],
}
report={'schema':'live-provider-probe-1','data_kind':'real','sources':{}}
for name,urls in SOURCES.items():
 attempts=[]
 for url in urls:
  try:
   with urlopen(Request(url,headers={'User-Agent':'global-ml-weather verified provider test'}),timeout=90) as r:
    data=r.read(80000001)
    if len(data)>80000000:raise ValueError('Source exceeds 80MB limit')
    info={'requested_url':url,'response_url':r.geturl(),'http_status':r.status,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'acquired_at':datetime.now(timezone.utc).isoformat(),'status':'downloaded'}
   path=OUT/(name+('.zip' if name=='igra' else '.txt'));path.write_bytes(data);info['path']=path.name
   if name=='igra':
    with zipfile.ZipFile(path) as z:
     info['members']=[{'name':x.filename,'bytes':x.file_size} for x in z.infolist()]
     with z.open(z.infolist()[0]) as f:info['preview']=f.read(1800).decode('ascii',errors='replace')
   else:info['preview']=data[:1600].decode('utf-8',errors='replace')
   report['sources'][name]=info;break
  except Exception as e:attempts.append({'url':url,'error':type(e).__name__+': '+str(e)[:200]})
 else:report['sources'][name]={'status':'failed','attempts':attempts}
 print(name,json.dumps(report['sources'][name]),flush=True)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
