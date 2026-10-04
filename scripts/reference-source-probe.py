"""Read official public reference sources; no credentials or repository writes."""
import hashlib
import json
from pathlib import Path
import re
from urllib.request import urlopen

OUT=Path('outputs/reference-probe'); OUT.mkdir(parents=True,exist_ok=True)

def read(url,limit):
    with urlopen(url,timeout=60) as r:
        data=r.read(limit+1)
    if len(data)>limit: raise ValueError('Source exceeds size bound')
    return data

report={}
for name in ('mean_by_level.nc','stddev_by_level.nc','diffs_stddev_by_level.nc'):
    url='https://storage.googleapis.com/dm_graphcast/graphcast/stats/'+name
    data=read(url,2000000); (OUT/name).write_bytes(data)
    import xarray as xr
    with xr.open_dataset(OUT/name) as ds:
        report[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'attributes':{k:str(v) for k,v in ds.attrs.items()},'data':ds.to_dict(data=True)}
base='https://oceania.generic-mapping-tools.org/server/earth/earth_relief/earth_relief_30s_p/'
try:
    index=read(base,2000000).decode()
    (OUT/'dem-index.txt').write_text(index)
    names=sorted(set(re.findall(r'href="([^"/]+\.(?:nc|grd|jp2))"',index)))
    report['dem']={'url':base,'count':len(names),'names':names[:10]}
except Exception as e:
    report['dem']={'error':str(e)}
(OUT/'report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
print(json.dumps(report,default=str),flush=True)
