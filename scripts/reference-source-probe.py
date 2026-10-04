"""Read official public sources and retain artifacts; no credentials or repository writes."""
import hashlib,json,re,subprocess,zipfile
from pathlib import Path
from urllib.request import urlopen
import xarray as xr
OUT=Path('outputs/reference-probe');OUT.mkdir(parents=True,exist_ok=True)
def read(url,limit):
    with urlopen(url,timeout=20) as r:data=r.read(limit+1)
    if len(data)>limit:raise ValueError('Source exceeds size bound')
    return data
report={}
for name in ('mean_by_level.nc','stddev_by_level.nc','diffs_stddev_by_level.nc'):
    url='https://storage.googleapis.com/dm_graphcast/graphcast/stats/'+name
    data=read(url,2000000);(OUT/name).write_bytes(data)
    with xr.open_dataset(OUT/name) as ds:
        report[name]={'url':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'attributes':{k:str(v) for k,v in ds.attrs.items()}}
mirrors=['https://opentopography.s3.sdsc.edu/gmtdata','https://www.star.nesdis.noaa.gov/data/socd3/lsa/gmtdata','https://generic-mapping-tools.iag.usp.br/gmtdata','https://www.earthbyte.org/webdav/gmt_mirror/gmtdata']
report['dem_mirrors']=[]
for base in mirrors:
    url=base+'/server/earth/earth_relief/earth_relief_30s_p/'
    entry={'url':url}
    try:
        index=read(url,2000000).decode();(OUT/('dem-index-'+str(len(report['dem_mirrors']))+'.txt')).write_text(index)
        names=sorted(set(re.findall(r'href="([^"/]+\.(?:nc|grd|jp2))"',index)))
        entry.update(count=len(names),names=names[:5])
        if names:
            data=read(url+names[0],16000000);path=OUT/'dem-sample.nc';path.write_bytes(data)
            with xr.open_dataset(path,decode_cf=False) as ds:
                entry['sample']={'name':names[0],'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'variables':{k:{'shape':list(v.shape),'dtype':str(v.dtype),'attrs':{a:str(b) for a,b in v.attrs.items()}} for k,v in ds.variables.items()},'attrs':{a:str(b) for a,b in ds.attrs.items()}}
    except Exception as e:entry['error']=str(e)
    report['dem_mirrors'].append(entry)
    print('MIRROR='+json.dumps(entry),flush=True)
    if 'sample' in entry:break
with zipfile.ZipFile(OUT/'source-snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in subprocess.check_output(['git','ls-files'],text=True).splitlines():
        path=Path(name)
        if path.is_file():archive.write(path,name)
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print('REPORT='+json.dumps(report),flush=True)
