"""NOAA IGRA 2.2 fixed-width archive adapter; actual sparse sounding values only."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import gzip
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import zipfile

from .http import download
from .upper_air import normalize_profile, specific_humidity
from ..pipeline.dataset import utc
from ..pipeline.io import atomic_json, digest, read_json, sha256

BASE='https://www.ncei.noaa.gov/pub/data/igra/'
FORMAT_URL=BASE+'data/igra2-data-format.txt'
LIST_FORMAT_URL=BASE+'igra2-list-format.txt'
ID=re.compile(r'[A-Z0-9]{11}')
MAX_SOURCE=100_000_000
MAX_EXPANDED=1024**3
MAX_CACHE=16*1024**3
MAX_OUTPUT=32*1024**3
START=utc('2021-01-01T00:00:00Z')
TRAIN_END=utc('2022-01-01T00:00:00Z')
VAL_END=utc('2022-07-01T00:00:00Z')
END=utc('2023-01-01T00:00:00Z')


def station_url(station):
    if not isinstance(station,str) or not ID.fullmatch(station):
        raise ValueError('Invalid IGRA station ID.')
    # NOAA publishes period-of-record archives, not separate 2021/2022 objects.
    return BASE+'data/data-por/'+station+'-data.txt.zip'


def station_inventory(path):
    stations=[]
    # NOAA's inventory uses UTF-8 station names and 88 CHARACTER columns.
    # Slice after decoding: e.g. Š occupies two bytes but one name column.
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        if len(line)!=88:raise ValueError('Invalid IGRA station inventory column count.')
        if not line[:41].isascii() or not line[71:].isascii():
            raise ValueError('Non-ASCII IGRA station inventory outside the station name.')
        identity=line[:11];lat=float(line[12:20]);lon=float(line[21:30])
        first,last=int(line[72:76]),int(line[77:81])
        if not ID.fullmatch(identity):raise ValueError('Invalid inventory ID.')
        # Mobile inventory positions are sentinels; preserve them only in raw catalog.
        if not -90<=lat<=90 or not -180<=lon<=180:continue
        if first<=2021<=last:
            elevation=float(line[31:37])
            stations.append({'id':identity,'latitude':lat,'longitude':lon,
                             'elevation_m':None if elevation in (-999.9,-998.8) else elevation,
                             'name':line[41:71].strip(),'first_year':first,'last_year':last})
    return stations


def spatial_candidates(stations):
    cells={}
    for station in stations:
        key=(min(11,int((station['latitude']+90)/15)),min(23,int((station['longitude']+180)/15)))
        cells.setdefault(key,[]).append(station)
    for rows in cells.values():rows.sort(key=lambda row:row['id'])
    ordered=[]
    # Far-apart longitude/latitude cells are visited before dense-cell replacements.
    keys=sorted(cells)
    def spread(items):
        if not items:return []
        mid=len(items)//2
        return [items[mid]]+spread(items[:mid])+spread(items[mid+1:])
    keys=spread(keys)
    for depth in range(max((len(rows) for rows in cells.values()),default=0)):
        ordered.extend(cells[key][depth] for key in keys if depth<len(cells[key]))
    return ordered


@contextmanager
def _stream(path):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size>MAX_SOURCE:
        raise ValueError('IGRA source must be a bounded regular file.')
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist()
            if len(infos)!=1:
                raise ValueError('IGRA ZIP requires exactly one text member.')
            item=infos[0]
            if (item.is_dir() or item.flag_bits&1 or '/' in item.filename or '\\' in item.filename or
                    not item.filename.endswith('-data.txt') or item.file_size>MAX_EXPANDED or
                    (item.external_attr>>16)&0o170000==0o120000):
                raise ValueError('Unsafe IGRA ZIP member.')
            with archive.open(item) as stream:yield stream
    elif path.suffix=='.gz':
        with gzip.open(path,'rb') as stream:yield stream
    elif path.suffix=='.txt':
        with path.open('rb') as stream:yield stream
    else:raise ValueError('Unsupported IGRA archive.')


def _lines(stream):
    total=0
    while True:
        raw=stream.readline(256)
        if not raw:break
        total+=len(raw)
        if total>MAX_EXPANDED or len(raw)>128:
            raise ValueError('IGRA uncompressed stream exceeds bounds.')
        yield raw.decode('ascii').rstrip('\r\n')


def _number(line,first,last):
    value=int(line[first-1:last])
    return None if value in (-9999,-8888) else value


def _header(line):
    if len(line)<71 or line[0]!='#':raise ValueError('Malformed IGRA header.')
    station=line[1:12]
    if not ID.fullmatch(station):raise ValueError('Invalid sounding station ID.')
    date=datetime(int(line[13:17]),int(line[18:20]),int(line[21:23]),tzinfo=timezone.utc)
    hour=int(line[24:26]);release=int(line[27:31]);count=int(line[32:36])
    if not 0<=count<=10000 or hour not in (*range(24),99):raise ValueError('Invalid IGRA header time or count.')
    nominal=date+timedelta(hours=hour) if hour!=99 else None
    launch=None
    if release!=9999:
        rh,rm=divmod(release,100)
        if not 0<=rh<=23 or rm not in (*range(60),99):raise ValueError('Invalid release HHMM.')
        if rm!=99:
            launch=date+timedelta(hours=rh,minutes=rm)
            if nominal is not None:
                launch=min((launch+timedelta(days=offset) for offset in (-1,0,1)),
                           key=lambda value:abs((value-nominal).total_seconds()))
    lat=int(line[55:62])/10000;lon=int(line[63:71])/10000
    profile={'provider':'NOAA_IGRA2','station_id':station,'nominal_time':nominal.isoformat() if nominal else None,
             'launch_time':launch.isoformat() if launch else None,'launch_latitude':lat,'launch_longitude':lon,
             'provider_message_id':f'IGRA2/{station}/{date:%Y%m%d}/{hour:02d}/{release:04d}',
             'pressure_source':line[37:45].strip(),'nonpressure_source':line[46:54].strip(),
             'release_date_basis':'nearest_nominal_day_inferred' if nominal is not None and launch else 'header_calendar_day',
             'release_time_raw':release,'levels':[]}
    return profile,count


def _level(line,actual_launch):
    if len(line)<51:raise ValueError('Truncated IGRA level.')
    if line[0] not in '123' or line[1] not in '012':raise ValueError('Invalid IGRA level type.')
    pressure=_number(line,10,15);pflag=line[15];zflag=line[21];tflag=line[27]
    if pressure is None or not 1<=pressure<=120000 or pflag not in ' AB':return None
    level={'pressure_pa':pressure,'igra_qc':{'PFLAG':pflag,'ZFLAG':zflag,'TFLAG':tflag,
                                         'LVLTYP1':line[0],'LVLTYP2':line[1]},'omitted_fields':[]}
    missing={}
    for name,first,last in [('ETIME',4,8),('GPH',17,21),('TEMP',23,27),('RH',29,33),
                            ('DPDP',35,39),('WDIR',41,45),('WSPD',47,51)]:
        raw=int(line[first-1:last])
        if raw in (-8888,-9999):missing[name]='quality_removed' if raw==-8888 else 'source_missing'
    level['igra_qc']['missing_fields']=missing
    elapsed=_number(line,4,8)
    if elapsed is not None:
        minutes,seconds=divmod(elapsed,100)
        if minutes<0 or seconds>=60 or minutes*60+seconds>86400:raise ValueError('Invalid elapsed MMMSS.')
        if actual_launch:level['elapsed_seconds']=minutes*60+seconds
        else:level['omitted_fields'].append('elapsed_time_without_precise_release')
    temperature=_number(line,23,27);height=_number(line,17,21)
    if temperature is not None and tflag in ' AB' and 150<=temperature/10+273.15<=350:
        level['temperature_k']=temperature/10+273.15
    if height is not None and zflag in ' AB' and -1000<=height<=100000:
        level['geopotential_height_m']=height
    rh=_number(line,29,33);dpdp=_number(line,35,39)
    if 'temperature_k' in level:
        kwargs={}
        if dpdp is not None and 0<=dpdp/10<=150 and 150<=level['temperature_k']-dpdp/10<=350:
            kwargs['dewpoint_k']=level['temperature_k']-dpdp/10
        elif rh is not None and 0<=rh<=1000:kwargs['relative_humidity']=rh/1000
        if kwargs:
            try:
                q=specific_humidity(pressure,temperature_k=level['temperature_k'],**kwargs)
                if 0<=q<=.2:level['specific_humidity']=q
                else:level['omitted_fields'].append('humidity_range')
            except ValueError:level['omitted_fields'].append('humidity_conversion_invalid')
    direction=_number(line,41,45);speed=_number(line,47,51)
    if speed==0:
        level.update(u_ms=0.,v_ms=0.)
    elif direction is not None and speed is not None and 0<=direction<=360 and 0<=speed/10<=200:
        level.update(wind_direction_deg=direction,wind_speed_ms=speed/10)
    # No speed-of-ascent estimate or standard atmosphere supplies omega or a missing value.
    if not any(key in level for key in ('temperature_k','specific_humidity','geopotential_height_m','wind_speed_ms','u_ms')):
        return None
    return level


def iter_profiles(path, *, years=(2021,2022), report=None, expected_station=None):
    """Read one bounded compressed source; retain incomplete actual pressure levels."""
    report=report if report is not None else Counter()
    with _stream(path) as stream:
        lines=iter(_lines(stream))
        for header in lines:
            if not header.strip():continue
            if len(header)<36 or not header.startswith('#'):
                raise ValueError('Malformed IGRA structural header.')
            year=int(header[13:17]);count=int(header[32:36]);wanted=year in years
            if not 0<=count<=10000:raise ValueError('Invalid IGRA level count.')
            if expected_station is not None and header[1:12]!=expected_station:
                raise ValueError('IGRA archive/header station identity mismatch.')
            profile,_=_header(header) if wanted else (None,count)
            for _ in range(count):
                try:line=next(lines)
                except StopIteration as exc:raise ValueError('Truncated IGRA sounding.') from exc
                if line.startswith('#'):raise ValueError('IGRA level count does not match header.')
                if wanted:
                    level=_level(line,profile['launch_time'] is not None)
                    if level is not None:profile['levels'].append(level)
                    else:report['omitted_levels']+=1
            if not wanted:continue
            if not profile['levels'] or not (profile['launch_time'] or profile['nominal_time']):
                report['profiles_without_values_or_time']+=1;continue
            if not -90<=profile['launch_latitude']<=90 or not -180<=profile['launch_longitude']<=180:
                report['profiles_without_position']+=1;continue
            report['profiles']+=1
            yield profile


def normalized_records(profile, *, acquired_at, archive_sha256, format_sha256):
    records=normalize_profile(profile,acquired_at=acquired_at,availability_mode='assumed_latency',latency_minutes=120)
    for record in records:
        level=profile['levels'][record['sequence']]
        record.update(provider_qc=level['igra_qc'],omitted_igra_fields=level['omitted_fields'],
                      pressure_source=profile['pressure_source'],nonpressure_source=profile['nonpressure_source'],
                      archive_sha256=archive_sha256,format_sha256=format_sha256,
                      release_date_basis=profile['release_date_basis'],assumed_latency_minutes=120,
                      actual_available_at=None,instrument_height_m=None,
                      position_limitation='header_station_position_no_balloon_drift_or_historical_fixed_station_relocation')
    return records


def group_split(records):
    if not records:return None
    lower=min(utc(row['observed_at']) for row in records)
    upper=max(utc(row['observed_at']) for row in records)
    guard=timedelta(hours=42)
    if START<=lower and upper<TRAIN_END-guard:return 'train'
    if TRAIN_END+guard<=lower and upper<VAL_END-guard:return 'validation'
    if VAL_END+guard<=lower and upper<END:return 'test'
    return None


def acquire(cache,output,*,network=False,station_limit=32,minimum_train_profiles=64):
    if type(station_limit) is not int or not 1<=station_limit<=128:
        raise ValueError('IGRA station limit is 1–128.')
    if type(minimum_train_profiles) is not int or not 1<=minimum_train_profiles<=732:
        raise ValueError('Invalid train-only sounding admission threshold.')
    cache=Path(cache).absolute();output=Path(output).absolute();cache.mkdir(parents=True,exist_ok=True)
    if output.exists():
        old=read_json(output/'manifest.json')
        if old['station_limit']!=station_limit or old['minimum_train_profiles']!=minimum_train_profiles:
            raise ValueError('Published IGRA admission differs from request.')
        for source in old['sources']:
            if sha256(source['path'])!=source['sha256']:raise ValueError('IGRA source changed.')
        if sha256(output/'observations.jsonl')!=old['observations_sha256']:
            raise ValueError('IGRA normalized observations changed.')
        return old
    def fetch(url,name,limit):
        path=cache/name
        used=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
        if used>MAX_CACHE:raise ValueError('IGRA 16 GiB cache budget exhausted.')
        held=path.stat().st_size if path.exists() else 0
        left=MAX_CACHE-used+held
        if left<=0:raise ValueError('IGRA 16 GiB cache budget exhausted.')
        receipt=download(url,path,max_bytes=min(limit,left),network=network)
        return path,receipt
    catalog,catalog_receipt=fetch(BASE+'igra2-station-list.txt','igra2-station-list.txt',20_000_000)
    format_path,format_receipt=fetch(FORMAT_URL,'igra2-data-format.txt',1_000_000)
    list_path,list_receipt=fetch(LIST_FORMAT_URL,'igra2-list-format.txt',1_000_000)
    sources=[{'path':str(path),'role':'format_or_inventory',**receipt} for path,receipt in
             ((catalog,catalog_receipt),(format_path,format_receipt),(list_path,list_receipt))]
    candidates=spatial_candidates(station_inventory(catalog));selected=[];failures=[]
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.igra-acquire-',dir=output.parent) as temporary:
        stage=Path(temporary);database=stage/'unique.sqlite';size=0
        with sqlite3.connect(database) as db, (stage/'observations.jsonl').open('wb') as result:
            db.execute('CREATE TABLE profiles(id TEXT PRIMARY KEY,hash TEXT)')
            for candidate in candidates:
                station=candidate['id']
                print(json.dumps({'stage':'igra','station':station,'accepted':len(selected)}),flush=True)
                try:
                    archive,receipt=fetch(station_url(station),station+'-data.txt.zip',MAX_SOURCE)
                    before=sha256(archive);train_ids=set();admission_report=Counter()
                    # This pass never normalizes or examines validation/test values.
                    for profile in iter_profiles(archive,years=(2021,),report=admission_report,expected_station=station):
                        records=normalized_records(profile,acquired_at=receipt['acquired_at'],
                            archive_sha256=before,format_sha256=format_receipt['sha256'])
                        if (records and min(utc(r['observed_at']) for r in records)>=START and
                                max(utc(r['observed_at']) for r in records)<TRAIN_END):
                            train_ids.add(records[0]['profile_id'])
                    train_count=len(train_ids)
                    if train_count<minimum_train_profiles:
                        failures.append({'station':station,'reason':'train_profile_coverage','train_profiles':train_count})
                        continue
                except ValueError as exc:
                    if 'budget' in str(exc):raise
                    failures.append({'station':station,'reason':'source_or_parser_error','detail':str(exc)})
                    continue
                source={'path':str(archive),'role':'input_target_train_norm',**receipt}
                sources.append(source);selected.append(dict(candidate,train_profiles=train_count,admission_qc=dict(admission_report)))
                report=Counter()
                for profile in iter_profiles(archive,report=report,expected_station=station):
                    records=normalized_records(profile,acquired_at=receipt['acquired_at'],
                        archive_sha256=before,format_sha256=format_receipt['sha256'])
                    split=group_split(records)
                    if split is None:continue
                    pid=records[0]['profile_id'];fingerprint=digest(records)
                    old=db.execute('SELECT hash FROM profiles WHERE id=?',(pid,)).fetchone()
                    if old:
                        if old[0]!=fingerprint:raise ValueError('Conflicting IGRA profile version.')
                        continue
                    db.execute('INSERT INTO profiles VALUES (?,?)',(pid,fingerprint))
                    for record in records:
                        record['group_split']=split
                        line=(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n').encode()
                        if size+len(line)>MAX_OUTPUT:raise ValueError('IGRA normalized JSONL budget exhausted.')
                        result.write(line);size+=len(line)
                if sha256(archive)!=before:raise ValueError('IGRA archive changed during processing.')
                if len(selected)>=station_limit:break
        if len(selected)<station_limit:
            atomic_json(cache/'admission-failure.json',{'requested':station_limit,'selected':selected,'failures':failures})
            raise ValueError('Insufficient stations passing train-only IGRA admission.')
        # The duplicate index is an implementation detail; scalar provenance is retained in JSONL.
        database.unlink()
        manifest={'schema':'igra-observation-archive-1','provider':'NOAA_IGRA2','data_kind':'real',
                  'station_limit':station_limit,'minimum_train_profiles':minimum_train_profiles,
                  'stations':selected,'failures':failures,'sources':sources,
                  'requested_years':[2021,2022],'archive_layout':'full_period_of_record_filtered_to_requested_years',
                  'admission_period':[START.isoformat(),TRAIN_END.isoformat()],
                  'source_roles':{'input':'NOAA_IGRA2','target':'NOAA_IGRA2','norm':'train_NOAA_IGRA2',
                                  'static':None,'external_verification':'ERA5_separate_only'},
                  'group_guard_hours':84,'availability':'assumed_120_minutes_not_historical_receipt',
                  'omega_supported':False,'balloon_drift_supported':False,
                  'observations_sha256':sha256(stage/'observations.jsonl'),'bytes':size,
                  'scientific_acceptance':'pending_real_profile_operator_and_training_checks'}
        atomic_json(stage/'manifest.json',manifest);stage.rename(output)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('acquire',))
    parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--allow-network',action='store_true');parser.add_argument('--station-limit',type=int,default=32)
    parser.add_argument('--minimum-train-profiles',type=int,default=64)
    args=parser.parse_args(argv)
    acquire(args.cache,args.output,network=args.allow_network,station_limit=args.station_limit,
            minimum_train_profiles=args.minimum_train_profiles)


if __name__=='__main__':main()
