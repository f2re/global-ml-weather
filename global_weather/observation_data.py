"""Bounded native-station research data; no reanalysis or invented sensor heights."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from .pipeline.dataset import utc
from .pipeline.io import atomic_json, digest, read_json, sha256
from .providers.ghcnh import read_records, qc_good

VARIABLES = ('t2m', 'td2m', 'u10', 'v10', 'surface_pressure', 'mslp')
NATIVE_VARIABLES = ('station_temperature', 'station_dew_point', 'station_eastward_wind',
                    'station_northward_wind', 'station_pressure', 'station_mean_sea_level_pressure')
UNITS = ('K', 'K', 'm s-1', 'm s-1', 'Pa', 'Pa')
START = utc('2021-01-01T00:00:00Z')
END = utc('2023-01-01T00:00:00Z')
TRAIN_END = utc('2022-01-01T00:00:00Z')
VAL_END = utc('2022-07-01T00:00:00Z')
HOURS = int((END-START).total_seconds()/3600)


def _sources(cache):
    if isinstance(cache,(list,tuple)):
        return sorted({path for root in cache for path in _sources(root)})
    root = Path(cache)
    if root.is_symlink():
        raise ValueError('Cache symlinks are forbidden.')
    paths = sorted(root.rglob('GHCNh_*_202[12].psv'))
    if not paths:
        paths = sorted(root.rglob('*.sqlite')) + sorted(root.rglob('*.jsonl'))
    if not paths:
        raise ValueError('No GHCNh observation cache found.')
    return paths


def _records(path):
    if path.is_symlink() or 'era5' in str(path).lower():
        raise ValueError('Only immutable GHCNh observation sources are admitted.')
    if path.suffix == '.psv':
        receipt = read_json(path.with_suffix('.psv.receipt.json'))
        if receipt.get('sha256') != sha256(path):
            raise ValueError('GHCNh source hash mismatch.')
        records, _ = read_records(path, acquired_at=receipt['acquired_at'],
                                  latency_minutes=60, start=START.isoformat(), end=END.isoformat())
        yield from records
    elif path.suffix == '.sqlite':
        with sqlite3.connect(path.resolve().as_uri()+'?mode=ro', uri=True) as connection:
            for row in connection.execute('SELECT record FROM observations ORDER BY id'):
                yield json.loads(row[0])
    elif path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError('Unsupported observation cache format.')


def _checked(record):
    if record.get('provider') != 'NOAA_GHCNh' or record.get('valid') is not True:
        return None
    variable = record.get('variable')
    if variable not in VARIABLES:
        return None
    v = VARIABLES.index(variable)
    if record.get('units') != UNITS[v] or not math.isfinite(record.get('value', math.nan)):
        raise ValueError('Invalid native observation units or value.')
    quality = record.get('provider_qc')
    if not isinstance(quality, dict) or not quality:
        raise ValueError('Missing provider QC provenance.')
    if any(not qc_good(q.get('Quality_Code', ''), q.get('Source_Code', '')) or
           q.get('Measurement_Code', '').strip() in {'E','D','I'} for q in quality.values()):
        return None
    identity = record.get('observation_id', '').split('/')
    if len(identity) != 4 or identity[0] != 'GHCNh':
        raise ValueError('Unknown native observation identity.')
    observed, available = utc(record['observed_at']), utc(record['available_at'])
    if available < observed:
        raise ValueError('Observation available before measurement.')
    # Trailing bins (t-1h,t]: exact hour measurements belong to that hour.
    seconds = (observed-START).total_seconds()
    hour = math.ceil(seconds/3600)
    if not 0 <= hour < HOURS:
        return None
    lat, lon = record['latitude'], record['longitude']
    if not math.isfinite(lat+lon) or abs(lat)>90 or abs(lon)>180:
        raise ValueError('Invalid station geometry.')
    return identity[1], hour, v, available, observed


def _unique(cache, database, *, train_only=False, retained=None):
    sources=[]
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE observations (id TEXT PRIMARY KEY, revision INTEGER, record TEXT, hash TEXT)')
        db.execute('CREATE TABLE seen_versions (id TEXT PRIMARY KEY, hash TEXT)')
        for path in _sources(cache):
            if train_only and path.suffix == '.psv' and not path.name.endswith('_2021.psv'):
                continue
            before = sha256(path)
            for record in _records(path):
                if train_only and not START <= utc(record['observed_at']) < TRAIN_END:
                    continue
                record_identity=record.get('observation_id','').split('/')
                if retained is not None and (len(record_identity)<2 or record_identity[1] not in retained):
                    continue
                if record.get('provider')=='NOAA_GHCNh' and record.get('revision',0)!=0:
                    raise ValueError('Pilot supports revision zero only; issue-aware revisions required.')
                if record.get('provider')=='NOAA_GHCNh' and record.get('valid') is not True:
                    raise ValueError('Revoked observation requires issue-aware revision handling.')
                key=record['observation_id'];fingerprint=digest(record)
                previous=db.execute('SELECT hash FROM seen_versions WHERE id=?',(key,)).fetchone()
                if previous is not None:
                    if previous[0]!=fingerprint:
                        raise ValueError('Conflicting observation version.')
                    continue
                db.execute('INSERT INTO seen_versions VALUES (?,?)',(key,fingerprint))
                checked = _checked(record)
                if checked is None:
                    continue
                station, _, _, _, observed = checked
                if (train_only and not START <= observed < TRAIN_END) or (retained is not None and station not in retained):
                    continue
                revision = record.get('revision', 0)
                if type(revision) is not int or revision < 0:
                    raise ValueError('Invalid observation revision.')
                key=record['observation_id']; fingerprint=digest(record)
                old=db.execute('SELECT revision,hash FROM observations WHERE id=?',(key,)).fetchone()
                if old and old[0] == revision and old[1] != fingerprint:
                    raise ValueError('Conflicting observation version.')
                if old is None or revision > old[0]:
                    db.execute('INSERT OR REPLACE INTO observations VALUES (?,?,?,?)',
                               (key,revision,json.dumps(record,allow_nan=False),fingerprint))
            if sha256(path) != before:
                raise ValueError('Observation source changed during preparation.')
            sources.append({'path':str(path.resolve()),'sha256':before,'role':'observations'})
    return sources


def admit(cache, output, *, station_limit=1024, minimum_coverage=.05):
    """Freeze station selection using 2021 QC and hourly completeness only."""
    import tempfile
    if type(station_limit) is not int or not 1<=station_limit<=1024 or not 0<minimum_coverage<=1:
        raise ValueError('Invalid station admission bounds.')
    stats={}; geometry={}
    with tempfile.TemporaryDirectory(prefix='station-admission-',dir=Path(output).parent) as temp:
        database=Path(temp)/'unique.sqlite'
        sources=_unique(cache,database,train_only=True)
        with sqlite3.connect(database) as db:
            for (text,) in db.execute('SELECT record FROM observations'):
                record=json.loads(text); station,hour,v,_,_=_checked(record)
                stats.setdefault(station,[set() for _ in VARIABLES])[v].add(hour)
                position=(record['latitude'],record['longitude'])
                if station in geometry and geometry[station]!=position:
                    raise ValueError('Station location changed; operator metadata needed.')
                geometry[station]=position
    train_hours=int((TRAIN_END-START).total_seconds()/3600)
    accepted=[]; rejected=[]
    for station in sorted(stats):
        coverage=[len(hours)/train_hours for hours in stats[station]]
        row={'id':station,'latitude':geometry[station][0],'longitude':geometry[station][1],
             'train_coverage':coverage,'instrument_height_m':None}
        if max(coverage)>=minimum_coverage and len(accepted)<station_limit:
            accepted.append(row)
        else:
            rejected.append(dict(row,reason='train_coverage_below_threshold' if max(coverage)<minimum_coverage else 'station_limit'))
    if not accepted:
        raise ValueError('No stations pass train-only admission.')
    report={'schema':'native-station-admission-1','stations':accepted,'rejected':rejected,
            'minimum_coverage':minimum_coverage,'period':[START.isoformat(),TRAIN_END.isoformat()],
            'selection':'any_native_variable_hourly_coverage_train_only_then_station_id',
            'sources':sources,'variables':list(NATIVE_VARIABLES)}
    atomic_json(output,report)
    return report


def _prepare(cache, admission_path, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    admission=read_json(admission_path); stations=admission['stations']
    if admission.get('schema')!='native-station-admission-1' or not 1<=len(stations)<=1024:
        raise ValueError('Invalid frozen station admission.')
    indices={station['id']:i for i,station in enumerate(stations)}
    if len(indices)!=len(stations):
        raise ValueError('Duplicate station IDs.')
    sources=_unique(cache,output/'observations.sqlite',retained=set(indices))
    for source in admission['sources']:
        if sha256(source['path'])!=source['sha256']:
            raise ValueError('Admission train data changed.')
    shape=(HOURS,len(stations),len(VARIABLES))
    sums=np.zeros(shape,dtype=np.float32); counts=np.zeros(shape,dtype=np.uint16)
    # Max availability within each bin allows causal masking even for delays.
    available=np.zeros(shape,dtype=np.int64)
    with sqlite3.connect(output/'observations.sqlite') as db:
        for (text,) in db.execute('SELECT record FROM observations'):
            record=json.loads(text); station,hour,v,ready,_=_checked(record); s=indices[station]
            if (record['latitude'],record['longitude'])!=(stations[s]['latitude'],stations[s]['longitude']):
                raise ValueError('Station geometry differs from frozen admission.')
            if counts[hour,s,v]==65535:
                raise ValueError('Hourly measurement count exceeds limit.')
            sums[hour,s,v]+=record['value']; counts[hour,s,v]+=1
            available[hour,s,v]=max(available[hour,s,v],math.ceil((ready-START).total_seconds()))
    mask=counts>0
    np.divide(sums,counts,out=sums,where=mask)
    train_end=int((TRAIN_END-START).total_seconds()/3600)
    mean=[];std=[]
    for v in range(len(VARIABLES)):
        measured=sums[:train_end,:,v][mask[:train_end,:,v]].astype(np.float64)
        if len(measured)<2 or not np.isfinite(measured).all() or measured.std()<=0:
            raise ValueError('Insufficient variable train statistics; no invented norms.')
        mean.append(float(measured.mean()));std.append(float(measured.std()))
    # Normalize unique hourly observations once, never overlapping issue windows.
    norm={'schema':'native-station-norm-1','mean':mean,'std':std,'units':UNITS,
          'variables':NATIVE_VARIABLES,'period':[START.isoformat(),TRAIN_END.isoformat()],
          'method':'unique_hourly_station_variable_means_population_std','data_kind':'real'}
    atomic_json(output/'norm.json',norm)
    np.savez(output/'hourly.npz',values=sums,mask=mask,available_seconds=available)
    samples=[]
    for split,begin,end in [('train',START,TRAIN_END),('validation',TRAIN_END,VAL_END),('test',VAL_END,END)]:
        issue=begin+timedelta(days=1)
        while issue+timedelta(hours=72)<end:
            samples.append({'issue':issue.isoformat(),'hour':int((issue-START).total_seconds()/3600),'split':split})
            issue+=timedelta(days=1)
    manifest={'schema':'native-station-dataset-1','data_kind':'real','stations':stations,
              'variables':NATIVE_VARIABLES,'units':UNITS,'samples':samples,'guard_hours':84,
              'input_hours':12,'lead_hours':list(range(3,73,3)),
              'operator':'station_ID_readout_hourly_mean_trailing_(t-1h,t]; native_instrument_height_unknown',
              'temporal_mismatch':'hourly_mean_target_not_instantaneous_point; no_height_conversion',
              'source_roles':{'input':'GHCNh','target':'GHCNh','norm':'train_GHCNh','static':None,'external_verification':'ERA5_separate_frozen_model_only'},
              'availability':'assumed_archive_latency_not_historical_receipt',
              'global_profile_acceptance':'BLOCKED_no_vertical_targets_and_unknown_instrument_heights',
              'sources':sources,'admission_sha256':sha256(admission_path),
              'norm_sha256':sha256(output/'norm.json'),'hourly_sha256':sha256(output/'hourly.npz'),
              'records_sha256':sha256(output/'observations.sqlite')}
    atomic_json(output/'dataset.json',manifest)
    return read_json(output/'dataset.json')


def prepare(cache, admission_path, output):
    """Publish complete data atomically; reuse only unchanged frozen artifacts."""
    import tempfile
    output=Path(output)
    if output.exists():
        dataset=ObservationDataset(output)
        if dataset.manifest['admission_sha256']!=sha256(admission_path):
            raise ValueError('Frozen admission changed; use a new dataset ID.')
        for source in dataset.manifest['sources']:
            if sha256(source['path'])!=source['sha256']:
                raise ValueError('Frozen raw observation source changed.')
        if {str(p.resolve()) for p in _sources(cache)}!={s['path'] for s in dataset.manifest['sources']}:
            raise ValueError('Observation source inventory changed.')
        return dataset.manifest
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.observation-prepare-',dir=output.parent) as temp:
        stage=Path(temp)/'dataset'
        manifest=_prepare(cache,admission_path,stage)
        stage.rename(output)
    return manifest


class ObservationDataset:
    def __init__(self,path):
        path=Path(path);self.root=path if path.is_dir() else path.parent
        self.manifest=read_json(self.root/'dataset.json')
        if self.manifest.get('schema')!='native-station-dataset-1':
            raise ValueError('Unknown observation dataset schema.')
        roles={'input':'GHCNh','target':'GHCNh','norm':'train_GHCNh','static':None,
               'external_verification':'ERA5_separate_frozen_model_only'}
        if (self.manifest.get('data_kind')!='real' or self.manifest.get('source_roles')!=roles or
                self.manifest.get('variables')!=list(NATIVE_VARIABLES) or
                self.manifest.get('units')!=list(UNITS)):
            raise ValueError('Observation source roles, units or native variables differ.')
        for file,key in [('norm.json','norm_sha256'),('hourly.npz','hourly_sha256'),('observations.sqlite','records_sha256')]:
            if sha256(self.root/file)!=self.manifest[key]:
                raise ValueError('Observation dataset artifact changed.')
        self.norm=read_json(self.root/'norm.json');self.mean=np.asarray(self.norm['mean'],dtype=np.float32)
        self.std=np.asarray(self.norm['std'],dtype=np.float32)
        if (self.norm.get('schema')!='native-station-norm-1' or self.norm.get('data_kind')!='real' or
                self.norm.get('method')!='unique_hourly_station_variable_means_population_std' or
                self.norm.get('period')!=[START.isoformat(),TRAIN_END.isoformat()] or
                self.norm.get('variables')!=list(NATIVE_VARIABLES) or self.norm.get('units')!=list(UNITS) or
                self.mean.shape!=(6,) or self.std.shape!=(6,) or not np.isfinite(self.mean).all() or
                not np.isfinite(self.std).all() or (self.std<=0).any()):
            raise ValueError('Invalid or non-train observation normalization.')
        with np.load(self.root/'hourly.npz',allow_pickle=False) as data:
            self.values=data['values'];self.mask=data['mask'];self.available=data['available_seconds']
        self.samples=self.manifest['samples']
        station_ids=[station['id'] for station in self.manifest['stations']]
        if not 1<=len(station_ids)<=1024 or len(set(station_ids))!=len(station_ids):
            raise ValueError('Invalid native station inventory.')
        shape=(HOURS,len(self.manifest['stations']),6)
        if (self.values.shape!=shape or self.mask.shape!=shape or self.available.shape!=shape or
                self.mask.dtype!=bool or not np.isfinite(self.values[self.mask]).all()):
            raise ValueError('Invalid hourly observation arrays.')
        boundaries={'train':(START,TRAIN_END),'validation':(TRAIN_END,VAL_END),'test':(VAL_END,END)}
        seen=set()
        for row in self.samples:
            issue=utc(row['issue']);split=row['split']
            if split not in boundaries:
                raise ValueError('Unknown chronological split.')
            begin,end=boundaries[split]
            if (type(row['hour']) is not int or not begin<=issue-timedelta(hours=12) or not issue+timedelta(hours=72)<end or
                    row['hour']!=(issue-START).total_seconds()/3600 or issue in seen):
                raise ValueError('Observation windows cross the frozen chronological split.')
            seen.add(issue)
        self.latlon=np.asarray([[s['latitude'],s['longitude']] for s in self.manifest['stations']],dtype=np.float32)

    def subset(self,split):
        return [i for i,row in enumerate(self.samples) if row['split']==split]

    def sample(self,index):
        row=self.samples[index];hour=row['hour'];inputs=np.arange(hour-11,hour+1)
        targets=hour+np.arange(3,73,3)
        x=self.values[inputs].copy();y=self.values[targets].copy()
        xm=self.mask[inputs] & (self.available[inputs]<=hour*3600);ym=self.mask[targets].copy()
        nx=np.where(xm,(x-self.mean)/self.std,0).astype(np.float32)
        ny=np.where(ym,(y-self.mean)/self.std,0).astype(np.float32)
        return {'input':np.where(xm,x,0),'normalized_input':nx,'input_mask':xm,'target':y,
                'normalized_target':ny,'target_mask':ym,'issue':row['issue'],'latlon':self.latlon}


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('admit','prepare'))
    parser.add_argument('--source-cache',required=True,action='append');parser.add_argument('--output',required=True)
    parser.add_argument('--admission');parser.add_argument('--station-limit',type=int,default=1024)
    parser.add_argument('--minimum-coverage',type=float,default=.05)
    args=parser.parse_args(argv)
    if args.command=='admit':
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        admit(args.source_cache,args.output,station_limit=args.station_limit,minimum_coverage=args.minimum_coverage)
    else:
        if not args.admission:parser.error('--admission required for preparation')
        prepare(args.source_cache,args.admission,args.output)


if __name__=='__main__':
    main()
