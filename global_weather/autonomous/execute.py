"""One resumable chain from provider files to an independently evaluated model.

A single process owns an experiment. Completed stages seal all output files.
Credentials are supplied at runtime, never placed in an experiment manifest.
"""
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta, datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from ..pipeline.io import atomic_json, read_json, sha256, digest, reference, artifact
from ..pipeline.dataset import utc, PreparedDataset
from .plan import parse_plan


def now():return datetime.now(timezone.utc).isoformat()


class Stages:
    def __init__(self, root, identity, cancelled=None):
        self.root=Path(root);self.identity=identity;self.cancelled=cancelled or (lambda:False)
        self.path=self.root/"stages.json"
        self.state=read_json(self.path) if self.path.exists() else {"fingerprint":identity,"stages":{}}
        if self.state["fingerprint"]!=identity:raise ValueError("План изменён; создайте новый эксперимент.")

    def progress(self, stage, **details):
        if self.cancelled():raise InterruptedError("Эксперимент остановлен пользователем.")
        row={"time":now(),"stage":stage,**details}
        atomic_json(self.root/"progress.json",row)
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)

    def run(self, name, fn):
        self.progress(name)
        old=self.state["stages"].get(name)
        directory=self.root/name
        if old and old.get("status")=="completed":
            for ref in old["files"]:artifact(directory,ref)
            self.progress(name,cached=True)
            return directory,old["result"]
        if directory.exists():
            # A crash can occur after publishing a sealed directory but before
            # updating the journal. Recover only a complete, matching capsule.
            seal=directory/'stage-completion.json'
            if not seal.is_file():
                raise ValueError("Неоднозначный этап без паспорта: "+name+". Сохранённые данные не перезаписаны.")
            complete=read_json(seal)
            if complete.get('fingerprint')!=self.identity or complete.get('name')!=name:
                raise ValueError('Опубликованный этап относится к другому плану.')
            for ref in complete['files']:artifact(directory,ref)
            self.state['stages'][name]={k:complete[k] for k in ('status','files','result','finished')}
            atomic_json(self.path,self.state)
            return directory,complete['result']
        self.state["stages"][name]={"status":"running","started":now()};atomic_json(self.path,self.state)
        # Temporary data belong only to this attempt. Raw cache is outside it.
        with tempfile.TemporaryDirectory(prefix=".stage-"+name+"-",dir=self.root) as temp:
            temp=Path(temp);result=fn(temp)
            if self.cancelled():raise InterruptedError("Эксперимент остановлен.")
            files=[reference(temp,p) for p in sorted(temp.rglob("*")) if p.is_file()]
            complete={"status":"completed","files":files,"result":result,"finished":now()}
            atomic_json(temp/'stage-completion.json',dict(complete,fingerprint=self.identity,name=name))
            temp.rename(directory)
            self.state["stages"][name]=complete
            atomic_json(self.path,self.state)
        return directory,result


def spatial_candidates(stations, count):
    """Spread catalog candidates spatially, never advertise this as global coverage."""
    cells={}
    for s in stations:
        key=(min(11,int((s['latitude']+90)/15)),min(23,int((s['longitude']+180)/15)))
        cells.setdefault(key,[]).append(s)
    for rows in cells.values():rows.sort(key=lambda s:(s['id'][2] not in 'MW',s['id']))
    ordered=[]
    for depth in range(5):
        for key in sorted(cells):
            if depth<len(cells[key]):ordered.append(cells[key][depth]['id'])
    if len(ordered)>count:
        import numpy as np
        indices=np.linspace(0,len(ordered)-1,count,dtype=int)
        ordered=[ordered[i] for i in indices]
    return ordered


def execute(plan_path, root, *, cds_key=None, cancelled=None):
    from ..providers.ghcnh import read_records, fetch_station, station_index, BASE
    from ..providers.http import download
    from ..providers.era5 import requests_for, retrieve
    from ..pipeline.era5 import prepare_targets, prepare_static
    from ..pipeline.runner import train, evaluate
    from ..pipeline.fit import fit_normalization
    from ..pipeline.coverage import check_coverage
    from ..pipeline.io import read_arrays
    from ..observations import Variable
    from ..vertical import PRESSURE_HPA
    root=Path(root).absolute()
    if any(p.is_symlink() for p in (root,*root.parents)):raise ValueError("Ссылка вместо каталога эксперимента.")
    root.mkdir(parents=True,exist_ok=True)
    experiment=parse_plan(read_json(plan_path));plan=experiment.checked()
    with (root/"experiment.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError("Эксперимент уже выполняется.") from exc
        stages=Stages(root,plan["fingerprint"],cancelled)
        if (root/'plan.json').exists() and read_json(root/'plan.json')!=plan:
            raise ValueError('План эксперимента изменён.')
        atomic_json(root/"plan.json",plan)
        cache=root/"cache";cache.mkdir(exist_ok=True)
        budget=int(experiment.max_download_gib*1024**3)
        def remaining(cached=None):
            used=sum(p.stat().st_size for p in cache.rglob('*') if p.is_file())
            if used>budget:raise ValueError('Исчерпан бюджет локального кэша.')
            # Reusing the same immutable object does not allocate its bytes again.
            held=0
            if cached is not None and cached.exists():
                held=sum(p.stat().st_size for p in cached.rglob('*') if p.is_file()) if cached.is_dir() else cached.stat().st_size
            left=budget-used+held
            if left<1:raise ValueError('Исчерпан бюджет локального кэша.')
            if shutil.disk_usage(root).free<256*1024**2:raise ValueError('Недостаточно свободного места.')
            return left
        def acquire_stations(out):
            chosen=list(experiment.stations)
            candidates=chosen
            if not chosen:
                catalog=cache/"ghcnh-stations.txt"
                download(BASE+"doc/ghcnh-station-list.txt",catalog,network=experiment.network,
                         max_bytes=min(20_000_000,remaining(catalog)),cancelled=stages.cancelled)
                candidates=spatial_candidates(station_index(catalog),experiment.station_count*5)
            start=utc(experiment.start)-timedelta(hours=12);end=utc(experiment.end)
            years=range(start.year,end.year+1);receipts=[];failures=[];accepted=[]
            database=out/"observations.sqlite"
            with sqlite3.connect(database) as db:
                db.execute('CREATE TABLE observations (id TEXT PRIMARY KEY, observed TEXT, available TEXT, record TEXT)')
                for station in candidates:
                    rows=[];station_receipts=[]
                    stages.progress('ghcnh',station=station,accepted_stations=len(accepted))
                    try:
                        for year in years:
                            path,receipt=fetch_station(station,year,cache/'ghcnh',network=experiment.network,
                                        max_bytes=min(100_000_000,remaining(cache/'ghcnh'/f"GHCNh_{station}_{year}.psv")),cancelled=stages.cancelled)
                            records,quality=read_records(path,acquired_at=receipt['acquired_at'],
                                latency_minutes=experiment.latency_minutes,start=start.isoformat(),end=end.isoformat())
                            rows.extend(records);station_receipts.append(dict(station=station,year=year,**receipt,quality=quality))
                        if not rows:raise ValueError('Нет пригодных данных выбранного периода.')
                    except ValueError as exc:
                        failures.append({'station':station,'reason':str(exc)})
                        if chosen:raise
                        continue
                    accepted.append(station);receipts.extend(station_receipts)
                    for row in rows:
                        db.execute('INSERT INTO observations VALUES (?,?,?,?)',(row['observation_id'],row['observed_at'],row['available_at'],json.dumps(row,ensure_ascii=False,allow_nan=False)))
                    if not chosen and len(accepted)>=experiment.station_count:break
                if len(accepted)<(len(chosen) or experiment.station_count):raise ValueError('Недостаточно пригодных станций выбранного периода. Измените их число или задайте список.')
                db.execute('CREATE INDEX times ON observations(observed,available)')
                count=db.execute('SELECT count(*) FROM observations').fetchone()[0]
            result={'stations':accepted,'records':count,'requested_station_count':len(chosen) or experiment.station_count,
                    'failures':failures,'sources':receipts,'historical_availability_known':False}
            atomic_json(out/'receipt.json',result);return result
        station_dir,station_report=stages.run('ghcnh',acquire_stations)
        def acquire_era5(out):
            files={'pressure':[],'surface':[],'static':[]};receipts=[]
            queries=requests_for(plan)
            for number,query in enumerate(queries):
                stages.progress('era5',request=query['id'],number=number+1,total=len(queries))
                paths,receipt=retrieve(query,cache/'era5',key=cds_key,network=experiment.network,
                                     max_bytes=min(2*1024**3,remaining(cache/'era5'/digest(query))),cancelled=stages.cancelled)
                category='static' if query['id']=='static' else 'pressure' if query['id'].endswith('pressure') else 'surface'
                files[category].extend([reference(root,p) for p in paths]);receipts.append(receipt)
            if len(files['static'])!=1:raise ValueError('Статические поля должны быть в одном согласованном файле.')
            result={'files':files,'receipts':receipts};atomic_json(out/'inventory.json',result);return result
        _,era=stages.run('era5',acquire_era5)
        source_paths={k:[artifact(root,ref) for ref in refs] for k,refs in era['files'].items()}
        by_date={}
        for receipt in era['receipts']:
            query=receipt['request'];name=query['id']
            if name=='static':continue
            day=name[:10];kind='pressure' if name.endswith('pressure') else 'surface'
            by_date.setdefault(day,{'pressure':[],'surface':[]})[kind].extend(
                cache/'era5'/digest(query)/f['name'] for f in receipt['files'])
        def prepare(out):
            units={'t2m':'K','td2m':'K','u10':'m s-1','v10':'m s-1','surface_pressure':'Pa','mslp':'Pa'}
            registry={name:asdict(Variable(unit,0.,1.,'surface')) for name,unit in units.items()}
            atomic_json(out/'registry.json',registry)
            prepare_static(source_paths['static'][0],out/'static.npz',mesh_level=experiment.mesh_level)
            samples=[];coverage=[]
            with sqlite3.connect(station_dir/'observations.sqlite') as db:
                for number,sample in enumerate(plan['samples']):
                    stages.progress('prepare',sample=sample['id'],number=number+1,total=len(plan['samples']))
                    issue=utc(sample['issue_time']);name=sample['id'];observations=out/(name+'.jsonl');target=out/(name+'.npz')
                    records=db.execute('SELECT record FROM observations WHERE observed>? AND observed<=? AND available<=? ORDER BY observed,id',
                                        ((issue-timedelta(hours=12)).isoformat(),issue.isoformat(),issue.isoformat())).fetchall()
                    if not records:raise ValueError('Нет доступных наблюдений выпуска '+name)
                    observations.write_text(''.join(row[0]+'\n' for row in records),encoding='utf-8')
                    if observations.stat().st_size>32*1024**2:raise ValueError('Вход выпуска превышает бюджет JSONL.')
                    needed={(issue+timedelta(hours=h)).date().isoformat() for h in range(experiment.horizon_hours+1)}
                    inputs={kind:[p for day in sorted(needed) for p in by_date[day][kind]] for kind in ('pressure','surface')}
                    prepare_targets(inputs['pressure'],inputs['surface'],target,issue_time=issue.isoformat(),
                                    mesh_level=experiment.mesh_level,horizon_hours=experiment.horizon_hours,
                                    step_hours=experiment.step_hours,confirm_utc=True,precipitation_kind='hourly_increment')
                    checked=check_coverage(read_arrays(target),experiment.minimum_target_coverage)
                    coverage_path=out/(name+'.coverage.json')
                    atomic_json(coverage_path,checked)
                    coverage.append({'sample':name,'records':len(records),'targets':reference(out,coverage_path)})
                    samples.append(dict(sample,observations=reference(out,observations),targets=reference(out,target),
                        provenance={'observations':'NOAA GHCNh; see provider receipts',
                                    'targets':'ECMWF ERA5 CDS; immutable request and checksum receipts',
                                    'availability':f'assumed archive latency {experiment.latency_minutes} minutes; NOT historical receipt',
                                    'target_operator':'bilinear_at_cell_centres_not_conservative'}))
            from ..grid import build_grid
            manifest={'schema':'global-weather-dataset-1','data_kind':'real','mesh_level':experiment.mesh_level,
                      'grid_fingerprint':build_grid(experiment.mesh_level).fingerprint,'pressure_hpa':list(PRESSURE_HPA),
                      'step_hours':experiment.step_hours,'horizon_hours':experiment.horizon_hours,
                      'registry':reference(out,out/'registry.json'),'static':reference(out,out/'static.npz'),
                      'normalization':None,'samples':samples,'license':'NOAA GHCNh; Copernicus ERA5 terms accepted by operator',
                      'target_coverage':{'minimum':experiment.minimum_target_coverage},
                      'availability_mode':'assumed_archive_latency_not_historical_receipt'}
            atomic_json(out/'unscaled.json',manifest);atomic_json(out/'coverage.json',coverage)
            # Bundled verified reference norms retain their own source period.
            from ..import_climatology import main as import_norms
            import_norms(['--bundled','--output',str(out/'base-norms.json')])
            fit_normalization(out/'unscaled.json',out/'dataset.json',base_path=out/'base-norms.json')
            ds=PreparedDataset(out/'dataset.json');report=ds.validate();atomic_json(out/'validation.json',report)
            return {'dataset':'dataset.json','samples':len(samples),'records':sum(c['records'] for c in coverage)}
        prepared,_=stages.run('prepared',prepare)
        dataset=prepared/'dataset.json';run=root/'training';cfg=experiment.train_config()
        stages.progress('training')
        if (run/'report.json').exists():
            report=read_json(run/'report.json')
            ds=PreparedDataset(dataset)
            if report['dataset_fingerprint']!=ds.fingerprint:raise ValueError('Обучение относится к другой выборке.')
            from ..pipeline.runner import load_trained
            load_trained(ds,run)
        else:
            if run.exists() and not (run/'latest.json').exists():
                # An interrupted first epoch has no usable checkpoint. Preserve it.
                run.rename(root/('interrupted-training-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')))
            train(dataset,run,cfg,resume=(run/'latest.json').exists(),progress=stages.progress,cancelled=stages.cancelled)
        def final_test(out):
            stages.progress('evaluate')
            return evaluate(dataset,run,out/'evaluation.json',split='test')
        _,evaluation=stages.run('evaluation',final_test)
        result={'status':'completed_research','data_kind':'real','plan_fingerprint':plan['fingerprint'],
                'dataset':reference(root,dataset),'evaluation':reference(root,root/'evaluation/evaluation.json'),
                'training':reference(root,run/'report.json'),'stations':station_report['stations'],
                'availability':'assumed_archive_latency_not_historical_receipt','meteorologically_validated':False}
        atomic_json(root/'result.json',result);stages.progress('completed');return result
