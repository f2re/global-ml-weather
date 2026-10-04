"""План загрузки. Сеть только по явному разрешению; спутники получает прежний поставщик."""
from __future__ import annotations
from datetime import timedelta
from pathlib import Path
import re
from .contracts import utc
from .io import write_json,read_json,sha256


def plan(start,end,*,stations=(),horizon_hours=72):
    start,end=utc(start),utc(end)
    if end<start or end-start>timedelta(days=31) or horizon_hours not in (3,6,12,24,48,72):
        raise ValueError('Неверный период; один план ограничен 31 сутками выпусков.')
    stations=tuple(stations)
    if len(stations)>50 or len(set(stations))!=len(stations) or any(not re.fullmatch(r'\d{11}',s) for s in stations):
        raise ValueError('Нужны уникальные идентификаторы NOAA, не более 50.')
    lower=start-timedelta(hours=12);upper=end+timedelta(hours=horizon_hours)
    jobs=[];day=lower.date()
    while day<=upper.date():
        for surface in (False,True):
            jobs.append({'kind':'era5','date':day.isoformat(),'surface':surface,
                         'output':f"era5/{day.isoformat()}-{'surface' if surface else 'pressure'}.nc"})
        day+=timedelta(days=1)
    for year in range(lower.year,upper.year+1):
        for station in stations:jobs.append({'kind':'noaa','station':station,'year':year,'output':f'noaa/{year}-{station}.csv'})
    return {'schema':'multimodal-acquisition-1','issue_start':start.isoformat(),'issue_end':end.isoformat(),
            'observation_start':lower.isoformat(),'target_end':upper.isoformat(),'horizon_hours':horizon_hours,
            'stations':list(stations),'jobs':jobs,'satellite_sources':'existing arktika-worker and SatDump; local immutable exports',
            'availability_warning':'Archive acquisition time is not historical operational availability.'}


def execute(plan_path,output,*,network=False,limit=4,offset=0):
    if not network:raise ValueError('Исполнение загрузки требует --network.')
    request=read_json(plan_path)
    expected=plan(request['issue_start'],request['issue_end'],stations=request['stations'],horizon_hours=request['horizon_hours'])
    if request!=expected:raise ValueError('План изменён: используйте только сформированные зарегистрированные задания.')
    if type(limit) is not int or not 1<=limit<=8:raise ValueError('Один запуск ограничен 1–8 запросами.')
    if type(offset) is not int or not 0<=offset<len(request['jobs']):raise ValueError('Неверное начало пакета.')
    root=Path(output).absolute()
    if root.exists() or root.is_symlink() or any(p.is_symlink() for p in root.parents):
        raise FileExistsError('Нужен новый каталог загрузки без символических ссылок.')
    root.mkdir(parents=True)
    from ..connectors.acquire import main as acquire
    results=[]
    for job in request['jobs'][offset:offset+limit]:
        path=root/job['output']
        args=['era5-download' if job['kind']=='era5' else 'noaa-isd','--network','--output',str(path)]
        args+=['--date',job['date']] if job['kind']=='era5' else ['--station',job['station'],'--year',str(job['year'])]
        if job.get('surface'):args.append('--surface')
        try:
            acquire(args);results.append({'job':job,'status':'completed','sha256':sha256(path)})
        except Exception as exc:
            results.append({'job':job,'status':'failed','error_type':type(exc).__name__})
            write_json(root/'acquisition-report.json',{'results':results,'remaining':len(request['jobs'])-offset-len(results)})
            raise
    report={'results':results,'remaining':len(request['jobs'])-offset-len(results),'status':'batch_complete_not_full_plan'}
    write_json(root/'acquisition-report.json',report)
    return report
