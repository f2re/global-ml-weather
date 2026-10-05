"""Complete ERA5 daily requests, separate accumulations and immutable receipts."""
from __future__ import annotations
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile
import zipfile
from ..connectors.acquire import era5_request
from ..pipeline.io import atomic_json, digest, sha256

CDS_URL = "https://cds.climate.copernicus.eu/api"


def requests_for(plan):
    result=[];grid=plan["spec"]["source_grid_degrees"]
    from ..pipeline.dataset import utc
    times={day:{'state':set(),'rain':set()} for day in plan['era5_dates']}
    for sample in plan['samples']:
        issue=utc(sample['issue_time'])
        for h in range(0,plan['spec']['horizon_hours']+1,plan['spec']['step_hours']):
            when=issue+timedelta(hours=h);times[when.date().isoformat()]['state'].add(when.strftime('%H:00'))
        for h in range(1,plan['spec']['horizon_hours']+1):
            when=issue+timedelta(hours=h);times[when.date().isoformat()]['rain'].add(when.strftime('%H:00'))
    for day in plan["era5_dates"]:
        p=era5_request(day);p["request"]["grid"]=[grid,grid]
        p['request']['time']=sorted(times[day]['state'])
        if times[day]['state']:result.append(dict(id=day+"-pressure",**p))
        s=era5_request(day,pressure_levels=False)
        s["request"]["grid"]=[grid,grid]
        wanted=s["request"]["variable"][:8]
        # Do not request unused SST/ice/snow; accumulations have a separate time type.
        for group,variables in (("surface",[v for v in wanted if v!="total_precipitation"]),("rain",["total_precipitation"])):
            if not times[day]["rain" if group=="rain" else "state"]:continue
            result.append({"id":day+"-"+group,"dataset":s["dataset"],"request":dict(s["request"],variable=variables,time=sorted(times[day]["rain" if group=="rain" else "state"]))})
    day=plan["era5_dates"][0];s=era5_request(day,pressure_levels=False)
    result.append({"id":"static","dataset":s["dataset"],"request":dict(s["request"],
                   variable=["geopotential","land_sea_mask"],time=["00:00"],grid=[grid,grid])})
    return result


def retrieve(query, cache, *, key=None, network=False, max_bytes=2*1024**3, cancelled=None):
    cache=Path(cache);cache.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() for p in (cache,*cache.parents)):raise ValueError("Ссылка вместо кэша ERA5.")
    identity=digest(query);directory=cache/identity;receipt=directory/"receipt.json"
    if receipt.exists():
        row=json.loads(receipt.read_text())
        if row.get("request")!=query:raise ValueError("Кэш ERA5 относится к другому запросу.")
        for f in row["files"]:
            path=directory/f["name"]
            if Path(f["name"]).name!=f["name"] or path.is_symlink() or not path.is_file() or sha256(path)!=f["sha256"]:
                raise ValueError("Кэш ERA5 повреждён.")
        return [directory/f["name"] for f in row["files"]],row
    if directory.exists():raise ValueError("Незавершённая публикация ERA5 без паспорта.")
    if not network:raise ValueError("Нет локальной ERA5; разрешите сеть при запуске.")
    try:import cdsapi
    except ImportError as exc:raise ValueError("Установите зависимости .[autonomous] для доступа ERA5.") from exc
    if cancelled and cancelled():raise InterruptedError("Загрузка отменена.")
    with tempfile.TemporaryDirectory(prefix=".cds-",dir=cache) as tmp:
        tmp=Path(tmp);raw=tmp/"response.nc"
        try:
            options={"url":CDS_URL,"quiet":True,"debug":False,"progress":False,"timeout":60,"retry_max":3}
            if key:options["key"]=key
            cdsapi.Client(**options).retrieve(query["dataset"],query["request"],str(raw))
        except Exception as exc:
            # Provider exception strings may include credential-bearing requests.
            raise ValueError("ERA5 не получена. Проверьте токен CDS, условия обоих наборов, сеть и квоту; повторите запуск.") from None
        if cancelled and cancelled():raise InterruptedError("Загрузка отменена.")
        if not raw.is_file() or not 0<raw.stat().st_size<=max_bytes:raise ValueError("Ответ ERA5 пуст или превышает бюджет.")
        unpack=tmp/"unpacked";unpack.mkdir()
        if zipfile.is_zipfile(raw):
            with zipfile.ZipFile(raw) as archive:
                members=archive.infolist()
                if not members or len(members)>16 or sum(m.file_size for m in members)>max_bytes:
                    raise ValueError("Архив ERA5 превышает бюджет распаковки.")
                for i,m in enumerate(members):
                    if m.is_dir() or not m.filename.endswith('.nc'):raise ValueError("Неожиданный файл в архиве ERA5.")
                    target=unpack/f"field-{i}.nc"
                    with archive.open(m) as src,target.open("xb") as dst:
                        import shutil
                        shutil.copyfileobj(src,dst,256*1024)
        else:raw.rename(unpack/"field-0.nc")
        files=sorted(unpack.glob("*.nc"))
        import xarray as xr
        for path in files:
            with xr.open_dataset(path) as ds:
                if not ds.data_vars:raise ValueError("Ответ CDS не содержит метеорологических полей.")
        row={"request":query,"acquired_at":datetime.now(timezone.utc).isoformat(),
             "files":[{"name":p.name,"sha256":sha256(p),"bytes":p.stat().st_size} for p in files],
             "source_truth_verified":False}
        atomic_json(unpack/"receipt.json",row);unpack.rename(directory)
    return [directory/f["name"] for f in row["files"]],row
