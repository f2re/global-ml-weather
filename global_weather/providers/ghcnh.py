"""NOAA GHCNh v1.1 PSV. See retained official documentation, 2026-03-10.

Six integrated QC fields are admitted conservatively. Reanalysis never fills
missing observations. Modeled historical latency is explicitly not measured.
"""
from __future__ import annotations
import csv
from datetime import datetime, timedelta, timezone
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
from .http import download
from ..pipeline.dataset import utc

BASE = "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/"
DOC = BASE + "doc/ghcnh_DOCUMENTATION.pdf"
STATION_PATTERN = r"[A-Z0-9-]{11}"
QC_ISD = {"313", "314", "315", "322", "335", "343", "344", "346"}
QC_COMMON = {"220", "221", "222", "223", "347", "348"}
FIELDS = {"temperature":("t2m","K",1,273.15),
          "dew_point_temperature":("td2m","K",1,273.15),
          "station_level_pressure":("surface_pressure","Pa",100,0),
          "sea_level_pressure":("mslp","Pa",100,0)}


def station_url(station, year):
    if not re.fullmatch(STATION_PATTERN, station) or type(year) is not int or not 1800 <= year <= datetime.now(timezone.utc).year:
        raise ValueError("Неверный идентификатор GHCNh или год.")
    return BASE + f"access/by-year/{year}/psv/GHCNh_{station}_{year}.psv"


def station_index(path):
    result = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            identity = line[:11]
            lat,lon = float(line[12:20]),float(line[21:30])
            if not re.fullmatch(STATION_PATTERN,identity) or not math.isfinite(lat+lon) or abs(lat)>90 or abs(lon)>180:
                continue
            result.append(dict(id=identity, latitude=lat, longitude=lon, name=line[41:71].strip()))
        except ValueError:
            continue
    if not result:
        raise ValueError("Каталог GHCNh не распознан.")
    return result


def qc_good(code, source):
    code = code.strip()
    if not code:
        return True  # No integrated failure flag, not an independent certification.
    if source in QC_ISD:
        return code in {"0","1","4","5","9"}
    if source in QC_COMMON:
        return code == "1"
    return False  # Unknown and compound flags must not pass by substring matching.


def read_records(path, *, acquired_at, latency_minutes, start=None, end=None):
    if type(latency_minutes) is not int or not 0 <= latency_minutes <= 720:
        raise ValueError("Предполагаемая задержка должна быть в пределах 0–720 минут.")
    acquired = utc(acquired_at)
    lower = utc(start) if start else None; upper = utc(end) if end else None
    counts = Counter(); selected = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="|")
        names = reader.fieldnames or []
        aliases = {k:next((n for n in choices if n in names),None) for k,choices in {
            "id":("STATION","Station_ID"),"lat":("LATITUDE","Latitude"),
            "lon":("LONGITUDE","Longitude"),"elevation":("ELEVATION","Elevation")}.items()}
        if not all(aliases.values()) or "temperature" not in names or not ("DATE" in names or all(k in names for k in ("Year","Month","Day","Hour","Minute"))):
            raise ValueError("Неизвестная схема GHCNh PSV; адаптер остановлен.")
        for row in reader:
            counts["rows"] += 1
            try:
                if row.get("DATE"):
                    t = datetime.fromisoformat(row["DATE"].replace("Z","+00:00"))
                    observed = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
                else:
                    observed = datetime(*(int(row[k]) for k in ("Year","Month","Day","Hour","Minute")),tzinfo=timezone.utc)
                if observed > acquired:
                    raise ValueError("Измерение позже загрузки архива.")
                if lower and observed < lower or upper and observed > upper:
                    continue
                identity=row[aliases["id"]]; lat=float(row[aliases["lat"]]); lon=float(row[aliases["lon"]])
                if not re.fullmatch(STATION_PATTERN,identity) or not math.isfinite(lat+lon) or abs(lat)>90 or abs(lon)>180:
                    raise ValueError("Некорректная станция.")
            except (ValueError,TypeError,KeyError):
                counts["invalid_row"] += 1; continue
            values={}
            for name in (*FIELDS,"wind_speed","wind_direction"):
                try:
                    value=float(row.get(name,""))
                    if not math.isfinite(value) or not qc_good(row.get(name+"_Quality_Code",""),row.get(name+"_Source_Code","")):
                        counts["qc_rejected"]+=1;continue
                    if row.get(name+"_Measurement_Code","").strip() in {"E","D","I"}:
                        counts["estimated_rejected"]+=1;continue
                    if name in ("temperature","dew_point_temperature") and not -100 <= value <= 65:
                        raise ValueError()
                    if name in ("station_level_pressure","sea_level_pressure") and not 100 <= value <= 1200:
                        raise ValueError()
                    if name=="wind_speed" and not 0<=value<=150 or name=="wind_direction" and not 0<=value<=360:
                        raise ValueError()
                    values[name]=value
                except (ValueError,TypeError):
                    counts["missing_or_range"]+=1
            physical = [(out,v*factor+offset,unit,[name]) for name,(out,unit,factor,offset) in FIELDS.items() if (v:=values.get(name)) is not None]
            speed=values.get("wind_speed"); direction=values.get("wind_direction")
            if speed==0 or speed is not None and direction is not None:
                rad=math.radians(direction or 0)
                physical.extend([(name,value,"m s-1",["wind_speed","wind_direction"]) for name,value in
                                 (("u10",-speed*math.sin(rad)),("v10",-speed*math.cos(rad)))])
            for name,value,unit,dependencies in physical:
                record=dict(observation_id=f"GHCNh/{identity}/{observed.isoformat()}/{name}",source="station",
                            variable=name,value=value,units=unit,latitude=lat,longitude=lon,
                            observed_at=observed.isoformat(),available_at=(observed+timedelta(minutes=latency_minutes)).isoformat(),
                            acquired_at=acquired.isoformat(),actual_available_at=None,
                            availability_basis="assumed_archive_latency_not_historical_receipt",
                            assumed_latency_minutes=latency_minutes,provider="NOAA_GHCNh",revision=0,valid=True,
                            height_reference="nominal_surface_height_requires_station_metadata",
                            provider_qc={n:{k:row.get(n+"_"+k,"") for k in ("Quality_Code","Measurement_Code","Source_Code","Report_Type","Source_Station_ID")} for n in dependencies})
                try:
                    elevation=float(row[aliases["elevation"]])
                    if math.isfinite(elevation) and elevation != -999.9:record["elevation_m"]=elevation
                except (ValueError,TypeError):pass
                key=record["observation_id"]
                if key in selected and selected[key] != record:
                    raise ValueError("Противоречащие дубликаты GHCNh: требуется согласование источника.")
                if key in selected:counts["identical_duplicates"]+=1
                selected[key]=record
    records=sorted(selected.values(),key=lambda x:(x["observed_at"],x["observation_id"]))
    return records,dict(counts,records=len(records),historical_availability_known=False)


def fetch_station(station, year, cache, **kwargs):
    path=Path(cache)/f"GHCNh_{station}_{year}.psv"
    receipt=download(station_url(station,year),path,**kwargs)
    return path,receipt
