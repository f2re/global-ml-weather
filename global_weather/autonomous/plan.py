"""A bounded, immutable experiment specification, independent of credentials."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, asdict, field
import math
import json
from ..pipeline.dataset import utc, integer
from ..pipeline.runner import config_from_json
from ..pipeline.io import digest


@dataclass(frozen=True)
class Experiment:
    start: str = "2020-01-01T12:00:00Z"
    end: str = "2020-01-31T12:00:00Z"
    issue_interval_hours: int = 24
    horizon_hours: int = 6
    step_hours: int = 3
    mesh_level: int = 2
    source_grid_degrees: float = 2.5
    station_count: int = 8
    stations: tuple = ()
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    latency_minutes: int = 60
    archive_assumption_accepted: bool = False
    network: bool = False
    max_download_gib: float = 8.0
    minimum_target_coverage: float = 0.99
    max_runtime_hours: int = 24
    training: dict = field(default_factory=lambda: {"device":"auto","epochs":5,"hidden":16,"threads":2})

    def checked(self):
        start,end=utc(self.start),utc(self.end)
        if not start<end or (end-start).days>366 or end+timedelta(hours=self.horizon_hours)>datetime.now(timezone.utc)-timedelta(days=90):
            raise ValueError("Нужен архивный период до 366 суток, заканчивающийся не менее 90 дней назад.")
        if start.minute or start.second or start.microsecond or end.minute or end.second or end.microsecond:
            raise ValueError("Сроки выпуска должны задаваться точным часом UTC.")
        integer(self.horizon_hours,1,72);integer(self.step_hours,1,6);integer(self.issue_interval_hours,1,168)
        if self.step_hours not in (1,3,6) or self.horizon_hours%self.step_hours:
            raise ValueError("Горизонт должен быть кратен шагу 1, 3 или 6 часов.")
        integer(self.mesh_level,0,5);integer(self.station_count,1,1024);integer(self.latency_minutes,0,720)
        integer(self.max_runtime_hours,1,336)
        if self.source_grid_degrees not in (0.25,0.5,1.,2.5):
            raise ValueError("Шаг исходной ERA5: 0.25, 0.5, 1 или 2.5 градуса.")
        for number in (self.train_fraction,self.validation_fraction,self.max_download_gib,self.minimum_target_coverage):
            if type(number) not in (float,int) or not math.isfinite(number):raise ValueError("Параметры должны быть конечными числами.")
        if not 0.1<=self.train_fraction<=0.9 or not 0.05<=self.validation_fraction<=0.4 or self.train_fraction+self.validation_fraction>=0.95:
            raise ValueError("Доли выборки не оставляют независимый тест.")
        if not 0<self.max_download_gib<=1024 or not 0<self.minimum_target_coverage<=1:
            raise ValueError("Неверный бюджет или порог покрытия.")
        if self.archive_assumption_accepted is not True:
            raise ValueError("Подтвердите архивный исследовательский режим: задержка поступления моделируется, а не измерена.")
        if type(self.network) is not bool:raise ValueError("Разрешение сети должно быть логическим.")
        from ..providers.ghcnh import station_url
        if not isinstance(self.stations,(tuple,list)) or len(self.stations)>1024 or len(set(self.stations))!=len(self.stations):
            raise ValueError("Неверный список станций.")
        for station in self.stations:station_url(station,start.year)
        cfg=self.train_config()
        samples=self.samples()
        if len(samples)>10000:raise ValueError("Слишком много выпусков.")
        class Shape:
            step=self.step_hours;horizon=self.horizon_hours;n_cells=10*4**self.mesh_level+2
        cfg.validate(Shape())
        if (self.horizon_hours//self.step_hours+1)*Shape.n_cells*230*5>500*1024**2:
            raise ValueError("Цели превышают текущий предел NPZ; уменьшите сетку или горизонт.")
        dates=self.dates(samples)
        points=(int(180/self.source_grid_degrees)+1)*int(360/self.source_grid_degrees)
        state_times={utc(s['issue_time'])+timedelta(hours=h) for s in samples for h in range(0,self.horizon_hours+1,self.step_hours)}
        rain_times={utc(s['issue_time'])+timedelta(hours=h) for s in samples for h in range(1,self.horizon_hours+1)}
        estimate=points*(len(state_times)*(37*6+7)+len(rain_times)+2)*4*1.5
        if estimate>self.max_download_gib*1024**3:
            raise ValueError(f"Предварительный объём ERA5 {estimate/1024**3:.2f} ГиБ превышает бюджет.")
        spec=json.loads(json.dumps(asdict(self),allow_nan=False))
        return {"schema":"autonomous-plan-1","spec":spec,"fingerprint":digest(spec),
                "samples":samples,"era5_dates":dates,"estimated_era5_bytes":int(estimate),
                "availability":"assumed_archive_latency_not_historical_receipt","meteorologically_validated":False}

    def train_config(self):
        value=dict(self.training)
        if "horizon_hours" in value and value["horizon_hours"]!=self.horizon_hours:
            raise ValueError("Горизонт обучения должен совпадать с планом.")
        value["horizon_hours"]=self.horizon_hours
        return config_from_json(value)

    def samples(self):
        start,end=utc(self.start),utc(self.end);delta=end-start
        v=start+delta*self.train_fraction;t=start+delta*(self.train_fraction+self.validation_fraction)
        guard=timedelta(hours=self.horizon_hours+12)
        result=[];issue=start
        while issue<=end:
            split="train" if issue<v else "validation" if issue>=v+guard and issue<t else "test" if issue>=t+guard else None
            if split:result.append({"id":issue.strftime("issue-%Y%m%dT%H"),"issue_time":issue.isoformat(),"split":split})
            issue+=timedelta(hours=self.issue_interval_hours)
        if any(sum(x["split"]==part for x in result)<2 for part in ("train","validation","test")):
            raise ValueError("После защитных интервалов требуется минимум два выпуска в каждой части. Увеличьте период.")
        return result

    def dates(self,samples=None):
        dates=set()
        for s in samples or self.samples():
            issue=utc(s["issue_time"])
            dates.update((issue+timedelta(hours=h)).date().isoformat() for h in range(self.horizon_hours+1))
        return sorted(dates)


def parse_plan(data):
    if not isinstance(data,dict) or set(data)-set(Experiment.__dataclass_fields__):
        raise ValueError("Неизвестные поля плана эксперимента.")
    return Experiment(**data)
