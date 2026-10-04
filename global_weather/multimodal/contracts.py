"""Физические каналы, фиксированные нормы и причинный отбор кадров.

Контракт не калибрует прибор. Каждая оптическая последовательность должна быть
заранее приведена на неизменную локальную сетку. Пролёты не дублируются по часам.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import re
import numpy as np
import torch
from torch import Tensor

SOURCES = ("electro_l", "arktika_m", "meteor_msu_mr", "meteor_mtvza")


def utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(dt, datetime) or dt.utcoffset() is None:
        raise ValueError("Требуется ISO-время с часовым поясом.")
    return dt.astimezone(timezone.utc)


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def hash_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Нужна SHA256 исходника.")
    return value


@dataclass(frozen=True)
class Channel:
    id: str
    quantity: str
    units: str
    mean: float | None
    std: float | None
    calibration_id: str

    def __post_init__(self):
        allowed = {"brightness_temperature": "K", "reflectance": "1",
                   "surface_reflectance": "1"}
        if (not self.id or not self.calibration_id or self.quantity not in allowed
                or self.units != allowed[self.quantity]):
            raise ValueError("Неизвестный канал, калибровка, единицы или нормы.")
        if self.mean is None and self.std is None: return
        if self.mean is None or self.std is None or not np.isfinite([self.mean,self.std]).all() or self.std <= 0:
            raise ValueError("Неизвестные или некорректные нормы.")


@dataclass(frozen=True)
class Sensor:
    id: str
    source: str
    platform: str
    kind: str
    channels: tuple[Channel, ...]
    normalization_sha256: str | None
    fit_end: str | None
    data_kind: str

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", self.id) or not self.platform:
            raise ValueError("Неверное имя адаптера или платформы.")
        if self.source not in SOURCES or self.kind not in ("imager", "microwave"):
            raise ValueError("Неизвестный источник или тип адаптера.")
        if (self.source == "meteor_mtvza") != (self.kind == "microwave"):
            raise ValueError("МТВЗА требует отдельного микроволнового адаптера.")
        if not 1 <= len(self.channels) <= 64 or len({x.id for x in self.channels}) != len(self.channels):
            raise ValueError("Нужны уникальные физические каналы.")
        if self.data_kind not in ("real", "synthetic"):
            raise ValueError("Неизвестное происхождение норм.")
        unfit = [c.mean is None for c in self.channels]
        if any(unfit):
            if not all(unfit) or self.normalization_sha256 is not None or self.fit_end is not None:
                raise ValueError("Неполный черновой реестр норм.")
        else:
            hash_value(self.normalization_sha256)
            utc(self.fit_end)

    @classmethod
    def from_dict(cls, item):
        data = dict(item)
        data["channels"] = tuple(Channel(**c) for c in data["channels"])
        return cls(**data)

    def require_normalization(self):
        if any(c.mean is None for c in self.channels) or self.fit_end is None:
            raise ValueError("Сначала рассчитайте нормы по обучающей части; инженерного замещения нет.")
        return self

    @property
    def measurement_signature(self):
        return fingerprint({"id":self.id,"source":self.source,"platform":self.platform,"kind":self.kind,
            "data_kind":self.data_kind,"channels":[{"id":c.id,"quantity":c.quantity,"units":c.units,
                        "calibration_id":c.calibration_id} for c in self.channels]})

    @property
    def signature(self):
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class Sequence:
    """Размеры values/valid: T,C,H,W; остальные пиксельные поля: T,H,W.

    Координаты: H,W, одни для всей последовательности. Эпохи времени: T.
    Микроволновые каналы заранее согласованы по пятну в пределах одного Sensor.
    Для разных пятен нужны разные Sensor и отдельные последовательности.
    """
    sensor_id: str
    values: Tensor
    valid: Tensor
    observed_unix: Tensor
    available_unix: Tensor
    latitude: Tensor
    longitude: Tensor
    view_zenith_deg: Tensor
    solar_zenith_deg: Tensor
    footprint_km: Tensor
    area_m2: Tensor
    frame_ids: tuple[str, ...]
    channel_ids: tuple[str, ...]
    grid_id: str
    geometry_reference: str
    source_sha256: str
    sensor_signature: str
    data_kind: str
    link_pixel: Tensor | None = None
    link_cell: Tensor | None = None
    link_weight: Tensor | None = None
    link_grid_fingerprint: str | None = None
    link_reference: str | None = None

    def validate(self, sensor: Sensor, *, n_cells=None, grid_fingerprint=None):
        if self.sensor_id != sensor.id or self.sensor_signature != sensor.measurement_signature:
            raise ValueError("Последовательность и реестр прибора не совпадают.")
        if self.data_kind != sensor.data_kind or self.channel_ids != tuple(c.id for c in sensor.channels):
            raise ValueError("Изменены происхождение или порядок физических каналов.")
        x, mask = self.values, self.valid
        if x.ndim != 4 or mask.dtype != torch.bool or mask.shape != x.shape or not x.is_floating_point():
            raise ValueError("Нужны values[T,C,H,W] и отдельная Boolean-маска.")
        t,c,h,w = x.shape
        if not 1 <= t <= 48 or c != len(sensor.channels) or not 1 <= h*w <= 262144:
            raise ValueError("Размер последовательности превышает проверяемый предел.")
        if sensor.kind=="imager" and (h<8 or w<8):
            raise ValueError("Для четырёхмасштабного кодировщика нужен фрагмент не меньше 8×8.")
        if not torch.isfinite(x[mask]).all():
            raise ValueError("Неконечное пригодное измерение.")
        if self.observed_unix.shape != (t,) or self.available_unix.shape != (t,):
            raise ValueError("Нужны времена каждого кадра.")
        times = self.observed_unix
        if (not torch.isfinite(times).all() or not torch.isfinite(self.available_unix).all()
                or (self.available_unix < times).any() or (torch.diff(times) <= 0).any()):
            raise ValueError("Неверные времена съёмки или готовности.")
        if len(self.frame_ids) != t or len(set(self.frame_ids)) != t:
            raise ValueError("Нужны уникальные идентификаторы версий кадров.")
        for ident in self.frame_ids: hash_value(ident)
        if not self.grid_id or not self.geometry_reference:
            raise ValueError("Геометрия должна быть согласована до кодирования.")
        hash_value(self.source_sha256)
        spatial_support=mask.any(0).any(0)
        for data in (self.latitude, self.longitude, self.area_m2):
            if data.shape != (h,w) or not torch.isfinite(data[spatial_support]).all():
                raise ValueError("Неверная геометрия или площадь исходного пикселя.")
        if ((self.latitude[spatial_support].abs() > 90).any() or (self.longitude[spatial_support].abs() > 180).any()
                or (self.area_m2[spatial_support] <= 0).any()):
            raise ValueError("Недопустимые координаты или площади.")
        support = mask.any(1)
        for data in (self.view_zenith_deg,self.footprint_km):
            if data.shape != (t,h,w) or not torch.isfinite(data[support]).all():
                raise ValueError("Неизвестная геометрия пригодного пикселя.")
        if (((self.view_zenith_deg[support] < 0) | (self.view_zenith_deg[support] >= 90)).any()
                or (self.footprint_km[support] <= 0).any()):
            raise ValueError("Неверные углы или размер пятна.")
        if self.solar_zenith_deg.shape != (t,h,w):
            raise ValueError("Неверная форма солнечной геометрии.")
        known = torch.isfinite(self.solar_zenith_deg)
        if ((self.solar_zenith_deg[known] < 0) | (self.solar_zenith_deg[known] > 180)).any():
            raise ValueError("Неверный солнечный угол.")
        for j,ch in enumerate(sensor.channels):
            observed = x[:,j][mask[:,j]]
            if ch.units == "K" and (observed <= 0).any():
                raise ValueError("Яркостная температура должна быть положительной.")
        if sensor.kind == "microwave":
            if any(v is None for v in (self.link_pixel,self.link_cell,self.link_weight)) or not self.link_reference:
                raise ValueError("МТВЗА требует явных связей антенного пятна, не одного ближайшего пикселя.")
            a,b,v = self.link_pixel,self.link_cell,self.link_weight
            if a.dtype != torch.long or b.dtype != torch.long or a.ndim != 1 or not 0 < len(a) <= 2_000_000 or a.shape != b.shape or a.shape != v.shape:
                raise ValueError("Неверная таблица антенных связей.")
            if ((a < 0) | (a >= h*w)).any() or not torch.isfinite(v).all() or (v <= 0).any():
                raise ValueError("Недопустимые антенные связи.")
            if (b < 0).any() or (n_cells is not None and (b >= n_cells).any()):
                raise ValueError("Антенные связи выходят за сетку.")
            hash_value(self.link_grid_fingerprint)
            if grid_fingerprint is not None and grid_fingerprint != self.link_grid_fingerprint:
                raise ValueError("Пятно связано с другой глобальной сеткой.")
            pairs = torch.stack((a,b),1)
            if len(torch.unique(pairs,dim=0)) != len(a):
                raise ValueError("Дублирование антенной связи.")
            sums = v.new_zeros(h*w).index_add(0,a,v)
            if not torch.allclose(sums[sums>0],torch.ones_like(sums[sums>0]),atol=1e-5):
                raise ValueError("Сумма весов каждого пятна должна равняться единице.")
            if (support.any(0).flatten() & (sums == 0)).any():
                raise ValueError("Пригодное пятно не имеет пространственной поддержки.")
        return self

    def causal(self, issue_time, sensor: Sensor):
        self.validate(sensor)
        issue = utc(issue_time).timestamp()
        use = ((self.observed_unix > issue-43200) & (self.observed_unix <= issue)
               & (self.available_unix <= issue))
        if not use.any(): return None
        ids = tuple(x for x,keep in zip(self.frame_ids,use.tolist()) if keep)
        changes = {key:getattr(self,key)[use] for key in ("values","valid","observed_unix","available_unix",
                    "view_zenith_deg","solar_zenith_deg","footprint_km")}
        return replace(self,**changes,frame_ids=ids)

    def to(self, device):
        return replace(self,**{k:v.to(device) for k,v in vars(self).items() if isinstance(v,Tensor)})

    def normalized(self, sensor: Sensor):
        sensor.require_normalization()
        mean = self.values.new_tensor([c.mean for c in sensor.channels])[None,:,None,None]
        std = self.values.new_tensor([c.std for c in sensor.channels])[None,:,None,None]
        mask = self.valid.clone()
        for j,c in enumerate(sensor.channels):
            if c.units == "1": mask[:,j] &= torch.isfinite(self.solar_zenith_deg) & (self.solar_zenith_deg < 90)
        x = (torch.where(mask,self.values,mean)-mean)/std
        return x,mask
