"""Finite masked fields, causal provenance, explicit retrieval assumptions."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntFlag
import hashlib
import json
import re
import numpy as np
from .catalog import CATALOG, SCHEMA


class QC(IntFlag):
    MISSING = 1
    DOMAIN = 2
    CONDITIONS = 4
    AMBIGUOUS = 8
    NO_SOLUTION = 16
    LOW_SENSITIVITY = 32
    BOUNDARY = 64


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Требуется время с часовым поясом.')
    return dt.astimezone(timezone.utc)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Требуется фактическая SHA256 исходника.')
    return value


@dataclass(frozen=True)
class Field:
    values: np.ndarray
    valid: np.ndarray
    quantity: str
    units: str
    grid_id: str
    observed_at: str
    available_at: str
    source_sha256: str
    metadata: dict = field(default_factory=dict)
    uncertainty: np.ndarray | None = None

    def __post_init__(self):
        x = np.asarray(self.values)
        m = np.asarray(self.valid)
        if x.dtype.kind not in 'fi' or not x.size or x.shape != m.shape or m.dtype != bool:
            raise ValueError('Численный массив и Boolean-маска должны совпадать.')
        if not np.isfinite(x[m]).all():
            raise ValueError('Пригодное измерение не может быть NaN или бесконечностью.')
        if not all(isinstance(s, str) and s for s in (self.quantity, self.units, self.grid_id)):
            raise ValueError('Нужны величина, единицы и идентификатор сетки.')
        if utc(self.available_at) < utc(self.observed_at):
            raise ValueError('Готовность раньше измерения.')
        require_hash(self.source_sha256)
        canonical(self.metadata)
        u = None if self.uncertainty is None else np.asarray(self.uncertainty, dtype=float)
        if u is not None and (u.shape != x.shape or (m & ((u < 0) | np.isinf(u))).any()):
            raise ValueError('Некорректная погрешность. Неизвестная погрешность обозначается NaN.')
        object.__setattr__(self, 'values', x.astype(float, copy=True))
        object.__setattr__(self, 'valid', m.copy())
        object.__setattr__(self, 'uncertainty', None if u is None else u.copy())

    def require(self, quantity, units):
        if self.quantity != quantity or self.units != units:
            raise ValueError(f'Нужно {quantity} [{units}], получено {self.quantity} [{self.units}].')
        return self

    def dependency(self):
        return dict(sha256=self.source_sha256, quantity=self.quantity,
                    observed_at=utc(self.observed_at).isoformat(),
                    available_at=utc(self.available_at).isoformat())


@dataclass(frozen=True)
class Product:
    name: str
    method: str
    values: np.ndarray
    valid: np.ndarray
    qc: np.ndarray
    uncertainty: np.ndarray
    metadata: dict

    def __post_init__(self):
        spec = CATALOG[self.name]
        x, m, q, u = map(np.asarray, (self.values, self.valid, self.qc, self.uncertainty))
        if self.method not in spec.methods or x.shape != m.shape or x.shape != q.shape or x.shape != u.shape:
            raise ValueError('Неверная схема продукции.')
        if m.dtype != bool or q.dtype.kind not in 'iu' or not np.isfinite(x[m]).all():
            raise ValueError('Неверные значения или маски продукции.')
        if (m & ((x < spec.limits[0]) | (x > spec.limits[1]) | (q != 0))).any():
            raise ValueError('Пригодное значение не прошло физическую границу или QC.')
        if (m & ((u < 0) | np.isinf(u))).any():
            raise ValueError('Некорректная погрешность продукции.')
        validate_metadata(self.metadata, name=self.name, method=self.method)


def aligned(*fields, max_skew_seconds=None):
    first = fields[0]
    if any(f.grid_id != first.grid_id or f.values.shape != first.values.shape for f in fields):
        raise ValueError('Сначала согласуйте сетки и геометрию; неявное пересэмплирование запрещено.')
    if max_skew_seconds is not None:
        times = [utc(f.observed_at) for f in fields]
        if (max(times)-min(times)).total_seconds() > max_skew_seconds:
            raise ValueError('Каналы не относятся к согласованному сроку.')
    return np.logical_and.reduce([f.valid for f in fields])


def condition(field, reference):
    field.require('eligibility_mask', '1')
    aligned(field, reference, max_skew_seconds=60.)
    if not np.isin(field.values[field.valid], [0, 1]).all():
        raise ValueError('Условия задаются явной маской 0/1.')
    return field.valid & (field.values == 1)


def result(name, method, values, valid, qc, primary, dependencies, *, available_at,
           source, platform, data_kind, assumptions, uncertainty=None, attributes=None):
    """Do not change acquisition time when processing or reusing a product."""
    if data_kind not in ('real', 'synthetic') or not platform:
        raise ValueError('Укажите происхождение и платформу.')
    if data_kind == 'real' and any(f.metadata.get('data_kind') != 'real' for f in dependencies):
        raise ValueError('Нельзя выдать синтетический или неидентифицированный вход за реальный.')
    deps = [f.dependency() for f in dependencies]
    ready = utc(available_at)
    if not deps or max(utc(d['available_at']) for d in deps) > ready:
        raise ValueError('Продукт не может быть готов раньше своих входов.')
    spec = CATALOG[name]
    x, m, flags = np.asarray(values, float), np.asarray(valid, bool).copy(), np.asarray(qc, np.uint16).copy()
    bad = ~np.isfinite(x) | (x < spec.limits[0]) | (x > spec.limits[1])
    flags[bad & m] |= int(QC.DOMAIN)
    m &= ~bad & (flags == 0)
    x = np.where(m, x, np.nan)
    u = np.full(x.shape, np.nan) if uncertainty is None else np.where(m, uncertainty, np.nan)
    metadata = dict(schema=SCHEMA, product=name, method=method, units=spec.units,
                    source=source, platform=platform, data_kind=data_kind, grid_id=primary.grid_id,
                    observed_at=utc(primary.observed_at).isoformat(), available_at=ready.isoformat(),
                    dependencies=deps, assumptions=list(assumptions),
                    uncertainty_scope='conditional_input_error_only; missing means unknown',
                    meteorologically_validated=False, primary_sha256=primary.source_sha256, attributes=attributes or {})
    return Product(name, method, x, m, flags, u, metadata)


def validate_metadata(meta, *, name, method, issue_time=None):
    spec = CATALOG[name]
    if meta.get('schema') != SCHEMA or meta.get('product') != name or meta.get('method') != method or meta.get('units') != spec.units:
        raise ValueError('Метаданные продукта не совпадают с реестром.')
    if meta.get('data_kind') not in ('real', 'synthetic') or not meta.get('platform') or not meta.get('grid_id'):
        raise ValueError('Нет происхождения продукции.')
    observed, ready = utc(meta['observed_at']), utc(meta['available_at'])
    if ready < observed:
        raise ValueError('Готовность раньше измерения.')
    deps = meta.get('dependencies')
    require_hash(meta.get('primary_sha256'))
    if not meta.get('source'): raise ValueError('Нет источника продукции.')
    if not isinstance(deps, list) or not deps or not isinstance(meta.get('assumptions'),list) or not meta['assumptions']:
        raise ValueError('Нужны исходники и допущения алгоритма.')
    if meta['primary_sha256'] not in {d.get('sha256') for d in deps}:
        raise ValueError('Главный источник отсутствует среди зависимостей.')
    for d in deps:
        require_hash(d['sha256'])
        if utc(d['observed_at']) > utc(d['available_at']) or utc(d['available_at']) > ready:
            raise ValueError('Нарушена причинность входов продукции.')
    if issue_time is not None and (ready > utc(issue_time) or observed > utc(issue_time)):
        raise ValueError('Будущий или ещё недоступный продукт.')
