"""Canonical identities and time roles, independent of request boundaries.

Monthly holdouts are research partitions, not proof of prospective forecast
independence. Normalisation and weight-selection periods still require C5.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import hashlib
import json
import re

from ..vertical import PRESSURE_HPA

UTC = timezone.utc
SPLIT_VERSION = 'monthly-ranked-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def utc_time(value):
    if not isinstance(value, str) or len(value) > 40:
        raise ValueError('Требуется время ISO 8601 с часовым поясом.')
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError as exc:
        raise ValueError('Неверное время ISO 8601.') from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Для времени требуется часовой пояс.')
    result = result.astimezone(UTC)
    if result.year < 2 or result.year > 9998:
        raise ValueError('Недостаточно календарного запаса для истории и целей.')
    return result


def calendar_date(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
        raise ValueError('Дата должна иметь формат ГГГГ-ММ-ДД.')
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError('Такой календарной даты не существует.') from exc
    if not 2 <= result.year <= 9998:
        raise ValueError('Недостаточно календарного запаса для истории и целей.')
    return result


def check_range(start, end, today):
    first, last = calendar_date(start), calendar_date(end)
    if first > last:
        raise ValueError('Начальная дата позже конечной.')
    if last > today:
        raise ValueError('Обучающий период не может включать будущие даты UTC.')
    return first.toordinal(), last.toordinal()


def default_contract():
    # This reserves the scientific identity; it does not allocate weights.
    return {
        'schema': 'continuous-model-contract-1',
        'architecture': 'adaptive', 'mesh_level': 2, 'hidden': 16,
        'latent_slots': 8, 'pressure_hpa': list(PRESSURE_HPA),
        'history_hours': 12, 'horizon_hours': 72, 'step_hours': 3,
        'issue_hours': [0, 6, 12, 18], 'normalization': None,
        'device_policy': 'auto_cuda_first',
        'split': {'version': SPLIT_VERSION, 'seed': 17, 'train_months': 8,
                  'validation_months': 2, 'test_months': 2},
    }


@lru_cache(maxsize=256)
def _year_roles(year, seed):
    # Fixed quotas avoid years without any test month. Hash ranking avoids
    # reserving the same meteorological season in every year.
    order = sorted(range(1, 13), key=lambda month: hashlib.sha256(
        f'{SPLIT_VERSION}/{seed}/{year:04d}-{month:02d}'.encode()).digest())
    result = {}
    for rank, month in enumerate(order):
        result[month] = 'train' if rank < 8 else 'validation' if rank < 10 else 'test'
    return result


def month_role(when, seed=17):
    return _year_roles(when.year, seed)[when.month]


def temporal_role(issue, history_hours=12, horizon_hours=72, seed=17):
    """Erode each role at boundaries by the full closed dependency interval.

    Checking the exact endpoint is deliberately conservative for open input
    windows. Shared endpoints belonging to different roles are not admitted.
    """
    nominal = month_role(issue, seed)
    lower = issue - timedelta(hours=history_hours)
    upper = issue + timedelta(hours=horizon_hours)
    cursor = lower
    while True:
        if month_role(cursor, seed) != nominal:
            return nominal, 'guard'
        following = (datetime(cursor.year + 1, 1, 1, tzinfo=UTC) if cursor.month == 12
                     else datetime(cursor.year, cursor.month + 1, 1, tzinfo=UTC))
        if following > upper:
            break
        cursor = following
    return nominal, nominal


def identifier(value, label='идентификатор', maximum=128):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}', value) or len(value) > maximum:
        raise ValueError(f'Неверный {label}.')
    if '://' in value:
        raise ValueError('Сохраните идентификатор объекта, а не URL с реквизитами.')
    return value


def asset_identity(value):
    if not isinstance(value, dict) or set(value) != {'provider', 'object_id', 'revision', 'sha256'}:
        raise ValueError('Источник требует provider, object_id, revision и sha256.')
    result = {k: identifier(value[k], k, 256 if k == 'object_id' else 128)
              for k in ('provider', 'object_id', 'revision')}
    if not isinstance(value['sha256'], str) or not re.fullmatch('[a-f0-9]{64}', value['sha256']):
        raise ValueError('Неверная контрольная сумма источника.')
    result['sha256'] = value['sha256']
    return result


def sample_identity(value, contract):
    expected = {'issue_time', 'inputs', 'targets', 'modalities', 'transform_sha256'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Неизвестные или отсутствующие поля описания примера.')
    issue = utc_time(value['issue_time'])
    if issue.minute or issue.second or issue.microsecond or issue.hour not in contract['issue_hours']:
        raise ValueError('Выпуск должен совпадать со сроком постоянной программы.')
    transform = value['transform_sha256']
    if not isinstance(transform, str) or not re.fullmatch('[a-f0-9]{64}', transform):
        raise ValueError('Нужен SHA256 преобразования.')
    modalities = value['modalities']
    if not isinstance(modalities, list) or not 1 <= len(modalities) <= 64:
        raise ValueError('Нужен ограниченный список модальностей.')
    modalities = [identifier(x, 'модальность') for x in modalities]
    if len(set(modalities)) != len(modalities):
        raise ValueError('Повторная модальность.')
    result = {'schema': 'continuous-sample-1', 'issue_time': issue.isoformat(),
              'history_hours': contract['history_hours'], 'horizon_hours': contract['horizon_hours'],
              'step_hours': contract['step_hours'], 'modalities': sorted(modalities),
              'transform_sha256': transform}
    for kind in ('inputs', 'targets'):
        rows = value[kind]
        if not isinstance(rows, list) or not 1 <= len(rows) <= 4096:
            raise ValueError('Список версий источников пуст или превышает предел описания.')
        rows = [asset_identity(row) for row in rows]
        keys = [(r['provider'], r['object_id'], r['revision']) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError('Повторные или противоречащие версии одного источника.')
        result[kind] = sorted(rows, key=canonical)
    if len(canonical(result).encode()) > 2 * 1024**2:
        raise ValueError('Описание примера слишком велико; используйте версионированный блок источников.')
    return result
