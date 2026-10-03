"""Local-only inspection. A structural pass is never a radiometric validation."""
from pathlib import Path
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from ..lab.contracts import sha256
from ..contracts import ObservationEvent
from ..observations import SOURCES

MAX_BYTES = 32*1024*1024


def inspect_file(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink(): raise ValueError('Нет обычного входного файла.')
    size = path.stat().st_size
    if size > MAX_BYTES: raise ValueError('Файл больше предела 32 МиБ; подготовьте отдельную выборку.')
    base = dict(name=path.name, bytes=size, sha256=sha256(path), physics_verified=False, status='inspected')
    if path.suffix.lower() == '.jsonl':
        reasons, sources = Counter(), Counter()
        times = []; valid = 0; total = 0
        with path.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip(): continue
                total += 1
                try:
                    r = json.loads(line)
                    event = ObservationEvent(str(r['observation_id']), r['source'],
                       datetime.fromisoformat(r['observed_at'].replace('Z', '+00:00')),
                       datetime.fromisoformat(r['available_at'].replace('Z', '+00:00')), int(r.get('revision', 0)))
                    values = [float(r[k]) for k in ('value', 'latitude', 'longitude')]
                    if r['source'] not in SOURCES or not all(map(math.isfinite, values)): raise ValueError()
                    if abs(values[1]) > 90 or abs(values[2]) > 180 or not r.get('units') or not r.get('variable'): raise ValueError()
                    if not isinstance(r.get('valid', True), bool): raise ValueError()
                    if not r.get('valid', True): reasons['marked_missing'] += 1; continue
                    times.append(event.observed_at.astimezone(timezone.utc).isoformat()); sources[event.source] += 1; valid += 1
                except (KeyError, TypeError, ValueError, OverflowError): reasons['invalid_contract'] += 1
        return dict(base, records=total, accepted_contract=valid, rejected=dict(reasons), sources=dict(sources),
                    first_observation=min(times) if times else None, last_observation=max(times) if times else None,
                    status='contract_only', operational_availability_verified=False,
                    note='Структура не удостоверяет источник, историческую доступность, единицы и радиометрическую калибровку.')
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            columns = reader.fieldnames or []
            count = sum(1 for _ in reader)
        return dict(base, rows=count, columns=columns, status='inventory_only')
    if path.suffix.lower() == '.json':
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict) and payload.get('kind') == 'global_level_zscore':
            from ..normalization import NormalizationBundle
            norm = NormalizationBundle(payload)
            missing = [k for k in ('td2m', 'surface_pressure', 'total_cloud_fraction', 'precipitation_step') if k not in norm.stats]
            interval = norm.stats['precipitation_step'].interval_hours if 'precipitation_step' in norm.stats else None
            return dict(base, status='normalization_contract_only', fingerprint=norm.fingerprint,
                        variables=sorted(norm.stats), missing_variables=missing,
                        precipitation_interval_hours=interval, fit_period=payload['provenance'].get('fit_period'),
                        surface_subset_complete_for_3h=not missing and interval == 3,
                        model_ready=False,
                        note='Совпадение состава не удостоверяет происхождение или независимость обучающего периода.')
        return dict(base, status='inventory_only', kind=payload.get('kind') if isinstance(payload, dict) else None)
    return dict(base, status='quarantine', reason='Формат требует отдельного декодера и физической проверки.')


def inventory(directory):
    directory = Path(directory)
    out = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and not path.is_symlink() and len(out) < 500:
            out.append(dict(name=path.name, bytes=path.stat().st_size,
                            status='quarantine' if path.suffix.lower() not in ('.csv', '.jsonl', '.json') else 'not_inspected'))
    return out
