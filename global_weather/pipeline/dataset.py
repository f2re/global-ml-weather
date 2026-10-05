"""Versioned prepared datasets; no implicit remapping, units or archive latency."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
import numpy as np
from .io import artifact, digest, read_json, read_arrays, parse_json, MAX_JSON
from ..products.ingest import maximum_history, record_history
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS


def utc(value):
    if not isinstance(value, str):
        raise ValueError('Время должно быть строкой ISO 8601.')
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError('Для времени требуется часовой пояс.')
    return dt.astimezone(timezone.utc)


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'Требуется целое число от {minimum} до {maximum}.')
    return value


def _same(actual, expected, label):
    if np.asarray(actual).tolist() != list(expected):
        raise ValueError(f'Несовместимые {label}.')


@dataclass
class Sample:
    id: str
    split: str
    issue: datetime
    observations: dict
    targets: dict
    provenance: dict


class PreparedDataset:
    def __init__(self, path, *, require_norm=True, max_cells=40962, max_samples=10000, inference=False):
        self.path = Path(path).absolute()
        self.root = self.path.parent
        self.manifest = read_json(path)
        m = self.manifest
        if not isinstance(m, dict):
            raise ValueError('Манифест должен быть объектом JSON.')
        self.is_input = m.get('schema') == 'global-weather-input-1'
        if (m.get('schema') != 'global-weather-dataset-1' and not (inference and self.is_input)) or m.get('data_kind') not in ('real', 'synthetic'):
            raise ValueError('Неизвестная схема или происхождение набора.')
        self.kind = m['data_kind']
        if not isinstance(m.get('license'), str) or not m['license'].strip():
            raise ValueError('Не указаны условия использования исходных данных.')
        self.level = integer(m.get('mesh_level'), 0, 7)
        self.n_cells = 10*4**self.level + 2
        if self.n_cells > max_cells:
            raise ValueError('Сетка превышает разрешённый размер.')
        self.step = integer(m.get('step_hours'), 1, 6)
        if self.step not in (1, 3, 6):
            raise ValueError('Допустимый шаг: 1, 3 или 6 часов.')
        self.horizon = integer(m.get('horizon_hours'), self.step, 72)
        if self.horizon % self.step:
            raise ValueError('Горизонт не кратен шагу.')
        _same(m.get('pressure_hpa'), PRESSURE_HPA, 'уровни давления')
        self.grid_fingerprint = m.get('grid_fingerprint')
        if not isinstance(self.grid_fingerprint, str) or not re.fullmatch('[a-f0-9]{64}', self.grid_fingerprint):
            raise ValueError('Нужен отпечаток глобальной сетки.')
        self.registry = read_json(artifact(self.root, m.get('registry'), limit=MAX_JSON))
        if not isinstance(self.registry, dict) or not self.registry:
            raise ValueError('Пустой реестр величин.')
        self.history_hours = maximum_history(self.registry)
        p = artifact(self.root, m.get('static'), limit=512*1024**2)
        st = read_arrays(p)
        _same(st.get('surface_units'), ('m', '1'), 'единицы статических полей')
        if str(st.get('grid_fingerprint')) != self.grid_fingerprint:
            raise ValueError('Статические поля принадлежат другой сетке.')
        self.elevation, self.land = st.get('elevation_m'), st.get('land_fraction')
        for a in (self.elevation, self.land):
            if a is None or a.shape != (self.n_cells,) or a.dtype.kind != 'f' or not np.isfinite(a).all():
                raise ValueError('Повреждены статические поля.')
        if ((self.land < 0) | (self.land > 1)).any():
            raise ValueError('Доля суши вне [0,1].')
        self.samples = []
        entries = m.get('samples')
        if not isinstance(entries, list) or not 1 <= len(entries) <= max_samples:
            raise ValueError('Недопустимое число примеров.')
        for entry in entries:
            if not isinstance(entry, dict) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', entry.get('id', '')):
                raise ValueError('Неверный ID примера.')
            split = entry.get('split')
            if split not in (('inference',) if self.is_input else ('train', 'validation', 'test')):
                raise ValueError('Неизвестная часть выборки.')
            provenance = entry.get('provenance', {})
            for key in (('observations', 'availability') if self.is_input else ('observations', 'targets', 'availability', 'target_operator')):
                if not isinstance(provenance.get(key), str) or not provenance[key].strip():
                    raise ValueError(f'Не указано происхождение: {key}')
            self.samples.append(Sample(entry['id'], split, utc(entry['issue_time']),
                                       entry['observations'], entry.get('targets'), provenance))
        if len({s.id for s in self.samples}) != len(self.samples) or len({s.issue for s in self.samples}) != len(self.samples):
            raise ValueError('Повторные ID или сроки выпуска.')
        self.samples.sort(key=lambda s: s.issue)
        self._check_splits()
        self.norm = None
        if m.get('normalization') is not None:
            from ..normalization import NormalizationBundle
            self.norm = NormalizationBundle.load(artifact(self.root, m['normalization'], limit=MAX_JSON))
            norm_kind = self.norm._payload['provenance'].get('data_kind')
            if norm_kind is not None and norm_kind != self.kind:
                raise ValueError('Нормы и выборка имеют разное происхождение.')
            for name, units in zip(PROFILE_VARIABLES, PROFILE_UNITS):
                self.norm.get(name, units).at(np.array(PRESSURE_HPA)*100)
            for name, units in zip(SURFACE_VARIABLES, SURFACE_UNITS):
                self.norm.get(name, units).at(interval_hours=self.step if name == 'precipitation_step' else None)
            for split in ('validation', 'test'):
                times = [s.issue for s in self.samples if s.split == split]
                if times:
                    self.norm.assert_independent_test((min(times)-timedelta(hours=self.history_hours)).isoformat())
        elif require_norm:
            raise ValueError('Набор не содержит фиксированных норм.')
        self.fingerprint = digest(m)

    def assert_unchanged(self):
        if digest(read_json(self.path)) != self.fingerprint:
            raise ValueError('Манифест изменился во время эксперимента.')
        for key in ('registry', 'static', 'normalization'):
            if self.manifest.get(key) is not None:
                artifact(self.root, self.manifest[key], limit=512*1024**2)

    def _check_splits(self):
        parts = [[s for s in self.samples if s.split == k] for k in ('train', 'validation', 'test')]
        parts = [p for p in parts if p]
        for left, right in zip(parts, parts[1:]):
            if max(s.issue for s in left) + timedelta(hours=self.horizon) >= min(s.issue for s in right) - timedelta(hours=self.history_hours):
                raise ValueError('Временные окна разных частей выборки пересекаются или нарушен порядок.')

    def subset(self, split):
        result = [s for s in self.samples if s.split == split]
        if not result:
            raise ValueError(f'Нет части выборки: {split}')
        return result

    def records(self, sample):
        path = artifact(self.root, sample.observations, limit=MAX_JSON)
        result = []
        with path.open(encoding='utf-8') as f:
            for number, text in enumerate(f, 1):
                if not text.strip():
                    continue
                r = parse_json(text)
                if not isinstance(r, dict) or r.get('variable') not in self.registry:
                    raise ValueError(f'Неизвестная величина в строке {number}.')
                if not isinstance(r.get('valid', True), bool):
                    raise ValueError('Маска наблюдения должна быть Boolean.')
                observed, available = utc(r['observed_at']), utc(r['available_at'])
                if available < observed:
                    raise ValueError('Готовность наблюдения раньше измерения.')
                if self.registry[r['variable']].get('product') and r.get('valid', True):
                    if r.get('derivation', {}).get('data_kind') != self.kind:
                        raise ValueError('Происхождение продукции и выборки различается.')
                if not r.get('valid', True):
                    result.append(r)
                    continue
                if r.get('units') != self.registry[r['variable']]['units']:
                    raise ValueError('Единицы наблюдения не совпадают с реестром.')
                if type(r.get('value')) not in (int, float) or not np.isfinite(r['value']):
                    raise ValueError('Неконечное пригодное наблюдение.')
                result.append(r)
        return result

    def eligible_records(self, sample, *, include_invalid=False):
        selected = {}
        for record in self.records(sample):
            lower = sample.issue-timedelta(hours=record_history(self.registry,record['variable']))
            observed, available = utc(record['observed_at']), utc(record['available_at'])
            if not lower < observed <= sample.issue or available > sample.issue:
                continue
            key = (record.get('source'), record.get('observation_id'))
            if not all(isinstance(x, str) and x for x in key):
                raise ValueError('Нужны источник и устойчивый ID наблюдения.')
            revision = integer(record.get('revision', 0), 0, 2**31-1)
            rank = (revision, available)
            old = selected.get(key)
            if old is not None and old[0] == rank and old[1] != record:
                raise ValueError('Противоречащие записи одной версии.')
            if old is None or rank > old[0]:
                selected[key] = (rank, record)
        return [r for _, r in selected.values() if include_invalid or r.get('valid', True)]

    def targets(self, sample):
        if sample.targets is None:
            raise ValueError('У входа прогноза нет будущих целей.')
        data = read_arrays(artifact(self.root, sample.targets, limit=512*1024**2))
        for key, expected in [('pressure_hpa', PRESSURE_HPA), ('profile_variables', PROFILE_VARIABLES),
                              ('profile_units', PROFILE_UNITS), ('surface_variables', SURFACE_VARIABLES),
                              ('surface_units', SURFACE_UNITS)]:
            _same(data.get(key), expected, key)
        if str(data.get('grid_fingerprint')) != self.grid_fingerprint or utc(str(data.get('issue_time'))) != sample.issue:
            raise ValueError('Цели имеют другую сетку или время выпуска.')
        leads = data.get('lead_hours')
        if leads is None or leads.dtype.kind not in 'iu' or leads.ndim != 1:
            raise ValueError('Некорректная ось заблаговременности.')
        if leads.tolist() != list(range(0, self.horizon+1, self.step)):
            raise ValueError('Цели должны содержать анализ и все сроки до горизонта набора.')
        k, n = len(leads), self.n_cells
        for value, mask, shape in [('profiles', 'profile_mask', (k, n, 37, 6)), ('surface', 'surface_mask', (k, n, 8))]:
            a, m = data.get(value), data.get(mask)
            if a is None or m is None or a.shape != shape or m.shape != shape or a.dtype.kind != 'f' or m.dtype != bool:
                raise ValueError('Неверные размеры или типы целевых массивов.')
            if not np.isfinite(a[m]).all():
                raise ValueError('Неконечная цель помечена пригодной.')
        p, pm, s, sm = [data[x] for x in ('profiles', 'profile_mask', 'surface', 'surface_mask')]
        if sm[0, :, 6].any():
            raise ValueError('Анализ не содержит будущих накопленных осадков.')
        any_profile = pm.any(axis=(2, 3))
        if (any_profile & ~sm[:, :, 4]).any():
            raise ValueError('Для профилей требуется целевое давление поверхности.')
        below = np.array(PRESSURE_HPA)[None, None, :] * 100 > s[:, :, 4, None]
        if (pm.any(-1) & below & sm[:, :, 4, None]).any():
            raise ValueError('Пригодные цели оказались ниже поверхности.')
        for a, valid in ((p[..., 0], pm[..., 0]), (s[..., 0], sm[..., 0]),
                         (s[..., 1], sm[..., 1]), (s[..., 4], sm[..., 4]), (s[..., 5], sm[..., 5])):
            if (a[valid] <= 0).any():
                raise ValueError('Температура в K или давление неположительны.')
        if (p[..., 1][pm[..., 1]] < 0).any() or (p[..., 1][pm[..., 1]] > 1).any():
            raise ValueError('Удельная влажность вне физического диапазона.')
        if (s[..., 6][sm[..., 6]] < 0).any() or ((s[..., 7][sm[..., 7]] < 0) | (s[..., 7][sm[..., 7]] > 1)).any():
            raise ValueError('Осадки или облачная доля вне допустимого диапазона.')
        policy = self.manifest.get('target_coverage')
        if policy is not None:
            if not isinstance(policy, dict) or set(policy) != {'minimum'}:
                raise ValueError('Неверная политика покрытия целей.')
            from .coverage import check_coverage
            check_coverage(data, policy['minimum'])
        return data

    def packed(self, sample):
        from ..observations import pack_observations, Variable
        grid = self.grid()
        variables = {k: Variable(**v) for k, v in self.registry.items()}
        packed = pack_observations(self.eligible_records(sample), grid, np.array(PRESSURE_HPA)*100,
                                    sample.issue, variables, normalization=self.norm)
        unacceptable = {k: v for k, v in packed.rejected.items() if k != 'not_available_in_12h_window' and v}
        if unacceptable:
            raise ValueError(f'Наблюдения не прошли допуск: {unacceptable}')
        if self.manifest.get('multimodal') is not None:
            from ..multimodal.integration import attach
            return attach(self, sample, packed)
        if not packed.accepted_records:
            raise ValueError('Нет пригодных наблюдений или зарегистрированного контекста поверхности.')
        return packed

    def grid(self):
        if not hasattr(self, '_grid'):
            from ..grid import build_grid
            self._grid = build_grid(self.level)
            if self._grid.fingerprint != self.grid_fingerprint:
                raise ValueError('Отпечаток геометрии не совпадает.')
        return self._grid

    def validate(self, *, targets=True):
        self.grid()
        rows = []
        for sample in self.samples:
            self.records(sample)
            if targets:
                self.targets(sample)
            count = None
            if self.norm is not None:
                count = self.packed(sample).accepted_records
            rows.append({'id': sample.id, 'split': sample.split, 'accepted_records': count})
        return {'schema': 'dataset-check-1', 'status': 'prepared_data_validated', 'data_kind': self.kind,
                'dataset_fingerprint': self.fingerprint, 'cells': self.n_cells, 'samples': rows,
                'meteorologically_validated': False, 'source_truth_verified': False,
                'note': 'Проверка структуры и причинности не удостоверяет источник данных или точность погоды.'}
