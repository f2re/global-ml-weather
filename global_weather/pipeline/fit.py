"""Streaming training-only statistics. Imported norms retain their own provenance."""
from __future__ import annotations
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import numpy as np
from .dataset import PreparedDataset, utc
from .io import atomic_json, reference, digest, sha256, read_json, local_path
from ..products.ingest import maximum_history, record_history
from ..vertical import PRESSURE_HPA, PROFILE_VARIABLES, PROFILE_UNITS, SURFACE_VARIABLES, SURFACE_UNITS


class Moments:
    """Parallel weighted Welford population moments, accumulated in float64."""
    def __init__(self, shape=()):
        self.weight = np.zeros(shape, np.float64)
        self.mean = np.zeros(shape, np.float64)
        self.m2 = np.zeros(shape, np.float64)

    def add(self, values, mask, weights):
        x = np.asarray(values, dtype=np.float64)
        m = np.asarray(mask)
        w = np.broadcast_to(np.asarray(weights, np.float64), x.shape)
        if m.dtype != bool or x.shape != m.shape or x.shape[1:] != self.mean.shape:
            raise ValueError('Неверные формы статистики.')
        if not np.isfinite(w).all() or (w < 0).any() or not np.isfinite(x[m]).all():
            raise ValueError('Неконечные значения или неверные веса статистики.')
        w = np.where(m, w, 0.); x = np.where(m, x, 0.)
        amount = w.sum(0)
        mean = (w*x).sum(0)/np.maximum(amount, np.finfo(float).tiny)
        m2 = (w*(x-mean)**2).sum(0)
        total = self.weight+amount
        delta = mean-self.mean
        safe = np.maximum(total, np.finfo(float).tiny)
        self.mean += delta*amount/safe
        self.m2 += m2+delta**2*self.weight*amount/safe
        self.weight = total

    def finish(self):
        if (self.weight <= 0).any():
            raise ValueError('Нет обучающих значений для части норм.')
        std = np.sqrt(self.m2/self.weight)
        if not np.isfinite(std).all() or (std <= 0).any():
            raise ValueError('Нулевая дисперсия. Добавьте данные; масштаб не заменяется константой.')
        return self.mean, std


def fit_normalization(dataset_path, output_manifest, *, base_path=None):
    """Fit missing variables using train only; output manifest must share the dataset root.

    Atmospheric statistics use target analyses (lead=0), precipitation uses the
    first complete forecast interval. Duplicate valid times are rejected upstream.
    Satellite statistics use unique training observations, not air-temperature norms.
    """
    from ..normalization import NormalizationBundle, ZStat
    ds = PreparedDataset(dataset_path, require_norm=False)
    output_manifest = Path(output_manifest).absolute()
    if output_manifest.parent != ds.root or output_manifest.exists():
        raise ValueError('Новый манифест должен находиться рядом с исходным и не существовать.')
    norm_path = output_manifest.with_name(output_manifest.stem+'.normalization.json')
    if norm_path.exists() or norm_path.is_symlink():
        raise FileExistsError('Нормы не перезаписываются.')
    train = ds.subset('train')
    # Never fit means or variances from validation/test targets, even during admission.
    area = ds.grid().areas_m2 / ds.grid().areas_m2.mean()
    stats = {}; origins = {}; components = []; fit_ends = []
    if base_path is not None:
        base = NormalizationBundle.load(base_path)
        components.append({'kind': 'imported', 'fingerprint': base.fingerprint,
                           'file_sha256': sha256(Path(base_path)), 'provenance': base._payload['provenance']})
        fit_ends.append(base._payload['provenance']['fit_period'])
        stats.update(base.stats)
        origins.update({name: 'imported:'+base.fingerprint for name in stats})
        # Incompatible accumulation is not inherited or scaled by a ratio.
        if 'precipitation_step' in stats and stats['precipitation_step'].interval_hours != ds.step:
            del stats['precipitation_step']; origins.pop('precipitation_step')
    required = dict(zip(PROFILE_VARIABLES, PROFILE_UNITS)) | dict(zip(SURFACE_VARIABLES, SURFACE_UNITS))
    required.update({name: v['units'] for name, v in ds.registry.items()})
    missing = set(required) - set(stats)
    profiles = {name: Moments((37,)) for name in missing if name in PROFILE_VARIABLES}
    surface = {name: Moments() for name in missing if name in SURFACE_VARIABLES}
    others = {name: Moments() for name in missing if name not in profiles and name not in surface}
    for name in others:
        v = ds.registry[name]
        if v.get('product'):
            from ..observations import Variable
            Variable(**v)  # Validate retrieval identity, units and maximum age.
            continue
        if v.get('vertical') != 'column' or not all(v.get(k) for k in ('source', 'platform', 'channel_id')):
            raise ValueError('Дополнительная норма требует привязки к физическому спутниковому каналу.')
    seen = {}; hashes = {}; used_end = max(s.issue for s in train)
    for s in train:
        ds.assert_unchanged()
        d = ds.targets(s)
        hashes['target:'+s.id] = s.targets['sha256']
        for name, moment in profiles.items():
            k = PROFILE_VARIABLES.index(name)
            moment.add(d['profiles'][0, :, :, k], d['profile_mask'][0, :, :, k], area[:, None])
        for name, moment in surface.items():
            k = SURFACE_VARIABLES.index(name); lead = 1 if k == 6 else 0
            moment.add(d['surface'][lead, :, k], d['surface_mask'][lead, :, k], area)
            if k == 6:
                used_end = max(used_end, s.issue+timedelta(hours=ds.step))
        if others:
            hashes['observations:'+s.id] = s.observations['sha256']
            for r in ds.eligible_records(s, include_invalid=True):
                if r['variable'] not in others or not s.issue-timedelta(hours=record_history(ds.registry,r['variable'])) < utc(r['observed_at']) <= s.issue or utc(r['available_at']) > s.issue:
                    continue
                key = (r.get('source'), r.get('observation_id'))
                if not all(key):
                    raise ValueError('У наблюдения нет устойчивого идентификатора.')
                old = seen.get(key)
                rank = (r.get('revision', 0), utc(r['available_at']))
                if old is None or rank > (old.get('revision', 0), utc(old['available_at'])):
                    seen[key] = r
                elif rank == (old.get('revision', 0), utc(old['available_at'])) and r != old:
                    raise ValueError('Противоречащие версии одного наблюдения.')
    if others:
        # Run physical/radiometric gate before computing channel statistics.
        from ..observations import Variable, pack_observations
        registry = {name: Variable(**v) for name, v in ds.registry.items()}
        for r in seen.values():
            if not r.get('valid', True):
                continue
            packed = pack_observations([r], ds.grid(), np.array(PRESSURE_HPA)*100, utc(r['available_at']), registry)
            if packed.accepted_records != 1:
                raise ValueError('Спутниковое наблюдение не прошло физический допуск для нормы.')
            others[r['variable']].add(np.array([r['value']]), np.array([True]), 1.)
    for group in (profiles, surface, others):
        for name, moment in group.items():
            mean, std = moment.finish()
            stats[name] = ZStat(required[name], tuple(np.atleast_1d(mean)), tuple(np.atleast_1d(std)),
                                tuple(np.array(PRESSURE_HPA)*100) if name in profiles else (),
                                ds.step if name == 'precipitation_step' else None)
            origins[name] = 'own_training_partition'
    period = {'start': (min(s.issue for s in train)-timedelta(hours=ds.history_hours)).isoformat(), 'end': used_end.isoformat()}
    fit_ends.append(period)
    if any(not isinstance(p, dict) or not p.get('end') or not p.get('start') for p in fit_ends):
        combined_period = None
    else:
        combined_period = {'start': min(utc(p['start']) for p in fit_ends).isoformat(),
                           'end': max(utc(p['end']) for p in fit_ends).isoformat()}
    components.append({'kind': 'own_train', 'data_kind': ds.kind, 'period': period,
                       'source_dataset_fingerprint': ds.fingerprint, 'sample_ids': [s.id for s in train]})
    if base_path is not None:
        hashes['imported_normalization'] = sha256(Path(base_path))
    payload = {'schema_version': 1, 'kind': 'global_level_zscore',
               'variables': {name: asdict(stat) for name, stat in sorted(stats.items())},
               'provenance': {'repository': 'https://github.com/f2re/global-ml-weather',
                              'revision': 'pipeline-normalization-1', 'data_family': 'prepared_training_targets_and_observations',
                              'license': ds.manifest.get('license', 'not_redistributable_without_source_review'),
                              'artifact_sha256': hashes, 'fit_period': combined_period,
                              'components': components, 'variable_origins': origins,
                              'spatial_weighting': 'actual_spherical_cell_area; satellite unique observation weight=1',
                              'data_kind': ds.kind}}
    bundle = NormalizationBundle(payload)
    # Completeness and unit/interval matching are checked BEFORE writing anything.
    for name, units in required.items():
        stat = bundle.get(name, units)
        pressure = np.array(PRESSURE_HPA)*100 if name in PROFILE_VARIABLES else None
        stat.at(pressure, interval_hours=ds.step if name == 'precipitation_step' else None)
    for split in ('validation', 'test'):
        times = [s.issue for s in ds.samples if s.split == split]
        if times:
            bundle.assert_independent_test((min(times)-timedelta(hours=ds.history_hours)).isoformat())
    ds.assert_unchanged()
    bundle.save(norm_path)
    manifest = dict(ds.manifest)
    manifest['normalization'] = reference(ds.root, norm_path)
    atomic_json(output_manifest, manifest)
    return {'status': 'normalization_frozen', 'normalization_fingerprint': bundle.fingerprint,
            'data_kind': ds.kind, 'manifest': output_manifest.name,
            'variables': sorted(stats), 'fit_period': combined_period, 'meteorologically_validated': False}
