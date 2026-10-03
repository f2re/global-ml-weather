"""Evaluate explicitly paired physical arrays; never infer a target from a forecast."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np
from .contracts import atomic_json, sha256
from .metrics import paired_scores


def evaluate(path, manifest):
    source = Path(path)
    if source.is_symlink() or source.stat().st_size > 128*1024*1024: raise ValueError('Недопустимый файл.')
    with zipfile.ZipFile(source) as archive:
        if sum(f.file_size for f in archive.infolist()) > 512*1024*1024: raise ValueError('Превышен размер распакованных массивов.')
    required = ('variable', 'units', 'target_source', 'forecast_source', 'data_kind', 'paired_sha256')
    if any(not manifest.get(k) for k in required): raise ValueError('Нет обязательных метаданных проверки.')
    if manifest['data_kind'] not in ('real', 'synthetic'): raise ValueError('Неизвестное происхождение данных.')
    if manifest['paired_sha256'] != sha256(source): raise ValueError('Массивы не соответствуют манифесту.')
    with np.load(source, allow_pickle=False) as data:
        p, y, mask, leads, area = (data[k] for k in ('prediction', 'target', 'mask', 'lead_hours', 'area_m2'))
        if p.ndim not in (2, 3) or p.shape[0] != len(leads) or area.shape != (p.shape[1],): raise ValueError('Нужна форма [срок, ячейка, (уровень)].')
        if not np.isfinite(leads).all() or not (np.diff(leads) > 0).all() or (leads < 0).any() or (leads > 72).any(): raise ValueError('Неверные сроки.')
        weights = area.reshape((1, len(area)) + ((1,) if p.ndim == 3 else ()))
        baseline = data['baseline'] if 'baseline' in data else None
        overall = paired_scores(p, y, mask, weights, baseline)
        scores = [dict(lead_hours=float(lead), **paired_scores(p[i], y[i], mask[i], weights[0], baseline[i] if baseline is not None else None)) for i, lead in enumerate(leads)]
    return dict(kind='paired_verification', data_kind=manifest['data_kind'], variable=manifest['variable'], units=manifest['units'],
                declared_sources={'target': manifest['target_source'], 'forecast': manifest['forecast_source']},
                paired_sha256=manifest['paired_sha256'], overall=overall, by_lead=scores,
                independent_test_certified=False,
                note='Метрики по явно сопоставленным массивам. Независимость периода, происхождение и оператор сопоставления требуют научного аудита.')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs', type=Path, required=True); p.add_argument('--manifest', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(argv)
    atomic_json(a.output, evaluate(a.pairs, json.loads(a.manifest.read_text(encoding='utf-8'))))


if __name__ == '__main__': main()
