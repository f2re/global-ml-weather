"""Observation-error calibration from saved departures (Desroziers et al. 2005, QJRMS 131, 3385-3396).

For every assimilated measurement y of a cycled analysis:

    d_ob = y - H(x_b)     background departure (previous cycle's +6 h forecast)
    d_oa = y - H(x_a)     analysis departure
    d_ab = H(x_a) - H(x_b)

With consistent error statistics, E[d_oa d_ob] = sigma_o^2 and
E[d_ab d_ob] = sigma_b^2 in observation space, and
E[d_ob^2] = sigma_o^2 + sigma_b^2. The diagnostics are computed per platform,
variable and pressure band on the TRAIN split only, from the frozen model;
validation and test are never read. The R11 analysis is a learned map, not a
BLUE, so the estimates are consistency diagnostics: the table is rewritten,
a new experiment is trained with it, and the diagnostics are repeated until
they stop changing. A model that copies its inputs drives d_oa to zero and
the estimate to zero; the fit ratio E[d_oa^2] / E[d_ob^2] is reported, and
an estimate is applied only within [floor_ratio, ceiling_ratio] of the current
sigma. Satellite channels keep their passport errors; the report suggests
values for the admission file but never rewrites it.
"""
from __future__ import annotations

import argparse
import copy
from datetime import timedelta
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..observation_training import digest, save
from .model import Rollout
from .operator import build_plan, observe
from .sources import seconds
from .tokens import PLATFORMS, PRESSURE_VARIABLES, VARIABLES, encode

METHOD = 'desroziers-2005-cycled-train-v1'
BAND_HPA = (1000., 925., 850., 700., 500., 400., 300., 250., 200., 150., 100., 70., 50., 30., 10., 1.)


def _band(pressure_pa: np.ndarray) -> np.ndarray:
    """Index of the nearest table level in ln p (NaN pressures -> -1)."""
    out = np.full(len(pressure_pa), -1, np.int64)
    known = np.isfinite(pressure_pa)
    if known.any():
        levels = np.log(np.asarray(BAND_HPA) * 100.)
        out[known] = np.abs(np.log(pressure_pa[known])[:, None] - levels[None]).argmin(1)
    return out


def departures(model, experiment, *, issues: int, seed: int = 0, cycle_hours: int = 6,
               window_hours: float | None = None) -> dict:
    """Collect y, H(x_b), H(x_a) for assimilated train measurements of cycled analyses."""
    from .training import background_state
    window_hours = experiment.config['data']['analysis_window_hours'] if window_hours is None else window_hours
    candidates = experiment.issues('train')
    available = set(candidates)
    candidates = [t for t in candidates if t - timedelta(hours=cycle_hours) in available]
    if not candidates:
        raise ValueError('No cycled train issues.')
    chosen = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(issues, len(candidates)), dtype=int)]
    columns = {k: [] for k in ('platform', 'variable', 'channel', 'pressure', 'value', 'sigma', 'hxa', 'hxb', 'issue')}
    noise, model.noise_scale = model.noise_scale, 0.0
    d = experiment.config['data']
    stats = {k: v for k, v in experiment.normalizer.statistics.items()
             if k.split('|')[1] in ('brightness_temperature', 'satellite_product')}
    with torch.no_grad():
        for index, issue in enumerate(chosen):
            background, previous = background_state(model, experiment, issue, 'train', cycle_hours=cycle_hours)
            if background is None:
                continue
            issue_s = seconds(issue)
            history = experiment._window(issue_s - 12 * 3600, issue_s, split='train', available_before=issue_s,
                                         rng=np.random.default_rng(seed + index),
                                         limit=d['max_input_tokens_per_source'])
            inputs = history.select(history.available > seconds(previous))
            tokens, usable = encode(inputs, issue_s, experiment.normalizer, experiment.errors.sigma(inputs),
                                    experiment.channels, device=experiment.device,
                                    channel_slots=experiment.channel_tables['slots'])
            assimilated = inputs.select(usable)
            near = np.abs(assimilated.observed - issue_s) <= window_hours * 3600
            targets = assimilated.select(near)
            if not len(targets):
                continue
            sigma = experiment.errors.sigma(targets)
            plan = build_plan(targets, sigma, issue_s, 0, experiment.mesh, analysis=np.ones(len(targets), bool),
                              analysis_window_hours=window_hours, column_statistics=stats,
                              channels=experiment.channels, device=experiment.device)
            if plan.count == 0:
                continue
            analysis = model(tokens, issue, 0, background=background)
            state, fields = model.constrain(background, None)
            first_guess = Rollout(issue, [0], [state], [fields])
            hxa = observe(model, analysis, plan).double().cpu().numpy()
            hxb = observe(model, first_guess, plan).double().cpu().numpy()
            kept = targets.select(plan.rows)
            columns['platform'].append(kept.platform); columns['variable'].append(kept.variable)
            columns['channel'].append(kept.channel); columns['pressure'].append(kept.pressure_pa)
            columns['value'].append(kept.value); columns['sigma'].append(sigma[plan.rows])
            columns['hxa'].append(hxa); columns['hxb'].append(hxb)
            columns['issue'].append(np.full(len(kept), issue_s))
    model.noise_scale = noise
    if not columns['value']:
        raise ValueError('No assimilated measurements inside the analysis window.')
    return {k: np.concatenate(v) for k, v in columns.items()}


def diagnose(data: dict, *, minimum_count: int = 200) -> list[dict]:
    """Desroziers statistics per (platform, variable, channel, pressure band)."""
    d_ob = data['value'] - data['hxb']
    d_oa = data['value'] - data['hxa']
    d_ab = data['hxa'] - data['hxb']
    band = _band(data['pressure'])
    keys = np.rec.fromarrays([data['platform'].astype(np.int64), data['variable'].astype(np.int64),
                              data['channel'].astype(np.int64), band])
    rows = []
    for key in np.unique(keys):
        mask = keys == key
        n = int(mask.sum())
        r = float(np.mean(d_oa[mask] * d_ob[mask]))
        b = float(np.mean(d_ab[mask] * d_ob[mask]))
        total = float(np.mean(d_ob[mask] ** 2))
        rows.append({'platform': PLATFORMS[key[0]], 'variable': VARIABLES[key[1]], 'channel': int(key[2]),
                     'pressure_hpa': BAND_HPA[key[3]] if key[3] >= 0 else None, 'count': n,
                     'sigma_current': float(np.sqrt(np.mean(data['sigma'][mask] ** 2))),
                     'sigma_o': math.sqrt(r) if r > 0 else None, 'sigma_b': math.sqrt(b) if b > 0 else None,
                     'omb_rms': math.sqrt(total), 'oma_rms': float(np.sqrt(np.mean(d_oa[mask] ** 2))),
                     'omb_bias': float(np.mean(d_ob[mask])),
                     'consistency': (r + b) / total if total > 0 else None,
                     'fit_ratio': float(np.mean(d_oa[mask] ** 2)) / total if total > 0 else None,
                     'enough': n >= minimum_count})
    return rows


def calibrated_table(payload: dict, rows: list[dict], *, floor_ratio: float = 0.5, ceiling_ratio: float = 2.0) -> tuple[dict, list]:
    """Rewrite error specifications from usable diagnostics; return (payload, applied changes)."""
    out = copy.deepcopy(payload)
    applied = []
    by_key = {}
    for row in rows:
        if row['channel'] >= 0 or not row['enough'] or row['sigma_o'] is None:
            continue
        ratio = row['sigma_o'] / row['sigma_current']
        if not floor_ratio <= ratio <= ceiling_ratio:
            continue
        by_key.setdefault((row['platform'], row['variable']), []).append(row)
    for (platform, variable), entries in by_key.items():
        spec = out['platforms'].get(platform, {}).get(variable)
        if spec is None or 'from_record' in spec or 'relative' in spec:
            continue
        if variable in PRESSURE_VARIABLES:
            levels = np.asarray(spec.get('pressure_hpa', BAND_HPA), float)
            current = (np.asarray(spec['sigma'], float) if 'pressure_hpa' in spec
                       else np.full(len(levels), float(spec['sigma'])))
            table = dict(zip(levels.tolist(), current.tolist()))
            for row in entries:
                table[row['pressure_hpa']] = row['sigma_o']
            pressures = sorted(table, reverse=True)
            out['platforms'][platform][variable] = {'pressure_hpa': pressures, 'sigma': [table[p] for p in pressures]}
        else:
            weighted = sum(r['sigma_o'] ** 2 * r['count'] for r in entries) / sum(r['count'] for r in entries)
            out['platforms'][platform][variable] = {'sigma': math.sqrt(weighted)}
        applied.append({'platform': platform, 'variable': variable, 'before': spec,
                        'after': out['platforms'][platform][variable]})
    return out, applied


def run(training, output_table, report_path, *, issues: int, minimum_count: int = 200, surface=None,
        profile=None, stores=(), static=None, device='cpu') -> dict:
    from .forecast import load_frozen
    model, experiment, provenance = load_frozen(training, surface=surface, profile=profile, stores=stores,
                                                static=static, device=device)
    data = departures(model, experiment, issues=issues)
    rows = diagnose(data, minimum_count=minimum_count)
    current = json.loads(experiment.error_path.read_text())
    table, applied = calibrated_table(current, rows)
    table['basis'] = (f'Desroziers et al. (2005) calibration ({METHOD}) of {experiment.error_path.name} '
                      f'from {len(data["value"])} assimilated train measurements; '
                      'retrain with this table and repeat until the estimates stop changing.')
    table['calibration'] = {'method': METHOD, 'split': 'train', 'source_table_sha256': digest(experiment.error_path),
                            'checkpoint_sha256': provenance['checkpoint_sha256'], 'applied': len(applied)}
    output_table, report_path = Path(output_table), Path(report_path)
    if output_table.exists():
        raise FileExistsError('Calibrated table exists; choose a new file.')
    save(output_table, table)
    channels = [r for r in rows if r['channel'] >= 0 and r['sigma_o'] is not None and r['enough']]
    report = {'method': METHOD, 'split': 'train', 'measurements': int(len(data['value'])),
              'issues': int(len(np.unique(data['issue']))), 'rows': rows, 'applied': applied,
              'satellite_channel_suggestions': [{'channel': experiment.channels[r['channel']],
                                                 'suggested_observation_error_k': r['sigma_o'],
                                                 'count': r['count']} for r in channels],
              'output_table': str(output_table), 'output_table_sha256': digest(output_table),
              **provenance, 'scientific_acceptance': False}
    save(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True, help='New observation error table (JSON).')
    parser.add_argument('--report', required=True)
    parser.add_argument('--issues', type=int, default=256)
    parser.add_argument('--minimum-count', type=int, default=200)
    parser.add_argument('--surface'); parser.add_argument('--profile')
    parser.add_argument('--store', action='append', default=[]); parser.add_argument('--static')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    report = run(args.training, args.output, args.report, issues=args.issues, minimum_count=args.minimum_count,
                 surface=args.surface, profile=args.profile, stores=args.store, static=args.static, device=args.device)
    print(json.dumps({'measurements': report['measurements'], 'applied': len(report['applied'])}))


if __name__ == '__main__':
    main()


__all__ = ['departures', 'diagnose', 'calibrated_table', 'run']
