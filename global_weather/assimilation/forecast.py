"""Frozen R11 forecasts and observation-space scores; never updates weights.

`forecast` writes, per issue, the latent-mesh fields (T, q, u, v, Phi, omega
on 37 levels with below-ground masks, p_s, MSLP, near-surface station-native
T/Td) for every requested member. `score` evaluates a frozen checkpoint on a
split, including the test split, which the trainer never reads.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from ..observations import utc
from ..observation_training import digest, save
from ..vertical import PRESSURE_HPA
from .objective import observation_objective, persistence_skill
from .training import Experiment, _members, checkpoint_path, load_config


def load_frozen(training, *, stage: int | None = None, surface=None, profile=None, stores=(), static=None,
                climatology=None, synthetic=False, device='cpu'):
    training = Path(training)
    completion = json.loads((training / 'complete.json').read_text())
    best = completion['best']
    if not best:
        raise ValueError('Training has no validated checkpoint.')
    key = str(max(int(k) for k in best) if stage is None else stage)
    reference = json.loads((training / f'best-stage-{key}.json').read_text())
    if reference['directory'] != best[key]['directory'] or reference['epoch'] != best[key]['epoch']:
        raise ValueError('Best checkpoint record differs from the completion record.')
    state = torch.load(checkpoint_path(training, reference), map_location=device, weights_only=True)
    config = load_config(state['identity']['config'])
    experiment = Experiment(config, surface=surface, profile=profile, stores=stores, static=static,
                            climatology=climatology, synthetic=synthetic, device=device)
    identity = experiment.identity()
    for field in ('architecture', 'config', 'sources', 'channels', 'normalization', 'input_statistics',
                  'observation_errors_sha256', 'satellite_admission_sha256', 'static', 'grid_fingerprint',
                  'source_sha256'):
        if identity[field] != state['identity'][field]:
            raise ValueError('Frozen R11 identity differs: ' + field)
    model = experiment.build_model()
    model.load_state_dict(state['model'])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, experiment, {'checkpoint_sha256': reference['sha256'], 'stage': int(key), 'epoch': reference['epoch']}


def score(model, experiment, split: str, hours: int, *, limit: int, seed: int = 0) -> dict:
    issues = experiment.issues(split)
    if len(issues) > limit:
        issues = [issues[i] for i in np.linspace(0, len(issues) - 1, limit, dtype=int)]
    totals, variables, skills = [], {}, {}
    with torch.no_grad():
        for index, issue in enumerate(issues):
            sample = experiment.sample(issue, split, hours, rng=np.random.default_rng(seed + index), training=False)
            if sample.plan.count == 0:
                continue
            predictions, _ = _members(model, experiment, sample, issue, hours, 1)
            loss, report = observation_objective(predictions, sample.plan)
            totals.append(float(loss))
            for name, row in report['variables'].items():
                entry = variables.setdefault(name, [0, 0.0, 0.0])
                entry[0] += row['count']; entry[1] += row['rmse'] ** 2 * row['count']; entry[2] += row['bias'] * row['count']
            for name, row in persistence_skill(sample.plan, predictions, sample.baseline).items():
                entry = skills.setdefault(name, [0, 0.0, 0.0])
                entry[0] += row['count']; entry[1] += row['model_rmse'] ** 2 * row['count']
                entry[2] += row['baseline_rmse'] ** 2 * row['count']
    return {'split': split, 'rollout_hours': hours, 'issues': len(totals), 'loss': float(np.mean(totals)),
            'variables': {k: {'count': v[0], 'rmse': (v[1] / v[0]) ** 0.5, 'bias': v[2] / v[0]} for k, v in variables.items()},
            'persistence': {k: {'count': v[0], 'model_rmse': (v[1] / v[0]) ** 0.5, 'persistence_rmse': (v[2] / v[0]) ** 0.5,
                                'rmse_skill': 1 - (v[1] / v[2]) ** 0.5 if v[2] else None} for k, v in skills.items()},
            'uncertainty': 'not computed; block bootstrap over weather episodes is a separate step',
            'scientific_acceptance': False}


def forecast(model, experiment, issue, output, *, hours: int = 72, members: int = 1, provenance: dict) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError('Forecast output exists.')
    issue = utc(issue)
    split = next(row['split'] for row in experiment.windows if utc(row['issue']) == issue)
    sample = experiment.sample(issue, split, hours, rng=np.random.default_rng(0), training=False)
    model.noise_scale = 1.0 if members > 1 else 0.0
    arrays = {}
    with torch.no_grad():
        for member in range(members):
            rollout = model(sample.tokens, issue, hours)
            frames = model.frames(rollout)
            arrays.setdefault('profiles', []).append(np.stack([f.profiles.cpu().numpy() for f in frames]))
            arrays.setdefault('profile_mask', []).append(np.stack([f.profile_mask.cpu().numpy() for f in frames]))
            arrays.setdefault('surface_pressure', []).append(np.stack([f.surface[:, 4].cpu().numpy() for f in frames]))
            arrays.setdefault('mslp', []).append(np.stack([f.surface[:, 5].cpu().numpy() for f in frames]))
            arrays.setdefault('near_surface_temperature', []).append(
                np.stack([f.near_surface['temperature'].cpu().numpy() for f in frames]))
            arrays.setdefault('near_surface_dew_point', []).append(
                np.stack([f.near_surface['dew_point'].cpu().numpy() for f in frames]))
    model.noise_scale = 0.0
    output.mkdir(parents=True)
    np.savez_compressed(output / 'forecast.npz', **{k: np.stack(v) for k, v in arrays.items()},
                        xyz=experiment.mesh.xyz, pressure_pa=np.asarray(PRESSURE_HPA, float) * 100.,
                        lead_hours=np.arange(0, hours + 1, 3))
    manifest = {'schema': 'physical-assimilation-forecast-1', 'issue_time': issue.isoformat(), 'members': members,
                'dimensions': {'profiles': ['member', 'lead', 'node', 'pressure', 'variable']},
                'profile_variables': ['temperature', 'specific_humidity', 'u', 'v', 'geopotential', 'omega'],
                'profile_units': ['K', 'kg kg-1', 'm s-1', 'm s-1', 'm2 s-2', 'Pa s-1'],
                'geopotential_origin': 'hydrostatic integration from surface geopotential g*z_s',
                'omega_origin': 'continuity with kinematic surface boundary (O’Brien correction)',
                'below_ground_mask': 'pressure greater than forecast surface pressure (product mask only)',
                'near_surface_convention': 'station-native, unknown instrument heights; not 2 m / 10 m products',
                'input_tokens': sample.inputs, 'grid_fingerprint': experiment.mesh.grid.fingerprint,
                'mesh_level': experiment.mesh.level,
                'wind_basis': 'local_enu_vector',
                'forecast_sha256': digest(output / 'forecast.npz'), **provenance,
                'data_kind': experiment.sources[0].data_kind, 'scientific_acceptance': False,
                'created_utc': datetime.now(timezone.utc).isoformat()}
    save(output / 'forecast.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('forecast', 'score'))
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--surface'); parser.add_argument('--profile'); parser.add_argument('--store', action='append', default=[])
    parser.add_argument('--static'); parser.add_argument('--stage', type=int)
    parser.add_argument('--issue', action='append', default=[])
    parser.add_argument('--split', default='validation', choices=('validation', 'test'))
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--hours', type=int, default=72)
    parser.add_argument('--members', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    model, experiment, provenance = load_frozen(args.training, stage=args.stage, surface=args.surface,
                                                profile=args.profile, stores=args.store, static=args.static,
                                                device=args.device)
    if args.command == 'score':
        result = score(model, experiment, args.split, args.hours, limit=args.limit)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        save(Path(args.output), result | provenance)
        print(json.dumps({'issues': result['issues'], 'loss': result['loss']}))
    else:
        for issue in args.issue:
            target = Path(args.output) / utc(issue).strftime('%Y%m%dT%H%M%SZ')
            print(json.dumps(forecast(model, experiment, issue, target, hours=args.hours, members=args.members,
                                      provenance=provenance)['forecast_sha256']))


if __name__ == '__main__':
    main()
