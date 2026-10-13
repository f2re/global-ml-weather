"""R11 acceptance gates 0-5 with thresholds pinned before the experiment.

Thresholds live in configs/physical_assimilation_gates_r11.json, schema
physical-assimilation-gates-1; changing them after a run requires a new
experiment identity. The gates use only R11 measurements and model output:
ERA5 is not read.

0. Program self-test: every one of the eight checks passed.
1. Domain of dependence: the multimesh lets one 3-hour step reach at least
   `min_reach_km` from every sampled start node (structural, exact BFS).
2. Analysis of withheld stations and launches (S1 rows inside the analysis
   window): normalised RMSE is below persistence and the train climatology,
   with the lower 95 % block-bootstrap bound of the skill above `min_skill`.
3. Dry-air mass: relative drift over every 72-hour rollout <= `max_drift`.
4. Forecast at +24/+48/+72 h (+-1.5 h): the same skill test against
   persistence and the train climatology.
5. Balance and spectra at 500 and 250 hPa: the divergent share of kinetic
   energy and the kinetic-energy spectrum in l in [l_low, l_high] keep at
   least the pinned ratio of the model's own analysis at the valid time.

Baselines:
* persistence: the latest earlier report of the same station, launch, flight
  or granule and variable (same pressure band), available at the issue; for
  withheld analysis rows only reports older than the analysis window count;
* climatology: train-split means per platform, variable, channel, month and
  pressure band, per station for surface families and per 10 degree latitude
  band otherwise. Validation and test never enter the climatology.
Errors are divided by the observation error, so families are comparable.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
from pathlib import Path

import numpy as np
import torch

from ..observation_training import digest, save
from .mesh import divergence
from .operator import observe
from .sources import seconds
from .tokens import FAMILY, PLATFORMS, VARIABLES, TokenBatch

SCHEMA = 'physical-assimilation-gates-1'
LEADS_SECONDS = 3600.0


# ----- thresholds -------------------------------------------------------------------------------

def load_thresholds(path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get('schema') != SCHEMA:
        raise ValueError('Unknown gate threshold schema.')
    g2, g4, g5 = payload['gate_2'], payload['gate_4'], payload['gate_5']
    if not payload['gate_1']['min_reach_km'] >= 1620:
        raise ValueError('Gate 1 cannot be weaker than 150 m/s over 3 h.')
    if not 0 < payload['gate_3']['max_relative_dry_mass_drift'] <= 1e-4:
        raise ValueError('Gate 3 drift bound must lie in (0, 1e-4].')
    for gate in (g2, g4):
        if not (0.5 < gate['confidence'] < 1 and gate['bootstrap'] >= 200 and gate['block_issues'] >= 1
                and gate['min_count'] >= 1 and set(gate['baselines']) == {'persistence', 'climatology'}):
            raise ValueError('Skill gates need confidence, bootstrap, blocks, counts and both baselines.')
    if sorted(g4['leads_hours']) != [24, 48, 72]:
        raise ValueError('Gate 4 scores +24, +48 and +72 h.')
    if not (0 < g5['min_divergent_fraction_ratio'] <= 1 and 0 < g5['min_band_energy_ratio'] <= 1
            and 1 <= g5['l_low'] < g5['l_high'] <= g5['l_max']):
        raise ValueError('Invalid spectral gate.')
    return payload


# ----- baselines --------------------------------------------------------------------------------

def _band(pressure: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(pressure), np.round(np.log(np.where(np.isfinite(pressure), pressure, 1.)) / 0.1), -1).astype(np.int64)


def _months(observed: np.ndarray) -> np.ndarray:
    return (np.asarray(observed, float).astype('datetime64[s]').astype('datetime64[M]').astype(np.int64) % 12) + 1


def _location(batch: TokenBatch) -> np.ndarray:
    surface = np.array([FAMILY[PLATFORMS[p]] == 'surface' for p in batch.platform], bool)
    latitude_band = np.floor((batch.latitude + 90) / 10).astype(np.int64)
    return np.where(surface, batch.group, -1 - latitude_band)


class TrainClimatology:
    """Train-only means; month-specific with a month-free fallback."""

    def __init__(self, experiment, *, stride_days: int = 1):
        from ..observation_data import START, TRAIN_END
        sums: dict = {}
        day = START
        while day < TRAIN_END:
            begin = seconds(day)
            for source in experiment.sources:
                kwargs = {'split': 'train'}
                if hasattr(source, 'channels'):
                    kwargs['channel_index'] = experiment.channel_index
                batch = source.window(begin, begin + 86400., **kwargs)
                if len(batch):
                    self._accumulate(sums, batch)
            day += timedelta(days=stride_days)
        self.table = {key: total / count for key, (total, count) in sums.items()}
        self.fingerprint = {'period': 'train', 'stride_days': stride_days, 'keys': len(self.table),
                            'sources': [s.fingerprint for s in experiment.sources]}

    @staticmethod
    def keys(batch: TokenBatch, month: bool = True) -> np.ndarray:
        """[P, 6] integer keys; the last column is the month or 0 for the month-free fallback."""
        months = _months(batch.observed) if month else np.zeros(len(batch), np.int64)
        return np.stack((batch.platform.astype(np.int64), batch.variable.astype(np.int64),
                         batch.channel.astype(np.int64), _location(batch), _band(batch.pressure_pa), months), 1)

    def _accumulate(self, sums, batch):
        for month in (True, False):
            unique, inverse = np.unique(self.keys(batch, month), axis=0, return_inverse=True)
            inverse = inverse.reshape(-1)
            totals = np.bincount(inverse, batch.value, minlength=len(unique))
            counts = np.bincount(inverse, minlength=len(unique))
            for key, total, count in zip(map(tuple, unique.tolist()), totals, counts):
                old = sums.get(key, (0.0, 0))
                sums[key] = (old[0] + float(total), old[1] + int(count))

    def lookup(self, batch: TokenBatch) -> np.ndarray:
        out = np.full(len(batch), np.nan)
        for month in (False, True):  # month-specific values overwrite the fallback
            unique, inverse = np.unique(self.keys(batch, month), axis=0, return_inverse=True)
            values = np.array([self.table.get(key, np.nan) for key in map(tuple, unique.tolist())])
            found = values[inverse.reshape(-1)]
            out = np.where(np.isfinite(found), found, out)
        return out


def persistence(history: TokenBatch, targets: TokenBatch, issue_s: float, analysis: np.ndarray,
                window_hours: float) -> np.ndarray:
    """Latest earlier report of the same group, variable, channel and pressure band."""
    out = np.full(len(targets), np.nan)
    if not len(history) or not len(targets):
        return out
    order = np.argsort(history.observed, kind='stable')
    keys = list(zip(history.group[order], history.variable[order], history.channel[order],
                    _band(history.pressure_pa)[order]))
    observed = history.observed[order]
    latest_any, latest_old = {}, {}
    cutoff = issue_s - window_hours * 3600.
    for key, when, value in zip(keys, observed, history.value[order]):
        latest_any[key] = value
        if when < cutoff:
            latest_old[key] = value
    target_keys = zip(targets.group, targets.variable, targets.channel, _band(targets.pressure_pa))
    for i, (key, is_analysis) in enumerate(zip(target_keys, analysis)):
        out[i] = (latest_old if is_analysis else latest_any).get(key, np.nan)
    return out


# ----- spectra and balance ----------------------------------------------------------------------

class HarmonicBasis:
    """Real orthonormal spherical harmonics on the latent nodes with area quadrature."""

    def __init__(self, xyz: np.ndarray, areas: np.ndarray, l_max: int):
        import scipy.special as special
        theta = np.arccos(np.clip(xyz[:, 2], -1, 1))
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        rows, degree = [], []
        for l in range(l_max + 1):
            for m in range(l + 1):
                if hasattr(special, 'sph_harm_y'):
                    y = special.sph_harm_y(l, m, theta, phi)
                else:  # SciPy < 1.15
                    y = special.sph_harm(m, l, phi, theta)
                if m == 0:
                    rows.append(y.real); degree.append(l)
                else:
                    rows.extend((math.sqrt(2) * y.real, math.sqrt(2) * y.imag)); degree.extend((l, l))
        self.matrix = np.asarray(rows)                     # [K, N]
        self.degree = np.asarray(degree)
        self.weights = 4 * np.pi * areas / areas.sum()
        self.l_max = l_max

    def coefficients(self, field: np.ndarray) -> np.ndarray:
        return self.matrix @ (field * self.weights)


def vorticity_divergence(model, wind: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Relative vorticity = -div(r x V) and divergence of a tangent Cartesian wind [N, 3], s^-1."""
    args = (model.xyz.double(), model.dual_pairs, model.dual_midpoint.double(), model.dual_normal.double(),
            model.dual_length.double(), model.areas)
    wind = wind.double()
    delta = divergence(wind, *args)  # dual lengths in metres, areas in m^2: already s^-1
    rotated = torch.cross(model.xyz.double().expand_as(wind), wind, dim=-1)
    zeta = -divergence(rotated, *args)
    return zeta, delta


def energy_spectrum(model, wind: torch.Tensor, basis: HarmonicBasis) -> dict:
    zeta, delta = (x.cpu().numpy() for x in vorticity_divergence(model, wind))
    cz, cd = basis.coefficients(zeta), basis.coefficients(delta)
    degree = basis.degree
    valid = degree > 0
    scale = np.where(valid, 1.0 / np.maximum(degree * (degree + 1), 1), 0.0)
    rotational = np.bincount(degree, cz ** 2 * scale, minlength=basis.l_max + 1)
    divergent = np.bincount(degree, cd ** 2 * scale, minlength=basis.l_max + 1)
    total = rotational + divergent
    return {'total': total, 'divergent_fraction': float(divergent.sum() / max(total.sum(), 1e-300))}


# ----- evaluation -------------------------------------------------------------------------------

def _skill_interval(per_issue: list, *, confidence: float, bootstrap: int, block: int, seed: int) -> dict:
    """Block bootstrap over consecutive issues of skill = 1 - sqrt(sum model^2 / sum baseline^2)."""
    model = np.array([r[0] for r in per_issue]); base = np.array([r[1] for r in per_issue])
    count = int(sum(r[2] for r in per_issue))
    if count == 0 or base.sum() <= 0:
        return {'count': count, 'skill': None, 'lower': None, 'upper': None}
    skill = 1 - math.sqrt(model.sum() / base.sum())
    rng = np.random.default_rng(seed)
    n = len(per_issue)
    starts = np.arange(0, n, block)
    draws = []
    for _ in range(bootstrap):
        chosen = rng.choice(starts, len(starts), replace=True)
        index = np.concatenate([np.arange(s, min(s + block, n)) for s in chosen])
        if base[index].sum() > 0:
            draws.append(1 - math.sqrt(model[index].sum() / base[index].sum()))
    tail = (1 - confidence) / 2
    return {'count': count, 'skill': skill, 'lower': float(np.quantile(draws, tail)),
            'upper': float(np.quantile(draws, 1 - tail))}


def evaluate(model, experiment, thresholds: dict, *, split: str, issues: int, seed: int = 0,
             climatology: TrainClimatology, selftest: dict | None = None) -> dict:
    if split not in ('validation', 'test'):
        raise ValueError('Gates are computed on validation, or once on test for the frozen model.')
    g2, g3, g4, g5 = (thresholds[k] for k in ('gate_2', 'gate_3', 'gate_4', 'gate_5'))
    window = experiment.config['data']['analysis_window_hours']
    candidates = experiment.issues(split)
    chosen = [candidates[i] for i in np.linspace(0, len(candidates) - 1, min(issues, len(candidates)), dtype=int)]
    valid_times = set(candidates)
    basis = HarmonicBasis(experiment.mesh.xyz, experiment.mesh.areas_m2, g5['l_max'])
    levels = [int(np.flatnonzero(np.isclose(model.pressure.cpu().numpy(), p * 100.))[0]) for p in g5['pressure_hpa']]
    rows2 = {b: [] for b in g2['baselines']}
    rows4 = {(lead, b): [] for lead in g4['leads_hours'] for b in g4['baselines']}
    drift, spectral = [], {lead: {'fraction': [], 'band': []} for lead in g4['leads_hours']}
    model.noise_scale = 0.0
    band = slice(g5['l_low'], g5['l_high'] + 1)
    with torch.no_grad():
        for index, issue in enumerate(chosen):
            sample = experiment.sample(issue, split, 72, rng=np.random.default_rng(seed + index), training=False)
            if sample.plan.count == 0:
                continue
            rollout = model(sample.tokens, issue, 72)
            masses = np.array([float(model.dry_mass(f)) for f in rollout.physics])
            drift.append(float(np.abs(masses - masses[0]).max() / masses[0]))
            plan = sample.plan
            targets = sample.targets.select(plan.rows)
            predicted = observe(model, rollout, plan).double().cpu().numpy()
            value, sigma = plan.value.double().cpu().numpy(), plan.sigma.double().cpu().numpy()
            lead = plan.lead_hours.double().cpu().numpy()
            analysis = plan.is_analysis.cpu().numpy()
            baselines = {'climatology': climatology.lookup(targets),
                         'persistence': persistence(sample.history, targets, seconds(issue), analysis, window)}
            error = ((predicted - value) / sigma) ** 2
            for name, base in baselines.items():
                base_error = ((base - value) / sigma) ** 2
                paired = np.isfinite(base_error)
                rows = analysis & paired
                if name in rows2:
                    rows2[name].append((float(error[rows].sum()), float(base_error[rows].sum()), int(rows.sum())))
                for hours in g4['leads_hours']:
                    rows = (~analysis) & paired & (np.abs(lead - hours) <= 1.5)
                    rows4[(hours, name)].append((float(error[rows].sum()), float(base_error[rows].sum()), int(rows.sum())))
            for hours in g4['leads_hours']:
                valid = issue + timedelta(hours=hours)
                if valid not in valid_times:
                    continue
                reference = experiment.sample(valid, split, 3, rng=np.random.default_rng(seed + index), training=False)
                analysed = model(reference.tokens, valid, 0).physics[0]['wind']
                forecast = rollout.physics[hours // 3]['wind']
                for level in levels:
                    a = energy_spectrum(model, analysed[:, level], basis)
                    f = energy_spectrum(model, forecast[:, level], basis)
                    spectral[hours]['fraction'].append(f['divergent_fraction'] / max(a['divergent_fraction'], 1e-300))
                    spectral[hours]['band'].append(float(f['total'][band].sum() / max(a['total'][band].sum(), 1e-300)))
    report = {'schema': SCHEMA, 'split': split, 'issues': len(drift), 'thresholds_sha256': None,
              'climatology': climatology.fingerprint, 'era5_used': False, 'gates': {}}
    gates = report['gates']
    if selftest is None:
        gates['0'] = {'passed': False, 'reason': 'self-test report not supplied'}
    else:
        gates['0'] = {'passed': bool(selftest.get('passed')) and all(selftest.get('checks', {}).values()),
                      'checks': selftest.get('checks')}
    reach = min(_reach(experiment.mesh, model, start) for start in _starts(experiment.mesh))
    gates['1'] = {'passed': reach >= thresholds['gate_1']['min_reach_km'], 'min_reach_km': reach,
                  'method': 'exact breadth-first search over multimesh edges, processor layers per step'}
    g2_rows = {name: _skill_interval(rows2[name], confidence=g2['confidence'], bootstrap=g2['bootstrap'],
                                     block=g2['block_issues'], seed=seed) for name in rows2}
    gates['2'] = {'passed': all(r['count'] >= g2['min_count'] and r['lower'] is not None and r['lower'] > g2['min_skill']
                                for r in g2_rows.values()), 'skill': g2_rows}
    gates['3'] = {'passed': bool(drift) and max(drift) <= g3['max_relative_dry_mass_drift'],
                  'max_relative_drift': max(drift) if drift else None}
    g4_rows = {f'{hours}h_{name}': _skill_interval(rows4[(hours, name)], confidence=g4['confidence'],
                                                  bootstrap=g4['bootstrap'], block=g4['block_issues'], seed=seed)
               for (hours, name) in rows4}
    gates['4'] = {'passed': all(r['count'] >= g4['min_count'] and r['lower'] is not None and r['lower'] > g4['min_skill']
                                for r in g4_rows.values()), 'skill': g4_rows}
    g5_rows = {f'{hours}h': {'divergent_fraction_ratio': float(np.median(v['fraction'])) if v['fraction'] else None,
                             'band_energy_ratio': float(np.median(v['band'])) if v['band'] else None,
                             'samples': len(v['band'])} for hours, v in spectral.items()}
    gates['5'] = {'passed': all(r['samples'] > 0 and r['divergent_fraction_ratio'] >= g5['min_divergent_fraction_ratio']
                                and r['band_energy_ratio'] >= g5['min_band_energy_ratio'] for r in g5_rows.values()),
                  'ratios': g5_rows, 'pressure_hpa': g5['pressure_hpa'], 'band_l': [g5['l_low'], g5['l_high']]}
    report['passed'] = all(g['passed'] for g in gates.values())
    report['scientific_acceptance'] = False
    return report


def _starts(mesh) -> list:
    return list(range(12)) + list(np.linspace(12, mesh.n_nodes - 1, 20, dtype=int))


def _reach(mesh, model, start: int) -> float:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    n = mesh.n_nodes
    hops = len(model.dynamics.layers)
    adjacency = csr_matrix((np.ones(len(mesh.senders)), (mesh.senders, mesh.receivers)), shape=(n, n))
    distance = shortest_path(adjacency, unweighted=True, indices=start, directed=True)
    angle = np.arccos(np.clip(mesh.xyz @ mesh.xyz[start], -1, 1))
    return float(6_371_000.0 * angle[distance <= hops].max() / 1000)


def run(training, thresholds_path, output, *, split: str, issues: int, selftest_path=None, climate_stride_days: int = 1,
        surface=None, profile=None, stores=(), static=None, device='cpu', synthetic=False) -> dict:
    from .forecast import load_frozen
    thresholds = load_thresholds(thresholds_path)
    model, experiment, provenance = load_frozen(training, surface=surface, profile=profile, stores=stores,
                                                static=static, device=device, synthetic=synthetic)
    climatology = TrainClimatology(experiment, stride_days=climate_stride_days)
    selftest = json.loads(Path(selftest_path).read_text()) if selftest_path else None
    report = evaluate(model, experiment, thresholds, split=split, issues=issues, climatology=climatology,
                      selftest=selftest)
    report['thresholds_sha256'] = digest(Path(thresholds_path))
    report.update(provenance)
    report['data_kind'] = 'synthetic' if synthetic else 'real'
    output = Path(output)
    if output.exists():
        raise FileExistsError('Gate report exists; choose a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--thresholds', default='configs/physical_assimilation_gates_r11.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--split', default='validation', choices=('validation', 'test'))
    parser.add_argument('--issues', type=int, default=120)
    parser.add_argument('--selftest', help='selftest.json written by the self-test command.')
    parser.add_argument('--climate-stride-days', type=int, default=1)
    parser.add_argument('--surface'); parser.add_argument('--profile')
    parser.add_argument('--store', action='append', default=[]); parser.add_argument('--static')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    report = run(args.training, args.thresholds, args.output, split=args.split, issues=args.issues,
                 selftest_path=args.selftest, climate_stride_days=args.climate_stride_days, surface=args.surface,
                 profile=args.profile, stores=args.store, static=args.static, device=args.device)
    print(json.dumps({'passed': report['passed'], 'gates': {k: v['passed'] for k, v in report['gates'].items()}}))


if __name__ == '__main__':
    main()


__all__ = ['load_thresholds', 'TrainClimatology', 'persistence', 'HarmonicBasis', 'energy_spectrum', 'evaluate', 'run']
