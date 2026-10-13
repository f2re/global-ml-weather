"""R11 trainer: S1 reconstruction of withheld measurements + S2 forecasts of future ones.

Every optimizer step samples real windows from ALL train issue times
(00/06/12/18 UTC), withholds whole stations/launches/flights as analysis
targets (forcing spatial spreading instead of copying), drops a fraction of
the remaining inputs, rolls the model forward and scores every planned target
in observation space. Rollouts grow by curriculum stages (6 h -> 24 h -> 72 h).
Validation uses fixed issues and holdouts; test is never read here.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch

from ..joint_observation_data import BOUNDARIES, common_windows
from ..observation_training import digest, save, sync_directory
from ..observations import utc
from .mesh import LatentMesh
from .model import ARCHITECTURE, PhysicalAssimilationModel, StateNormalization
from .objective import observation_objective, persistence_skill
from .operator import build_plan, observe
from .sources import GhcnhSource, IgraSource, SyntheticSource, TokenStore, seconds
from .tokens import OPERATOR, VARIABLES, ErrorModel, Normalizer, TokenBatch, encode

CONFIG_SCHEMA = 'physical-assimilation-config-1'
STATUS = 'physical_assimilation_research_trained'
SOURCE_FILES = ('__init__.py', 'thermo.py', 'geometry.py', 'mesh.py', 'tokens.py', 'model.py', 'operator.py', 'objective.py',
                'sources.py', 'drift.py', 'satellite.py', 'static.py', 'training.py')

DEFAULT_CONFIG = {
    'schema': CONFIG_SCHEMA, 'experiment_id': 'physical-assimilation-r11', 'allow_synthetic': False,
    'mesh_level': 5, 'multimesh': True, 'hidden': 256, 'message_dim': 64, 'processor_layers': 8,
    'analysis_layers': 8, 'encoder_neighbours': 7, 'noise_dim': 16, 'conserve_dry_mass': True,
    'checkpoint_steps': True, 'precision': 'bf16', 'seed': 2026, 'threads': 8, 'deterministic': True,
    'issue_stride_hours': 6, 'input_hours': 12, 'horizon_hours': 72,
    'optimizer': {'betas': [0.9, 0.95], 'weight_decay': 0.1, 'gradient_clip': 1.0, 'eps': 1e-8},
    'stages': [
        {'name': 'S1+S2 short', 'rollout_hours': 6, 'epochs': 40, 'issues_per_epoch': 1024, 'issues_per_step': 4,
         'peak_lr': 3e-4, 'min_lr': 3e-6, 'warmup_steps': 500, 'schedule': 'cosine', 'ensemble_members': 1,
         'patience': 8, 'cycling': False},
        {'name': 'S2 24h', 'rollout_hours': 24, 'epochs': 15, 'issues_per_epoch': 512, 'issues_per_step': 4,
         'peak_lr': 3e-5, 'min_lr': 3e-5, 'warmup_steps': 0, 'schedule': 'constant', 'ensemble_members': 1,
         'patience': 5, 'cycling': False},
        {'name': 'S2 72h', 'rollout_hours': 72, 'epochs': 10, 'issues_per_epoch': 256, 'issues_per_step': 4,
         'peak_lr': 1e-5, 'min_lr': 1e-5, 'warmup_steps': 0, 'schedule': 'constant', 'ensemble_members': 1,
         'patience': 4, 'cycling': True}],
    'data': {'max_input_tokens_per_source': 150000, 'max_target_tokens_per_source': 200000, 'input_dropout': 0.2,
             'holdout_fraction': 0.15, 'analysis_window_hours': 1.5, 'analysis_weight': 0.5,
             'family_weights': {'surface': 1.0, 'upper_air': 1.0, 'limb': 1.0, 'radiance': 1.0, 'product': 1.0},
             'satellite_supervision': False, 'validation_issues': 64, 'sonde_drift': True},
    'huber_delta': 3.0,
    'observation_errors': 'configs/observation_errors_r11.json',
    'satellite_admission': None,
}


def _merge(base, update):
    out = dict(base)
    for key, value in update.items():
        if key not in base:
            raise ValueError('Unknown R11 configuration key: ' + key)
        out[key] = _merge(base[key], value) if isinstance(base[key], dict) and isinstance(value, dict) else value
    return out


def load_config(payload: dict) -> dict:
    config = _merge(DEFAULT_CONFIG, payload)
    if config['schema'] != CONFIG_SCHEMA:
        raise ValueError('Unknown R11 configuration schema.')
    ints = ('mesh_level', 'hidden', 'message_dim', 'processor_layers', 'analysis_layers', 'encoder_neighbours',
            'noise_dim', 'seed', 'threads', 'issue_stride_hours', 'input_hours', 'horizon_hours')
    for key in ints:
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError('Invalid integer configuration: ' + key)
    if not 1 <= config['mesh_level'] <= 7 or config['hidden'] < 8 or config['processor_layers'] < 1:
        raise ValueError('Mesh level 1..7, hidden >= 8 and at least one processor layer are required.')
    if config['input_hours'] != 12 or config['horizon_hours'] != 72:
        raise ValueError('R11 keeps the 12-hour observation window and the 72-hour horizon.')
    if config['issue_stride_hours'] not in (1, 3, 6, 12, 24):
        raise ValueError('Issue stride must divide a day.')
    if config['precision'] not in ('fp32', 'bf16'):
        raise ValueError('Precision must be fp32 or bf16.')
    if not config['stages']:
        raise ValueError('At least one curriculum stage is required.')
    previous = 0
    for stage in config['stages']:
        if set(stage) != set(DEFAULT_CONFIG['stages'][0]):
            raise ValueError('Every stage needs exactly the documented fields.')
        hours = stage['rollout_hours']
        if type(hours) is not int or hours % 3 or not 3 <= hours <= config['horizon_hours'] or hours < previous:
            raise ValueError('Stage rollouts must be non-decreasing multiples of 3 h within the horizon.')
        previous = hours
        if stage['schedule'] not in ('cosine', 'constant') or not 0 < stage['min_lr'] <= stage['peak_lr'] <= 1e-2:
            raise ValueError('Invalid learning-rate schedule.')
        for key in ('epochs', 'issues_per_epoch', 'issues_per_step', 'ensemble_members', 'patience'):
            if type(stage[key]) is not int or stage[key] < 1:
                raise ValueError('Invalid stage bound: ' + key)
        if stage['ensemble_members'] > 1 and config['noise_dim'] == 0:
            raise ValueError('A CRPS ensemble stage needs noise_dim > 0.')
        if type(stage['cycling']) is not bool:
            raise ValueError('Stage cycling must be Boolean.')
    data = config['data']
    if not (0 <= data['input_dropout'] < 1 and 0 < data['holdout_fraction'] < 1
            and 0 < data['analysis_window_hours'] <= 3 and data['analysis_weight'] >= 0):
        raise ValueError('Invalid data sampling configuration.')
    if type(data['sonde_drift']) is not bool:
        raise ValueError('sonde_drift must be Boolean.')
    if data['satellite_supervision'] and not config['satellite_admission']:
        raise ValueError('Satellite targets require a physical channel admission file.')
    return config


@dataclass
class Sample:
    tokens: object
    plan: object
    baseline: np.ndarray
    inputs: int
    report: dict
    targets: object = None   # TokenBatch the plan rows index into
    history: object = None   # every causal report of the input window, withheld groups included


class Experiment:
    """Sources, norms, errors and geometry shared by training, validation and forecasts."""

    def __init__(self, config: dict, *, surface=None, profile=None, stores=(), static=None, climatology=None,
                 synthetic: bool = False, root: Path | None = None, device='cpu'):
        self.config = config
        self.device = torch.device(device)
        self.root = root or Path(__file__).resolve().parents[2]
        self.sources = []
        if synthetic:
            if not config['allow_synthetic']:
                raise ValueError('Synthetic sources are refused in a real experiment configuration.')
            self.sources.append(SyntheticSource(config['seed']))
        if surface:
            self.sources.append(GhcnhSource(surface))
        if profile:
            self.sources.append(IgraSource(profile, drift=config['data']['sonde_drift']))
        self.sources.extend(TokenStore(path) for path in stores)
        if not self.sources:
            raise ValueError('At least one observation source is required.')
        if any(s.data_kind != 'real' for s in self.sources) and not config['allow_synthetic']:
            raise ValueError('Only real measurements may train a real experiment.')
        self.channels = sorted({c for s in self.sources for c in getattr(s, 'channels', [])})
        self.channel_index = {c: i for i, c in enumerate(self.channels)}
        self.mesh = LatentMesh(config['mesh_level'], multimesh=config['multimesh'])
        self.normalization = StateNormalization.from_graphcast()
        statistics = {}
        for source in self.sources:
            statistics.update(source.statistics())
        self.normalizer = Normalizer(np.asarray(self.normalization.profile_mean),
                                     np.asarray(self.normalization.profile_std), statistics)
        error_path = Path(config['observation_errors'])
        error_path = error_path if error_path.is_absolute() else self.root / error_path
        self.error_path = error_path
        self.errors = ErrorModel(json.loads(error_path.read_text()))
        from . import satellite
        self.admission, self.admission_path, self.satellite_report = None, None, {}
        if config['satellite_admission']:
            admission_path = Path(config['satellite_admission'])
            self.admission_path = admission_path if admission_path.is_absolute() else self.root / admission_path
            self.admission = satellite.load(self.admission_path)
            stores_only = [s for s in self.sources if isinstance(s, TokenStore)]
            self.satellite_report = satellite.check_against_stores(self.admission, stores_only, statistics)
        self.channel_tables = satellite.channel_tables(self.admission, self.channels)
        self.static, self.static_identity = self._static(static)
        self.climatology = climatology
        self.windows = common_windows(config['horizon_hours'], config['issue_stride_hours'])

    def _static(self, static):
        if static is None:
            if not self.config['allow_synthetic']:
                raise ValueError('A verified static terrain artifact is required.')
            zeros = np.zeros(self.mesh.n_nodes)
            return {'elevation_m': zeros, 'land_fraction': zeros}, {'kind': 'synthetic_flat'}
        from .static import load_latent_static
        fields, identity = load_latent_static(static, self.mesh.grid)
        return fields, identity

    def identity(self) -> dict:
        package = Path(__file__).parent
        return {'architecture': ARCHITECTURE, 'config': self.config,
                'sources': [s.fingerprint for s in self.sources], 'channels': self.channels,
                'normalization': self.normalization.payload(), 'input_statistics': self.normalizer.statistics,
                'observation_errors_sha256': digest(self.error_path), 'static': self.static_identity,
                'satellite_admission_sha256': digest(self.admission_path) if self.admission_path else None,
                'climatology': getattr(self.climatology, 'fingerprint', None),
                'mesh_level': self.mesh.level, 'grid_fingerprint': self.mesh.grid.fingerprint,
                'multimesh_edges': int(len(self.mesh.senders)),
                'source_sha256': {name: digest(package / name) for name in SOURCE_FILES if (package / name).exists()},
                'commit': _commit(self.root), 'python': platform.python_version(), 'torch': str(torch.__version__),
                'numpy': np.__version__, 'device': str(self.device),
                'cuda': torch.version.cuda, 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

    def build_model(self) -> PhysicalAssimilationModel:
        c = self.config
        return PhysicalAssimilationModel(
            self.mesh, self.normalization, self.static, channels=max(1, len(self.channels)), hidden=c['hidden'],
            message_dim=c['message_dim'], processor_layers=c['processor_layers'],
            analysis_layers=c['analysis_layers'], encoder_neighbours=c['encoder_neighbours'],
            noise_dim=c['noise_dim'], conserve_dry_mass=c['conserve_dry_mass'], climatology=self.climatology,
            checkpoint_steps=c['checkpoint_steps'], channel_tables=self.channel_tables).to(self.device)

    def issues(self, split: str) -> list:
        return [utc(row['issue']) for row in self.windows if row['split'] == split]

    def verify(self):
        for source in self.sources:
            source.verify()

    def _window(self, begin, end, *, split, available_before, rng, limit):
        batches = []
        for source in self.sources:
            kwargs = dict(split=split, available_before=available_before, rng=rng, limit=limit)
            if isinstance(source, TokenStore):
                kwargs['channel_index'] = self.channel_index
            batches.append(source.window(begin, end, **kwargs))
        return TokenBatch.concatenate(batches).validate()

    def sample(self, issue, split: str, rollout_hours: int, *, rng, training: bool,
               background_issue=None) -> Sample:
        """Inputs, withheld analysis targets and future targets for one issue.

        With a background from the previous cycle, only reports that became
        available after that cycle's issue are assimilated again (late reports
        included), so no version is used twice by the cycled analysis.
        """
        d = self.config['data']
        issue_s = seconds(issue)
        history = self._window(issue_s - 12 * 3600, issue_s, split=split, available_before=issue_s, rng=rng,
                               limit=d['max_input_tokens_per_source'])
        future = self._window(issue_s, issue_s + rollout_hours * 3600, split=split, available_before=None, rng=rng,
                              limit=d['max_target_tokens_per_source'])
        groups = np.unique(history.group)
        held = groups[rng.random(len(groups)) < d['holdout_fraction']]
        withheld = np.isin(history.group, held)
        inputs = history.select(~withheld)
        if background_issue is not None:
            inputs = inputs.select(inputs.available > seconds(background_issue))
        if training and d['input_dropout'] > 0 and len(inputs):
            inputs = inputs.select(rng.random(len(inputs)) >= d['input_dropout'])
        targets = TokenBatch.concatenate([history.select(withheld), future])
        analysis = np.r_[np.ones(int(withheld.sum()), bool), np.zeros(len(future), bool)]
        column = np.array([OPERATOR[VARIABLES[v]] == 'column' for v in targets.variable], bool)
        if d['satellite_supervision']:
            # Only physically admitted target channels are scored; products never are.
            admitted = (self.channel_tables['target'][targets.channel.astype(np.int64) + 1]
                        & (targets.variable == VARIABLES.index('brightness_temperature')))
            column &= ~admitted
        targets, analysis = targets.select(~column), analysis[~column]
        tokens, usable = encode(inputs, issue_s, self.normalizer, self.errors.sigma(inputs), self.channels,
                                device=self.device, channel_slots=self.channel_tables['slots'])
        stats = {k: v for k, v in self.normalizer.statistics.items() if k.split('|')[1] in ('brightness_temperature', 'satellite_product')}
        plan = build_plan(targets, self.errors.sigma(targets), issue_s, rollout_hours, self.mesh, analysis=analysis,
                          analysis_window_hours=d['analysis_window_hours'], column_statistics=stats,
                          channels=self.channels, analysis_weight=d['analysis_weight'],
                          family_weights=d['family_weights'], device=self.device)
        baseline = _persistence(inputs, targets)[plan.rows]
        report = {'issue': issue.isoformat(), 'history_tokens': len(history), 'input_tokens': len(tokens),
                  'inputs_without_statistics': int((~usable).sum()), 'withheld_groups': int(len(held)),
                  'targets': plan.count, 'rejected': plan.rejected}
        return Sample(tokens, plan, baseline, len(tokens), report, targets, history)


def _persistence(inputs: TokenBatch, targets: TokenBatch) -> np.ndarray:
    """Last available input of the same station/platform group and variable (surface only)."""
    out = np.full(len(targets), np.nan)
    if not len(inputs) or not len(targets):
        return out
    surface = np.array([OPERATOR[VARIABLES[v]] == 'surface' for v in inputs.variable])
    order = np.argsort(inputs.observed[surface], kind='stable')
    latest = {}
    for group, variable, value in zip(inputs.group[surface][order], inputs.variable[surface][order],
                                      inputs.value[surface][order]):
        latest[(int(group), int(variable))] = float(value)
    for i, (group, variable) in enumerate(zip(targets.group, targets.variable)):
        out[i] = latest.get((int(group), int(variable)), np.nan)
    return out


def checkpoint_path(root: Path, reference: dict) -> Path:
    """Resolve an R11 checkpoint reference; refuse renamed, linked or modified states."""
    name = reference.get('directory')
    if name != f"stage-{int(reference['stage'])}-epoch-{int(reference['epoch']):04d}":
        raise ValueError('Invalid R11 checkpoint reference.')
    folder = Path(root) / name
    if folder.is_symlink() or (folder / 'state.pt').is_symlink():
        raise ValueError('Checkpoint symlinks are forbidden.')
    if digest(folder / 'state.pt') != reference['sha256']:
        raise ValueError('Checkpoint hash mismatch.')
    return folder / 'state.pt'


def _commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _parameter_groups(model: torch.nn.Module, weight_decay: float):
    decay, plain = [], []
    for name, parameter in model.named_parameters():
        (plain if parameter.ndim < 2 or 'embedding' in name or name.endswith('bias') else decay).append(parameter)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': plain, 'weight_decay': 0.0}]


def learning_rate(stage: dict, step: int, total: int) -> float:
    """Linear warmup to peak, then cosine to min (or constant)."""
    if stage['warmup_steps'] and step < stage['warmup_steps']:
        return stage['peak_lr'] * (step + 1) / stage['warmup_steps']
    if stage['schedule'] == 'constant':
        return stage['peak_lr']
    progress = min(1.0, (step - stage['warmup_steps']) / max(1, total - stage['warmup_steps']))
    return stage['min_lr'] + 0.5 * (stage['peak_lr'] - stage['min_lr']) * (1 + math.cos(math.pi * progress))


def background_state(model, experiment, issue, split, *, cycle_hours: int = 6):
    """Previous-cycle forecast valid at `issue`, or None when that cycle lies outside the split."""
    previous = issue - timedelta(hours=cycle_hours)
    if previous not in set(experiment.issues(split)):
        return None, None
    sample = experiment.sample(previous, split, cycle_hours, rng=np.random.default_rng(0), training=False)
    noise, model.noise_scale = model.noise_scale, 0.0
    with torch.no_grad():
        state = model(sample.tokens, previous, cycle_hours).states[-1].detach()
    model.noise_scale = noise
    return state, previous


def _members(model, experiment, sample, issue, hours, members, background=None):
    predictions = []
    model.noise_scale = 1.0 if members > 1 else 0.0
    for _ in range(members):
        rollout = model(sample.tokens, issue, hours, background=background)
        predictions.append(observe(model, rollout, sample.plan))
    model.noise_scale = 0.0
    return torch.stack(predictions), rollout


def evaluate(model, experiment: Experiment, split: str, hours: int, *, limit: int, seed: int,
             cycling: bool = False) -> dict:
    """Fixed issues spread over all issue hours; fixed holdouts; no dropout; no gradient."""
    if split == 'test':
        raise ValueError('Test is evaluated only by the separate frozen-model procedure.')
    issues = experiment.issues(split)
    if len(issues) > limit:
        issues = [issues[i] for i in np.linspace(0, len(issues) - 1, limit, dtype=int)]
    model.eval()
    totals, weights, variables, skills = [], [], {}, {}
    with torch.no_grad():
        for index, issue in enumerate(issues):
            background, previous = background_state(model, experiment, issue, split) if cycling else (None, None)
            sample = experiment.sample(issue, split, hours, rng=np.random.default_rng(seed + index), training=False,
                                       background_issue=previous)
            if sample.plan.count == 0:
                continue
            predictions, _ = _members(model, experiment, sample, issue, hours, 1, background)
            loss, report = observation_objective(predictions, sample.plan, huber_delta=experiment.config['huber_delta'])
            totals.append(float(loss)); weights.append(float(sample.plan.weight.sum()))
            for key, row in report['variables'].items():
                entry = variables.setdefault(key, {'count': 0, 'squared': 0.0})
                entry['count'] += row['count']; entry['squared'] += row['rmse'] ** 2 * row['count']
            for key, row in persistence_skill(sample.plan, predictions, sample.baseline).items():
                entry = skills.setdefault(key, {'count': 0, 'model': 0.0, 'baseline': 0.0})
                entry['count'] += row['count']; entry['model'] += row['model_rmse'] ** 2 * row['count']
                entry['baseline'] += row['baseline_rmse'] ** 2 * row['count']
    if not totals:
        raise ValueError('No supported validation targets.')
    return {'split': split, 'rollout_hours': hours, 'issues': len(totals), 'cycling': cycling,
            'loss': float(np.average(totals, weights=weights)),
            'rmse': {k: math.sqrt(v['squared'] / v['count']) for k, v in variables.items()},
            'persistence': {k: {'count': v['count'], 'model_rmse': math.sqrt(v['model'] / v['count']),
                                'persistence_rmse': math.sqrt(v['baseline'] / v['count']),
                                'rmse_skill': 1 - math.sqrt(v['model'] / v['baseline']) if v['baseline'] else None}
                            for k, v in skills.items()}}


def train(config: dict, output, *, surface=None, profile=None, stores=(), static=None, climatology=None,
          synthetic=False, device='auto', max_steps: int | None = None) -> dict:
    config = load_config(config)
    selected = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    if config['deterministic']:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    experiment = Experiment(config, surface=surface, profile=profile, stores=stores, static=static,
                            climatology=climatology, synthetic=synthetic, device=selected)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        model = experiment.build_model()
        optimizer = torch.optim.AdamW(_parameter_groups(model, config['optimizer']['weight_decay']),
                                      lr=config['stages'][0]['peak_lr'], betas=tuple(config['optimizer']['betas']),
                                      eps=config['optimizer']['eps'])
        identity = experiment.identity()
        state = {'stage': 0, 'epoch': 0, 'stage_step': 0, 'global_step': 0, 'best': {}, 'stale': 0}
        if (output / 'latest.json').exists():
            reference = json.loads((output / 'latest.json').read_text())
            saved = torch.load(checkpoint_path(output, reference), map_location=selected, weights_only=True)
            if saved['identity'] != identity:
                raise ValueError('R11 data, norms, errors, sources, config or numerical environment changed.')
            model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
            state = saved['progress']
            random.setstate(saved['python_rng'])
            generator = saved['numpy_rng']
            np.random.set_state((generator[0], np.asarray(generator[1], dtype=np.uint32), *generator[2:]))
            torch.set_rng_state(saved['torch_rng'])
            if selected.type == 'cuda':
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
        rng = np.random.default_rng(config['seed'] + state['global_step'])
        train_issues = experiment.issues('train')
        autocast = (torch.autocast(selected.type, dtype=torch.bfloat16) if config['precision'] == 'bf16'
                    and selected.type == 'cuda' else torch.autocast(selected.type, enabled=False))
        history = []
        for stage_index in range(state['stage'], len(config['stages'])):
            stage = config['stages'][stage_index]
            steps_per_epoch = math.ceil(stage['issues_per_epoch'] / stage['issues_per_step'])
            total = steps_per_epoch * stage['epochs']
            for epoch in range(state['epoch'] + 1, stage['epochs'] + 1):
                experiment.verify()
                model.train()
                started, losses, reports = time.monotonic(), [], []
                gradients = {name: False for name, _ in model.named_parameters()}
                order = rng.permutation(len(train_issues))[:stage['issues_per_epoch']]
                for start in range(0, len(order), stage['issues_per_step']):
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate(stage, state['stage_step'], total)
                    optimizer.zero_grad(set_to_none=True)
                    batch_losses = []
                    for index in order[start:start + stage['issues_per_step']]:
                        issue = train_issues[index]
                        background, previous = (background_state(model, experiment, issue, 'train')
                                                if stage['cycling'] else (None, None))
                        sample = experiment.sample(issue, 'train', stage['rollout_hours'], rng=rng, training=True,
                                                   background_issue=previous)
                        if sample.plan.count == 0:
                            continue
                        with autocast:
                            predictions, _ = _members(model, experiment, sample, issue, stage['rollout_hours'],
                                                      stage['ensemble_members'], background)
                        loss, report = observation_objective(predictions.float(), sample.plan,
                                                             huber_delta=config['huber_delta'])
                        (loss / stage['issues_per_step']).backward()
                        batch_losses.append(float(loss.detach()))
                        reports.append({k: report[k] for k in ('targets', 'rejected')} | sample.report)
                    if not batch_losses:
                        continue
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            if not torch.isfinite(parameter.grad).all():
                                raise FloatingPointError('Nonfinite gradient: ' + name)
                            gradients[name] = gradients[name] or bool(parameter.grad.abs().sum() > 0)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['gradient_clip'],
                                                   error_if_nonfinite=True)
                    optimizer.step()
                    state['stage_step'] += 1; state['global_step'] += 1
                    losses.append(float(np.mean(batch_losses)))
                    save(output / 'progress.json', {'stage': stage['name'], 'epoch': epoch,
                                                     'global_step': state['global_step'], 'loss': losses[-1],
                                                     'lr': optimizer.param_groups[0]['lr'],
                                                     'seconds': time.monotonic() - started})
                    if max_steps is not None and state['global_step'] >= max_steps:
                        break
                validation = evaluate(model, experiment, 'validation', stage['rollout_hours'],
                                      limit=config['data']['validation_issues'], seed=config['seed'],
                                      cycling=stage['cycling'])
                key = str(stage_index)
                improved = validation['loss'] < state['best'].get(key, {'loss': math.inf})['loss']
                state['stale'] = 0 if improved else state['stale'] + 1
                state['epoch'] = epoch
                if improved:
                    # Recorded before saving so a resumed run keeps its best references.
                    state['best'][key] = {'loss': validation['loss'], 'epoch': epoch,
                                          'directory': f'stage-{stage_index}-epoch-{epoch:04d}'}
                inactive = sorted(name for name, active in gradients.items() if not active)
                metrics = {'stage': stage['name'], 'stage_index': stage_index, 'epoch': epoch,
                           'train_loss': float(np.mean(losses)) if losses else None, 'validation': validation,
                           'optimizer_steps': state['global_step'], 'seconds': time.monotonic() - started,
                           'parameters_without_gradient': inactive, 'samples': reports[-4:],
                           'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(selected) if selected.type == 'cuda' else None,
                           'scientific_acceptance': False}
                directory = _save_checkpoint(output, model, optimizer, identity, state, metrics, stage_index, epoch)
                if improved:
                    record = {'stage': stage_index, **state['best'][key], 'sha256': digest(directory / 'state.pt')}
                    save(output / f'best-stage-{stage_index}.json', record)
                    save(output / 'best.json', record)
                history.append(metrics)
                print(json.dumps({k: metrics[k] for k in ('stage', 'epoch', 'train_loss', 'optimizer_steps')}
                                 | {'validation_loss': validation['loss']}), flush=True)
                if state['stale'] >= stage['patience'] or (max_steps is not None and state['global_step'] >= max_steps):
                    break
            state.update(stage=stage_index + 1, epoch=0, stage_step=0, stale=0)
            if max_steps is not None and state['global_step'] >= max_steps:
                break
        completion = {'identity': identity, 'status': STATUS, 'best': state['best'], 'scientific_acceptance': False,
                      'omega': 'continuity_diagnostic_with_kinematic_surface_boundary',
                      'geopotential': 'hydrostatic_from_surface_geopotential',
                      'dry_mass': 'conserved_by_uniform_surface_pressure_corrector' if config['conserve_dry_mass'] else 'not_constrained',
                      'test_independence': 'test split not read by the trainer'}
        save(output / 'complete.json', completion)
        return {'history': history, 'completion': completion}


def _save_checkpoint(output: Path, model, optimizer, identity, state, metrics, stage_index, epoch) -> Path:
    temporary = output / f'.stage-{stage_index}-epoch-{epoch:04d}'
    temporary.mkdir()
    rng = np.random.get_state()
    torch.save({'identity': identity, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'progress': state, 'python_rng': random.getstate(), 'numpy_rng': (rng[0], rng[1].tolist(), *rng[2:]),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, temporary / 'state.pt')
    with (temporary / 'state.pt').open('rb') as file:
        os.fsync(file.fileno())
    save(temporary / 'metrics.json', metrics)
    sync_directory(temporary)
    final = output / f'stage-{stage_index}-epoch-{epoch:04d}'
    temporary.rename(final)
    sync_directory(output)
    save(output / 'latest.json', {'directory': final.name, 'stage': stage_index, 'epoch': epoch,
                                  'sha256': digest(final / 'state.pt')})
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--surface', help='Admitted GHCNh native-station dataset directory.')
    parser.add_argument('--profile', help='Admitted IGRA measured-profile dataset directory.')
    parser.add_argument('--store', action='append', default=[], help='Prepared token store (any other platform).')
    parser.add_argument('--static', help='Latent-mesh static terrain artifact.')
    parser.add_argument('--climatology', help='Seasonal climatology prepared on the latent mesh level.')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args(argv)
    climatology = None
    if args.climatology:
        from ..seasonal_climatology import SeasonalClimatology
        payload = json.loads(Path(args.config).read_text())
        from ..grid import build_grid
        climatology = SeasonalClimatology(args.climatology, build_grid(payload.get('mesh_level', 5)))
    result = train(json.loads(Path(args.config).read_text()), args.output, surface=args.surface, profile=args.profile,
                   stores=args.store, static=args.static, climatology=climatology, device=args.device)
    print(json.dumps({'status': result['completion']['status'], 'best': result['completion']['best']}))


if __name__ == '__main__':
    main()
