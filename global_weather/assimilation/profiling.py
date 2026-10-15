"""Memory and time of one R11 training sample per curriculum stage, on the device that will train.

For each distinct (rollout, ensemble members, cycling) stage the command
builds one real train sample, runs forward and backward exactly as the
trainer does (bf16 autocast on CUDA, activation checkpointing as configured)
and records:

* CUDA: peak allocated and reserved bytes (torch.cuda.max_memory_*), device
  total, and whether the peak leaves `headroom` free;
* CPU: process peak RSS (monotonic, so stages are run from short to long);
* forward and backward seconds, input tokens and planned targets.

The trainer accumulates gradients over `issues_per_step` samples
sequentially, so one sample bounds the activation memory of a step. No
optimizer step is taken and no checkpoint is written.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ..observation_training import save
from .objective import observation_objective
from .training import Experiment, _members, background_state, load_config

SCHEMA = 'physical-assimilation-profile-1'


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def profile(config: dict, *, surface=None, profile=None, stores=(), static=None, synthetic=False,
            device='auto', headroom: float = 0.1, seed: int = 0) -> dict:
    config = load_config(config)
    selected = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
    experiment = Experiment(config, surface=surface, profile=profile, stores=stores, static=static,
                            synthetic=synthetic, device=selected)
    model = experiment.build_model()
    model.train()
    autocast = (torch.autocast(selected.type, dtype=torch.bfloat16) if config['precision'] == 'bf16'
                and selected.type == 'cuda' else torch.autocast(selected.type, enabled=False))
    issues = experiment.issues('train')
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    for stage in sorted(config['stages'], key=lambda s: (s['rollout_hours'], s['ensemble_members'])):
        key = (stage['rollout_hours'], stage['ensemble_members'], stage['cycling'])
        if key in seen:
            continue
        seen.add(key)
        issue = issues[len(issues) // 2]
        background, previous = (background_state(model, experiment, issue, 'train') if stage['cycling']
                                else (None, None))
        sample = experiment.sample(issue, 'train', stage['rollout_hours'], rng=rng, training=True,
                                   background_issue=previous)
        if selected.type == 'cuda':
            torch.cuda.synchronize(selected)
            torch.cuda.reset_peak_memory_stats(selected)
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with autocast:
            predictions, _ = _members(model, experiment, sample, issue, stage['rollout_hours'],
                                      stage['ensemble_members'], background)
        loss, _ = observation_objective(predictions.float(), sample.plan, huber_delta=config['huber_delta'])
        if selected.type == 'cuda':
            torch.cuda.synchronize(selected)
        forward = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        if selected.type == 'cuda':
            torch.cuda.synchronize(selected)
        backward = time.perf_counter() - started
        row = {'stage': stage['name'], 'rollout_hours': stage['rollout_hours'],
               'ensemble_members': stage['ensemble_members'], 'cycling': stage['cycling'],
               'input_tokens': sample.inputs, 'targets': sample.plan.count,
               'forward_seconds': forward, 'backward_seconds': backward,
               'peak_rss_bytes': _rss_bytes()}
        if selected.type == 'cuda':
            total = torch.cuda.get_device_properties(selected).total_memory
            peak = torch.cuda.max_memory_reserved(selected)
            row.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(selected),
                       cuda_peak_reserved_bytes=peak, cuda_total_bytes=total,
                       fits_with_headroom=bool(peak <= (1 - headroom) * total))
        rows.append(row)
    model.zero_grad(set_to_none=True)
    return {'schema': SCHEMA, 'device': str(selected),
            'device_name': torch.cuda.get_device_name(selected) if selected.type == 'cuda' else 'cpu',
            'precision': config['precision'] if selected.type == 'cuda' else 'fp32 (bf16 only on CUDA)',
            'checkpoint_steps': config['checkpoint_steps'],
            'parameters': int(sum(p.numel() for p in model.parameters())),
            'mesh_level': config['mesh_level'], 'hidden': config['hidden'],
            'data_kind': 'synthetic' if synthetic else 'real', 'headroom': headroom, 'stages': rows,
            'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'scientific_acceptance': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--surface'); parser.add_argument('--profile')
    parser.add_argument('--store', action='append', default=[]); parser.add_argument('--static')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--headroom', type=float, default=0.1)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Profile report exists; choose a new file.')
    report = profile(json.loads(Path(args.config).read_text()), surface=args.surface, profile=args.profile,
                     stores=args.store, static=args.static, device=args.device, headroom=args.headroom)
    output.parent.mkdir(parents=True, exist_ok=True)
    save(output, report)
    print(json.dumps({'device': report['device'], 'stages': [
        {k: r.get(k) for k in ('rollout_hours', 'ensemble_members', 'peak_rss_bytes', 'cuda_peak_reserved_bytes',
                               'fits_with_headroom')} for r in report['stages']]}))


if __name__ == '__main__':
    main()
