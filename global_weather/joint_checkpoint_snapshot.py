"""Freeze a verified best completed epoch while its parent training may continue.

The snapshot publication is complete; that statement does not mean the planned
parent experiment has completed or that forecast skill has been accepted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from .observation_training import checkpoint_path, digest, save, sync_directory

STATUS = 'joint_measured_observation_research_trained'
SCHEMA = 'joint-completed-epoch-snapshot-1'


def _reference(path):
    if path.is_symlink():
        raise ValueError('Checkpoint reference symlinks are forbidden.')
    value = json.loads(path.read_text())
    if (type(value.get('epoch')) is not int or value['epoch'] < 1
            or not isinstance(value.get('sha256'), str) or len(value['sha256']) != 64):
        raise ValueError('Need an explicit positive completed epoch and SHA256.')
    return value


def _validated_state(state, metrics, epoch):
    if (state.get('epoch') != epoch or state.get('best_epoch') != epoch
            or metrics.get('epoch') != epoch):
        raise ValueError('Selected best completed epoch identity differs.')
    identity = state.get('identity')
    if (not isinstance(identity, dict) or not isinstance(identity.get('config'), dict)
            or identity.get('architecture') != 'joint-measured-hydrostatic-v1'):
        raise ValueError('Checkpoint training identity or config is missing.')
    config = identity['config']
    planned = config.get('epochs')
    horizon = identity.get('joint_fingerprint', {}).get('horizon_hours')
    if (type(planned) is not int or planned < epoch or type(horizon) is not int
            or horizon < 3 or horizon > 72 or horizon % 3):
        raise ValueError('Checkpoint planned budget or forecast horizon is invalid.')
    gradients = metrics.get('nonzero_gradients')
    if (not isinstance(gradients, dict) or not gradients
            or any(not isinstance(name, str) or not name or flag is not True
                   for name, flag in gradients.items())):
        raise ValueError('Missing or incomplete nonzero-gradient evidence.')
    for key, size in (('profile_target_counts', 5), ('native_target_counts', 6)):
        counts = metrics.get(key)
        if (not isinstance(counts, list) or len(counts) != size
                or any(type(value) is not int or value <= 0 for value in counts)):
            raise ValueError('Missing measured target coverage: ' + key)
    for value in (metrics.get('train_loss'), metrics.get('validation', {}).get('joint'), state.get('best')):
        if type(value) not in (int, float) or not np.isfinite(value):
            raise ValueError('Missing or nonfinite completed epoch loss.')
    if state['best'] != metrics['validation']['joint']:
        raise ValueError('Validation best metric differs from completed epoch.')
    support = np.asarray(state.get('trained_variable_mask'))
    if support.shape != (37, 5) or support.dtype != np.bool_ or not support.any(axis=0).all():
        raise ValueError('Missing or inconsistent trained-variable coverage.')
    weights = state.get('model')
    if not isinstance(weights, dict) or not weights:
        raise ValueError('Missing trained model weights.')
    # The fixed joint architecture uses linear/GRU/normalization weights and
    # biases plus the explicitly learned pressure-level embedding. Its remaining
    # state entries are fixed buffers, including sparse graph geometry.
    parameter_names = {
        name for name in weights if isinstance(name, str)
        and (name == 'levels' or name.rsplit('.', 1)[-1].startswith(('weight', 'bias')))
    }
    if not parameter_names or set(gradients) != parameter_names:
        raise ValueError('Nonzero-gradient evidence does not cover exactly the learned parameters.')
    for name, value in weights.items():
        if not isinstance(value, torch.Tensor):
            raise ValueError('Model weight is not a tensor: ' + str(name))
        if value.is_floating_point():
            entries = value.coalesce().values() if value.is_sparse else value
            if not torch.isfinite(entries).all():
                raise ValueError('Nonfinite model weights: ' + name)
    return identity, planned, horizon


def freeze_best_completed_epoch(training, output):
    """Copy immutable epoch state/metrics; leave all parent artifacts untouched."""
    training, output = Path(training), Path(output)
    if training.is_symlink() or any(parent.is_symlink() for parent in training.parents):
        raise ValueError('Training directory symlink is forbidden.')
    if (output.exists() or output.is_symlink()
            or any(parent.is_symlink() for parent in output.parents)):
        raise FileExistsError('Choose a new snapshot directory without symlink parents.')
    if output.resolve().is_relative_to(training.resolve()):
        raise ValueError('Snapshot must be outside the parent training directory.')
    captured = datetime.now(timezone.utc).isoformat()
    best = _reference(training / 'best.json')
    latest = _reference(training / 'latest.json')
    if latest['epoch'] < best['epoch']:
        raise ValueError('Best epoch has not been committed to the completed epoch cursor.')
    source_state = checkpoint_path(training, best)
    checkpoint_path(training, latest)
    source_metrics = source_state.parent / 'metrics.json'
    if source_metrics.is_symlink():
        raise ValueError('Completed epoch metrics symlink is forbidden.')
    metrics_hash = digest(source_metrics)
    metrics = json.loads(source_metrics.read_text())
    state = torch.load(source_state, map_location='cpu', weights_only=True)
    identity, planned, horizon = _validated_state(state, metrics, best['epoch'])
    if latest['epoch'] > planned:
        raise ValueError('Completed epoch cursor exceeds the declared experiment budget.')
    parent_completed = False
    parent_completion_hash = None
    parent_completion_path = training / 'complete.json'
    if parent_completion_path.exists():
        if parent_completion_path.is_symlink():
            raise ValueError('Parent completion symlink is forbidden.')
        completion = json.loads(parent_completion_path.read_text())
        if completion.get('status') != STATUS or completion.get('identity') != identity:
            raise ValueError('Parent completion has a different training identity.')
        parent_completed = True
        parent_completion_hash = digest(parent_completion_path)
    report = {
        'schema': SCHEMA, 'snapshot_publication_status': 'complete',
        'parent_training': str(training.resolve()),
        'parent_training_completed': parent_completed,
        'parent_completion_sha256': parent_completion_hash,
        'epochs_planned': planned, 'completed_at_freeze': latest['epoch'],
        'selected_epoch': best['epoch'], 'checkpoint_sha256': best['sha256'],
        'metrics_sha256': metrics_hash, 'captured_utc': captured,
        'scientific_acceptance': False,
        'limitations': ['snapshot freezes a completed validated epoch, not the entire planned parent experiment',
                        'best selection uses validation only; no independent forecast skill is asserted',
                        'snapshot is for inference only and does not authorize optimizer continuation'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.joint-snapshot-', dir=output.parent) as staging:
        staged = Path(staging) / 'snapshot'
        epoch = staged / best['directory']
        epoch.mkdir(parents=True)
        shutil.copyfile(source_state, epoch / 'state.pt')
        shutil.copyfile(source_metrics, epoch / 'metrics.json')
        if (digest(epoch / 'state.pt') != best['sha256'] or digest(source_state) != best['sha256']
                or digest(epoch / 'metrics.json') != metrics_hash or digest(source_metrics) != metrics_hash):
            raise ValueError('Immutable epoch artifact changed during snapshot publication.')
        for path in (epoch / 'state.pt', epoch / 'metrics.json'):
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
        save(staged / 'best.json', best)
        save(staged / 'latest.json', best)
        # Compatibility with frozen inference concerns a completely published
        # trained epoch. Parent experiment completion is explicitly separate.
        save(staged / 'complete.json', {
            'identity': identity, 'status': STATUS, 'best_epoch': best['epoch'],
            'scientific_acceptance': False, 'trained_variable_mask': state['trained_variable_mask'],
            'lead_hours': list(range(3, horizon + 1, 3)),
            'snapshot_schema': SCHEMA, 'parent_training_completed': parent_completed,
            'epochs_planned': planned, 'completed_at_freeze': latest['epoch'],
            'test_independence': 'previously seen periods; diagnostic only',
        })
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(staged / 'snapshot.json', report)
        sync_directory(epoch)
        sync_directory(staged)
        if output.exists() or output.is_symlink():
            raise FileExistsError('Snapshot output appeared during publication.')
        staged.rename(output)
        sync_directory(output.parent)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    print(json.dumps(freeze_best_completed_epoch(args.training, args.output), allow_nan=False))


if __name__ == '__main__':
    main()
