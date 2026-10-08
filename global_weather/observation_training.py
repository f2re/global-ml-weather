"""Station-native research training; no reanalysis loader or network access."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.nn import functional as F

from .grid import build_pyramid
from .observation_data import ObservationDataset
from .observation_model import StationObservationModel


# Preserve the new main planning API alongside the executable native pilot.
from .observation_stages import (TrainingStage, default_training_stages,
    stage_plan_payload, partition_observation_groups, validate_stage_transition)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def checkpoint_path(root, ref):
    name = ref['directory']
    if not isinstance(name, str) or name != f"epoch-{int(ref['epoch']):04d}":
        raise ValueError('Invalid checkpoint reference')
    folder = root / name
    if folder.is_symlink() or (folder / 'state.pt').is_symlink():
        raise ValueError('Checkpoint symlinks forbidden')
    path = folder / 'state.pt'
    if digest(path) != ref['sha256']: raise ValueError('Checkpoint hash mismatch')
    return path


def verify_dataset(dataset):
    for name, key in [('hourly.npz', 'hourly_sha256'), ('norm.json', 'norm_sha256'), ('observations.sqlite', 'records_sha256')]:
        if digest(dataset.root / name) != dataset.manifest[key]:
            raise ValueError('Dataset changed: ' + name)


def loss(prediction, target, mask):
    """Equal task weights, equal station/lead records within each task."""
    terms = []
    for variable in range(6):
        valid = mask[..., variable]
        if valid.any():
            terms.append(F.huber_loss(prediction[..., variable][valid], target[..., variable][valid]))
    if not terms:
        raise ValueError('No observed target')
    return torch.stack(terms).mean()


def tensors(sample, device):
    return [torch.as_tensor(sample[name], device=device, dtype=(torch.bool if 'mask' in name else torch.float32))
            for name in ('input', 'input_mask', 'normalized_target', 'target_mask')]


def identity(dataset, config, device):
    root = Path(__file__).resolve().parents[1]
    return {'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
            'dataset_manifest_sha256': digest(dataset.root / 'dataset.json'),
            'config': config, 'device': str(device), 'python': platform.python_version(),
            'torch': str(torch.__version__), 'numpy': np.__version__,
            'cuda': torch.version.cuda, 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.machine()}


def create_model(dataset, config, device):
    return StationObservationModel(build_pyramid(config['mesh_level']), dataset.latlon,
                                   dataset.mean, dataset.std, hidden=config['hidden']).to(device)


def validation(model, dataset, device, split):
    values = []
    model.eval()
    with torch.no_grad():
        for index in dataset.subset(split):
            x, xm, y, ym = tensors(dataset.sample(index), device)
            if ym.any():
                values.append(float(loss(model(x, xm).native_normalized, y, ym)))
    if not values:
        raise ValueError('No observed targets in ' + split)
    return float(np.mean(values))


def _train(dataset_path, output, config):
    dataset = ObservationDataset(dataset_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = create_model(dataset, config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    expected = identity(dataset, config, device)
    latest_path = output / 'latest.json'
    completed = 0
    best = float('inf'); best_epoch = None; stale = 0
    if latest_path.exists():
        latest = json.loads(latest_path.read_text())
        path = checkpoint_path(output, latest)
        state = torch.load(path, map_location=device, weights_only=True)
        if state['identity'] != expected:
            raise ValueError('Resume source/data/config/numerical environment differs')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        completed, best, best_epoch, stale = state['epoch'], state['best'], state['best_epoch'], state['stale']
        random.setstate(state['python_rng']); np.random.set_state((state['numpy_rng'][0], np.array(state['numpy_rng'][1], dtype=np.uint32), *state['numpy_rng'][2:]))
        torch.set_rng_state(state['torch_rng'].cpu())
        if device.type == 'cuda': torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        if best_epoch is not None:
            best_ref = latest if best_epoch == completed else state['best_ref']
            checkpoint_path(output, best_ref)
            save(output / 'best.json', best_ref)
    if any(output.glob('.epoch-*')) or (output / f'epoch-{completed + 1:04d}').exists():
        raise ValueError('Ambiguous interrupted epoch; preserve it and inspect before resume')
    for epoch in range(completed + 1, config['epochs'] + 1):
        verify_dataset(dataset)
        if identity(dataset, config, device) != expected: raise ValueError('Identity changed')
        started = time.monotonic(); model.train(); train_losses = []
        gradient_report = {}
        indices = list(dataset.subset('train')); random.shuffle(indices)
        for index in indices:
            x, xm, y, ym = tensors(dataset.sample(index), device)
            if not ym.any() or not xm.any(): continue
            optimizer.zero_grad(set_to_none=True)
            result = model(x, xm)
            objective = loss(result.native_normalized, y, ym)
            if not torch.isfinite(objective): raise FloatingPointError('Nonfinite loss')
            objective.backward()
            current_nonzero = False
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all(): raise FloatingPointError('Nonfinite gradient: ' + name)
                    nonzero = bool(parameter.grad.abs().sum() > 0)
                    current_nonzero |= nonzero
                    gradient_report[name] = gradient_report.get(name, False) or nonzero
                else: gradient_report.setdefault(name, False)
            if not current_nonzero: raise FloatingPointError('No finite nonzero gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); train_losses.append(float(objective.detach()))
        if not train_losses: raise ValueError('No training observations')
        score = validation(model, dataset, device, 'validation')
        improved = score < best
        if improved: best, best_epoch, stale = score, epoch, 0
        else: stale += 1
        folder = output / f'.epoch-{epoch:04d}'; folder.mkdir()
        numpy_rng = np.random.get_state()
        state = {'identity': expected, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                 'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'stale': stale,
                 'best_ref': None if improved else json.loads((output / 'best.json').read_text()),
                 'python_rng': random.getstate(), 'numpy_rng': (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
        torch.save(state, folder / 'state.pt')
        with (folder / 'state.pt').open('rb') as checkpoint: os.fsync(checkpoint.fileno())
        report = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'validation_loss': score,
                  'seconds': time.monotonic() - started, 'nonzero_gradients': gradient_report,
                  'scientific_acceptance': False, 'profile_supervision': False, 'era5_used': False}
        save(folder / 'metrics.json', report)
        verify_dataset(dataset)
        sync_directory(folder)
        final = output / f'epoch-{epoch:04d}'; folder.rename(final); sync_directory(output)
        ref = {'directory': final.name, 'sha256': digest(final / 'state.pt'), 'epoch': epoch}
        save(latest_path, ref)
        if improved: save(output / 'best.json', ref)
        print(json.dumps(report), flush=True)
        if stale >= config['patience']: break
    save(output / 'complete.json', {'identity': expected, 'best_epoch': best_epoch,
         'scientific_acceptance': False, 'status': 'station_native_research_trained',
         'limitations': ['unknown instrument heights', 'no observed profile targets', 'no terrain context', 'ERA5 verification pending']})


def train(dataset_path, output, config):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _train(dataset_path, output, config)


def evaluate(dataset_path, training, output):
    dataset = ObservationDataset(dataset_path); training = Path(training)
    ref = json.loads((training / 'best.json').read_text()); path = checkpoint_path(training, ref)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state = torch.load(path, map_location=device, weights_only=True)
    config = state['identity']['config']; torch.set_num_threads(config['threads'])
    if state['identity'] != identity(dataset, config, device): raise ValueError('Evaluation identity differs')
    model = create_model(dataset, config, device); model.load_state_dict(state['model']); model.eval()
    sums = np.zeros((24, 6, 4)); persistence = np.zeros_like(sums)
    with torch.no_grad():
        for index in dataset.subset('test'):
            sample = dataset.sample(index); x, xm, y, ym = tensors(sample, device)
            predicted = model(x, xm).native_normalized.cpu().numpy() * dataset.std + dataset.mean
            target = sample['target']; mask = sample['target_mask']
            # Last actually available observation per station/variable, with its own mask.
            base = np.zeros_like(sample['input'][0]); base_mask = np.zeros_like(sample['input_mask'][0])
            for row, valid in zip(sample['input'], sample['input_mask']):
                base[valid] = row[valid]; base_mask |= valid
            for lead in range(24):
                for variable in range(6):
                    common = mask[lead, :, variable] & base_mask[:, variable]
                    for prediction, accumulator in ((predicted[lead,:,variable], sums), (base[:,variable], persistence)):
                        error = prediction[common] - target[lead, common, variable]
                        accumulator[lead, variable] += [np.abs(error).sum(), np.square(error).sum(), error.sum(), len(error)]
    def metrics(array):
        return [[None if not cell[3] else {'mae': float(cell[0]/cell[3]), 'rmse': float(np.sqrt(cell[1]/cell[3])),
                    'bias': float(cell[2]/cell[3]), 'count': int(cell[3])} for cell in row] for row in array]
    save(output, {'checkpoint': ref, 'split': 'test', 'leads': list(range(3,73,3)),
                  'model': metrics(sums), 'persistence': metrics(persistence), 'common_target_masks': True,
                  'scientific_acceptance': False, 'era5_verification': 'blocked_native_height_operator_not_admitted'})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('train'); p.add_argument('--dataset', required=True); p.add_argument('--output', required=True); p.add_argument('--config', required=True)
    p = sub.add_parser('test'); p.add_argument('--dataset', required=True); p.add_argument('--training', required=True); p.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if args.command == 'train': train(args.dataset, args.output, json.loads(Path(args.config).read_text()))
    else: evaluate(args.dataset, args.training, args.output)


if __name__ == '__main__': main()
