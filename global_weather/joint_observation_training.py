"""Bounded fresh joint experiment; measured targets and atomic C2 epochs only."""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from .grid import build_pyramid
from .joint_observation_data import JointObservationDataset
from .joint_observation_model import ARCHITECTURE, JointObservationModel
from .observation_data import START, TRAIN_END
from .observation_training import checkpoint_path, digest, save, sync_directory
from .observations import utc
from .profile_normalization import load_normalization
from .profile_physics import parse_policy, screen_humidity
from .profile_training import configuration, identity, _restore_rng
from .profile_training_v2 import objective
from .station_anomaly_model import anomaly_loss
from .station_climatology import StationClimatology

STATUS = 'joint_measured_observation_research_trained'


def _config(value):
    value = dict(value)
    policy = parse_policy(value.pop('physical_policy', {'architecture': 'hydrostatic_flow'}))
    if policy.architecture != 'hydrostatic_flow':
        raise ValueError('Joint experiment uses reviewed hydrostatic flow architecture.')
    surface_weight = value.pop('surface_weight', .5)
    if type(surface_weight) not in (int, float) or not 0 < surface_weight < 1:
        raise ValueError('Both measured branches require a positive loss weight.')
    result = configuration(value)
    if result['hidden'] > 128:
        raise ValueError('Joint hidden exceeds reviewed budget.')
    result.update(surface_weight=surface_weight, physical_policy=policy.payload())
    return result


def _device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)


def _build(dataset, norms, config, station_climate, device, climate_path=None, static_path=None):
    grid = build_pyramid(config['mesh_level'])[0]
    climate = None
    if climate_path is not None:
        from .seasonal_climatology import SeasonalClimatology
        climate = SeasonalClimatology(climate_path, grid=grid)
    static = None
    if static_path is not None:
        from .joint_static import load_static
        loaded = load_static(static_path, grid)
        static = {'terrain_elevation_m': loaded['elevation_m'],
                  'terrain_land_fraction': loaded['land_mask'].astype(np.float32),
                  'terrain_slope': loaded['slope'], 'fingerprint': loaded['static_sha256']}
    model = JointObservationModel(grid, norms, dataset.surface.latlon, station_climate.std,
                                  station_climate.monthly, config['hidden'], climate, static).to(device)
    model.static_path = static_path
    model.static_fingerprint = static.get('fingerprint') if static is not None else None
    model.physical_policy = parse_policy(config['physical_policy'])
    return model


def normalized_history(model, raw, mask, issue):
    origin = utc(issue)
    months = torch.tensor([(origin - timedelta(hours=11-h)).month-1 for h in range(12)], device=model.xyz.device)
    x = torch.as_tensor(raw, dtype=torch.float32, device=model.xyz.device)
    m = torch.as_tensor(mask, dtype=torch.bool, device=model.xyz.device)
    baseline = model.surface_monthly[months]
    return torch.where(m, (x-baseline)/model.surface_scales, torch.zeros_like(x)), m


def _batch(model, dataset, issue, limit):
    raw, mask, target, target_mask, inputs, targets = dataset.sample(issue, limit)
    x, m = normalized_history(model, raw, mask, issue)
    policy = model.physical_policy
    inputs, _ = screen_humidity(inputs, policy)
    targets, report = screen_humidity(targets, policy)
    frames = model(inputs, issue, surface_inputs=x, surface_mask=m, horizon_hours=dataset.horizon_hours)
    profile_loss, counts = objective(model, frames, targets)
    prediction = torch.stack([f.native_normalized for f in frames[1:]])
    y = torch.as_tensor(target, dtype=torch.float32, device=model.xyz.device)
    ym = torch.as_tensor(target_mask, dtype=torch.bool, device=model.xyz.device)
    months = torch.tensor([f.valid_time.month-1 for f in frames[1:]], device=model.xyz.device)
    anomalies = (y-model.surface_monthly[months])/model.surface_scales
    surface_loss = anomaly_loss(prediction, anomalies, ym)
    return profile_loss, surface_loss, counts, targets, ym.sum((0, 1)).cpu().numpy(), report


def score(model, dataset, config, split):
    model.eval()
    rows = []
    with torch.no_grad():
        for issue in dataset.issues(split, config['max_'+split+'_issues']):
            p, s, *_ = _batch(model, dataset, issue, config['max_records_per_window'])
            rows.append((float(p), float(s)))
    if not rows:
        raise ValueError('No joint '+split+' support.')
    means = np.mean(rows, axis=0)
    return {'profile': float(means[0]), 'surface': float(means[1]),
            'joint': float((1-config['surface_weight'])*means[0]+config['surface_weight']*means[1]),
            'issues': len(rows), 'horizon_hours': dataset.horizon_hours}


def _state(model, state):
    for name, buffer in model.named_buffers():
        if name not in state:
            raise ValueError('Checkpoint fixed buffer differs: '+name)
        restored = state[name].to(buffer.device)
        if buffer.is_sparse or restored.is_sparse:
            equal = buffer.is_sparse and restored.is_sparse and buffer.shape == restored.shape
            if equal:
                current, saved = buffer.coalesce(), restored.coalesce()
                equal = (torch.equal(current.indices(), saved.indices())
                         and torch.equal(current.values(), saved.values()))
        else:
            equal = torch.equal(buffer, restored)
        if not equal:
            raise ValueError('Checkpoint fixed buffer differs: '+name)
    model.load_state_dict(state, strict=True)


def _verify(dataset, norms, norm_path, norm_hash, model, station_path, station_hash):
    dataset.verify()
    norms.verify_sources()
    if digest(norm_path) != norm_hash or digest(station_path) != station_hash:
        raise ValueError('Joint fixed normalization changed.')
    if hasattr(model, 'climatology'):
        model.climatology.verify_sources()
    if model.static_path is not None:
        from .joint_static import load_static
        if load_static(model.static_path, model.grid)['static_sha256'] != model.static_fingerprint:
            raise ValueError('Immutable terrain changed.')


def train(dataset_path, norm_path, output, config, climatology_path=None, static_path=None):
    config = _config(config)
    dataset = JointObservationDataset(dataset_path)
    if dataset.horizon_hours != 72:
        raise ValueError('New joint experiment requires the reviewed 72-hour horizon.')
    norm_path = Path(norm_path).resolve()
    norm_hash = digest(norm_path)
    norms = load_normalization(json.loads(norm_path.read_text()))
    if norms.payload['source_identity'] != {
            'dataset_manifest_sha256': dataset.profile.manifest_sha256,
            **{key: dataset.profile.manifest[key] for key in ('database_sha256', 'source_sha256', 'admission_sha256')}}:
        raise ValueError('GraphCast norms belong to another admitted profile dataset.')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output/'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = _device()
        torch.set_num_threads(config['threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        station_path = output/'station_climatology.npz'
        if station_path.exists():
            station_climate = StationClimatology.load(station_path)
        else:
            if (output/'latest.json').exists():
                raise ValueError('Resume station climatology is missing.')
            timestamps = [START+timedelta(hours=h) for h in range(len(dataset.surface.values))]
            station_climate = StationClimatology.fit(dataset.surface.values, dataset.surface.mask, timestamps,
                                                    np.array([t < TRAIN_END for t in timestamps]))
            station_climate.save(station_path)
        station_hash = digest(station_path)
        model = _build(dataset, norms, config, station_climate, device, climatology_path, static_path)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        expected = {**identity(dataset.profile, config, device), 'architecture': ARCHITECTURE,
                    'joint_dataset_path': str(Path(dataset_path).resolve()), 'joint_fingerprint': dataset.fingerprint,
                    'norm_path': str(norm_path), 'norm_sha256': norm_hash,
                    'station_climatology_path': str(station_path), 'station_climatology_sha256': station_hash,
                    'climatology_path': str(Path(climatology_path).resolve()) if climatology_path else None,
                    'climatology_sha256': model.climatology.fingerprint if climatology_path else None,
                    'static_path': str(Path(static_path).resolve()) if static_path else None,
                    'static_fingerprint': model.static_fingerprint,
                    'joint_source_sha256': {name: digest(Path(__file__).parent/name) for name in
                        ('joint_observation_model.py', 'joint_observation_training.py', 'joint_observation_data.py',
                         'profile_hydrostatic_model.py', 'profile_model_v2.py', 'profile_physics.py')}}
        completed, best, best_epoch, stale = 0, float('inf'), None, 0
        best_ref = None
        trained = np.zeros((37, 5), dtype=bool)
        if (output/'latest.json').exists():
            ref = json.loads((output/'latest.json').read_text())
            previous = torch.load(checkpoint_path(output, ref), map_location=device, weights_only=True)
            if previous['identity'] != expected:
                raise ValueError('Joint C2 source/data/norm/optimizer/numerical identity changed.')
            _state(model, previous['model']); optimizer.load_state_dict(previous['optimizer']); _restore_rng(previous, device)
            completed, best, best_epoch, stale = previous['epoch'], previous['best'], previous['best_epoch'], previous['stale']
            best_ref = ref if best_epoch == completed else previous.get('best_ref')
            if best_ref is None and (output/'best.json').exists():
                best_ref = json.loads((output/'best.json').read_text())
            if best_ref is None or best_ref['epoch'] != best_epoch:
                raise ValueError('Joint C2 best checkpoint reference is missing or inconsistent.')
            checkpoint_path(output, best_ref)
            save(output/'best.json', best_ref)
            trained = np.asarray(previous['trained_variable_mask'], dtype=bool)
        for epoch in range(completed+1, config['epochs']+1):
            _verify(dataset, norms, norm_path, norm_hash, model, station_path, station_hash)
            started = time.monotonic(); model.train(); losses = []; gradients = {}; counts = np.zeros(5, int); native_counts = np.zeros(6, int)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            issues = dataset.issues('train', config['max_train_issues']); random.shuffle(issues)
            for issue in issues:
                optimizer.zero_grad(set_to_none=True)
                p, s, coverage, targets, nc, qc = _batch(model, dataset, issue, config['max_records_per_window'])
                loss = (1-config['surface_weight'])*p+config['surface_weight']*s
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite joint loss.')
                loss.backward(); counts += coverage; native_counts += nc
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all(): raise FloatingPointError('Nonfinite gradient: '+name)
                    gradients[name] = gradients.get(name, False) or (param.grad is not None and bool(param.grad.abs().sum() > 0))
                from .profile_training import VARIABLES
                for target in targets:
                    index = VARIABLES.index(target['variable'])
                    level = int(np.abs(np.log(norms.pressure_pa)-np.log(target['pressure_pa'])).argmin())
                    trained[level, index] = True
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
                save(output/'progress.json', {'epoch': epoch, 'optimizer_steps': len(losses), 'train_issues': len(issues),
                    'loss': losses[-1], 'profile_loss': float(p.detach()), 'surface_loss': float(s.detach()),
                    'seconds': time.monotonic()-started, 'device': str(device), 'source_commit': expected['commit'], 'qc': qc})
            if not losses or not all(gradients.values()) or not (counts > 0).all() or not (native_counts > 0).all():
                raise ValueError('Incomplete joint branch or measured variable gradient support.')
            validation = score(model, dataset, config, 'validation')
            improved = validation['joint'] < best
            if improved: best, best_epoch, stale = validation['joint'], epoch, 0
            else: stale += 1
            _verify(dataset, norms, norm_path, norm_hash, model, station_path, station_hash)
            temporary = output/f'.epoch-{epoch:04d}'; temporary.mkdir()
            rng = np.random.get_state()
            state = {'identity': expected, 'epoch': epoch, 'best': best, 'best_epoch': best_epoch, 'stale': stale,
                     'best_ref': best_ref,
                     'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'trained_variable_mask': trained.tolist(),
                     'python_rng': random.getstate(), 'numpy_rng': (rng[0], rng[1].tolist(), *rng[2:]),
                     'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []}
            torch.save(state, temporary/'state.pt')
            with (temporary/'state.pt').open('rb') as file: os.fsync(file.fileno())
            metrics = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'validation': validation,
                       'seconds': time.monotonic()-started, 'nonzero_gradients': gradients,
                       'profile_target_counts': counts.tolist(), 'native_target_counts': native_counts.tolist(),
                       'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                       'parameters': sum(p.numel() for p in model.parameters()), 'scientific_acceptance': False}
            save(temporary/'metrics.json', metrics); sync_directory(temporary)
            final = output/f'epoch-{epoch:04d}'; temporary.rename(final); sync_directory(output)
            ref = {'directory': final.name, 'epoch': epoch, 'sha256': digest(final/'state.pt')}
            save(output/'latest.json', ref)
            if improved:
                best_ref = ref
                save(output/'best.json', best_ref)
            print(json.dumps(metrics), flush=True)
            if stale >= config['patience']: break
        save(output/'complete.json', {'identity': expected, 'status': STATUS, 'best_epoch': best_epoch,
            'scientific_acceptance': False, 'omega': 'unavailable_no_measured_targets',
            'below_ground_issuance': 'unknown_global_surface_pressure',
            'test_independence': 'previously seen periods; diagnostic only',
            'trained_variable_mask': trained.tolist(), 'lead_hours': dataset.lead_hours})


def load_frozen(dataset_path, training, device='auto'):
    training = Path(training)
    completion = json.loads((training/'complete.json').read_text())
    if completion['status'] != STATUS: raise ValueError('Joint training is incomplete.')
    ref = json.loads((training/'best.json').read_text())
    state = torch.load(checkpoint_path(training, ref), map_location='cpu', weights_only=True)
    expected = completion['identity']
    if state['identity'] != expected or ref['epoch'] != completion['best_epoch']: raise ValueError('Joint best identity differs.')
    if expected.get('joint_source_sha256') != {name: digest(Path(__file__).parent/name)
                                              for name in expected.get('joint_source_sha256', {})}:
        raise ValueError('Frozen joint model source changed.')
    dataset = JointObservationDataset(dataset_path)
    if dataset.fingerprint != expected['joint_fingerprint']: raise ValueError('Frozen joint dataset changed.')
    norms = load_normalization(json.loads(Path(expected['norm_path']).read_text()))
    station = StationClimatology.load(expected['station_climatology_path'])
    model = _build(dataset, norms, expected['config'], station, _device(device), expected['climatology_path'], expected.get('static_path'))
    _verify(dataset, norms, expected['norm_path'], expected['norm_sha256'], model,
            expected['station_climatology_path'], expected['station_climatology_sha256'])
    if model.static_fingerprint != expected.get('static_fingerprint'):
        raise ValueError('Frozen terrain fingerprint differs.')
    if hasattr(model, 'climatology') and model.climatology.fingerprint != expected['climatology_sha256']:
        raise ValueError('Frozen climate fingerprint differs.')
    _state(model, state['model']); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    model.trained_variable_mask = np.asarray(state['trained_variable_mask'], dtype=bool)
    model.frozen_metadata = {'config': expected['config'], 'epoch': state['epoch'],
                             'identity': expected, 'trained_variable_mask': state['trained_variable_mask']}
    model.frozen_checkpoint_sha256 = ref['sha256']
    return model, dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True); parser.add_argument('--norm', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--config', required=True)
    parser.add_argument('--climatology'); parser.add_argument('--static')
    args = parser.parse_args(argv)
    train(args.dataset, args.norm, args.output, json.loads(Path(args.config).read_text()), args.climatology, args.static)


if __name__ == '__main__': main()
