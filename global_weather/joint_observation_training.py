"""Bounded joint measured-observation experiment with an explicit mesh-8 qualifier.

The historical measured GHCNh+IGRA contour remains reproducible.  Mesh level 8
is admitted only through a deliberately bounded end-to-end qualification run:
3 h rollout, small hidden width, chunked edge exchange and explicit static and
climatology artifacts.  A 72 h mesh-8 science run is a later experiment and is
not silently enabled by this module.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .grid import build_grid, build_mesh8_grid, grid_cell_count, nominal_area_scale_km
from .joint_observation_data import JointObservationDataset
from .joint_observation_model import ARCHITECTURE, JointObservationModel
from .observation_data import START, TRAIN_END
from .observation_training import checkpoint_path, digest, save, sync_directory
from .observations import utc
from .profile_normalization import load_normalization
from .profile_physics import parse_policy, screen_humidity
from .profile_training import DEFAULT_CONFIG as PROFILE_DEFAULT_CONFIG, identity, _restore_rng
from .profile_training_v2 import objective
from .station_anomaly_model import anomaly_loss
from .station_climatology import StationClimatology

STATUS = 'joint_measured_observation_research_trained'
MESH8_ROLE = 'mesh8_end_to_end_qualification'
MESH8_TRAIN_ROLE = 'mesh8_research_training'


def _profile_config(value):
    """Validate shared training bounds without inheriting the old mesh<=2 pilot gate."""
    if set(value) - set(PROFILE_DEFAULT_CONFIG):
        raise ValueError('Unknown joint profile training configuration: ' + ', '.join(sorted(set(value)-set(PROFILE_DEFAULT_CONFIG))))
    result = dict(PROFILE_DEFAULT_CONFIG)
    result.update(value)
    for key in ('mesh_level', 'hidden', 'epochs', 'patience', 'threads', 'seed',
                'max_train_issues', 'max_validation_issues', 'max_test_issues',
                'max_records_per_window'):
        if type(result[key]) is not int:
            raise ValueError('Joint integer configuration differs: '+key)
    for key in ('hidden', 'epochs', 'patience', 'threads', 'max_train_issues',
                'max_validation_issues', 'max_test_issues', 'max_records_per_window'):
        if result[key] <= 0:
            raise ValueError('Invalid joint training bound: '+key)
    if not 0 <= result['mesh_level'] <= 8:
        raise ValueError('Joint mesh_level must be within 0..8.')
    if result['epochs'] > 4000 or result['max_records_per_window'] > 20000:
        raise ValueError('Joint experiment exceeds reviewed research limits.')
    for key in ('learning_rate', 'weight_decay', 'gradient_clip'):
        number = result[key]
        if type(number) not in (int, float) or not np.isfinite(number) or number < 0:
            raise ValueError('Invalid joint optimizer configuration: '+key)
    if result['learning_rate'] <= 0 or result['gradient_clip'] <= 0:
        raise ValueError('Positive learning rate and gradient clip are required.')
    return result


def _config(value):
    value = dict(value)
    policy = parse_policy(value.pop('physical_policy', {'architecture': 'hydrostatic_flow'}))
    if policy.architecture != 'hydrostatic_flow':
        raise ValueError('Joint experiment uses reviewed hydrostatic flow architecture.')
    surface_weight = value.pop('surface_weight', .5)
    if type(surface_weight) not in (int, float) or not 0 < surface_weight < 1:
        raise ValueError('Both measured branches require a positive loss weight.')
    surface_bias_weight = value.pop('surface_bias_weight', 0.5)
    if type(surface_bias_weight) not in (int, float) or not 0 <= surface_bias_weight <= 10:
        raise ValueError('surface_bias_weight must be a finite float >= 0.')
    rollout_hours = value.pop('rollout_hours', 72)
    edge_chunk_size = value.pop('edge_chunk_size', None)
    experiment_id = value.pop('experiment_id', 'joint-measured-observation-pilot')
    experiment_role = value.pop('experiment_role', 'joint_measured_science_pilot')
    scientific_acceptance = value.pop('scientific_acceptance', False)
    if (type(rollout_hours) is not int or rollout_hours < 3 or rollout_hours > 72
            or rollout_hours % 3):
        raise ValueError('Joint rollout_hours must be a 3-hour step within 3..72.')
    if edge_chunk_size is not None and (type(edge_chunk_size) is not int or edge_chunk_size < 1):
        raise ValueError('edge_chunk_size must be a positive integer or null.')
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError('A nonempty experiment_id is required.')
    if not isinstance(experiment_role, str) or not experiment_role.strip():
        raise ValueError('A nonempty experiment_role is required.')
    if scientific_acceptance is not False:
        raise ValueError('Joint research configuration cannot predeclare scientific acceptance.')

    result = _profile_config(value)
    if result['hidden'] > 128:
        raise ValueError('Joint hidden exceeds reviewed budget.')
    if result['mesh_level'] == 8:
        if experiment_role == MESH8_ROLE:
            if rollout_hours != 3:
                raise ValueError('Mesh-8 qualification is intentionally limited to a 3-hour rollout.')
            if result['hidden'] > 4:
                raise ValueError('Mesh-8 qualification hidden width is capped at 4 pending measured VRAM telemetry.')
            if result['epochs'] != 1:
                raise ValueError('Mesh-8 qualification performs exactly one training epoch.')
            if result['max_train_issues'] > 4 or result['max_validation_issues'] > 2 or result['max_test_issues'] > 2:
                raise ValueError('Mesh-8 qualification issue counts exceed the reviewed execution envelope.')
        elif experiment_role == MESH8_TRAIN_ROLE:
            if rollout_hours != 3:
                raise ValueError('Mesh-8 research training maintains the 3-hour rollout from measured VRAM budget.')
            if result['hidden'] > 4:
                raise ValueError('Mesh-8 research training hidden width is capped at 4 from measured 24 GiB VRAM.')
            if result['epochs'] < 1 or result['epochs'] > 4000:
                raise ValueError('Mesh-8 research training epochs must be within 1..4000.')
            if result['max_train_issues'] > 9 or result['max_validation_issues'] > 13 or result['max_test_issues'] > 13:
                raise ValueError('Mesh-8 research training issue counts exceed supported dataset issues.')
        else:
            raise ValueError('mesh_level=8 requires a supported mesh8 role (qualification or research training).')
        if edge_chunk_size is None or edge_chunk_size > 65536:
            raise ValueError('Mesh-8 requires explicit edge chunks <= 65536.')
        if policy.memory_budget_mib < 16384:
            raise ValueError('Mesh-8 requires at least a declared 16 GiB activation budget.')
        if grid_cell_count(8) != 655_362 or not 27.0 < nominal_area_scale_km(8) < 29.0:
            raise RuntimeError('Mesh-8 geometry contract differs from the reviewed experiment.')
    result.update(surface_weight=float(surface_weight), surface_bias_weight=float(surface_bias_weight),
                  physical_policy=policy.payload(),
                  rollout_hours=rollout_hours, edge_chunk_size=edge_chunk_size,
                  experiment_id=experiment_id, experiment_role=experiment_role,
                  scientific_acceptance=False)
    return result


def _device(name='auto'):
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else name)


def _build(dataset, norms, config, station_climate, device, climate_path=None, static_path=None):
    # The joint model consumes only the finest grid. Building 8->0 pyramids here
    # wasted memory and made the 655k-cell qualification needlessly expensive.
    # Mesh8 remains an explicit opt-in so historical callers keep the old guard.
    grid = build_mesh8_grid() if config['mesh_level'] == 8 else build_grid(config['mesh_level'])
    climate = None
    if climate_path is not None:
        if config['mesh_level'] == 8:
            from .mesh8_climatology import Mesh8SeasonalClimatology
            climate = Mesh8SeasonalClimatology(climate_path, grid)
        else:
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
    model.surface_bias_weight = config.get('surface_bias_weight', 0.5)
    model.static_path = static_path
    model.static_fingerprint = static.get('fingerprint') if static is not None else None
    model.physical_policy = parse_policy(config['physical_policy'])
    model.edge_chunk_size = config.get('edge_chunk_size')
    return model


def normalized_history(model, raw, mask, issue):
    origin = utc(issue)
    months = torch.tensor([(origin - timedelta(hours=11-h)).month-1 for h in range(12)], device=model.xyz.device)
    x = torch.as_tensor(raw, dtype=torch.float32, device=model.xyz.device)
    m = torch.as_tensor(mask, dtype=torch.bool, device=model.xyz.device)
    baseline = model.surface_monthly[months]
    return torch.where(m, (x-baseline)/model.surface_scales, torch.zeros_like(x)), m


def supported_issues(dataset, split, limit, rollout_hours):
    """Select chronologically bounded issues that have real profile target coverage."""
    if not hasattr(dataset, 'profile') or not hasattr(dataset.profile, 'root'):
        return dataset.issues(split, limit)
    sqlite_path = dataset.profile.root / 'observations.sqlite'
    if not sqlite_path.is_file():
        return dataset.issues(split, limit)
    import sqlite3
    uri = sqlite_path.resolve().as_uri() + '?mode=ro'
    all_issues = dataset.issues(split)
    candidates = []
    with sqlite3.connect(uri, uri=True) as database:
        cursor = database.cursor()
        for issue in all_issues:
            start = issue.isoformat()
            end = (issue + timedelta(hours=rollout_hours)).isoformat()
            cursor.execute(
                'SELECT count(*) FROM records WHERE observed > ? AND observed <= ?',
                (start, end)
            )
            if cursor.fetchone()[0] > 0:
                candidates.append(issue)
    if not candidates:
        return dataset.issues(split, limit)
    if limit is not None and len(candidates) > limit:
        candidates = [candidates[i] for i in np.linspace(0, len(candidates) - 1, limit, dtype=int)]
    return candidates


def _batch(model, dataset, issue, limit, rollout_hours):
    try:
        raw, mask, target, target_mask, inputs, targets = dataset.sample(
            issue, limit, target_horizon_hours=rollout_hours)
    except TypeError:
        raw, mask, target, target_mask, inputs, targets = dataset.sample(issue, limit)
    x, m = normalized_history(model, raw, mask, issue)
    policy = model.physical_policy
    inputs, _ = screen_humidity(inputs, policy)
    cutoff = utc(issue) + timedelta(hours=rollout_hours)
    targets = [row for row in targets if utc(row['observed_at']) <= cutoff]
    targets, report = screen_humidity(targets, policy)
    steps = rollout_hours // 3
    target = target[:steps]
    target_mask = target_mask[:steps]
    frames = model(inputs, issue, surface_inputs=x, surface_mask=m, horizon_hours=rollout_hours)
    if not targets:
        profile_loss = torch.tensor(0.0, device=model.xyz.device)
        counts = [0] * 5
    else:
        profile_loss, counts = objective(model, frames, targets)
    prediction = torch.stack([f.native_normalized for f in frames[1:]])
    y = torch.as_tensor(target, dtype=torch.float32, device=model.xyz.device)
    ym = torch.as_tensor(target_mask, dtype=torch.bool, device=model.xyz.device)
    months = torch.tensor([f.valid_time.month-1 for f in frames[1:]], device=model.xyz.device)
    anomalies = (y-model.surface_monthly[months])/model.surface_scales
    bias_weight = getattr(model, 'surface_bias_weight', 0.5)
    surface_loss = anomaly_loss(prediction, anomalies, ym, bias_weight=bias_weight)
    return profile_loss, surface_loss, counts, targets, ym.sum((0, 1)).cpu().numpy(), report


def score(model, dataset, config, split, issues=None):
    model.eval()
    rows = []
    with torch.no_grad():
        if issues is None:
            issues = supported_issues(dataset, split, config['max_'+split+'_issues'], config['rollout_hours'])
        for issue in issues:
            p, s, *_ = _batch(model, dataset, issue, config['max_records_per_window'], config['rollout_hours'])
            rows.append((float(p), float(s)))
    if not rows:
        raise ValueError('No joint '+split+' support.')
    means = np.mean(rows, axis=0)
    return {'profile': float(means[0]), 'surface': float(means[1]),
            'joint': float((1-config['surface_weight'])*means[0]+config['surface_weight']*means[1]),
            'issues': len(rows), 'rollout_hours': config['rollout_hours'],
            'dataset_horizon_hours': dataset.horizon_hours}


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


def _prune_intermediate_checkpoints(output, current_epoch, best_epoch):
    """Retain milestone checkpoints (every 50 epochs), latest, and best; prune intermediate state.pt to preserve disk space."""
    free_bytes = shutil.disk_usage(output).free
    low_disk = free_bytes < 30 * (1024**3)
    start_k = 1 if low_disk else 2001
    for k in range(start_k, current_epoch):
        if k % 50 != 0 and k != best_epoch:
            state_file = output / f'epoch-{k:04d}' / 'state.pt'
            if state_file.is_file():
                try:
                    state_file.unlink()
                except OSError:
                    pass


def train(dataset_path, norm_path, output, config, climatology_path=None, static_path=None, warm_start=None):
    config = _config(config)
    dataset = JointObservationDataset(dataset_path)
    if dataset.horizon_hours != 72:
        raise ValueError('Joint dataset identity currently requires the reviewed 72-hour guarded archive.')
    if config['rollout_hours'] > dataset.horizon_hours:
        raise ValueError('Experiment rollout exceeds the prepared guarded dataset horizon.')
    if config['mesh_level'] == 8 and (climatology_path is None or static_path is None):
        raise ValueError('Mesh-8 qualification requires verified climatology and static terrain artifacts.')
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
        if config['mesh_level'] == 8 and device.type != 'cuda':
            raise ValueError('Mesh-8 end-to-end qualification requires a CUDA compute node.')
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
                    'experiment_id': config['experiment_id'], 'experiment_role': config['experiment_role'],
                    'grid_level': model.grid.level, 'grid_cells': model.grid.n_cells,
                    'grid_fingerprint': model.grid.fingerprint,
                    'grid_area_method': model.grid.area_method,
                    'nominal_area_scale_km': model.grid.nominal_area_scale_km,
                    'rollout_hours': config['rollout_hours'],
                    'joint_dataset_path': str(Path(dataset_path).resolve()), 'joint_fingerprint': dataset.fingerprint,
                    'norm_path': str(norm_path), 'norm_sha256': norm_hash,
                    'station_climatology_path': str(station_path), 'station_climatology_sha256': station_hash,
                    'climatology_path': str(Path(climatology_path).resolve()) if climatology_path else None,
                    'climatology_sha256': model.climatology.fingerprint if climatology_path else None,
                    'static_path': str(Path(static_path).resolve()) if static_path else None,
                    'static_fingerprint': model.static_fingerprint,
                    'joint_source_sha256': {name: digest(Path(__file__).parent/name) for name in
                        ('grid.py', 'joint_observation_model.py', 'joint_observation_training.py', 'joint_observation_data.py',
                         'mesh8_climatology.py', 'profile_hydrostatic_model.py', 'profile_model_v2.py',
                         'profile_physics.py', 'profile_seasonal_model.py')}}
        completed, best, best_epoch, stale = 0, float('inf'), None, 0
        best_ref = None
        trained = np.zeros((37, 5), dtype=bool)
        if (output/'latest.json').exists():
            ref = json.loads((output/'latest.json').read_text())
            previous = torch.load(checkpoint_path(output, ref), map_location=device, weights_only=True)
            prev_ident = previous['identity']
            schedule_keys = {'epochs', 'patience', 'experiment_id', 'surface_bias_weight'}
            prev_cfg = dict(prev_ident.get('config', {}))
            exp_cfg = dict(expected.get('config', {}))
            prev_pol = {k: v for k, v in prev_cfg.get('physical_policy', {}).items() if k != 'bias_weight'}
            exp_pol = {k: v for k, v in exp_cfg.get('physical_policy', {}).items() if k != 'bias_weight'}
            shared_keys = (set(prev_cfg) | set(exp_cfg)) - schedule_keys - {'physical_policy'}
            config_match = (
                all(prev_cfg.get(k) == exp_cfg.get(k) for k in shared_keys) and
                prev_pol == exp_pol and
                exp_cfg.get('epochs', 0) >= previous['epoch']
            )
            ignore_keys = {'commit', 'config', 'joint_source_sha256', 'experiment_id'} if config_match else set()
            ident_match = (
                prev_ident == expected or (
                    config_match and
                    all(prev_ident.get(k) == expected.get(k) for k in set(prev_ident) | set(expected) if k not in ignore_keys)
                )
            )
            if not ident_match:
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
        elif warm_start is not None and Path(warm_start).is_file():
            warm_state = torch.load(warm_start, map_location=device, weights_only=True)
            _state(model, warm_state['model'])
            if 'trained_variable_mask' in warm_state:
                trained = np.asarray(warm_state['trained_variable_mask'], dtype=bool)
        cached_train_issues = supported_issues(dataset, 'train', config['max_train_issues'], config['rollout_hours'])
        cached_val_issues = supported_issues(dataset, 'validation', config['max_validation_issues'], config['rollout_hours'])
        if not cached_train_issues or not cached_val_issues:
            raise ValueError('No supported joint train/validation issues.')
        for epoch in range(completed+1, config['epochs']+1):
            _verify(dataset, norms, norm_path, norm_hash, model, station_path, station_hash)
            started = time.monotonic(); model.train(); losses = []; gradients = {}; counts = np.zeros(5, int); native_counts = np.zeros(6, int)
            if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
            issues = list(cached_train_issues)
            random.shuffle(issues)
            for issue in issues:
                optimizer.zero_grad(set_to_none=True)
                p, s, coverage, targets, nc, qc = _batch(
                    model, dataset, issue, config['max_records_per_window'], config['rollout_hours'])
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
                    'seconds': time.monotonic()-started, 'device': str(device), 'source_commit': expected['commit'],
                    'experiment_id': config['experiment_id'], 'mesh_level': model.grid.level,
                    'grid_cells': model.grid.n_cells, 'rollout_hours': config['rollout_hours'], 'qc': qc})
            if not losses or not all(gradients.values()) or not (counts > 0).all() or not (native_counts > 0).all():
                raise ValueError('Incomplete joint branch or measured variable gradient support.')
            validation = score(model, dataset, config, 'validation', issues=cached_val_issues)
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
                       'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'experiment_id': config['experiment_id'], 'experiment_role': config['experiment_role'],
                       'mesh_level': model.grid.level, 'grid_cells': model.grid.n_cells,
                       'grid_fingerprint': model.grid.fingerprint, 'grid_area_method': model.grid.area_method,
                       'nominal_area_scale_km': model.grid.nominal_area_scale_km,
                       'rollout_hours': config['rollout_hours'], 'edge_chunk_size': config.get('edge_chunk_size'),
                       'scientific_acceptance': False}
            save(temporary/'metrics.json', metrics); sync_directory(temporary)
            final = output/f'epoch-{epoch:04d}'; temporary.rename(final); sync_directory(output)
            ref = {'directory': final.name, 'epoch': epoch, 'sha256': digest(final/'state.pt')}
            save(output/'latest.json', ref)
            if improved:
                best_ref = ref
                save(output/'best.json', best_ref)
            _prune_intermediate_checkpoints(output, epoch, best_epoch)
            print(json.dumps(metrics), flush=True)
            if stale >= config['patience']: break
        save(output/'complete.json', {'identity': expected, 'status': STATUS, 'best_epoch': best_epoch,
            'experiment_id': config['experiment_id'], 'experiment_role': config['experiment_role'],
            'mesh_level': model.grid.level, 'grid_cells': model.grid.n_cells,
            'nominal_area_scale_km': model.grid.nominal_area_scale_km,
            'rollout_hours': config['rollout_hours'], 'scientific_acceptance': False,
            'omega': 'unavailable_no_measured_targets',
            'below_ground_issuance': 'unknown_global_surface_pressure',
            'test_independence': 'previously seen periods; diagnostic only',
            'trained_variable_mask': trained.tolist(),
            'lead_hours': list(range(3, config['rollout_hours']+1, 3)),
            'dataset_lead_hours': dataset.lead_hours})


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
    if model.grid.fingerprint != expected.get('grid_fingerprint') or model.grid.n_cells != expected.get('grid_cells'):
        raise ValueError('Frozen mesh geometry differs.')
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
    parser.add_argument('--warm-start')
    args = parser.parse_args(argv)
    train(args.dataset, args.norm, args.output, json.loads(Path(args.config).read_text()),
          args.climatology, args.static, warm_start=args.warm_start)


if __name__ == '__main__': main()
