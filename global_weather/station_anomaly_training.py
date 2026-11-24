"""Training pipeline for station anomaly models.

Learns surface atmospheric dynamics by supervising departures from climatology.
All operations are fully deterministic and track active gradient propagation.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import timedelta
import fcntl
import json
import os
from pathlib import Path
import random

import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from .grid import build_grid, build_mesh8_grid
from .observation_data import ObservationDataset, campaign_bounds
from .observation_model import LEAD_HOURS
from .observations import utc
from .observation_training import (
    checkpoint_path,
    digest,
    save,
    sync_directory,
)
from .station_anomaly_model import (
    StationAnomalyModel,
    anomaly_loss,
)
from .station_climatology import StationClimatology
from .pipeline.io import read_json

DEFAULT_CONFIG = {
    'mesh_level': 0,
    'hidden': 16,
    'epochs': 10,
    'patience': 3,
    'threads': 4,
    'seed': 42,
    'learning_rate': 1e-3,
    'weight_decay': 0.01,
    'gradient_clip': 1.0,
    'max_train_samples': 48,
    'max_val_samples': 16,
}

SATELLITE_STATUS = 'station_satellite_anomaly_research_trained'


@dataclass(frozen=True)
class SatelliteWarmStart:
    manifest: Path
    base_training: Path
    checkpoint_sha256: str
    climatology_sha256: str
    dataset_sha256: str


def create_model(dataset: ObservationDataset, climatology: StationClimatology, config: dict, device: torch.device,
                 *, sensors=None):
    mesh_level = config.get('mesh_level', 0)
    grid = build_mesh8_grid() if mesh_level == 8 else build_grid(mesh_level)
    if sensors is not None:
        from .station_anomaly_model import SatelliteStationAnomalyModel

        return SatelliteStationAnomalyModel([grid], dataset.latlon, climatology.std,
                                             sensors=sensors, hidden=config['hidden']).to(device)
    return StationAnomalyModel(
        [grid],
        dataset.latlon,
        climatology.std,
        hidden=config['hidden'],
    ).to(device)


def _sample(dataset, climatology, idx, device, *, require_station_input):
    row = dataset.sample(idx)
    x_mask, y_mask = row['input_mask'], row['target_mask']
    if not y_mask.any() or (require_station_input and not x_mask.any()):
        return None
    issue = utc(row['issue'])
    c_inputs = climatology.for_window([issue + timedelta(hours=h) for h in range(-11, 1)])
    c_targets = climatology.for_window([issue + timedelta(hours=h) for h in LEAD_HOURS])
    x = np.where(x_mask, (row['input'] - c_inputs) / climatology.std, 0.0).astype(np.float32)
    y = np.where(y_mask, (row['target'] - c_targets) / climatology.std, 0.0).astype(np.float32)
    return tuple(torch.as_tensor(a, device=device) for a in (x, x_mask, y, y_mask))


def _predict(model, dataset, idx, x, x_mask, satellite_data):
    if satellite_data is None:
        return model(x, x_mask)
    result = model(x, x_mask, issue_time=dataset.samples[idx]['issue'],
                   satellite_sequences=satellite_data.sequences(idx))
    if result.satellite_support is None or not result.satellite_support.any():
        raise ValueError('The assigned satellite sequence has no effective support on the model grid.')
    return result


def evaluate(model: StationAnomalyModel, dataset: ObservationDataset, climatology: StationClimatology,
             split: str, max_samples: int, device: torch.device, *, indices=None, satellite_data=None) -> float:
    model.eval()
    losses = []
    indices = dataset.subset(split)[:max_samples] if indices is None else indices
    with torch.no_grad():
        for idx in indices:
            sample = _sample(dataset, climatology, idx, device, require_station_input=satellite_data is not None)
            if sample is None:
                continue
            tx_anom, tx_mask, ty_anom, ty_mask = sample
            forecast = _predict(model, dataset, idx, tx_anom, tx_mask, satellite_data)
            loss = anomaly_loss(forecast.predicted_anomalies, ty_anom, ty_mask)
            losses.append(float(loss.detach().cpu()))

    return float(np.mean(losses)) if losses else float('inf')


def _fit_climatology(dataset):
    start = campaign_bounds(dataset.campaign)[0]
    timestamps = [start + timedelta(hours=h) for h in range(len(dataset.values))]
    train_hours = np.zeros(len(dataset.values), dtype=bool)
    for idx in dataset.subset('train'):
        h = dataset.samples[idx]['hour']
        train_hours[max(0, h - 11):min(len(dataset.values), h + 73)] = True
    return StationClimatology.fit(dataset.values, dataset.mask, timestamps, train_hours)


def _satellite_parent(dataset, cfg, warm_start):
    training = warm_start.base_training
    best = read_json(training / 'best.json')
    completion = read_json(training / 'complete.json')
    climate_path = training / 'station_climatology.npz'
    if (completion['status'] != 'surface_climatology_anomaly_research_trained'
            or completion['best_epoch'] != best['epoch'] or best['sha256'] != warm_start.checkpoint_sha256
            or digest(climate_path) != warm_start.climatology_sha256
            or digest(dataset.root / 'dataset.json') != warm_start.dataset_sha256):
        raise ValueError('The declared station-only parent, climatology, or dataset hashes do not match.')
    checkpoint = torch.load(checkpoint_path(training, best), map_location='cpu', weights_only=True)
    if (checkpoint['epoch'] != best['epoch'] or checkpoint['best_loss'] != completion['best_loss']
            or any(checkpoint['config'][key] != cfg[key] for key in ('mesh_level', 'hidden'))):
        raise ValueError('The parent checkpoint differs in epoch, grid, hidden width, or validation state.')
    return checkpoint['model'], StationClimatology.load(climate_path)


def _satellite_indices(dataset, satellite_data, split, limit):
    selected = []
    for idx in satellite_data.indices(split):
        hour = dataset.samples[idx]['hour']
        window = slice(hour - 11, hour + 1)
        if (dataset.mask[window] & (dataset.available[window] <= hour * 3600)).any() and (
                dataset.mask[hour + 3:hour + 73:3]).any():
            selected.append(idx)
            if len(selected) == limit:
                break
    if not selected:
        raise ValueError(f'No eligible {split} issue has station input, station targets and admitted satellite data.')
    return selected


def train(dataset_path: str | Path, output: str | Path, config: dict | None = None,
          climatology_path: str | Path | None = None, *, warm_start: SatelliteWarmStart | None = None) -> None:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    output = Path(output)
    satellite_data = None
    if warm_start is not None:
        from .station_satellite_data import StationSatelliteData

        if climatology_path is not None:
            raise ValueError('The satellite run must use the verified parent station climatology.')
        if any(type(cfg[key]) is not int or cfg[key] < 1 for key in (
                'epochs', 'patience', 'max_train_samples', 'max_val_samples')):
            raise ValueError('Epoch and sample limits must be positive integers.')
        satellite_data = StationSatelliteData(warm_start.manifest, dataset_path)
        if satellite_data.station_sha != warm_start.dataset_sha256 or not all(
                satellite_data.indices(split) for split in ('train', 'validation', 'test')):
            raise ValueError('The declared dataset and all three verified satellite partitions are required.')
    output.mkdir(parents=True, exist_ok=warm_start is None)
    if warm_start is not None:
        print('Verifying the frozen station observation data', flush=True)
    dataset = ObservationDataset(dataset_path)

    with (output / 'training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(cfg['threads'])
        torch.use_deterministic_algorithms(True)
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])

        identity = {}
        train_indices = dataset.subset('train')[:cfg['max_train_samples']]
        val_indices = None
        baseline_loss = None
        gradient_prefixes = ()
        if warm_start is not None:
            from shutil import copyfile

            train_indices = _satellite_indices(dataset, satellite_data, 'train', cfg['max_train_samples'])
            val_indices = _satellite_indices(dataset, satellite_data, 'validation', cfg['max_val_samples'])
            parent, climatology = _satellite_parent(dataset, cfg, warm_start)
            model = create_model(dataset, climatology, cfg, device, sensors=satellite_data.sensors)
            model.initialize_station_core(parent)
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(name.startswith(model.satellite_prefixes))
            gradient_prefixes = ('source_fusion.', *(f'satellite_bank.encoders.{sensor.id}.'
                                                    for sensor in satellite_data.sensors))
            baseline_loss = evaluate(model, dataset, climatology, 'validation', cfg['max_val_samples'],
                                     device, indices=val_indices)
            if not np.isfinite(baseline_loss):
                raise ValueError('The matched station-only validation loss is not finite.')
            copyfile(warm_start.base_training / 'station_climatology.npz', output / 'station_climatology.npz')
            if digest(output / 'station_climatology.npz') != warm_start.climatology_sha256:
                raise ValueError('The station climatology changed during copying.')
            identity = {
                'base_checkpoint_sha256': warm_start.checkpoint_sha256,
                'station_climatology_sha256': warm_start.climatology_sha256,
                'station_dataset_sha256': warm_start.dataset_sha256,
                'satellite_input_signature': satellite_data.signature,
                'station_core_frozen': True,
            }
            print(json.dumps(dict(identity, train_issues=len(train_indices), validation_issues=len(val_indices),
                                  station_only_val_anomaly_loss=baseline_loss)), flush=True)
        else:
            if climatology_path and Path(climatology_path).exists():
                climatology = StationClimatology.load(climatology_path)
            else:
                climatology = _fit_climatology(dataset)
                climatology.save(output / 'station_climatology.npz')
            model = create_model(dataset, climatology, cfg, device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

        best_loss = float('inf')
        best_epoch = None
        stale = 0
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            epoch_losses = []
            gradients = {}
            random.shuffle(train_indices)

            for idx in train_indices:
                sample = _sample(dataset, climatology, idx, device, require_station_input=True)
                if sample is None:
                    continue
                tx_anom, tx_mask, ty_anom, ty_mask = sample
                optimizer.zero_grad(set_to_none=True)
                forecast = _predict(model, dataset, idx, tx_anom, tx_mask, satellite_data)
                loss = anomaly_loss(forecast.predicted_anomalies, ty_anom, ty_mask)

                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite anomaly loss.")
                loss.backward()

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = gradients.get(name, False) or bool(param.grad.abs().sum() > 0)

                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                              error_if_nonfinite=warm_start is not None)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))

            if not all(any(value for name, value in gradients.items() if name.startswith(prefix))
                       for prefix in gradient_prefixes):
                raise FloatingPointError('Satellite encoder and fusion must both receive nonzero gradients.')
            val_loss = evaluate(model, dataset, climatology, 'validation', cfg['max_val_samples'], device,
                                indices=val_indices, satellite_data=satellite_data)
            if warm_start is not None and not np.isfinite(val_loss):
                raise FloatingPointError('The satellite validation loss is not finite.')
            improved = val_loss < best_loss
            if improved:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1

            # Save epoch checkpoint
            temp_dir = output / f'.epoch-{epoch:04d}'
            temp_dir.mkdir(exist_ok=True)
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss,
                'best_epoch': best_epoch,
                'config': cfg,
            }
            if identity:
                state['identity'] = identity
            torch.save(state, temp_dir / 'state.pt')
            metrics = {
                'epoch': epoch,
                'train_anomaly_loss': float(np.mean(epoch_losses)) if epoch_losses else None,
                'val_anomaly_loss': val_loss,
                'improved': improved,
                'nonzero_gradients': gradients,
                'prediction_target': 'dimensionless anomaly departure from climatology',
                'scientific_acceptance': False,
            }
            if identity:
                metrics.update(station_only_val_anomaly_loss=baseline_loss, train_issues=len(train_indices),
                               validation_issues=len(val_indices))
            save(temp_dir / 'metrics.json', metrics)
            sync_directory(temp_dir)
            final_dir = output / f'epoch-{epoch:04d}'
            temp_dir.rename(final_dir)
            sync_directory(output)

            ref = {'directory': final_dir.name, 'epoch': epoch, 'sha256': digest(final_dir / 'state.pt')}
            save(output / 'latest.json', ref)
            if improved:
                save(output / 'best.json', ref)

            print(json.dumps(metrics), flush=True)
            if stale >= cfg['patience']:
                break

        completion = {
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'status': 'surface_climatology_anomaly_research_trained',
            'prediction_target': 'dimensionless anomaly departure from climatology',
            'climatology_source': 'station-monthly-climatology-1',
            'scientific_acceptance': False,
        }
        if identity:
            completion.update(status=SATELLITE_STATUS, identity=identity, station_only_val_anomaly_loss=baseline_loss)
        save(output / 'complete.json', completion)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config')
    parser.add_argument('--climatology')
    parser.add_argument('--satellite-manifest', type=Path)
    parser.add_argument('--base-training', type=Path)
    parser.add_argument('--expected-base-sha')
    parser.add_argument('--expected-climatology-sha')
    parser.add_argument('--expected-dataset-sha')
    args = parser.parse_args(argv)
    config = json.loads(Path(args.config).read_text()) if args.config else None
    fields = (args.satellite_manifest, args.base_training, args.expected_base_sha,
              args.expected_climatology_sha, args.expected_dataset_sha)
    if any(fields) and (not all(fields) or args.climatology or not args.config):
        parser.error('Satellite training requires a config, a satellite manifest, a base training directory, '
                     'and the three expected hashes. Omit --climatology.')
    satellite = SatelliteWarmStart(*fields) if all(fields) else None
    train(args.dataset, args.output, config, args.climatology, warm_start=satellite)


if __name__ == '__main__':
    main()
